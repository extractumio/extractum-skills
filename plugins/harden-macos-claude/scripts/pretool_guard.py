#!/usr/bin/env python3
"""PreToolUse security guard — stops secret exfiltration before it happens.

Reads a single JSON hook payload on stdin and exits:
  0 — allow
  2 — BLOCK (stderr becomes the reason visible to Claude)

Design goals:
  * Never break legitimate workflows silently: every block raises a macOS
    alert + log entry via alert.raise_alert.
  * Conservative on local reads; aggressive on combined "sensitive + outbound".
  * Handles Bash, Write/Edit/MultiEdit/NotebookEdit, WebFetch/WebSearch, and
    outbound-shaped MCP tools (gmail send, slack post, webhook, upload, …).
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

# Make sibling modules importable regardless of how the hook is launched.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Plugin root: CLAUDE_PLUGIN_ROOT is set by Claude Code's hook runner; for the
# test suite and out-of-plugin invocations we derive it from __file__.
_PLUGIN_ROOT = Path(os.environ.get("CLAUDE_PLUGIN_ROOT") or str(_HERE.parent)).resolve()
_SCRIPTS_DIR = _PLUGIN_ROOT / "scripts"

import patterns as P        # noqa: E402
import alert as A           # noqa: E402
import access_log as AL     # noqa: E402


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

def _read_payload() -> dict:
    raw = ""
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return {}
        return json.loads(raw)
    except Exception as exc:
        # Fail-closed: a crafted payload that parses differently in Claude vs
        # here could slip a block.  If we got non-empty stdin and can't parse
        # it, BLOCK rather than allow.
        A.raise_alert(
            severity="block",
            event="selfdefense.bad_payload",
            summary="pretool_guard received malformed JSON on stdin; failing closed.",
            details=(raw[:400] if isinstance(raw, str) else str(raw)[:400]),
            session_id=None,
            tool_name="",
            findings=[],
        )
        sys.stderr.write(f"[security] pretool_guard: bad payload ({exc}) — BLOCKED (fail-closed)\n")
        sys.exit(2)


def _tool_input_text(tool_input: dict) -> str:
    """Flatten tool_input into a scannable blob."""
    try:
        return json.dumps(tool_input, ensure_ascii=False)
    except Exception:
        return str(tool_input)


# ---------------------------------------------------------------------------
# Per-tool evaluators — each returns (severity, event, summary, findings)
# severity: "block" | "warn" | "ok"
# ---------------------------------------------------------------------------

def _eval_bash(cmd: str) -> tuple[str, str, str, list[P.Finding]]:
    if not cmd:
        return "ok", "", "", []

    findings: list[P.Finding] = []

    net = P.find_network_send(cmd)
    envdump = P.find_env_dump(cmd)
    privpath = P.find_private_key_paths(cmd)
    senspath = P.find_sensitive_paths(cmd)
    remote_exec = P.find_remote_exec(cmd)
    secrets = P.find_secrets(cmd)

    # 1a. curl|sh / wget|bash / eval $(curl …) — always block.
    #     Piping downloaded content into a shell is unsafe regardless of
    #     domain (CDN compromise / TLS MITM turn it into RCE).
    remote_exec_shell = P.find_remote_exec_shell(cmd)
    if remote_exec_shell:
        findings += remote_exec_shell
        return ("block", "exfil.remote_exec",
                "Remote-fetch-then-shell pattern detected (curl|sh, eval $(curl…), bash <(curl…)).",
                findings)

    # 1b. curl|python / curl|node / curl|perl / curl|ruby — block UNLESS
    #     every URL in the command targets an allow-listed host AND no
    #     sensitive source or secret is present.  The allow-list lives at
    #     ~/.claude/hooks/security/allowlist.txt.
    remote_exec_interp = P.find_remote_exec_interp(cmd)
    if remote_exec_interp:
        sensitive_here = privpath or senspath or envdump or secrets
        if P.all_hosts_allowlisted(cmd) and not sensitive_here:
            # Allow-listed data-processing pipeline.  Emit a warn so it shows
            # up in the log, but let the command run.
            return ("warn", "remote_exec_interp.allowlisted",
                    "curl|interpreter pipeline to allow-listed host — permitted.",
                    remote_exec_interp)
        findings += remote_exec_interp
        return ("block", "exfil.remote_exec",
                "Remote fetch piped into an interpreter (python/node/perl/ruby) from a non-allow-listed host or with sensitive source.",
                findings)

    # 2. Any known paste/webhook/tunnel domain is an exfil target by nature
    for sub in P.EXFIL_DOMAIN_SUBSTRINGS:
        if sub in cmd:
            findings.append(P.Finding("network-send", f"exfil domain ({sub})", sub))
            return ("block", "exfil.known_domain",
                    f"Outbound request to a paste/webhook/tunnel service ({sub}).",
                    findings)

    # 3. env / printenv / keychain dump + any outbound step → block
    if envdump and net:
        findings += envdump + net
        return ("block", "exfil.env_dump",
                "Environment / keychain dump combined with an outbound network command.",
                findings)

    # 4. Private-key or cloud-credential path + outbound → block
    if privpath and net:
        findings += privpath + net
        return ("block", "exfil.private_key",
                "SSH / cloud / keychain file is being piped into a network command.",
                findings)

    # 5. Any other sensitive file path + outbound → block
    if senspath and net:
        findings += senspath + net
        return ("block", "exfil.sensitive_path",
                "Sensitive file path combined with an outbound network command.",
                findings)

    # 6. A literal secret string appears in the command + outbound → block
    if secrets and (net or any(sub in cmd for sub in P.EXFIL_DOMAIN_SUBSTRINGS)):
        findings += secrets + net
        return ("block", "exfil.literal_secret",
                "A high-entropy secret appears in the command and an outbound step is present.",
                findings)

    # 7. DNS-based exfil (long subdomain to odd resolver) — block
    for rx in P.DNS_EXFIL_RES:
        m = rx.search(cmd)
        if m:
            findings.append(P.Finding("network-send", "dns exfil", cmd[max(0, m.start()-10):m.end()+10]))
            return ("block", "exfil.dns",
                    "DNS-based exfiltration pattern detected.",
                    findings)

    # 7b. Symlink-from-sensitive: ln -s ~/.ssh/id_rsa /tmp/k
    m = P.SYMLINK_SENSITIVE_RES.search(cmd)
    if m:
        return ("block", "exfil.symlink_sensitive",
                "ln -s from a sensitive source (SSH / AWS / GnuPG / Keychain).",
                [P.Finding("sensitive-path", "symlink-from-sensitive",
                           cmd[max(0, m.start()-10):m.end()+10])])

    # 7c. Direct $VAR_KEY/TOKEN/SECRET dereference + outbound → block
    env_refs: list[P.Finding] = []
    for rx in P.ENV_REF_RES:
        m = rx.search(cmd)
        if m:
            env_refs.append(P.Finding("env-dump", "sensitive env-var ref",
                                      cmd[max(0, m.start()-10):m.end()+10]))
            break
    if env_refs and net:
        return ("block", "exfil.env_ref",
                "Direct dereference of a secret-named environment variable combined with an outbound step.",
                env_refs + net)

    # 7d. Persistence / autostart command + (network or sensitive ref) → block
    pers_cmd: list[P.Finding] = []
    for rx in P.PERSISTENCE_CMD_RES:
        m = rx.search(cmd)
        if m:
            pers_cmd.append(P.Finding("persistence", "autostart/background cmd",
                                      cmd[max(0, m.start()-10):m.end()+10]))
            break
    if pers_cmd and (net or senspath or privpath or secrets or env_refs):
        return ("block", "persistence.cmd",
                "Persistence / background-execution command combined with a sensitive or outbound step.",
                pers_cmd + net + senspath + privpath + secrets + env_refs)

    # 7e. Writing into a cloud-sync folder via shell redirect → block if it
    # involves sensitive content.  (The Write/Edit evaluator handles the
    # tool-native path.)
    for rx in P.SYNC_DIR_RES:
        if rx.search(cmd) and (privpath or senspath or secrets or envdump or env_refs):
            return ("block", "exfil.sync_folder",
                    "Redirect of sensitive content into a cloud-sync folder (iCloud/Dropbox/OneDrive/…).",
                    [P.Finding("sync-folder", "sync-dir redirect", cmd[:200])] + secrets + privpath)

    # 8. Warn-only: local read of a private key / credentials file with no send step
    if privpath and not net:
        return ("warn", "local.private_key_read",
                "Local read of a private-key / credentials file (no network step).",
                privpath)

    # 9. Warn-only: bare `env` / `printenv` with no send step
    if envdump and not net:
        return ("warn", "local.env_dump",
                "Environment dump with no network step — verify intent.",
                envdump)

    # 10. Attempt to disable / edit the security hooks or the charter.
    #     Only triggers on *write/delete/replace* operations targeting the
    #     protected paths — plain reads, greps, lists, and backups elsewhere
    #     are fine.
    # Protected targets in bash commands:
    #   * user-level CLAUDE.md / settings.json / settings.local.json
    #   * the plugin's own scripts dir (wherever the plugin is installed)
    #   * legacy ~/.claude/hooks/security/ (for pre-plugin installs)
    _t_user   = r"\.claude/(?:CLAUDE\.md\b|settings\.json\b|settings\.local\.json\b)"
    _t_legacy = r"\.claude/hooks/security\b"
    _t_plugin = re.escape(str(_SCRIPTS_DIR))
    _target = rf"(?:{_t_user}|{_t_legacy}|{_t_plugin})"
    _tamper_res = [
        re.compile(r"(?:^|[\s|;&])(?:>{1,2})\s*[\"'`]?\S*" + _target),            # > / >> target
        re.compile(r"\btee\b(?:\s+-[\w-]+)?\s+[^|]*?" + _target),                 # tee target
        re.compile(r"\b(?:rm|rmdir|chmod|chown|chflags)\b[^\n]*?" + _target),     # rm/chmod target
        re.compile(r"\bsed\s+-i[^\n]*?" + _target),                                # sed -i target
        re.compile(r"\b(?:echo|printf|cat)\b[^|\n]*?>\s*\S*" + _target),           # echo/printf > target
        re.compile(r"\b(?:mv|cp|install|ln)\b[^\n]*?\S+\s+\S*" + _target),         # mv/cp SRC DEST(target)
        re.compile(r"\btruncate\b[^\n]*?" + _target),                              # truncate target
    ]
    for rx in _tamper_res:
        m = rx.search(cmd)
        if m:
            return ("block", "selfdefense.tamper",
                    "Attempt to modify or remove the security hooks / global CLAUDE.md.",
                    [P.Finding("selfdefense", "hook/charter tamper", cmd[max(0, m.start()-20):m.end()+20])])

    return "ok", "", "", []


_SCRIPTS_DIR_ESC = re.escape(str(_SCRIPTS_DIR))
_HARDEN_PROTECTED_WRITE_RES = [
    # Plugin scripts dir (except the tests/ subtree).
    re.compile(_SCRIPTS_DIR_ESC + r"/(?!tests/)"),
    # The test runner itself IS protected (tampering it could fake quiet-mode).
    re.compile(_SCRIPTS_DIR_ESC + r"/tests/run_tests\.py$"),
    # Legacy layout — kept so pre-plugin installs remain covered.
    re.compile(r"/\.claude/hooks/security/(?!tests/)"),
    re.compile(r"/\.claude/hooks/security/tests/run_tests\.py$"),
    # User-level security charter and settings.
    re.compile(r"/\.claude/CLAUDE\.md$"),
    re.compile(r"/\.claude/settings(?:\.local)?\.json$"),
]

_HARDEN_MARKER = Path(os.path.expanduser("~/.claude/.harden-skill-active"))


def _skill_active() -> bool:
    """True when the user has explicitly unlocked the hardening plugin in this
    session (via the marker file or env var).  Required to edit protected
    paths (hook scripts, CLAUDE.md, settings.json)."""
    if os.environ.get("HARDEN_MACOS_CLAUDE_ACTIVE") == "1":
        return True
    try:
        return _HARDEN_MARKER.exists()
    except Exception:
        return False


def _eval_write(path: str, content: str) -> tuple[str, str, str, list[P.Finding]]:
    # Absolute-deny tier — any Write/Edit of these paths.  CLAUDE.md-weakening
    # content gets a more specific event below; everything else (including
    # wholesale rewrites) is simply refused.  Reads and backups elsewhere
    # remain fine; this only fires on in-place Write/Edit.
    for rx in _HARDEN_PROTECTED_WRITE_RES:
        if rx.search(path):
            if _skill_active():
                break  # skill-scoped edits allowed; fall through to content checks
            # Let the charter-weaken check below produce a more specific event
            # if the content is obviously malicious; otherwise hard-deny.
            if not (
                path.endswith("/.claude/CLAUDE.md")
                and re.search(r"(?i)(?:ignore|disable|remove|delete|override|bypass).{0,30}(?:security|hook|charter|sentinel)", content or "")
            ):
                return ("block", "selfdefense.protected_write",
                        f"Write/Edit of protected security path is refused: {path}",
                        [P.Finding("selfdefense", "protected path write", path)])

    if not content:
        return "ok", "", "", []

    findings: list[P.Finding] = []
    secrets = P.find_secrets(content)
    injections = P.find_injection(content)

    # Writing content that contains a secret is risky when the destination is a
    # staging / public / upload-shaped path, or outside the user's own home.
    looks_like_staging = bool(re.search(
        r"(?:^|/)(?:/tmp/|/var/tmp/|/Volumes/|.*/(?:public|share|shared|upload|uploads|drop|outbox|Desktop)/.*(?:leak|exfil|send|upload|debug|share)|.*\.(?:public|share)\.)",
        path, re.IGNORECASE,
    ))
    outside_home = not path.startswith(os.path.expanduser("~")) and not path.startswith("./") and not path.startswith("../") and path.startswith("/")

    if secrets and (looks_like_staging or outside_home):
        return ("block", "exfil.write_secret_to_staging",
                f"Attempt to write a secret to a staging / out-of-home path: {path}",
                secrets)

    # Always block writing a secret into an attachment-shaped filename
    if secrets and re.search(r"(?:attach(?:ment)?|outgoing|mail|slack|discord|telegram|webhook|gist|pastebin)", path, re.IGNORECASE):
        return ("block", "exfil.write_secret_to_outbound_shape",
                f"Secret being written to a file whose name suggests outbound delivery: {path}",
                secrets)

    # Block writing a secret into a cloud-sync folder (iCloud/Dropbox/etc.) —
    # that's exfil via the sync client, even if the path is under $HOME.
    if secrets and any(rx.search(path) for rx in P.SYNC_DIR_RES):
        return ("block", "exfil.write_secret_to_sync_folder",
                f"Secret being written into a cloud-sync folder: {path}",
                secrets)

    # Block writes into autostart/persistence surfaces.  These keep a payload
    # running after the session; never legitimate from an AI agent mid-task.
    if any(rx.search(path) for rx in P.PERSISTENCE_PATH_RES):
        return ("block", "persistence.write",
                f"Write/Edit to a persistence / autostart surface: {path}",
                [P.Finding("persistence", "autostart surface", path)])

    # Editing the global CLAUDE.md is allowed only through the skill; flag suspicious edits.
    if path.endswith("/.claude/CLAUDE.md") or path.endswith("/.claude/settings.json") or path.endswith("/.claude/settings.local.json"):
        if re.search(r"(?i)(?:ignore|disable|remove|delete|override|bypass).{0,30}(?:security|hook|charter|sentinel)", content):
            return ("block", "selfdefense.charter_weaken",
                    f"Edit to {path} attempts to weaken security charter.",
                    [P.Finding("selfdefense", "charter weaken", content[:160])])

    if secrets:
        findings += secrets
    if injections:
        findings += injections
    if findings:
        return ("warn", "write.contains_secret_or_injection",
                f"Write to {path} contains secret or injection-shaped content.",
                findings)

    return "ok", "", "", []


def _eval_webfetch(url: str, prompt: str) -> tuple[str, str, str, list[P.Finding]]:
    if not url:
        return "ok", "", "", []

    for sub in P.EXFIL_DOMAIN_SUBSTRINGS:
        if sub in url:
            return ("block", "exfil.webfetch_domain",
                    f"WebFetch target is a paste/webhook/tunnel service: {sub}",
                    [P.Finding("network-send", f"exfil domain ({sub})", url[:200])])

    secrets_in_url = P.find_secrets(url)
    if secrets_in_url:
        return ("block", "exfil.webfetch_secret_in_url",
                "WebFetch URL contains a high-entropy secret (likely exfil via query string).",
                secrets_in_url)

    # Prompt that tells Claude to read sensitive files & embed them in request
    if prompt:
        if P.find_private_key_paths(prompt) or any(rx.search(prompt) for rx in P.ENV_DUMP_RES):
            return ("block", "exfil.webfetch_prompt_wants_secrets",
                    "WebFetch prompt instructs Claude to include local secrets.",
                    P.find_private_key_paths(prompt) or [P.Finding("env-dump", "env dump prompt", prompt[:160])])

    return "ok", "", "", []


def _eval_mcp_outbound(tool_name: str, blob: str) -> tuple[str, str, str, list[P.Finding]]:
    """MCP tools whose name suggests outbound delivery (email/slack/post/upload)."""
    outbound_hint = re.compile(
        r"(?:send|post|publish|create.*(?:message|email|issue|comment|gist|paste)|upload|share|attach|mail|tweet|dm)",
        re.IGNORECASE,
    )
    if not outbound_hint.search(tool_name):
        return "ok", "", "", []

    secrets = P.find_secrets(blob)
    privpath = P.find_private_key_paths(blob)
    envhit = any(rx.search(blob) for rx in P.ENV_DUMP_RES)

    if secrets:
        return ("block", "exfil.mcp_outbound_secret",
                f"Outbound MCP tool '{tool_name}' payload contains a secret.",
                secrets)
    if privpath:
        return ("block", "exfil.mcp_outbound_private_key_ref",
                f"Outbound MCP tool '{tool_name}' references a private-key / credentials path.",
                privpath)
    if envhit:
        return ("block", "exfil.mcp_outbound_env_dump",
                f"Outbound MCP tool '{tool_name}' is carrying an env/keychain dump.",
                [P.Finding("env-dump", "env dump in mcp payload", blob[:160])])
    return "ok", "", "", []


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def evaluate(tool_name: str, tool_input: dict) -> tuple[str, str, str, list[P.Finding]]:
    tn = tool_name or ""
    ti = tool_input or {}

    if tn == "Bash":
        return _eval_bash(ti.get("command") or "")

    if tn in ("Write",):
        return _eval_write(ti.get("file_path") or "", ti.get("content") or "")
    if tn in ("Edit",):
        merged = (ti.get("new_string") or "") + "\n" + (ti.get("old_string") or "")
        return _eval_write(ti.get("file_path") or "", merged)
    if tn in ("MultiEdit",):
        path = ti.get("file_path") or ""
        merged = "\n".join(
            (e.get("new_string") or "") + "\n" + (e.get("old_string") or "")
            for e in (ti.get("edits") or [])
        )
        return _eval_write(path, merged)
    if tn in ("NotebookEdit",):
        return _eval_write(ti.get("notebook_path") or "", ti.get("new_source") or "")

    if tn in ("WebFetch",):
        return _eval_webfetch(ti.get("url") or "", ti.get("prompt") or "")
    if tn in ("WebSearch",):
        q = ti.get("query") or ""
        if P.find_secrets(q):
            return ("block", "exfil.websearch_secret",
                    "WebSearch query contains a high-entropy secret.",
                    P.find_secrets(q))
        return "ok", "", "", []

    # Heuristic: MCP / third-party outbound
    return _eval_mcp_outbound(tn, _tool_input_text(ti))


def main() -> int:
    payload = _read_payload()
    tool_name = payload.get("tool_name") or ""
    tool_input = payload.get("tool_input") or {}
    session_id = payload.get("session_id")

    severity, event, summary, findings = evaluate(tool_name, tool_input)

    # Audit log: "allow" reads/writes of a §2 path (e.g. Read ~/.ssh/id_rsa.pub,
    # Edit on ~/.zshrc) would otherwise leave no trace because raise_alert only
    # fires on block/warn. Only log here when the call was allowed through — a
    # block/warn will already get a richer record via raise_alert below.
    if severity == "ok":
        try:
            AL.log_if_sensitive(
                tool_name=tool_name, tool_input=tool_input,
                session_id=session_id, phase="pre", decision="allow",
            )
        except Exception:
            pass

    if severity == "block":
        A.raise_alert(
            severity="block",
            event=event,
            summary=summary,
            details=_tool_input_text(tool_input)[:600],
            session_id=session_id,
            tool_name=tool_name,
            findings=findings,
        )
        sys.stderr.write(A.format_reason(event, summary, findings) + "\n")
        return 2

    if severity == "warn":
        A.raise_alert(
            severity="warn",
            event=event,
            summary=summary,
            details=_tool_input_text(tool_input)[:400],
            session_id=session_id,
            tool_name=tool_name,
            findings=findings,
        )
        # Warnings don't block — emit note to stderr for log visibility but exit 0.
        sys.stderr.write(f"[security] warn: {event} — {summary}\n")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
