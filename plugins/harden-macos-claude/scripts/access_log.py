"""Audit logger for sensitive-source access.

Complements alert.raise_alert (which only fires on BLOCK/WARN). This module
records every tool call that TOUCHES a §2 sensitive source even if the call
is allowed through — the goal is a durable audit trail in ~/.claude/security.log
that a human can grep later.

What gets logged:
  - Read/Edit/Write/MultiEdit/NotebookEdit targeting a sensitive path
    (private key, cloud creds, .env*, keychain, .claude/*, shell history,
    browser profile, Mail/Messages, etc.)
  - Bash commands that reference a sensitive path or match env-dump / keychain
    dump shapes
  - WebFetch whose URL or prompt embeds a secret or sensitive path
  - Any tool_input blob that matches a high-entropy secret fingerprint

What gets stripped before log write:
  - Every substring matching a SECRET_SIGNATURES regex → ▇▇▇REDACTED-<label>▇▇▇
  - Entire payload is truncated to ~1KB after redaction

The logger is stdlib-only, best-effort, and never raises. If the log path is
unwritable we drop the record silently — audit is secondary to the main tool
call.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path
from typing import Any, Iterable

import patterns as P  # type: ignore  # noqa: E402


def _log_path() -> Path:
    # Allow tests (and users who relocate ~/.claude) to redirect the log.
    # This only changes WHERE we write — it cannot disable logging nor weaken
    # any enforcement rule, so honouring the env var is safe.
    override = os.environ.get("CLAUDE_SECURITY_LOG")
    if override:
        return Path(os.path.expanduser(override))
    return Path(os.path.expanduser("~/.claude/security.log"))


LOG_PATH = _log_path()
_MAX_DETAILS = 1000


def _flatten(value: Any, out: list[str]) -> None:
    if value is None:
        return
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, (int, float, bool)):
        out.append(str(value))
    elif isinstance(value, list):
        for v in value:
            _flatten(v, out)
    elif isinstance(value, dict):
        for v in value.values():
            _flatten(v, out)


def _payload_text(tool_input: dict | None) -> str:
    parts: list[str] = []
    _flatten(tool_input or {}, parts)
    return "\n".join(parts)


def _redact(blob: str) -> str:
    """Replace secret-shaped substrings with REDACTED placeholders."""
    if not blob:
        return blob
    redacted, _ = P.redact_secrets(blob)
    if len(redacted) > _MAX_DETAILS:
        redacted = redacted[: _MAX_DETAILS - 1] + "…"
    return redacted


def _tool_target_path(tool_name: str, tool_input: dict) -> str:
    if tool_name in ("Read", "Write", "Edit", "MultiEdit"):
        return tool_input.get("file_path") or ""
    if tool_name == "NotebookEdit":
        return tool_input.get("notebook_path") or ""
    if tool_name == "Bash":
        return tool_input.get("command") or ""
    if tool_name in ("WebFetch", "WebSearch"):
        return (tool_input.get("url") or "") + " " + (tool_input.get("prompt") or tool_input.get("query") or "")
    return ""


def _classify(tool_name: str, tool_input: dict | None) -> tuple[str, list[P.Finding]]:
    """Return (event_tag, findings) for any §2 match, or ("", [])."""
    tool_input = tool_input or {}
    target = _tool_target_path(tool_name, tool_input)
    blob = _payload_text(tool_input)

    privkey = P.find_private_key_paths(target) or P.find_private_key_paths(blob)
    if privkey:
        return "access.private_key_path", privkey

    sensitive = P.find_sensitive_paths(target) or P.find_sensitive_paths(blob)
    if sensitive:
        return "access.sensitive_path", sensitive

    env_dump = P.find_env_dump(blob)
    if env_dump:
        return "access.env_dump", env_dump

    secret_hits = P.find_secrets(blob)
    if secret_hits:
        return "access.secret_shape", secret_hits

    return "", []


def _write(record: dict) -> None:
    try:
        path = _log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:
        # Best-effort; never fail a tool call because logging failed.
        pass


def log_if_sensitive(
    *,
    tool_name: str,
    tool_input: dict | None,
    session_id: str | None = None,
    phase: str = "pre",                        # "pre" | "post"
    decision: str = "allow",                   # "allow" | "block" | "warn"
) -> bool:
    """Log a sensitive-source access event. Returns True if a record was written."""
    event, findings = _classify(tool_name, tool_input)
    if not event:
        return False

    blob = _payload_text(tool_input)
    record = {
        "ts": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "severity": "info",
        "phase": phase,
        "decision": decision,
        "event": event,
        "summary": _event_summary(event, tool_name, _tool_target_path(tool_name, tool_input or {})),
        "tool_name": tool_name,
        "session_id": session_id,
        "details": _redact(blob),
        "findings": [
            {"category": f.category, "label": f.label, "snippet": _redact(f.snippet)[:200]}
            for f in findings[:6]
        ],
        "pid": os.getpid(),
    }
    _write(record)
    return True


def _event_summary(event: str, tool_name: str, target: str) -> str:
    # The target string can embed a raw prompt / query / shell cmd, any of
    # which may contain a secret.  Redact before composing the summary.
    tgt_raw = (target or "").strip().splitlines()[0] if target else ""
    tgt = _redact(tgt_raw) if tgt_raw else ""
    if len(tgt) > 160:
        tgt = tgt[:159] + "…"
    labels = {
        "access.private_key_path": "private-key / credentials path referenced",
        "access.sensitive_path":   "sensitive §2 path referenced",
        "access.env_dump":         "environment / keychain dump pattern",
        "access.secret_shape":     "secret-shaped token in payload",
    }
    human = labels.get(event, event)
    if tgt:
        return f"{tool_name}: {human} ({tgt})"
    return f"{tool_name}: {human}"
