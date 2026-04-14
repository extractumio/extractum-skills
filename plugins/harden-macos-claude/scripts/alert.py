"""macOS alerting for security events. Stdlib only.

Three channels, best-effort and non-blocking:
  1. A blocking `display alert` modal via osascript (only for high severity)
  2. A non-blocking notification via osascript / terminal-notifier
  3. Append to ~/.claude/security.log (always)

Callers can also pass `speak=True` to trigger `say` for an audible cue.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Iterable

LOG_PATH = Path(os.path.expanduser("~/.claude/security.log"))
_MAX_MSG = 500

# Plugin layout: scripts/ lives under the plugin root.  CLAUDE_PLUGIN_ROOT is
# set by Claude Code when invoking hooks; fall back to __file__ for the test
# runner case (and for users running the scripts out-of-plugin).
_PLUGIN_ROOT = Path(
    os.environ.get("CLAUDE_PLUGIN_ROOT")
    or str(Path(__file__).resolve().parent.parent)
).resolve()
_SCRIPTS_DIR = _PLUGIN_ROOT / "scripts"

# Quiet mode: UI channels (notification, modal, say) are suppressed when this
# hook process is a child of the real test runner at its pinned path.  An
# attacker cannot forge the caller identity — they'd have to become our parent
# process — and the test runner path itself is covered by the absolute-deny
# write rule in pretool_guard.py.  No env var is trusted here.
_TEST_RUNNER_PATH = str(_SCRIPTS_DIR / "tests" / "run_tests.py")
_TEST_RUNNER_SUFFIXES = (
    "/scripts/tests/run_tests.py",              # plugin layout
    "/.claude/hooks/security/tests/run_tests.py",  # legacy layout
)


def _ancestor_cmdlines(max_depth: int = 4) -> list[str]:
    """Walk up the process tree, returning each ancestor's cmdline.

    macOS `ps` sometimes omits the script path for Python subprocesses (e.g.
    when invoked via `python3 -c ...`), so we check several ancestors rather
    than just the immediate parent.
    """
    out: list[str] = []
    try:
        pid = os.getppid()
        for _ in range(max_depth):
            if pid in (0, 1):
                break
            r = subprocess.run(
                ["/bin/ps", "-p", str(pid), "-o", "ppid=,args="],
                capture_output=True, text=True, timeout=1,
            )
            line = (r.stdout or "").strip()
            if not line:
                break
            parts = line.split(None, 1)
            if len(parts) < 2:
                break
            try:
                pid = int(parts[0])
            except ValueError:
                break
            out.append(parts[1])
    except Exception:
        pass
    return out


def _is_test_context() -> bool:
    """Quiet iff the hook is running under the pinned test runner.

    We accept the ancestry if any ancestor is a python process whose argv
    contains the pinned test-runner path.  The path itself is covered by the
    absolute-deny write rule in pretool_guard.py, so an attacker can't rename
    something else to that path without first subverting the write guard.
    """
    for cmd in _ancestor_cmdlines():
        if not (
            _TEST_RUNNER_PATH in cmd
            or any(suf in cmd for suf in _TEST_RUNNER_SUFFIXES)
        ):
            continue
        try:
            head = shlex.split(cmd)[0] if cmd else ""
        except ValueError:
            head = cmd.split(None, 1)[0] if cmd else ""
        if "python" in os.path.basename(head).lower():
            return True
    return False


QUIET = _is_test_context()


def _truncate(text: str, n: int = _MAX_MSG) -> str:
    text = (text or "").replace("\x00", "")
    if len(text) <= n:
        return text
    return text[: n - 1] + "…"


def _osascript(script: str) -> None:
    try:
        subprocess.Popen(
            ["/usr/bin/osascript", "-e", script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
    except Exception:
        pass


def _notify(title: str, subtitle: str, message: str, sound: str = "Basso") -> None:
    """Non-blocking banner notification."""
    t = _truncate(title, 120).replace('"', "'")
    s = _truncate(subtitle, 120).replace('"', "'")
    m = _truncate(message, 300).replace('"', "'")
    _osascript(
        f'display notification "{m}" with title "{t}" subtitle "{s}" sound name "{sound}"'
    )
    # If terminal-notifier is installed, also fire it (more visible persistent banner).
    tn = "/opt/homebrew/bin/terminal-notifier"
    if not os.path.exists(tn):
        tn = "/usr/local/bin/terminal-notifier"
    if os.path.exists(tn):
        try:
            subprocess.Popen(
                [
                    tn,
                    "-title", t,
                    "-subtitle", s,
                    "-message", m,
                    "-sound", sound,
                    "-group", "claude-security",
                    "-ignoreDnD",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
            )
        except Exception:
            pass


def _alert_modal(title: str, message: str) -> None:
    """Blocking critical alert modal (forces user acknowledgement)."""
    t = _truncate(title, 120).replace('"', "'")
    m = _truncate(message, 800).replace('"', "'")
    script = (
        f'tell application "System Events" to '
        f'display alert "{t}" message "{m}" as critical '
        f'buttons {{"Dismiss"}} default button "Dismiss"'
    )
    _osascript(script)


def _say(phrase: str) -> None:
    try:
        subprocess.Popen(
            ["/usr/bin/say", _truncate(phrase, 120)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
    except Exception:
        pass


def _log(record: dict) -> None:
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


def raise_alert(
    *,
    severity: str,                     # "block" | "warn"
    event: str,                        # short machine tag e.g. "exfil.bash.curl-env"
    summary: str,                      # one-line human description
    details: str = "",                 # longer description / snippet
    session_id: str | None = None,
    tool_name: str | None = None,
    findings: Iterable = (),           # iterable of patterns.Finding
    speak: bool = False,
    modal: bool | None = None,         # override default (default: modal when "block")
) -> None:
    """Fire notification, optional modal, optional speech, always log."""
    sev = (severity or "warn").lower()
    is_block = sev == "block"

    record = {
        "ts": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "severity": sev,
        "event": event,
        "summary": summary,
        "details": _truncate(details, 2000),
        "session_id": session_id,
        "tool_name": tool_name,
        "findings": [
            {"category": f.category, "label": f.label, "snippet": _truncate(f.snippet, 200)}
            for f in findings
        ],
        "pid": os.getpid(),
    }
    _log(record)

    if QUIET:
        return  # log only — skip notification, modal, speech

    title = "⚠️ Claude Security — BLOCKED" if is_block else "⚠️ Claude Security — warning"
    subtitle = event
    msg = summary
    if details:
        msg += " — " + _truncate(details, 200)

    _notify(title, subtitle, msg, sound=("Basso" if is_block else "Pop"))

    should_modal = modal if modal is not None else is_block
    if should_modal:
        _alert_modal(title, f"{summary}\n\n{_truncate(details, 600)}")

    if speak or is_block:
        _say("Security alert. Claude blocked a suspicious action.")


def format_reason(event: str, summary: str, findings: Iterable) -> str:
    """Produce the stderr message that Claude receives on block (exit 2).

    The wording is deliberately instructive so the model understands what to do
    next instead of retrying around the hook.
    """
    fs = list(findings)
    lines = [
        "⚠️ SECURITY HOOK BLOCKED this tool call.",
        f"Event: {event}",
        f"Reason: {summary}",
    ]
    if fs:
        lines.append("Detections:")
        for f in fs[:6]:
            lines.append(f"  • [{f.category}] {f.label} — {f.snippet[:120]}")
    lines.append("")
    lines.append("Per ~/.claude/CLAUDE.md you must STOP. Do not retry a variant.")
    lines.append("Surface this event to the user in chat (prefix '⚠️ SECURITY ALERT —'),")
    lines.append("explain what was attempted, and wait for explicit human instruction.")
    return "\n".join(lines)
