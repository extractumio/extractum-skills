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

# patterns is a sibling module; loaded lazily so the alert module can be used
# in contexts where patterns.py isn't importable (test shims, early errors).
try:
    import patterns as _P  # type: ignore
except Exception:
    _P = None

def _log_path() -> Path:
    # Allow tests (and users who relocate ~/.claude) to redirect the log.
    # Logging destination only — does not change any enforcement behaviour.
    override = os.environ.get("CLAUDE_SECURITY_LOG")
    if override:
        return Path(os.path.expanduser(override))
    return Path(os.path.expanduser("~/.claude/security.log"))


LOG_PATH = _log_path()
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
# We accept either the absolute path or any path whose tail matches these
# suffixes.  `endswith` handles absolute (`/Users/x/.../scripts/tests/run_tests.py`)
# and relative (`scripts/tests/run_tests.py`) invocations alike.  The path
# itself is covered by the absolute-deny write rule in pretool_guard.py, so
# an attacker can't rename something else to that path without first
# subverting the write guard.
_TEST_RUNNER_TAILS = (
    "scripts/tests/run_tests.py",                  # plugin layout
    ".claude/hooks/security/tests/run_tests.py",   # legacy layout
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

    Accept the ancestry if any ancestor is a python process whose argv includes
    an arg ending with one of `_TEST_RUNNER_TAILS` (absolute or relative path
    to run_tests.py).  Writes to that path are already absolute-deny in
    pretool_guard, so an attacker can't rename something else to that path
    without first subverting the write guard.
    """
    for cmd in _ancestor_cmdlines():
        try:
            argv = shlex.split(cmd)
        except ValueError:
            argv = cmd.split()
        if not argv:
            continue
        head = os.path.basename(argv[0]).lower()
        if "python" not in head:
            continue
        for arg in argv[1:]:
            a = arg.lstrip("./").rstrip()
            if a == _TEST_RUNNER_PATH or arg == _TEST_RUNNER_PATH:
                return True
            if a.endswith(_TEST_RUNNER_TAILS) or arg.endswith(_TEST_RUNNER_TAILS):
                return True
    return False


QUIET = _is_test_context()


def _truncate(text: str, n: int = _MAX_MSG) -> str:
    text = (text or "").replace("\x00", "")
    if len(text) <= n:
        return text
    return text[: n - 1] + "…"


def _redact(text: str) -> str:
    """Strip secret-shaped substrings before they reach the log/UI."""
    if not text or _P is None:
        return text
    try:
        redacted, _ = _P.redact_secrets(text)
        return redacted
    except Exception:
        return text


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
        path = _log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
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

    # Secrets must never reach the log file or the UI modal.  Redact every
    # string we persist or display.  Summary stays human-readable because it's
    # generated by our own code and doesn't embed raw tool input.
    safe_details = _redact(_truncate(details, 2000))
    safe_findings = [
        {"category": f.category, "label": f.label,
         "snippet": _redact(_truncate(f.snippet, 200))}
        for f in findings
    ]
    record = {
        "ts": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "severity": sev,
        "event": event,
        "summary": summary,
        "details": safe_details,
        "session_id": session_id,
        "tool_name": tool_name,
        "findings": safe_findings,
        "pid": os.getpid(),
    }
    _log(record)

    if QUIET:
        return  # log only — skip notification, modal, speech

    title = "⚠️ Claude Security — BLOCKED" if is_block else "⚠️ Claude Security — warning"
    subtitle = event
    msg = summary
    if safe_details:
        msg += " — " + _truncate(safe_details, 200)

    _notify(title, subtitle, msg, sound=("Basso" if is_block else "Pop"))

    should_modal = modal if modal is not None else is_block
    if should_modal:
        _alert_modal(title, f"{summary}\n\n{_truncate(safe_details, 600)}")

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
            lines.append(f"  • [{f.category}] {f.label} — {_redact(f.snippet)[:120]}")
    lines.append("")
    lines.append("Per ~/.claude/CLAUDE.md you must STOP. Do not retry a variant.")
    lines.append("Surface this event to the user in chat (prefix '⚠️ SECURITY ALERT —'),")
    lines.append("explain what was attempted, and wait for explicit human instruction.")
    return "\n".join(lines)
