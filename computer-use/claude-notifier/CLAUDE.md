# Claude Code Notifier

macOS notification hooks for Claude Code running in iTerm2. Notifies when long tasks finish, permissions are needed, or input is required — only when you're not already looking at the right terminal window.

## Features

- **Window-aware**: identifies the exact iTerm2 session via `ITERM_SESSION_ID` UUID. If you have multiple iTerm2 windows, notifications only fire when *that specific* session isn't focused — not just "any iTerm2".
- **Click-to-focus**: clicking a notification activates the correct iTerm2 window and tab via AppleScript.
- **Clean text**: strips markdown, control characters, and `\n` artifacts from notification balloons. Truncates to 120 chars.
- **Grouped**: repeated notifications for the same session replace each other (no stacking).
- **Remote support**: Linux/tmux hooks emit marker strings that iTerm2 Triggers catch on the local Mac.

## Install

```
bash install.sh
```

Requires: macOS, iTerm2, `terminal-notifier` (auto-installed via Homebrew if missing), `python3`.

## Manage

```
bash install.sh --status      # check installed files and hook entries
bash install.sh --uninstall   # remove hooks and settings entries
```

## What it installs

**Files** copied to `~/.claude/hooks/`:

| File | Purpose |
|------|---------|
| `mark-start.sh` | Captures timestamp + iTerm2 session UUID on prompt submit |
| `notify.sh` | Main notifier: cleans text, checks session focus, sends notification |
| `focus-session.sh` | AppleScript helper: finds session by UUID, brings its window to front |
| `local-notify.sh` | iTerm2 Trigger handler for remote (tmux) notification pipeline |

**Hook entries** added to `~/.claude/settings.json`:

| Event | Script |
|-------|--------|
| `UserPromptSubmit` | `mark-start.sh` |
| `Stop` | `notify.sh` (fires after 2+ min tasks) |
| `PermissionRequest` | `notify.sh` (fires immediately) |
| `Elicitation` | `notify.sh` (fires immediately) |

Existing hooks and settings are never modified — the installer only appends its own entries and backs up originals with a `.pre-notifier` suffix on first install.

## Remote (Linux/tmux)

On Linux, the installer installs `remote-mark-start.sh` and `remote-notify.sh` instead. These emit `@@CLAUDE_NOTIFY|type|subtitle|message@@` markers to the tmux pane TTY. Configure an iTerm2 Trigger on your Mac to catch them:

- **Regex**: `@@CLAUDE_NOTIFY\|([^|]+)\|([^|]+)\|([^@]+)@@`
- **Action**: Run Command — `~/.claude/hooks/local-notify.sh \1 \2 \3`

## Disclaimer

This software is provided as-is, with no warranty of any kind. Use it at your own risk. The authors are not responsible for any damage, data loss, or unintended behavior resulting from its use. The installer modifies `~/.claude/settings.json` and writes files to `~/.claude/hooks/` — review the changes before running in production environments.

## License

MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
