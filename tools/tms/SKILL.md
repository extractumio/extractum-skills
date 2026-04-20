---
name: tms
description: Connect to a remote tmux session over SSH using iTerm2's native tmux control mode (-CC), creating the session if missing and detaching any stale client. Works on macOS with iTerm2.
author: Greg Z.
author_email: info@extractum.io
author_url: https://www.linkedin.com/in/gregzem/
---

# tms — tmux over SSH with iTerm2 integration

A tiny wrapper that attaches to (or creates) a tmux session on a remote host and surfaces it through iTerm2's native tmux integration — so each remote tmux window becomes an iTerm2 tab with real scrollback, native copy/paste, and normal window controls.

## Usage

```
tms connect <host> <session>
```

Example:

```
tms connect 10.195.48.28 main
```

## What it does

Runs, on the remote host:

```
tmux -CC new-session -A -D -s <session>
```

- `-CC` — iTerm2 tmux control mode; required for native integration.
- `-A` — attach to the session if it already exists; otherwise create it.
- `-D` — detach any other client currently attached, so a new iTerm2 window can take over cleanly.

## Requirements

- iTerm2 with tmux integration enabled (default).
- SSH access to `<host>` (key-based recommended).
- `tmux` installed on the remote host.

## Install

Drop the script on your `PATH` and make it executable:

```
install -m 0755 scripts/tms ~/bin/tms
```

## Troubleshooting

- **"Cannot Attach — already attached"**: another client is still holding the session. The `-D` flag should force-detach it; if the dialog still appears, click **Force Detach Other** once.
- **Scroll doesn't work / windows linger after `exit`**: you're in a plain tmux session, not CC mode. Verify the script contains `tmux -CC` on the ssh line.
- **iTerm2 doesn't open tabs**: check *Preferences → General → tmux* — "Open tmux windows as" should be set to *Tabs in the attaching window* (or your preference).
