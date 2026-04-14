# macOS Native Notifications for Claude Code

Send native macOS notification banners when Claude Code needs your attention — long-running tasks finishing, permission requests, and MCP input prompts. Notifications are suppressed when iTerm2 is already focused, and include the terminal session identity so you know exactly which session needs you.

Works both **locally on macOS** and **remotely on Linux** (Ubuntu/Debian) via iTerm2's tmux -CC integration.

## What You Get

| Event | Notification | When |
|---|---|---|
| **Stop** | Task completed with elapsed time | Only if task took > 2 minutes |
| **PermissionRequest** | Claude is blocked waiting for approval | Always |
| **Elicitation** | MCP tool needs your input | Always |

All notifications are suppressed when iTerm2 is the frontmost app. Clicking a notification brings iTerm2 to focus, even across macOS Spaces/desktops.

---

# Part 1: Local macOS Setup

For Claude Code running directly on your Mac.

## Prerequisites

- **macOS** (tested on Sequoia)
- **iTerm2**
- **Homebrew**
- **Claude Code** with hooks support
- **Python 3** (pre-installed on macOS)

## Installation

### 1. Install terminal-notifier

```bash
brew install terminal-notifier
```

### 2. Configure macOS Notification Settings

Go to **System Settings > Notifications > terminal-notifier** and set:

- **Allow Notifications**: On
- **Alert style**: **Alerts** (stays on screen until clicked; use **Banners** if you prefer auto-dismiss)

Also ensure **Do Not Disturb / Focus Mode** is off, or terminal-notifier is allowed through your Focus filter.

### 3. Create the hook scripts

```bash
mkdir -p ~/.claude/hooks
```

#### `~/.claude/hooks/mark-start.sh`

Records the timestamp and iTerm2 tab title when the user submits a prompt.

```bash
#!/bin/bash
# Writes a start timestamp when user submits a prompt
input=$(cat)
session_id=$(/usr/bin/python3 -c "import sys,json; print(json.load(sys.stdin).get('session_id',''))" <<< "$input" 2>/dev/null)
stamp="/tmp/.claude-task-start-${session_id}"

# Clean up stale files older than 24h from crashed sessions
find /tmp -name ".claude-task-start-*" -mmin +1440 -delete 2>/dev/null

# Capture timestamp and iTerm2 tab/window title
tab_title=$(osascript -e '
tell application "iTerm2"
    tell current session of current tab of current window
        return name
    end tell
end tell
' 2>/dev/null)

printf '%s\n%s' "$(date +%s)" "$tab_title" > "$stamp"
```

#### `~/.claude/hooks/notify.sh`

Unified notification handler. Parses JSON from stdin, checks whether a notification is warranted, and sends it via `terminal-notifier`.

```bash
#!/bin/bash
# Unified Claude Code notification hook
# Handles: Stop (long tasks), PermissionRequest, Elicitation

input=$(cat)

eval "$(/usr/bin/python3 -c "
import sys, json
d = json.load(sys.stdin)

event = d.get('hook_event_name', '')
cwd = d.get('cwd', '')
session_id = d.get('session_id', '')
message = d.get('last_assistant_message', '')
tool_name = d.get('tool_name', '')
error = d.get('error', '')
mcp_server = d.get('mcp_server_name', '')

def trunc(s, n=200):
    return s[:n].rsplit(' ', 1)[0] + '...' if len(s) > n else s
message = trunc(message)
error = trunc(error)

print(f'EVENT={repr(event)}')
print(f'CWD={repr(cwd)}')
print(f'SESSION_ID={repr(session_id)}')
print(f'MESSAGE={repr(message)}')
print(f'TOOL_NAME={repr(tool_name)}')
print(f'TOOL_ERROR={repr(error)}')
print(f'MCP_SERVER={repr(mcp_server)}')
" <<< "$input" 2>/dev/null)"

project=$(basename "$CWD")
terminal_notifier=$(which terminal-notifier)

# Skip notification if iTerm is the frontmost app (user is already looking at it)
frontmost=$(osascript -e 'tell application "System Events" to get name of first process whose frontmost is true' 2>/dev/null)
[ "$frontmost" = "iTerm2" ] && exit 0

case "$EVENT" in
  Stop)
    stamp="/tmp/.claude-task-start-${SESSION_ID}"
    if [ -f "$stamp" ]; then
      start_time=$(head -1 "$stamp")
      tab_title=$(tail -1 "$stamp")
      now=$(date +%s)
      elapsed=$((now - start_time))
      rm -f "$stamp"
      [ "$elapsed" -lt 120 ] && exit 0
      mins=$((elapsed / 60))
      secs=$((elapsed % 60))
      "$terminal_notifier" \
        -title "Done" \
        -subtitle "${tab_title:-$project} -- ${mins}m ${secs}s" \
        -message "${MESSAGE:-Task completed}" \
        -activate com.googlecode.iterm2
    fi
    ;;

  PermissionRequest|Elicitation)
    # Fetch current tab title live for blocking events
    tab_title=$(osascript -e '
    tell application "iTerm2"
        tell current session of current tab of current window
            return name
        end tell
    end tell
    ' 2>/dev/null)
    if [ "$EVENT" = "PermissionRequest" ]; then
      "$terminal_notifier" \
        -title "Needs Approval" \
        -subtitle "${tab_title:-$project}" \
        -message "Permission needed for: $TOOL_NAME" \
        -activate com.googlecode.iterm2
    else
      "$terminal_notifier" \
        -title "Input Needed" \
        -subtitle "${tab_title:-$project} -- ${MCP_SERVER:-MCP}" \
        -message "Tool $TOOL_NAME needs your input" \
        -activate com.googlecode.iterm2
    fi
    ;;
esac
```

Make both scripts executable:

```bash
chmod +x ~/.claude/hooks/mark-start.sh ~/.claude/hooks/notify.sh
```

### 4. Configure Claude Code hooks

Add the following `hooks` block to your `~/.claude/settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "~/.claude/hooks/mark-start.sh"
          }
        ]
      }
    ],
    "Stop": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "~/.claude/hooks/notify.sh"
          }
        ]
      }
    ],
    "PermissionRequest": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "~/.claude/hooks/notify.sh"
          }
        ]
      }
    ],
    "Elicitation": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "~/.claude/hooks/notify.sh"
          }
        ]
      }
    ]
  }
}
```

No restart required — Claude Code picks up settings changes automatically.

---

# Part 2: Remote Linux Setup (via iTerm2 + tmux -CC)

For Claude Code running on a remote Ubuntu/Debian server, accessed via iTerm2's native tmux integration (`tmux -CC`).

## How It Works

```
Remote Server (Ubuntu/Debian)              Local Mac (iTerm2)
================================           ================================
Claude Code hook fires
  |
  v
remote-notify.sh
  |
  v
Writes marker string to
tmux pane TTY:
  @@CLAUDE_NOTIFY|type|sub|msg@@
  |
  +--- flows through tmux -CC ---------->  iTerm2 Trigger catches regex
                                             |
                                             v
                                           Runs local-notify.sh
                                             |
                                             v
                                           terminal-notifier shows
                                           native macOS notification
```

The transport is plain text through the tmux session. No reverse SSH, no extra ports, no daemons.

## Prerequisites

**On the remote server (Ubuntu 22/24, Debian 12+):**
- Python 3 (`apt install python3`)
- tmux
- Claude Code with hooks support

**On your local Mac:**
- Everything from Part 1 (terminal-notifier, iTerm2)
- iTerm2 Trigger configured (see below)

## Remote Installation

### 1. Create hook scripts on the remote server

```bash
mkdir -p ~/.claude/hooks
```

#### `~/.claude/hooks/mark-start.sh`

Records the timestamp, tmux pane title, and pane TTY path when the user submits a prompt.

```bash
#!/bin/bash
# Writes a start timestamp when user submits a prompt (remote/Linux version)
# Captures tmux session:window.pane identity and pane title for notifications

input=$(cat)
session_id=$(python3 -c "import sys,json; print(json.load(sys.stdin).get('session_id',''))" <<< "$input" 2>/dev/null)
stamp="/tmp/.claude-task-start-${session_id}"

# Clean up stale files older than 24h from crashed sessions
find /tmp -maxdepth 1 -name ".claude-task-start-*" -mmin +1440 -delete 2>/dev/null

# Capture tmux pane title and identity
pane_title=""
pane_tty=""
if [ -n "$TMUX" ]; then
    pane_title=$(tmux display-message -p '#{session_name}:#{window_name}' 2>/dev/null)
    if [ -n "$TMUX_PANE" ]; then
        pane_tty=$(tmux display-message -t "$TMUX_PANE" -p '#{pane_tty}' 2>/dev/null)
    else
        pane_tty=$(tmux display-message -p '#{pane_tty}' 2>/dev/null)
    fi
fi

# Store: line 1 = timestamp, line 2 = pane title, line 3 = pane tty
printf '%s\n%s\n%s' "$(date +%s)" "$pane_title" "$pane_tty" > "$stamp"
```

#### `~/.claude/hooks/notify.sh`

Outputs a marker string to the tmux pane TTY. iTerm2 Trigger on your Mac catches it and calls `local-notify.sh`.

```bash
#!/bin/bash
# Remote Claude Code notification hook (Ubuntu/Debian + tmux -CC -> iTerm2)
#
# Outputs a marker string to the tmux pane TTY.
# iTerm2 Trigger on the local Mac catches the marker and calls local-notify.sh.
#
# Handles: Stop (long tasks), PermissionRequest, Elicitation

input=$(cat)

eval "$(python3 -c "
import sys, json
d = json.load(sys.stdin)

event = d.get('hook_event_name', '')
cwd = d.get('cwd', '')
session_id = d.get('session_id', '')
message = d.get('last_assistant_message', '')
tool_name = d.get('tool_name', '')
mcp_server = d.get('mcp_server_name', '')

def trunc(s, n=200):
    return s[:n].rsplit(' ', 1)[0] + '...' if len(s) > n else s

def clean(s):
    return trunc(s).replace('|', '/').replace('\n', ' ').replace('\r', '')

print(f'EVENT={repr(event)}')
print(f'CWD={repr(cwd)}')
print(f'SESSION_ID={repr(session_id)}')
print(f'MESSAGE={repr(clean(message))}')
print(f'TOOL_NAME={repr(tool_name)}')
print(f'MCP_SERVER={repr(mcp_server)}')
" <<< "$input" 2>/dev/null)"

project=$(basename "$CWD")
stamp="/tmp/.claude-task-start-${SESSION_ID}"

# Resolve the TTY to write the marker to
# Priority: stored TTY from mark-start, then current tmux pane, then /dev/tty
resolve_tty() {
    # Try stored TTY from mark-start (line 3 of stamp file)
    if [ -f "$stamp" ]; then
        local stored_tty
        stored_tty=$(sed -n '3p' "$stamp")
        if [ -n "$stored_tty" ] && [ -w "$stored_tty" ]; then
            echo "$stored_tty"
            return
        fi
    fi
    # Try tmux
    if [ -n "$TMUX_PANE" ]; then
        local tty
        tty=$(tmux display-message -t "$TMUX_PANE" -p '#{pane_tty}' 2>/dev/null)
        if [ -n "$tty" ] && [ -w "$tty" ]; then
            echo "$tty"
            return
        fi
    fi
    if [ -n "$TMUX" ]; then
        local tty
        tty=$(tmux display-message -p '#{pane_tty}' 2>/dev/null)
        if [ -n "$tty" ] && [ -w "$tty" ]; then
            echo "$tty"
            return
        fi
    fi
    echo "/dev/tty"
}

notify_tty=$(resolve_tty)

send_marker() {
    local type="$1" subtitle="$2" message="$3"
    printf '@@CLAUDE_NOTIFY|%s|%s|%s@@\n' "$type" "$subtitle" "$message" > "$notify_tty" 2>/dev/null
}

case "$EVENT" in
  Stop)
    if [ -f "$stamp" ]; then
      start_time=$(sed -n '1p' "$stamp")
      pane_title=$(sed -n '2p' "$stamp")
      now=$(date +%s)
      elapsed=$((now - start_time))
      rm -f "$stamp"
      [ "$elapsed" -lt 120 ] && exit 0
      mins=$((elapsed / 60))
      secs=$((elapsed % 60))
      send_marker "DONE" "${pane_title:-$project} -- ${mins}m ${secs}s" "${MESSAGE:-Task completed}"
    fi
    ;;

  PermissionRequest)
    send_marker "APPROVAL" "$project" "Permission needed for: $TOOL_NAME"
    ;;

  Elicitation)
    send_marker "INPUT" "$project -- ${MCP_SERVER:-MCP}" "Tool $TOOL_NAME needs your input"
    ;;
esac
```

Make both executable:

```bash
chmod +x ~/.claude/hooks/mark-start.sh ~/.claude/hooks/notify.sh
```

### 2. Configure Claude Code on the remote server

Add the same hooks block to the remote `~/.claude/settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "~/.claude/hooks/mark-start.sh"
          }
        ]
      }
    ],
    "Stop": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "~/.claude/hooks/notify.sh"
          }
        ]
      }
    ],
    "PermissionRequest": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "~/.claude/hooks/notify.sh"
          }
        ]
      }
    ],
    "Elicitation": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "~/.claude/hooks/notify.sh"
          }
        ]
      }
    ]
  }
}
```

### 3. Create the local receiver script on your Mac

#### `~/.claude/hooks/local-notify.sh`

Called by the iTerm2 Trigger when it catches the marker pattern. Runs `terminal-notifier` locally.

```bash
#!/bin/bash
# Called by iTerm2 Trigger when it catches @@CLAUDE_NOTIFY|...|...|...@@
# Arguments: $1 = type (DONE/APPROVAL/INPUT), $2 = subtitle, $3 = message

type="$1"
subtitle="$2"
message="$3"

# Skip notification if iTerm is the frontmost app
frontmost=$(osascript -e 'tell application "System Events" to get name of first process whose frontmost is true' 2>/dev/null)
[ "$frontmost" = "iTerm2" ] && exit 0

terminal_notifier=$(which terminal-notifier)
[ -z "$terminal_notifier" ] && terminal_notifier="/opt/homebrew/bin/terminal-notifier"

case "$type" in
  DONE)
    "$terminal_notifier" \
      -title "Claude Code -- Done" \
      -subtitle "$subtitle" \
      -message "$message" \
      -activate com.googlecode.iterm2
    ;;
  APPROVAL)
    "$terminal_notifier" \
      -title "Claude Code -- Needs Approval" \
      -subtitle "$subtitle" \
      -message "$message" \
      -activate com.googlecode.iterm2
    ;;
  INPUT)
    "$terminal_notifier" \
      -title "Claude Code -- Input Needed" \
      -subtitle "$subtitle" \
      -message "$message" \
      -activate com.googlecode.iterm2
    ;;
esac
```

```bash
chmod +x ~/.claude/hooks/local-notify.sh
```

### 4. Configure the iTerm2 Trigger

This is the bridge that catches marker text flowing through tmux and runs the local script.

1. Open **iTerm2 > Settings > Profiles** (select your profile) **> Advanced**
2. Scroll to **Triggers** and click **Edit**
3. Click **+** to add a new trigger with these values:

| Field | Value |
|---|---|
| **Regular Expression** | `@@CLAUDE_NOTIFY\|([^|]*)\|([^|]*)\|([^|]*)@@` |
| **Action** | Run Command... |
| **Parameters** | `$HOME/.claude/hooks/local-notify.sh "\1" "\2" "\3"` |
| **Instant** | Checked |
| **Enabled** | Checked |

4. Click **Close**

**Important**: The **Instant** checkbox must be checked so the trigger fires immediately when the marker text appears, without waiting for a newline or cursor movement.

---

# Reference

## Hook data available via stdin (JSON)

| Field | Available in | Description |
|---|---|---|
| `session_id` | All events | Unique session identifier |
| `cwd` | All events | Current working directory |
| `hook_event_name` | All events | Event type (Stop, PermissionRequest, etc.) |
| `last_assistant_message` | Stop | Claude's final response text |
| `tool_name` | PermissionRequest, Elicitation | Tool requesting permission/input |
| `error` | PostToolUseFailure | Error description |
| `mcp_server_name` | Elicitation | MCP server name |

## Marker format (remote only)

```
@@CLAUDE_NOTIFY|type|subtitle|message@@
```

- **type**: `DONE`, `APPROVAL`, or `INPUT`
- **subtitle**: session identity + elapsed time
- **message**: human-readable description (truncated to 200 chars, pipes and newlines stripped)

## Temp file format

**Local** (`/tmp/.claude-task-start-{session_id}`):
```
<unix_timestamp>
<iTerm2 tab title>
```

**Remote** (`/tmp/.claude-task-start-{session_id}`):
```
<unix_timestamp>
<tmux session:window name>
<pane tty path>
```

## Temp file lifecycle

- **Created**: On every `UserPromptSubmit` (overwritten per turn)
- **Deleted**: By `notify.sh` when Stop fires
- **Stale cleanup**: `mark-start.sh` deletes any files older than 24h on each run
- **Reboot**: `/tmp/` is cleared on restart

## Customization

### Change the minimum duration threshold

In `notify.sh`, change `120` (seconds) to your preference:

```bash
[ "$elapsed" -lt 120 ] && exit 0
```

### Adapt for Terminal.app instead of iTerm2

1. Replace `-activate` bundle ID:
   ```bash
   -activate com.apple.Terminal
   ```
2. Replace AppleScript for tab title:
   ```applescript
   tell application "Terminal" to return name of front window
   ```
3. Update frontmost check:
   ```bash
   [ "$frontmost" = "Terminal" ] && exit 0
   ```

### Add sound to notifications

Add `-sound Glass` to `terminal-notifier` calls:

```bash
"$terminal_notifier" \
  -title "Done" \
  -subtitle "..." \
  -message "..." \
  -sound Glass \
  -activate com.googlecode.iterm2
```

### Use Banners instead of Alerts

Change alert style in **System Settings > Notifications > terminal-notifier** from **Alerts** to **Banners** for auto-dismissing notifications.

## Testing

### Local

```bash
# Test Stop (simulates a 4-minute task)
stamp="/tmp/.claude-task-start-test1"
printf '%s\n%s' "$(( $(date +%s) - 240 ))" "my-project -- user@host" > "$stamp"
echo '{"hook_event_name":"Stop","cwd":"/tmp","session_id":"test1","last_assistant_message":"Refactored auth module and all tests pass."}' | ~/.claude/hooks/notify.sh

# Test PermissionRequest
echo '{"hook_event_name":"PermissionRequest","cwd":"/tmp","session_id":"test2","tool_name":"Bash(rm -rf /tmp/stuff)"}' | ~/.claude/hooks/notify.sh

# Test Elicitation
echo '{"hook_event_name":"Elicitation","cwd":"/tmp","session_id":"test3","tool_name":"authenticate","mcp_server_name":"github-mcp"}' | ~/.claude/hooks/notify.sh
```

If iTerm2 is focused, notifications will be suppressed by design. Switch to another app first, or temporarily comment out the frontmost check.

### Remote (test marker output)

On the remote server:

```bash
# Test that marker reaches the terminal
stamp="/tmp/.claude-task-start-test1"
printf '%s\n%s\n%s' "$(( $(date +%s) - 240 ))" "dev:claude" "/dev/pts/0" > "$stamp"
echo '{"hook_event_name":"Stop","cwd":"/home/user/project","session_id":"test1","last_assistant_message":"All tests pass."}' | ~/.claude/hooks/notify.sh
# You should see: @@CLAUDE_NOTIFY|DONE|dev:claude -- 4m 0s|All tests pass.@@
```

Then check that the iTerm2 Trigger fires and you get a macOS notification.

### Test local-notify.sh directly

```bash
~/.claude/hooks/local-notify.sh "DONE" "dev:claude -- 3m 20s" "Refactored auth module"
~/.claude/hooks/local-notify.sh "APPROVAL" "my-project" "Permission needed for: Bash(rm -rf /tmp)"
~/.claude/hooks/local-notify.sh "INPUT" "my-project -- github-mcp" "Tool authenticate needs your input"
```

## Troubleshooting

| Problem | Solution |
|---|---|
| No notification appears | Check **System Settings > Notifications > terminal-notifier** is enabled |
| Notification appears but disappears instantly | Change alert style from Banners to **Alerts** |
| No notifications while in Focus/DND mode | Allow terminal-notifier through your Focus filter |
| `terminal-notifier` not found | Run `brew install terminal-notifier` |
| Tab title shows as empty | Ensure iTerm2 is running and has an active session |
| Stale temp files accumulating | Auto-clean after 24h; or `rm /tmp/.claude-task-start-*` |
| Remote marker not reaching iTerm2 | Check `tmux display-message -p '#{pane_tty}'` returns a valid TTY |
| iTerm2 Trigger not firing | Verify **Instant** is checked and regex is correct |
| Marker text visible in terminal | Expected; it flows through quickly. Can be hidden with iTerm2 Trigger's "Instant" + scroll |

## Files Summary

| File | Location | Purpose |
|---|---|---|
| `mark-start.sh` | `~/.claude/hooks/` (local or remote) | Records turn start timestamp |
| `notify.sh` | `~/.claude/hooks/` (local) | Sends notification via terminal-notifier |
| `notify.sh` | `~/.claude/hooks/` (remote) | Outputs marker to tmux TTY |
| `local-notify.sh` | `~/.claude/hooks/` (local Mac only) | Called by iTerm2 Trigger for remote notifications |
