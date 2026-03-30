# macOS Native Notifications for Claude Code

Send native macOS notification banners when Claude Code needs your attention — long-running tasks finishing, permission requests, and MCP input prompts. Notifications are suppressed when iTerm2 is already focused, and include the iTerm tab title so you know exactly which session needs you.

## What You Get

| Event | Notification | When |
|---|---|---|
| **Stop** | Task completed with elapsed time | Only if task took > 2 minutes |
| **PermissionRequest** | Claude is blocked waiting for approval | Always |
| **Elicitation** | MCP tool needs your input | Always |

All notifications are suppressed when iTerm2 is the frontmost app (you're already looking at it). Clicking a notification brings iTerm2 to focus, even across macOS Spaces/desktops.

## Prerequisites

- **macOS** (tested on Sequoia)
- **iTerm2** (for tab title detection and click-to-focus; adaptable to Terminal.app)
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

Create the directory:

```bash
mkdir -p ~/.claude/hooks
```

#### `~/.claude/hooks/mark-start.sh`

Records the timestamp and iTerm2 tab title when the user submits a prompt. This is used later by the Stop hook to calculate elapsed time and identify the session.

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

Unified notification handler for all hook events. Parses the JSON payload from stdin, checks whether a notification is warranted, and sends it via `terminal-notifier`.

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

## How It Works

### Architecture

```
UserPromptSubmit  -->  mark-start.sh  -->  /tmp/.claude-task-start-{session_id}
                                            (stores: unix timestamp + iTerm tab title)

Stop / PermissionRequest / Elicitation  -->  notify.sh
                                              1. Parse JSON from stdin
                                              2. Check if iTerm2 is focused -> skip if yes
                                              3. For Stop: check elapsed time -> skip if < 2 min
                                              4. Send notification via terminal-notifier
```

### Hook data available via stdin (JSON)

| Field | Available in | Description |
|---|---|---|
| `session_id` | All events | Unique session identifier |
| `cwd` | All events | Current working directory |
| `hook_event_name` | All events | Event type (Stop, PermissionRequest, etc.) |
| `last_assistant_message` | Stop | Claude's final response text |
| `tool_name` | PermissionRequest, Elicitation | Tool requesting permission/input |
| `error` | PostToolUseFailure | Error description |
| `mcp_server_name` | Elicitation | MCP server name |

### Temp file lifecycle

- **Created**: On every `UserPromptSubmit` (overwritten per turn)
- **Deleted**: By `notify.sh` when Stop fires
- **Stale cleanup**: `mark-start.sh` deletes any files older than 24h on each run
- **Reboot**: `/tmp/` is cleared by macOS on restart

## Customization

### Change the minimum duration threshold

In `notify.sh`, find this line and change `120` (seconds) to your preference:

```bash
[ "$elapsed" -lt 120 ] && exit 0
```

### Adapt for Terminal.app instead of iTerm2

1. Replace the `-activate` bundle ID:
   ```bash
   -activate com.apple.Terminal
   ```

2. Replace the AppleScript for tab title:
   ```applescript
   tell application "Terminal"
       return name of front window
   end tell
   ```

3. Update the frontmost check:
   ```bash
   [ "$frontmost" = "Terminal" ] && exit 0
   ```

### Add sound to notifications

Add `-sound Glass` (or any sound from System Preferences > Sound) to the `terminal-notifier` calls:

```bash
"$terminal_notifier" \
  -title "Done" \
  -subtitle "..." \
  -message "..." \
  -sound Glass \
  -activate com.googlecode.iterm2
```

### Use Banners instead of Alerts

If you prefer auto-dismissing notifications, change the alert style in **System Settings > Notifications > terminal-notifier** from **Alerts** to **Banners**.

## Testing

Test each notification type manually:

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

Note: If iTerm2 is focused, notifications will be suppressed by design. Switch to another app first, or temporarily comment out the frontmost check in `notify.sh` for testing.

## Troubleshooting

| Problem | Solution |
|---|---|
| No notification appears | Check **System Settings > Notifications > terminal-notifier** is enabled |
| Notification appears but disappears instantly | Change alert style from Banners to **Alerts** |
| No notifications while in Focus/DND mode | Allow terminal-notifier through your Focus filter |
| `terminal-notifier` not found | Run `brew install terminal-notifier` |
| Tab title shows as empty | Ensure iTerm2 is running and has an active session |
| Stale temp files accumulating | They auto-clean after 24h; or run `rm /tmp/.claude-task-start-*` |
