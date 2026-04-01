#!/bin/bash
# Writes a start timestamp and iTerm2 session UUID when user submits a prompt
input=$(cat)
session_id=$(/usr/bin/python3 -c "import sys,json; print(json.load(sys.stdin).get('session_id',''))" <<< "$input" 2>/dev/null)
stamp="/tmp/.claude-task-start-${session_id}"

# Clean up stale files older than 24h from crashed sessions
find /tmp -maxdepth 1 -name ".claude-task-start-*" -mmin +1440 -delete 2>/dev/null

# Extract iTerm2 session UUID from environment (stable, unique per session)
iterm_uuid="${ITERM_SESSION_ID#*:}"

# Capture iTerm2 tab/window title
tab_title=$(osascript -e '
tell application "iTerm2"
    tell current session of current tab of current window
        return name
    end tell
end tell
' 2>/dev/null)

# Store: line 1=timestamp, line 2=tab_title, line 3=iterm_session_uuid
printf '%s\n%s\n%s\n' "$(date +%s)" "$tab_title" "$iterm_uuid" > "$stamp"
