#!/bin/bash
# Focus the iTerm2 window/tab containing the session with the given UUID
# Called when user clicks a notification
# Usage: focus-session.sh <session-uuid>
target_uuid="$1"

if [ -z "$target_uuid" ]; then
    osascript -e 'tell application "iTerm2" to activate'
    exit 0
fi

osascript <<EOF
tell application "iTerm2"
    repeat with w in windows
        repeat with t in tabs of w
            repeat with s in sessions of t
                if id of s is "$target_uuid" then
                    select t
                    set index of w to 1
                    activate
                    return
                end if
            end repeat
        end repeat
    end repeat
    -- Fallback: just activate iTerm2
    activate
end tell
EOF
