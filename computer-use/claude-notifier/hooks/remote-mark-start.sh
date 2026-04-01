#!/bin/bash
# Writes a start timestamp when user submits a prompt (remote Linux/tmux version)
# Captures tmux session:window.pane identity and pane TTY for notifications

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
printf '%s\n%s\n%s\n' "$(date +%s)" "$pane_title" "$pane_tty" > "$stamp"
