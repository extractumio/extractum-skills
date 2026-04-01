#!/bin/bash
# Remote Claude Code notification hook (Linux + tmux -> iTerm2 via triggers)
#
# Outputs a marker string to the tmux pane TTY.
# iTerm2 Trigger on the local Mac catches the marker and calls local-notify.sh.
#
# Handles: Stop (long tasks), PermissionRequest, Elicitation

input=$(cat)

eval "$(python3 -c "
import sys, json, re, shlex

d = json.load(sys.stdin)

event = d.get('hook_event_name') or ''
cwd = d.get('cwd') or ''
session_id = d.get('session_id') or ''
message = d.get('last_assistant_message') or ''
tool_name = d.get('tool_name') or ''
mcp_server = d.get('mcp_server_name') or ''

def clean(s, n=120):
    if not s: return ''
    s = re.sub(r'\x60{3}[\s\S]*?\x60{3}', ' ', s)
    s = re.sub(r'\x60[^\x60]+\x60', ' ', s)
    s = re.sub(r'^#{1,6}\s+', '', s, flags=re.M)
    s = re.sub(r'[*_]{1,3}', '', s)
    s = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', s)
    s = re.sub(r'^\s*[-*+]\s+', '', s, flags=re.M)
    s = re.sub(r'[\x00-\x1f\x7f]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    # Pipe chars would break the marker protocol
    s = s.replace('|', '/')
    if len(s) > n:
        s = s[:n].rsplit(' ', 1)[0] + '...'
    return s

sq = shlex.quote
print(f'EVENT={sq(event)}')
print(f'CWD={sq(cwd)}')
print(f'SESSION_ID={sq(session_id)}')
print(f'MESSAGE={sq(clean(message))}')
print(f'TOOL_NAME={sq(tool_name)}')
print(f'MCP_SERVER={sq(mcp_server)}')
" <<< "$input" 2>/dev/null)"

project=$(basename "$CWD")
stamp="/tmp/.claude-task-start-${SESSION_ID}"

# Resolve the TTY to write the marker to
# Priority: stored TTY from mark-start > current tmux pane > /dev/tty
resolve_tty() {
    if [ -f "$stamp" ]; then
        local stored_tty
        stored_tty=$(sed -n '3p' "$stamp")
        if [ -n "$stored_tty" ] && [ -w "$stored_tty" ]; then
            echo "$stored_tty"; return
        fi
    fi
    if [ -n "$TMUX_PANE" ]; then
        local tty
        tty=$(tmux display-message -t "$TMUX_PANE" -p '#{pane_tty}' 2>/dev/null)
        if [ -n "$tty" ] && [ -w "$tty" ]; then
            echo "$tty"; return
        fi
    fi
    if [ -n "$TMUX" ]; then
        local tty
        tty=$(tmux display-message -p '#{pane_tty}' 2>/dev/null)
        if [ -n "$tty" ] && [ -w "$tty" ]; then
            echo "$tty"; return
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
    send_marker "APPROVAL" "$project" "Permission: $TOOL_NAME"
    ;;
  Elicitation)
    send_marker "INPUT" "$project" "Input needed: ${TOOL_NAME:-tool}"
    ;;
esac
