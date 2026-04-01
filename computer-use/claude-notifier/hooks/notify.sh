#!/bin/bash
# Unified Claude Code notification hook (macOS + iTerm2)
# Handles: Stop (long tasks), PermissionRequest, Elicitation
# Identifies the specific iTerm2 window via session UUID from ITERM_SESSION_ID

input=$(cat)

eval "$(/usr/bin/python3 -c "
import sys, json, re, shlex

d = json.load(sys.stdin)

event = d.get('hook_event_name') or ''
cwd = d.get('cwd') or ''
session_id = d.get('session_id') or ''
message = d.get('last_assistant_message') or ''
tool_name = d.get('tool_name') or ''
error = d.get('error') or ''
mcp_server = d.get('mcp_server_name') or ''

def clean(s, n=120):
    if not s: return ''
    # Strip markdown code blocks and inline code
    s = re.sub(r'\x60{3}[\s\S]*?\x60{3}', ' ', s)
    s = re.sub(r'\x60[^\x60]+\x60', ' ', s)
    # Strip markdown headers, bold/italic markers
    s = re.sub(r'^#{1,6}\s+', '', s, flags=re.M)
    s = re.sub(r'[*_]{1,3}', '', s)
    # Links -> text only
    s = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', s)
    # Strip bullet and numbered list prefixes
    s = re.sub(r'^\s*[-*+]\s+', '', s, flags=re.M)
    s = re.sub(r'^\s*\d+\.\s+', '', s, flags=re.M)
    # Replace all control chars / newlines with space
    s = re.sub(r'[\x00-\x1f\x7f]+', ' ', s)
    # Collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    if len(s) > n:
        s = s[:n].rsplit(' ', 1)[0] + chr(8230)
    return s

sq = shlex.quote
print(f'EVENT={sq(event)}')
print(f'CWD={sq(cwd)}')
print(f'SESSION_ID={sq(session_id)}')
print(f'MESSAGE={sq(clean(message))}')
print(f'TOOL_NAME={sq(tool_name)}')
print(f'TOOL_ERROR={sq(clean(error, 80))}')
print(f'MCP_SERVER={sq(mcp_server)}')
" <<< "$input" 2>/dev/null)"

project=$(basename "$CWD")
terminal_notifier=$(which terminal-notifier 2>/dev/null)
[ -z "$terminal_notifier" ] && exit 0
focus_script="$HOME/.claude/hooks/focus-session.sh"

# Read stamp file: line 1=start_time, line 2=tab_title, line 3=iterm_session_uuid
stamp="/tmp/.claude-task-start-${SESSION_ID}"
start_time=""
tab_title=""
session_uuid=""
if [ -f "$stamp" ]; then
    { read -r start_time; read -r tab_title; read -r session_uuid; } < "$stamp"
fi

# Check if this specific iTerm2 session is currently visible and focused
if [ -n "$session_uuid" ]; then
    focused=$(osascript <<EOF 2>/dev/null
tell application "System Events"
    if name of first process whose frontmost is true is not "iTerm2" then return "no"
end tell
tell application "iTerm2"
    try
        if id of (current session of current tab of current window) is "$session_uuid" then return "yes"
    end try
end tell
return "no"
EOF
    )
    [ "$focused" = "yes" ] && exit 0
else
    # Fallback: skip if any iTerm2 is frontmost
    frontmost=$(osascript -e 'tell application "System Events" to get name of first process whose frontmost is true' 2>/dev/null)
    [ "$frontmost" = "iTerm2" ] && exit 0
fi

# Build click command: focuses the exact iTerm2 window/tab for this session
click_cmd=""
if [ -n "$session_uuid" ] && [ -x "$focus_script" ]; then
    click_cmd="$focus_script $session_uuid"
fi

notify() {
    local title="$1" subtitle="$2" msg="$3"
    if [ -n "$click_cmd" ]; then
        "$terminal_notifier" \
            -title "$title" \
            -subtitle "$subtitle" \
            -message "$msg" \
            -group "claude-${SESSION_ID}" \
            -execute "$click_cmd"
    else
        "$terminal_notifier" \
            -title "$title" \
            -subtitle "$subtitle" \
            -message "$msg" \
            -group "claude-${SESSION_ID}" \
            -activate com.googlecode.iterm2
    fi
}

case "$EVENT" in
    Stop)
        if [ -n "$start_time" ]; then
            now=$(date +%s)
            elapsed=$((now - start_time))
            rm -f "$stamp"
            [ "$elapsed" -lt 120 ] && exit 0
            mins=$((elapsed / 60))
            secs=$((elapsed % 60))
            notify \
                "✅ Claude — Done" \
                "${tab_title:-$project} — ${mins}m ${secs}s" \
                "${MESSAGE:-Task completed}"
        fi
        ;;
    PermissionRequest)
        notify \
            "🔷 Claude — Approval" \
            "${tab_title:-$project}" \
            "Permission: $TOOL_NAME"
        ;;
    Elicitation)
        notify \
            "🔷 Claude — Input" \
            "${tab_title:-$project}" \
            "Input needed: ${TOOL_NAME:-tool}"
        ;;
esac
