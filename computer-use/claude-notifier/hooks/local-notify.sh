#!/bin/bash
# Called by iTerm2 Trigger when it catches @@CLAUDE_NOTIFY|...|...|...@@
# Used for remote -> local notification pipeline (tmux on remote -> iTerm2 trigger -> this script)
# Arguments: $1 = type (DONE/APPROVAL/INPUT), $2 = subtitle, $3 = message

type="$1"
subtitle="$2"
message="$3"

# Skip notification if iTerm is the frontmost app
frontmost=$(osascript -e 'tell application "System Events" to get name of first process whose frontmost is true' 2>/dev/null)
[ "$frontmost" = "iTerm2" ] && exit 0

terminal_notifier=$(which terminal-notifier 2>/dev/null)
[ -z "$terminal_notifier" ] && terminal_notifier="/opt/homebrew/bin/terminal-notifier"

case "$type" in
  DONE)
    "$terminal_notifier" \
      -title "✅ Claude — Done" \
      -subtitle "$subtitle" \
      -message "$message" \
      -activate com.googlecode.iterm2
    ;;
  APPROVAL)
    "$terminal_notifier" \
      -title "⏸ Claude — Approval" \
      -subtitle "$subtitle" \
      -message "$message" \
      -activate com.googlecode.iterm2
    ;;
  INPUT)
    "$terminal_notifier" \
      -title "❓ Claude — Input" \
      -subtitle "$subtitle" \
      -message "$message" \
      -activate com.googlecode.iterm2
    ;;
esac
