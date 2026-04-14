#!/usr/bin/env bash
# Uninstaller for harden-macos-claude.
#
# What it does:
#   1. Removes the managed charter block from ~/.claude/CLAUDE.md
#      (anything you added outside the markers is preserved).
#   2. Removes the session unlock marker ~/.claude/.harden-skill-active.
#   3. Prints the Claude Code command to disable the plugin.
#
# It does NOT touch ~/.claude/security.log (your audit trail stays).
# It does NOT uninstall the plugin itself from Claude Code — plugins are
# loaded via `claude --plugin-dir ...` / `claude plugin install`, and
# you disable them the same way (stop passing the flag, or
# `claude plugin remove harden-macos-claude`).
#
# Usage:
#   ./uninstall.sh            # interactive (prompts once)
#   ./uninstall.sh --yes      # non-interactive
#
# Safety:
#   Run from your shell, NOT from inside a Claude Code session — the hooks
#   (if still active) will (correctly) block writes to ~/.claude/CLAUDE.md.
#   If you must run through Claude, first create the unlock marker:
#       touch ~/.claude/.harden-skill-active

set -euo pipefail

PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="$HOME/.claude/CLAUDE.md"
MARKER="$HOME/.claude/.harden-skill-active"
BEGIN_MARK="<!-- BEGIN harden-macos-claude charter (managed block — do not edit by hand) -->"
END_MARK="<!-- END harden-macos-claude charter -->"

ASSUME_YES=0
if [[ "${1:-}" == "--yes" || "${1:-}" == "-y" ]]; then
    ASSUME_YES=1
fi

echo "harden-macos-claude uninstaller"
echo "  Charter target:  $TARGET"
echo "  Session marker:  $MARKER"
echo

if [[ "$ASSUME_YES" -ne 1 ]]; then
    read -r -p "Proceed with uninstall? [y/N] " reply
    case "$reply" in
        [yY]|[yY][eE][sS]) ;;
        *) echo "Aborted."; exit 0 ;;
    esac
fi

# 1. Remove managed charter block.
if [[ -f "$TARGET" ]]; then
    TS="$(date +%Y%m%d-%H%M%S)"
    cp "$TARGET" "$TARGET.bak-$TS"
    /usr/bin/python3 - "$TARGET" "$BEGIN_MARK" "$END_MARK" <<'PY'
import re, sys
target, begin, end = sys.argv[1:4]
with open(target, "r", encoding="utf-8") as f:
    existing = f.read()
pattern = re.compile(
    r"(?:\n{0,2})" + re.escape(begin) + r".*?" + re.escape(end) + r"\n?",
    re.DOTALL,
)
cleaned = pattern.sub("", existing).rstrip()
out = cleaned + ("\n" if cleaned else "")
with open(target, "w", encoding="utf-8") as f:
    f.write(out)
removed = existing != out
print(f"charter {'removed' if removed else 'not present'} in {target}")
PY
    echo "Backup: $TARGET.bak-$TS"
else
    echo "No CLAUDE.md at $TARGET — skipping."
fi

# 2. Remove session unlock marker if present.
if [[ -e "$MARKER" ]]; then
    rm -f "$MARKER"
    echo "Removed unlock marker: $MARKER"
fi

# 3. Instructions for detaching the plugin itself.
cat <<EOF

Charter removed. To fully disable the plugin in Claude Code:

  - If you loaded it with --plugin-dir: just stop passing that flag.
      (started with: claude --plugin-dir "$PLUGIN_DIR")

  - If you installed it via the plugin marketplace:
      claude plugin remove harden-macos-claude

Notes:
  - ~/.claude/security.log is left in place as your audit trail.
  - The hook scripts under this plugin directory are untouched; delete
    the plugin directory manually if you want the files gone.
  - Any backups created by install/uninstall live at:
      ~/.claude/CLAUDE.md.bak-<timestamp>
EOF
