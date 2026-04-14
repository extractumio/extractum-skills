#!/usr/bin/env bash
# Idempotent installer for harden-macos-claude.
#
# What it does:
#   1. Ensures ~/.claude/ exists.
#   2. Installs / updates the security charter inside a delimited, managed
#      block in ~/.claude/CLAUDE.md.  Your own content outside the block is
#      left untouched.  Re-running replaces the managed block with the
#      current plugin version (idempotent, upgrade-safe).
#   3. Prints the Claude Code command to enable the plugin.
#
# Usage:
#   ./install.sh            # install / update
#   ./install.sh --remove   # remove the managed block, keep your own content
#
# Safety notes:
#   * Run this from your shell, NOT from inside a Claude Code session — the
#     hooks may be active and (correctly) block writes to ~/.claude/CLAUDE.md.
#   * A timestamped backup is written to ~/.claude/CLAUDE.md.bak-<ts> before
#     every change.
#   * The managed block is bounded by HTML comment markers; edit outside them.

set -euo pipefail

PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHARTER_SRC="$PLUGIN_DIR/CLAUDE.md"
TARGET="$HOME/.claude/CLAUDE.md"
BEGIN_MARK="<!-- BEGIN harden-macos-claude charter (managed block — do not edit by hand) -->"
END_MARK="<!-- END harden-macos-claude charter -->"

MODE="install"
if [[ "${1:-}" == "--remove" || "${1:-}" == "-r" ]]; then
    MODE="remove"
fi

if [[ ! -f "$CHARTER_SRC" ]]; then
    echo "error: charter source not found at $CHARTER_SRC" >&2
    exit 1
fi

mkdir -p "$HOME/.claude"
: > /dev/null  # shellcheck noop
[[ -f "$TARGET" ]] || touch "$TARGET"

# Backup.
TS="$(date +%Y%m%d-%H%M%S)"
cp "$TARGET" "$TARGET.bak-$TS"

# Do the work in Python (trivial text munging, safer than sed for multi-line).
/usr/bin/python3 - "$TARGET" "$CHARTER_SRC" "$BEGIN_MARK" "$END_MARK" "$MODE" <<'PY'
import re, sys
target, charter_path, begin, end, mode = sys.argv[1:6]

with open(target, "r", encoding="utf-8") as f:
    existing = f.read()

# Strip any previously managed block (plus surrounding blank lines).
pattern = re.compile(
    r"(?:\n{0,2})"
    + re.escape(begin)
    + r".*?"
    + re.escape(end)
    + r"\n?",
    re.DOTALL,
)
cleaned = pattern.sub("", existing).rstrip()

if mode == "remove":
    out = cleaned + ("\n" if cleaned else "")
    action = "removed"
else:
    with open(charter_path, "r", encoding="utf-8") as f:
        charter = f.read().strip()
    block = f"{begin}\n{charter}\n{end}\n"
    sep = "\n\n" if cleaned else ""
    out = cleaned + sep + block
    action = "installed/updated"

with open(target, "w", encoding="utf-8") as f:
    f.write(out)

print(f"charter {action} in {target}")
PY

echo
echo "Backup: $TARGET.bak-$TS"
if [[ "$MODE" == "install" ]]; then
    echo
    echo "Next: enable the plugin in Claude Code ->"
    echo "    claude --plugin-dir \"$PLUGIN_DIR\""
    echo
    echo "Verify:"
    echo "    /usr/bin/python3 \"$PLUGIN_DIR/scripts/tests/run_tests.py\""
fi
