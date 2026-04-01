#!/bin/bash
set -euo pipefail

# ── Claude Code Notifier ──────────────────────────────────────────────
# macOS: notifications via terminal-notifier, click lands on exact iTerm2 window
# Linux: marker strings via tmux TTY, caught by iTerm2 Triggers on Mac
#
# Usage:
#   bash install.sh              Install / upgrade
#   bash install.sh --uninstall  Remove hooks and settings entries
#   bash install.sh --status     Check installation state
# ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOOKS_DIR="$HOME/.claude/hooks"
SETTINGS="$HOME/.claude/settings.json"
PLATFORM="$(uname -s)"
BACKUP_TAG=".pre-notifier"

# Hook files shipped in this package, per platform
MACOS_FILES=(mark-start.sh notify.sh focus-session.sh local-notify.sh)
LINUX_FILES=(remote-mark-start.sh remote-notify.sh)
ALL_FILES=(mark-start.sh notify.sh focus-session.sh local-notify.sh remote-mark-start.sh remote-notify.sh)

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()  { printf "  ${CYAN}::${NC} %s\n" "$*"; }
ok()    { printf "  ${GREEN}OK${NC} %s\n" "$*"; }
warn()  { printf "  ${YELLOW}!!${NC} %s\n" "$*"; }
fail()  { printf "  ${RED}ERR${NC} %s\n" "$*"; exit 1; }

# ── Cleanup trap: remove orphan temp files on any exit ────────────────
cleanup() { rm -f "$HOME"/.claude/.settings.*.tmp 2>/dev/null || true; }
trap cleanup EXIT

# ── Shared Python: atomic settings.json manipulation ──────────────────
# Handles: read → clean old hooks → optionally add new → validate → atomic rename
# Arguments via env: NOTIFIER_ACTION (install|uninstall), NOTIFIER_SETTINGS, NOTIFIER_PLATFORM
update_settings() {
    NOTIFIER_ACTION="$1" \
    NOTIFIER_SETTINGS="$SETTINGS" \
    NOTIFIER_PLATFORM="$PLATFORM" \
    python3 <<'PYEOF'
import json, os, sys, tempfile

action   = os.environ["NOTIFIER_ACTION"]
path     = os.environ["NOTIFIER_SETTINGS"]
platform = os.environ["NOTIFIER_PLATFORM"]

# Every hook command this package has ever registered (current + legacy).
# Upgrade removes all of these before re-adding the current set.
ALL_OUR_COMMANDS = {
    "~/.claude/hooks/mark-start.sh",
    "~/.claude/hooks/notify.sh",
    "~/.claude/hooks/notify-stop.sh",       # legacy v0
    "~/.claude/hooks/remote-mark-start.sh",
    "~/.claude/hooks/remote-notify.sh",
}

# Current version — what we want installed
if platform == "Darwin":
    DESIRED = {
        "UserPromptSubmit": "~/.claude/hooks/mark-start.sh",
        "Stop":              "~/.claude/hooks/notify.sh",
        "PermissionRequest": "~/.claude/hooks/notify.sh",
        "Elicitation":       "~/.claude/hooks/notify.sh",
    }
else:
    DESIRED = {
        "UserPromptSubmit": "~/.claude/hooks/remote-mark-start.sh",
        "Stop":              "~/.claude/hooks/remote-notify.sh",
        "PermissionRequest": "~/.claude/hooks/remote-notify.sh",
        "Elicitation":       "~/.claude/hooks/remote-notify.sh",
    }

# ── Read existing settings ────────────────────────────────────────────
cfg = {}
if os.path.exists(path):
    with open(path) as f:
        content = f.read().strip()
    if content:
        try:
            cfg = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"  ERROR: {path} is not valid JSON (line {e.lineno}: {e.msg})")
            print(f"  Fix it manually, then re-run the installer.")
            sys.exit(1)
    if not isinstance(cfg, dict):
        print(f"  ERROR: {path} root is {type(cfg).__name__}, expected object")
        sys.exit(1)

# Normalise: hooks must be a dict
if "hooks" in cfg and not isinstance(cfg["hooks"], dict):
    print(f"  WARN: hooks field was {type(cfg['hooks']).__name__}, resetting to dict")
    cfg["hooks"] = {}
hooks = cfg.setdefault("hooks", {})

# ── Step 1: Remove every trace of this package (clean slate for upgrade) ─
removed = []
for event in list(hooks.keys()):
    entries = hooks[event]
    if not isinstance(entries, list):
        continue
    cleaned = []
    for entry in entries:
        if not isinstance(entry, dict):
            cleaned.append(entry)
            continue
        hook_list = entry.get("hooks", [])
        if not isinstance(hook_list, list):
            cleaned.append(entry)
            continue
        remaining = []
        for h in hook_list:
            cmd = h.get("command", "") if isinstance(h, dict) else ""
            if cmd in ALL_OUR_COMMANDS:
                removed.append(f"{event} -> {cmd}")
            else:
                remaining.append(h)
        if remaining:
            entry["hooks"] = remaining
            cleaned.append(entry)
    if cleaned:
        hooks[event] = cleaned
    else:
        del hooks[event]

for r in removed:
    print(f"  Cleaned: {r}")

# ── Step 2: Add current hooks (install only) ─────────────────────────
if action == "install":
    for event, command in DESIRED.items():
        entries = hooks.setdefault(event, [])
        entries.append({
            "matcher": "",
            "hooks": [{"type": "command", "command": command}]
        })
        print(f"  Added:   {event} -> {command}")
else:
    if not removed:
        print("  Nothing to remove (not installed).")
        sys.exit(0)
    # Clean up empty hooks dict after uninstall
    if not hooks:
        del cfg["hooks"]

# ── Step 3: Atomic write (temp → verify → rename) ────────────────────
dir_path = os.path.dirname(path) or "."
os.makedirs(dir_path, exist_ok=True)

fd, tmp_path = tempfile.mkstemp(dir=dir_path, prefix=".settings.", suffix=".tmp")
try:
    with os.fdopen(fd, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")

    # Validate: re-read and parse what we just wrote
    with open(tmp_path) as f:
        verified = json.load(f)
    assert isinstance(verified, dict), "verification failed: root is not a dict"

    # Preserve original file permissions
    if os.path.exists(path):
        os.chmod(tmp_path, os.stat(path).st_mode)

    # Atomic rename (same filesystem guaranteed by mkstemp in same dir)
    os.rename(tmp_path, path)
    print("  Settings saved (atomic write).")

except BaseException as exc:
    # Clean up temp file on any failure
    try:
        os.unlink(tmp_path)
    except OSError:
        pass
    print(f"  ERROR writing settings: {exc}")
    sys.exit(1)
PYEOF
}

# ── Copy one hook file with safe backup ───────────────────────────────
# First-install backup uses ".pre-notifier" suffix and is never overwritten,
# so the user's original is always recoverable.
install_hook_file() {
    local name="$1"
    local src="$SCRIPT_DIR/hooks/$name"
    local dst="$HOOKS_DIR/$name"

    [ -f "$src" ] || fail "Missing source: hooks/$name"

    if [ -f "$dst" ]; then
        if cmp -s "$src" "$dst"; then
            ok "$name (up to date)"
            return 0
        fi
        # Preserve the user's *original* file — only once
        if [ ! -f "${dst}${BACKUP_TAG}" ]; then
            cp -p "$dst" "${dst}${BACKUP_TAG}"
            info "$name  original saved -> ${name}${BACKUP_TAG}"
        fi
    fi

    cp "$src" "$dst"
    chmod +x "$dst"
    ok "$name"
}

# ══════════════════════════════════════════════════════════════════════
# ── --status ──────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════
if [[ "${1:-}" == "--status" ]]; then
    echo ""
    printf "${BOLD}Claude Code Notifier — Status${NC}\n\n"

    installed=0
    for f in "${ALL_FILES[@]}"; do
        if [ -f "$HOOKS_DIR/$f" ]; then
            if [ -f "$SCRIPT_DIR/hooks/$f" ] && cmp -s "$SCRIPT_DIR/hooks/$f" "$HOOKS_DIR/$f"; then
                ok "$f (current)"
            else
                warn "$f (modified or outdated)"
            fi
            installed=1
        fi
    done
    [ "$installed" -eq 0 ] && info "No hook files installed."

    echo ""
    if [ -f "$SETTINGS" ]; then
        NOTIFIER_SETTINGS="$SETTINGS" python3 <<'PYEOF'
import json, os
path = os.environ["NOTIFIER_SETTINGS"]
with open(path) as f:
    cfg = json.load(f)
hooks = cfg.get("hooks", {})
our = {"~/.claude/hooks/mark-start.sh","~/.claude/hooks/notify.sh",
       "~/.claude/hooks/notify-stop.sh","~/.claude/hooks/remote-mark-start.sh",
       "~/.claude/hooks/remote-notify.sh"}
found = False
for event, entries in hooks.items():
    for entry in entries:
        for h in entry.get("hooks",[]):
            if h.get("command") in our:
                print(f"  Hook: {event} -> {h['command']}")
                found = True
if not found:
    print("  No notifier hooks in settings.json")
PYEOF
    else
        info "No settings.json found."
    fi
    echo ""
    exit 0
fi

# ══════════════════════════════════════════════════════════════════════
# ── --uninstall ───────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════
if [[ "${1:-}" == "--uninstall" ]]; then
    echo ""
    printf "${BOLD}Claude Code Notifier — Uninstall${NC}\n\n"

    # Remove hook files (only ours)
    for f in "${ALL_FILES[@]}"; do
        if [ -f "$HOOKS_DIR/$f" ]; then
            rm -f "$HOOKS_DIR/$f"
            ok "Removed $f"
        fi
    done

    # Remove hook entries from settings.json
    echo ""
    if [ -f "$SETTINGS" ]; then
        update_settings "uninstall"
    else
        info "No settings.json — nothing to clean."
    fi

    echo ""
    backups=$(find "$HOOKS_DIR" -name "*${BACKUP_TAG}" 2>/dev/null | head -5)
    if [ -n "$backups" ]; then
        info "Original backups preserved (${BACKUP_TAG} files)."
        info "Remove manually if not needed."
    fi
    echo ""
    ok "Uninstalled. Restart Claude Code for changes to take effect."
    exit 0
fi

# ══════════════════════════════════════════════════════════════════════
# ── Install / Upgrade ─────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════
if [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
    echo "Usage: bash install.sh [--uninstall | --status | --help]"
    exit 0
fi

echo ""
printf "${BOLD}Claude Code Notifier — Install${NC}\n\n"

if [[ "$PLATFORM" == "Darwin" ]]; then
    info "Platform: macOS (local iTerm2 notifications)"
else
    info "Platform: Linux (remote tmux -> iTerm2 trigger pipeline)"
fi
echo ""

# ── Prerequisites ─────────────────────────────────────────────────────

if ! command -v python3 &>/dev/null; then
    fail "python3 is required but not found."
fi

if [[ "$PLATFORM" == "Darwin" ]]; then
    if ! command -v osascript &>/dev/null; then
        fail "osascript not found — this installer requires macOS."
    fi
    if ! command -v terminal-notifier &>/dev/null; then
        warn "terminal-notifier not found."
        if command -v brew &>/dev/null; then
            info "Installing via Homebrew..."
            brew install terminal-notifier
            ok "terminal-notifier installed"
        else
            echo ""
            fail "Install it first:  brew install terminal-notifier"
        fi
    else
        ok "terminal-notifier"
    fi
    if [ -d "/Applications/iTerm.app" ]; then
        ok "iTerm2"
    else
        warn "iTerm2 not in /Applications — notifications require iTerm2."
    fi
else
    if ! command -v tmux &>/dev/null; then
        warn "tmux not found — remote hooks require tmux."
    else
        ok "tmux"
    fi
fi

echo ""

# ── Copy hook files ───────────────────────────────────────────────────

mkdir -p "$HOOKS_DIR"

if [[ "$PLATFORM" == "Darwin" ]]; then
    FILES=("${MACOS_FILES[@]}")
else
    FILES=("${LINUX_FILES[@]}")
fi

for f in "${FILES[@]}"; do
    install_hook_file "$f"
done

echo ""

# ── First-install backup of settings.json (never overwritten) ─────────

if [ -f "$SETTINGS" ] && [ ! -f "${SETTINGS}${BACKUP_TAG}" ]; then
    cp -p "$SETTINGS" "${SETTINGS}${BACKUP_TAG}"
    info "Original settings.json saved -> settings.json${BACKUP_TAG}"
fi

# ── Update settings.json (atomic, upgrade-aware) ─────────────────────

update_settings "install"

# ── Success ───────────────────────────────────────────────────────────

echo ""
if [[ "$PLATFORM" == "Darwin" ]]; then
    printf "${GREEN}${BOLD}Installed.${NC} Restart Claude Code for hooks to take effect.\n"
    cat <<'MSG'

  How it works:
    Tasks over 2 min    -> macOS notification on completion
    Permission / input  -> immediate notification
    Focus detection     -> only notifies when your specific iTerm2 session is not focused
    Click notification  -> lands on the exact iTerm2 window and tab

MSG
else
    printf "${GREEN}${BOLD}Installed.${NC} Restart Claude Code for hooks to take effect.\n"
    cat <<'MSG'

  How it works:
    Tasks over 2 min    -> marker string emitted to tmux pane TTY
    Permission / input  -> immediate marker
    iTerm2 Trigger      -> catches markers and shows macOS notification

  iTerm2 Trigger setup (on your Mac):
    Regex:   @@CLAUDE_NOTIFY\|([^|]+)\|([^|]+)\|([^@]+)@@
    Action:  Run Command...  ~/.claude/hooks/local-notify.sh \1 \2 \3

MSG
fi
printf "  Manage:  bash %s/install.sh [--status | --uninstall]\n\n" "$SCRIPT_DIR"
