#!/usr/bin/env bash
# init_project.sh — scaffold the standard CLAUDE.md project structure.
#
# Idempotent: existing files are skipped unless --force is given.
# See SKILL.md (one level up) for full docs.

set -euo pipefail

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMPLATES_DIR="$SKILL_DIR/templates"
VERSION_FILE="$SKILL_DIR/VERSION"

if [[ ! -d "$TEMPLATES_DIR" ]]; then
  echo "ERROR: templates dir not found at $TEMPLATES_DIR" >&2
  exit 1
fi
if [[ ! -f "$VERSION_FILE" ]]; then
  echo "ERROR: VERSION file not found at $VERSION_FILE" >&2
  exit 1
fi

TARGET_DIR="$PWD"
FORCE=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: init_project.sh [--target DIR] [--force] [--dry-run]

Scaffolds the standard CLAUDE.md structure into TARGET (default: $PWD).

Options:
  --target DIR  Where to scaffold. Defaults to current working directory.
  --force       Overwrite existing files. Default: skip.
  --dry-run     Show what would be done without writing files.
  -h, --help    This help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      [[ $# -ge 2 ]] || { echo "ERROR: --target needs a value" >&2; exit 2; }
      TARGET_DIR="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
done

TODAY="$(date +%Y-%m-%d)"
SKILL_VERSION="$(tr -d '[:space:]' < "$VERSION_FILE")"

# Color output if attached to a terminal
if [[ -t 1 ]]; then
  C_GREEN=$'\033[0;32m'
  C_YELLOW=$'\033[1;33m'
  C_BLUE=$'\033[0;34m'
  C_DIM=$'\033[2m'
  C_RESET=$'\033[0m'
else
  C_GREEN= C_YELLOW= C_BLUE= C_DIM= C_RESET=
fi

mkdir -p "$TARGET_DIR"
TARGET_DIR="$(cd "$TARGET_DIR" && pwd)"  # absolute

echo "${C_DIM}Scaffolding into:${C_RESET} $TARGET_DIR"
echo "${C_DIM}Skill version:   ${C_RESET} $SKILL_VERSION"
echo "${C_DIM}Date:            ${C_RESET} $TODAY"
[[ $DRY_RUN -eq 1 ]] && echo "${C_BLUE}(dry-run — no files will be written)${C_RESET}"
[[ $FORCE   -eq 1 ]] && echo "${C_YELLOW}(force — existing files will be overwritten)${C_RESET}"
echo

created=0; skipped=0; overwritten=0

copy_template() {
  local src="$1"  # absolute
  local rel="$2"  # relative path inside target
  local dst="$TARGET_DIR/$rel"
  local was_present=0
  [[ -e "$dst" ]] && was_present=1

  if [[ $was_present -eq 1 && $FORCE -eq 0 ]]; then
    printf '  %sskip%s    %s %s(exists)%s\n' "$C_YELLOW" "$C_RESET" "$rel" "$C_DIM" "$C_RESET"
    skipped=$((skipped+1))
    return
  fi

  if [[ $DRY_RUN -eq 1 ]]; then
    if [[ $was_present -eq 1 ]]; then
      printf '  %swould overwrite%s %s\n' "$C_YELLOW" "$C_RESET" "$rel"
    else
      printf '  %swould create%s    %s\n' "$C_GREEN" "$C_RESET" "$rel"
    fi
    return
  fi

  mkdir -p "$(dirname "$dst")"

  # Substitute placeholders. Pipe `|` is the delimiter since none of our
  # placeholders or values contain it.
  sed \
    -e "s|__DATE__|$TODAY|g" \
    -e "s|__SCAFFOLD_VERSION__|$SKILL_VERSION|g" \
    "$src" > "$dst"

  if [[ $was_present -eq 1 ]]; then
    printf '  %soverwrite%s  %s\n' "$C_YELLOW" "$C_RESET" "$rel"
    overwritten=$((overwritten+1))
  else
    printf '  %screate%s  %s\n' "$C_GREEN" "$C_RESET" "$rel"
    created=$((created+1))
  fi
}

# Capture pre-state of CLAUDE.md / AGENTS.md so the hardlink logic below is
# driven by what the user had BEFORE the scaffold ran, not by what's on disk
# after (which would conflate template-created files with pre-existing user files).
claude_path="$TARGET_DIR/CLAUDE.md"
agent_path="$TARGET_DIR/AGENTS.md"
had_claude=0; had_agent=0
[[ -e "$claude_path" ]] && had_claude=1
[[ -e "$agent_path" ]] && had_agent=1

# If the user has AGENTS.md but no CLAUDE.md, treat AGENTS.md as the source of
# truth: skip the CLAUDE.md template and let the hardlink step point
# CLAUDE.md at AGENTS.md.
skip_claude_template=0
if [[ $had_agent -eq 1 && $had_claude -eq 0 && $FORCE -eq 0 ]]; then
  skip_claude_template=1
fi

# Walk every template file. Strip the .template suffix so
# `CLAUDE.md.template` lands as `CLAUDE.md`.
while IFS= read -r -d '' src; do
  rel="${src#$TEMPLATES_DIR/}"
  out="${rel%.template}"
  if [[ "$out" == "CLAUDE.md" && $skip_claude_template -eq 1 ]]; then
    printf '  %sskip%s    CLAUDE.md %s(will hardlink to existing AGENTS.md)%s\n' \
      "$C_BLUE" "$C_RESET" "$C_DIM" "$C_RESET"
    skipped=$((skipped+1))
    continue
  fi
  copy_template "$src" "$out"
done < <(find "$TEMPLATES_DIR" -type f -print0 | sort -z)

# Ensure CLAUDE.md and AGENTS.md are the same file (hardlinked), so edits to
# one show up in the other. The two filenames serve different tool ecosystems
# (Claude Code vs. the AGENTS.md spec) but the content should never diverge.
inode_of() {
  # Portable: macOS uses `stat -f %i`, GNU uses `stat -c %i`.
  stat -f %i "$1" 2>/dev/null || stat -c %i "$1" 2>/dev/null
}

link_pair() {
  local src="$1" dst="$2"  # create dst as a hardlink to src
  if [[ $DRY_RUN -eq 1 ]]; then
    printf '  %swould link%s  %s <-hardlink-> %s\n' \
      "$C_GREEN" "$C_RESET" "$(basename "$dst")" "$(basename "$src")"
    return
  fi
  ln "$src" "$dst"
  printf '  %slink%s    %s <-hardlink-> %s\n' \
    "$C_GREEN" "$C_RESET" "$(basename "$dst")" "$(basename "$src")"
}

if [[ $had_claude -eq 1 && $had_agent -eq 1 ]]; then
  # Both pre-existed. Nothing we did should have touched AGENTS.md, so checking
  # inodes on disk is still meaningful.
  same_inode=0
  if [[ "$(inode_of "$claude_path")" == "$(inode_of "$agent_path")" ]]; then
    same_inode=1
  fi
  if [[ $same_inode -eq 1 ]]; then
    : # already linked — no-op
  elif [[ $FORCE -eq 1 ]]; then
    # With --force: keep CLAUDE.md as canonical, re-link AGENTS.md to it.
    if [[ $DRY_RUN -eq 1 ]]; then
      printf '  %swould relink%s AGENTS.md <-hardlink-> CLAUDE.md %s(prior AGENTS.md would be replaced)%s\n' \
        "$C_YELLOW" "$C_RESET" "$C_DIM" "$C_RESET"
    else
      rm -f "$agent_path"
      ln "$claude_path" "$agent_path"
      printf '  %srelink%s  AGENTS.md <-hardlink-> CLAUDE.md %s(prior AGENTS.md replaced)%s\n' \
        "$C_YELLOW" "$C_RESET" "$C_DIM" "$C_RESET"
    fi
  else
    printf '  %swarn%s    CLAUDE.md and AGENTS.md both exist with different content; leaving as-is %s(--force to relink)%s\n' \
      "$C_YELLOW" "$C_RESET" "$C_DIM" "$C_RESET"
  fi
elif [[ $had_claude -eq 1 && $had_agent -eq 0 ]]; then
  # Only CLAUDE.md existed — link AGENTS.md to it.
  link_pair "$claude_path" "$agent_path"
elif [[ $had_claude -eq 0 && $had_agent -eq 1 ]]; then
  # Only AGENTS.md existed — link CLAUDE.md to AGENTS.md (preserving user content).
  link_pair "$agent_path" "$claude_path"
else
  # Neither existed — CLAUDE.md was just scaffolded from the template;
  # mirror AGENTS.md onto it.
  link_pair "$claude_path" "$agent_path"
fi

# Always (unless dry-run) refresh the scaffold marker.
marker="$TARGET_DIR/.claude/.scaffold-version"
if [[ $DRY_RUN -eq 0 ]]; then
  mkdir -p "$(dirname "$marker")"
  cat > "$marker" <<EOF
# Generated by start-a-project skill. Do not edit by hand.
version: $SKILL_VERSION
scaffolded: $TODAY
EOF
  printf '  %sstamp%s   %s\n' "$C_BLUE" "$C_RESET" ".claude/.scaffold-version"
fi

echo
echo "${C_DIM}---${C_RESET}"
printf 'Created: %d   Overwritten: %d   Skipped: %d   Target: %s\n' \
  "$created" "$overwritten" "$skipped" "$TARGET_DIR"
if [[ $skipped -gt 0 && $FORCE -eq 0 ]]; then
  echo "${C_DIM}(skipped files already exist; re-run with --force to overwrite)${C_RESET}"
fi
