---
name: claude-orphan-cleanup
description: Find and terminate orphaned Claude Code CLI processes that consume CPU resources. Works on macOS and Linux systems.
---

# Claude Code Orphan Process Cleanup

## Background

When using Claude Code CLI (`claude` command), the tool spawns bash subprocesses for various operations:
- **Shell snapshots**: Captures shell environment (functions, aliases, PATH)
- **MCP tool calls**: Network requests, file operations via Model Context Protocol
- **Command execution**: Running user-requested terminal commands

If the main Claude process crashes, is interrupted (Ctrl+C), or terminates unexpectedly, these child processes can become **orphaned** — adopted by the init process (PID 1 on Linux, launchd on macOS) and left running indefinitely.

### Symptoms of Orphaned Processes

- Multiple `bash` processes consuming 100% CPU
- High CPU usage when Claude Code is not running
- Processes with PPID=1 containing `claude`, `shell-snapshot`, or `claude_agent_sdk` in their command line
- Fan noise / battery drain on laptops

## Instructions

When this skill is invoked, follow these steps to identify and clean up orphaned Claude Code processes:

### Step 1: Identify Orphaned Processes

Run the cleanup tool in dry-run mode (default) to list orphaned processes:

```bash
claude-cleanup
```

The tool searches for processes matching these criteria:
- **PPID=1** (orphaned, adopted by init/launchd)
- Command line contains Claude-specific patterns:
  - `shell-snapshot` — shell environment capture mechanism
  - `claude_agent_sdk` — internal SDK components
  - `_bundled/claude` — bundled binaries
  - `claude-[a-f0-9]+-cwd` — temp files with hex IDs
  - `SNAPSHOT_FILE=*/.claude/` — snapshot paths (any OS)
  - `claude --ripgrep` — bundled ripgrep wrapper
  - `/.claude/shell-snapshots` — snapshot directory references

### Step 2: Review Process List

The output displays:
- **PID**: Process ID
- **CPU%**: Current CPU usage (red if >50%)
- **ELAPSED**: How long the process has been running
- **COMMAND**: Truncated command line showing what the process was doing

Example output:
```
=== Claude Code Orphan Process Finder ===

Found 3 orphaned Claude Code process(es):

PID      CPU%     ELAPSED      COMMAND (truncated)
--------------------------------------------------------------------------------
11409    100.0%   16:23:34     /bin/bash -c -l { shopt -u extglob || setopt NO_EXTENDED_GLO...
96647    99.1%    17:16:44     /bin/bash -c -l SNAPSHOT_FILE=/Users/greg/.claude/shell-snap...
98091    99.2%    01-11:38:47  /bin/bash -c -l SNAPSHOT_FILE=/Users/greg/.claude/shell-snap...

Run with --kill to terminate these processes
Run with --force to kill without confirmation
```

### Step 3: Terminate Orphaned Processes

Once you've confirmed the processes are orphaned Claude processes, kill them:

**With confirmation prompt:**
```bash
claude-cleanup --kill
```

**Without confirmation (for scripts/automation):**
```bash
claude-cleanup --force
```

### Step 4: Verify Cleanup

Run the tool again to confirm all orphans have been terminated:

```bash
claude-cleanup
```

Expected output:
```
=== Claude Code Orphan Process Finder ===

✓ No orphaned Claude Code processes found
```

---

## Examples

### Example 1: Routine Cleanup After Heavy Claude Usage

**Scenario**: You've been using Claude Code extensively and notice high CPU usage even when Claude is idle.

**Execution**:

```bash
# Check for orphans
claude-cleanup

# Output shows 5 stuck processes at 100% CPU
# Kill them with confirmation
claude-cleanup --kill
# Type 'y' when prompted

# Verify cleanup
claude-cleanup
# Output: ✓ No orphaned Claude Code processes found
```

**Result**:
- CPU usage returns to normal
- System resources freed
- No impact on active Claude sessions

---

### Example 2: Automated Cleanup via Cron

**Scenario**: You want to automatically clean up orphaned processes every hour.

**Setup**:

```bash
# Edit crontab
crontab -e

# Add hourly cleanup job
0 * * * * /Users/greg/bin/claude-cleanup --force >> /tmp/claude-cleanup.log 2>&1
```

**Benefits**:
- Automatic cleanup without manual intervention
- Log file for auditing
- Prevents accumulation of orphaned processes

---

### Example 3: Quick One-Liner Without Script

**Scenario**: You need to clean up orphans but don't have the script installed.

**Execution**:

```bash
# Find and display orphaned Claude processes
ps -eo pid,ppid,pcpu,args | awk '$2==1' | grep -E 'claude|shell-snapshot|SNAPSHOT_FILE'

# Kill all matching processes
ps -eo pid,ppid,args | awk '$2==1' | grep -E 'claude|shell-snapshot' | awk '{print $1}' | xargs kill -9
```

**Note**: The script version is safer as it uses more specific patterns and provides confirmation.

---

### Example 4: Diagnosing a Specific Orphan

**Scenario**: You want to understand what a specific orphaned process was doing before killing it.

**Execution**:

```bash
# Get full command line for a specific PID
ps -p 11409 -o pid,ppid,lstart,args

# Check what files/resources the process has open
lsof -p 11409

# Check the working directory
lsof -p 11409 | grep cwd
```

**Output interpretation**:
- `shell-snapshot` processes were capturing shell environment
- `curl` commands were MCP tool network requests
- Working directory shows which project Claude was working on

---

### Example 5: macOS LaunchAgent for Automatic Cleanup

**Scenario**: You prefer macOS-native scheduling over cron.

**Setup**:

1. Create LaunchAgent plist:

```bash
cat > ~/Library/LaunchAgents/com.user.claude-cleanup.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.user.claude-cleanup</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/greg/bin/claude-cleanup</string>
        <string>--force</string>
    </array>
    <key>StartInterval</key>
    <integer>3600</integer>
    <key>StandardOutPath</key>
    <string>/tmp/claude-cleanup.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/claude-cleanup.log</string>
</dict>
</plist>
EOF
```

2. Load the agent:

```bash
launchctl load ~/Library/LaunchAgents/com.user.claude-cleanup.plist
```

3. Verify it's running:

```bash
launchctl list | grep claude-cleanup
```

**Benefits**:
- Survives system restarts
- Native macOS scheduling
- Runs even when not logged in (if configured)

---

## Command Reference

| Command | Description |
|---------|-------------|
| `claude-cleanup` | List orphaned processes (dry run) |
| `claude-cleanup --kill` or `-k` | Kill processes with confirmation |
| `claude-cleanup --force` or `-f` | Kill processes without confirmation |
| `claude-cleanup --help` or `-h` | Show help message |

---

## Implementation Reference

Below is the complete bash script that implements this skill:

```bash
#!/bin/bash
#
# claude-cleanup - Find and kill orphaned Claude Code processes
#
# Orphaned Claude processes typically have:
#   - PPID=1 (adopted by init/launchd after parent died)
#   - Command patterns related to claude_agent_sdk, shell-snapshots, or MCP tools
#   - Often consuming 100% CPU in stuck state
#
# Usage:
#   claude-cleanup          # List orphaned Claude processes (dry run)
#   claude-cleanup --kill   # Kill all found orphaned processes
#   claude-cleanup --force  # Kill without confirmation
#

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

KILL_MODE=false
FORCE_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --kill|-k)
            KILL_MODE=true
            shift
            ;;
        --force|-f)
            KILL_MODE=true
            FORCE_MODE=true
            shift
            ;;
        --help|-h)
            echo "Usage: claude-cleanup [OPTIONS]"
            echo ""
            echo "Find and kill orphaned Claude Code processes"
            echo ""
            echo "Options:"
            echo "  --kill, -k    Kill found orphaned processes (with confirmation)"
            echo "  --force, -f   Kill without confirmation"
            echo "  --help, -h    Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Patterns that identify Claude Code processes (cross-platform)
CLAUDE_PATTERNS=(
    # Shell environment snapshot mechanism
    "shell-snapshot"
    # Claude Agent SDK internals
    "claude_agent_sdk"
    "_bundled/claude"
    # Temp files for tracking CWD (hex ID pattern)
    "claude-[a-f0-9]+-cwd"
    # Snapshot files in .claude directory (any OS home path)
    "SNAPSHOT_FILE=.*/\\.claude/"
    # Claude's bundled ripgrep wrapper
    "claude --ripgrep"
    # Generic .claude config directory in command args
    "/\\.claude/shell-snapshots"
)

# Build grep pattern
GREP_PATTERN=$(IFS='|'; echo "${CLAUDE_PATTERNS[*]}")

echo -e "${BLUE}=== Claude Code Orphan Process Finder ===${NC}"
echo ""

# Find orphaned processes (PPID=1) matching Claude patterns
# Using ps to get all processes, then filtering
ORPHANS=$(ps -eo pid,ppid,pcpu,etime,args 2>/dev/null | \
    awk '$2 == 1' | \
    grep -E "$GREP_PATTERN" 2>/dev/null || true)

if [[ -z "$ORPHANS" ]]; then
    echo -e "${GREEN}✓ No orphaned Claude Code processes found${NC}"
    exit 0
fi

# Count and display
COUNT=$(echo "$ORPHANS" | wc -l | tr -d ' ')
echo -e "${YELLOW}Found $COUNT orphaned Claude Code process(es):${NC}"
echo ""

# Header
printf "${BLUE}%-8s %-8s %-12s %s${NC}\n" "PID" "CPU%" "ELAPSED" "COMMAND (truncated)"
echo "--------------------------------------------------------------------------------"

# Display each orphan
echo "$ORPHANS" | while read -r line; do
    PID=$(echo "$line" | awk '{print $1}')
    CPU=$(echo "$line" | awk '{print $3}')
    ELAPSED=$(echo "$line" | awk '{print $4}')
    CMD=$(echo "$line" | awk '{$1=$2=$3=$4=""; print substr($0, 5, 60)}')
    
    # Color based on CPU usage
    if (( $(echo "$CPU > 50" | bc -l 2>/dev/null || echo 0) )); then
        printf "${RED}%-8s %-8s %-12s %s...${NC}\n" "$PID" "$CPU%" "$ELAPSED" "$CMD"
    else
        printf "%-8s %-8s %-12s %s...\n" "$PID" "$CPU%" "$ELAPSED" "$CMD"
    fi
done

echo ""

if [[ "$KILL_MODE" == true ]]; then
    PIDS=$(echo "$ORPHANS" | awk '{print $1}' | tr '\n' ' ')
    
    if [[ "$FORCE_MODE" == false ]]; then
        echo -e "${YELLOW}Kill these processes? [y/N]${NC} "
        read -r CONFIRM
        if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
            echo "Aborted."
            exit 0
        fi
    fi
    
    echo -e "${RED}Killing processes: $PIDS${NC}"
    for pid in $PIDS; do
        if kill -9 "$pid" 2>/dev/null; then
            echo -e "${GREEN}  ✓ Killed PID $pid${NC}"
        else
            echo -e "${YELLOW}  ⚠ Could not kill PID $pid (may have already exited)${NC}"
        fi
    done
    echo ""
    echo -e "${GREEN}Done!${NC}"
else
    echo -e "${BLUE}Run with --kill to terminate these processes${NC}"
    echo -e "${BLUE}Run with --force to kill without confirmation${NC}"
fi
```

---

## Troubleshooting

### Script reports "No orphaned processes" but CPU is still high

Check for non-orphaned Claude processes:
```bash
ps aux | grep -E 'claude|node.*claude' | grep -v grep
```

These may be active Claude sessions. Close them normally or kill if unresponsive.

### Permission denied when killing processes

Ensure you have permission to kill the process (same user or root):
```bash
sudo claude-cleanup --force
```

### Script not found

Ensure the script is in your PATH:
```bash
# Check if ~/bin is in PATH
echo $PATH | grep -q "$HOME/bin" && echo "OK" || echo "Add ~/bin to PATH"

# Or run with full path
/Users/greg/bin/claude-cleanup
```
