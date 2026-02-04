---
name: create-claude-agent
description: Create production-ready Claude agent projects using the Claude Agent SDK.
---

# Create Claude Agent

## Instructions

1. Ask user about agent's purpose and required tools (Bash, Read, Write, Edit, Grep, Glob, WebFetch, Skill)
2. **Detect working folder** (see below)
3. **Check for naming conflicts** (existing agent with same name)
4. Create project structure using templates
5. Customize `task.md` for agent's task
6. Customize `.claude/settings.local.json` permissions based on required tools
7. Initialize git repository if not exists
8. **Create virtual environment and install dependencies**:
   - Run `python3 -m venv .venv`
   - Activate venv and run `pip install -r requirements.txt`
9. Output concise confirmation with next steps (no .md explanation files)

---

## Prerequisites

Before creating an agent, verify:
- Python 3.10+ is installed
- pip is available
- Claude CLI is installed (`npm install -g @anthropic-ai/claude-code` or `brew install claude`)
- `ANTHROPIC_API_KEY` is set in `.env` file

---

## Working Folder Detection

Before creating files, detect the project structure:

1. **Check if folder is empty or has existing structure** (look for `src/`, `agents/`, `scripts/`, etc.)
2. **Determine agent location based on structure:**

| Condition | Agent Location | Project Root |
|-----------|---------------|--------------|
| Empty folder | `./src/`, `./logs/`, `./claude_agent.py` | Current dir |
| Has `src/` | `src/agents/{agent-name}/` | Current dir |
| Has `agents/` | `agents/{agent-name}/` | Current dir |
| Has `scripts/` | `scripts/{agent-name}/` | Current dir |
| Other | `./{agent-name}/` | Current dir |

3. **Check for naming conflicts** — if agent location exists, prompt user to overwrite, rename, or abort
4. **Place project-level files at root:**
   - `.vscode/launch.json` → project root (merge if exists)
   - `requirements.txt` → project root (merge if exists)
   - `.gitignore` → project root (merge if exists)
   - `.env.example` → project root (create if not exists)

---

## Merging Existing Files

When project-level files already exist, merge rather than overwrite:

- **`.vscode/launch.json`**: Append new configuration to `configurations` array (avoid duplicates by name)
- **`requirements.txt`**: Append only missing packages
- **`.gitignore`**: Append only missing patterns

---

## Post-Creation Steps

After creating files:

1. **Initialize git** (if not already a repo)
2. **Create virtual environment**:
   ```bash
   python3 -m venv .venv
   ```
3. **Activate and install dependencies**:
   ```bash
   source .venv/bin/activate  # Unix/macOS
   pip install -r requirements.txt
   ```

---

## Project Structure

```
{project-root}/
├── .vscode/
│   └── launch.json                 # Debug config
├── .gitignore
├── requirements.txt
├── {agent-location}/               # Adaptive (see above)
│   ├── .claude/
│   │   └── settings.local.json
│   ├── src/
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   ├── schemas.py
│   │   └── exceptions.py
│   ├── logs/                       # Execution logs
│   ├── output/                     # Message history (JSONL)
│   ├── task.md
│   └── claude_agent.py
```

---

## File Templates

### requirements.txt

```txt
claude-agent-sdk
pydantic
python-dotenv
colorlog
```

> **Note**: Use `claude-agent-sdk`. The SDK wraps the Claude CLI.

### .env.example

```env
# Required: Your Anthropic API key
ANTHROPIC_API_KEY=your-api-key-here

# Optional: Agent configuration
# AGENT_MODEL=claude-sonnet-4-5
# AGENT_MAX_TURNS=50
# AGENT_TIMEOUT_SECONDS=1800
# AGENT_MAX_BUDGET_USD=10.0
# AGENT_PERMISSION_MODE=bypassPermissions  # default: bypassPermissions (allows all tools)
```

### .gitignore

```gitignore
# Agent logs and output
logs/
output/
*.jsonl

# Environment and secrets
.env
.env.local

# Python
__pycache__/
*.pyc
*.pyo
.venv/
venv/
*.egg-info/

# IDE
.idea/
*.swp
*.swo
```

### .claude/settings.local.json

Customize permissions based on agent's required capabilities:

```json
{
  "permissions": {
    "allow": [
      "Bash(curl:*)",
      "Bash(wget:*)",
      "Bash(git:*)",
      "Bash(find:*)",
      "Bash(grep:*)",
      "Bash(ls:*)",
      "Bash(cat:*)",
      "Bash(python:*)",
      "Bash(jq:*)",
      "Bash(date:*)"
    ],
    "deny": [
      "Bash(rm -rf:*)",
      "Bash(sudo:*)",
      "Bash(chmod 777:*)",
      "Bash(> /dev:*)"
    ]
  }
}
```

**Permission Customization Guide:**

| Agent Type | Additional Allow | Additional Deny |
|------------|-----------------|-----------------|
| Web scraper | `Bash(node:*)` | `Bash(ssh:*)` |
| Data processor | `Bash(jq:*)`, `Bash(awk:*)` | `Bash(curl:*)` |
| DevOps | `Bash(docker:*)`, `Bash(kubectl:*)` | — |
| File manager | `Bash(mv:*)`, `Bash(cp:*)` | `Bash(curl:*)` |

> **Security**: Start with minimal permissions, add as needed. Never allow `sudo` or destructive operations without explicit user confirmation.

### .vscode/launch.json

Use `{agent-path}` relative to project root. Uses cross-platform Python path.

> **Critical**: Do NOT use `envFile` - the SDK uses Claude CLI authentication, not environment variables.

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Run {agent-name}",
      "type": "debugpy",
      "request": "launch",
      "python": "${command:python.interpreterPath}",
      "program": "${workspaceFolder}/{agent-path}/claude_agent.py",
      "args": [],
      "console": "integratedTerminal",
      "justMyCode": true,
      "env": {
        "PYTHONPATH": "${workspaceFolder}/{agent-path}",
        "PYTHONUNBUFFERED": "1",
      },
      "cwd": "${workspaceFolder}/{agent-path}",
      "envFile": "${workspaceFolder}/.env"
    }
  ]
}
```

> **Cross-platform note**: Using `${command:python.interpreterPath}` ensures compatibility with Windows and Unix systems. Requires Python extension in VS Code/Cursor.

### task.md

```markdown
# Agent Task

## Objective
Describe what the agent should accomplish.

## Instructions
1. Step one
2. Step two
3. Step three

## Expected Output
Describe what output.json should contain.
```

### src/__init__.py

```python
"""Agent source package."""
```

### src/exceptions.py

```python
"""Agent exceptions."""


class AgentError(Exception):
    """Base exception."""
    pass


class ConfigurationError(AgentError):
    """Configuration error."""
    pass


class SessionIncompleteError(AgentError):
    """Session did not complete."""
    pass


class MaxTurnsExceededError(AgentError):
    """Exceeded turn limit."""
    pass


class ServerError(AgentError):
    """API error."""
    pass
```

### src/schemas.py

```python
"""Data models."""
from enum import StrEnum
from typing import Any, Optional

from pydantic import BaseModel, Field


class TaskStatus(StrEnum):
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    ERROR = "ERROR"


class LLMMetrics(BaseModel):
    """Session metrics."""
    model: str
    duration_ms: int
    num_turns: int
    session_id: str
    total_cost_usd: Optional[float] = None


class AgentResult(BaseModel):
    """Agent execution result."""
    status: TaskStatus
    summary: str
    details: Optional[str] = None
    data: dict[str, Any] = Field(default_factory=dict)
    metrics: Optional[LLMMetrics] = None
    error: Optional[str] = None
    message_history: list[dict[str, Any]] = Field(default_factory=list)
```

### src/agent.py

```python
"""Agent execution logic."""
import asyncio
import logging
import os
import time
from typing import Any, Optional

from claude_agent_sdk import query
from claude_agent_sdk.types import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    SystemMessage,
    UserMessage,
)

from .exceptions import (
    AgentError,
    MaxTurnsExceededError,
    ServerError,
    SessionIncompleteError,
)
from .schemas import AgentResult, LLMMetrics, TaskStatus

logger = logging.getLogger(__name__)

# ANSI color codes for verbose output
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
RESET = "\033[0m"
BOLD = "\033[1m"


def serialize_block(block: Any) -> dict[str, Any]:
    """Convert SDK message block to serializable dict."""
    block_type = getattr(block, 'type', None)
    if block_type == 'tool_use':
        return {
            "type": "tool_use",
            "name": getattr(block, 'name', ''),
            "input": getattr(block, 'input', {}),
        }
    elif block_type == 'tool_result':
        content = getattr(block, 'content', '')
        return {
            "type": "tool_result",
            "is_error": getattr(block, 'is_error', False),
            "content": str(content)[:500] if content else "",
        }
    elif block_type == 'text':
        return {"type": "text", "text": getattr(block, 'text', '')}
    return {"type": str(block_type), "raw": str(block)[:200]}


def print_verbose(prefix: str, content: str, color: str = RESET) -> None:
    """Print verbose output with colored prefix."""
    print(f"{color}{BOLD}[{prefix}]{RESET} {content}", flush=True)


async def run_agent_with_timeout(
    task: str,
    working_dir: str = ".",
    additional_dirs: Optional[list[str]] = None,
    verbose: bool = True,
) -> AgentResult:
    """Run the agent with configuration from environment."""
    max_turns = int(os.getenv("AGENT_MAX_TURNS", "50"))
    timeout_seconds = int(os.getenv("AGENT_TIMEOUT_SECONDS", "1800"))
    max_budget = float(os.getenv("AGENT_MAX_BUDGET_USD", "10.0"))
    model = os.getenv("AGENT_MODEL", "claude-sonnet-4-5")
    permission_mode = os.getenv("AGENT_PERMISSION_MODE", "bypassPermissions")

    logger.info(f"Starting agent with model={model}, max_turns={max_turns}")

    # add_dirs grants the agent read/write access to directories outside cwd.
    # Use this when the agent needs to access external resources, configs, or data.
    options = ClaudeAgentOptions(
        model=model,
        cwd=working_dir,
        permission_mode=permission_mode,
        max_turns=max_turns,
        max_budget_usd=max_budget,
        add_dirs=additional_dirs or [],  # Additional allowed directories
    )

    try:
        result_text = ""
        session_id = ""
        num_turns = 0
        total_cost = None
        message_history: list[dict[str, Any]] = []

        async def run_query():
            nonlocal result_text, session_id, num_turns, total_cost
            turn_count = 0

            async for message in query(prompt=task, options=options):
                if isinstance(message, AssistantMessage):
                    turn_count += 1
                    content = getattr(message, 'message', None)
                    serialized_blocks: list[dict[str, Any]] = []
                    if content:
                        msg_content = getattr(content, 'content', None)
                        if msg_content:
                            for block in msg_content:
                                serialized_blocks.append(serialize_block(block))
                                if verbose:
                                    block_type = getattr(block, 'type', None)
                                    if block_type == 'text':
                                        text = getattr(block, 'text', '')
                                        if text:
                                            preview = text[:500] + ("..." if len(text) > 500 else "")
                                            print_verbose("ASSISTANT", preview, GREEN)
                                    elif block_type == 'tool_use':
                                        tool_name = getattr(block, 'name', 'unknown')
                                        tool_input = getattr(block, 'input', {})
                                        print_verbose("TOOL", f"{tool_name}", YELLOW)
                                        if tool_name == 'Bash' and 'command' in tool_input:
                                            print_verbose("CMD", tool_input['command'][:200], MAGENTA)
                    message_history.append({
                        "turn": turn_count,
                        "role": "assistant",
                        "timestamp": time.time(),
                        "content": serialized_blocks,
                    })
                    if verbose:
                        print_verbose(f"TURN {turn_count}", "", CYAN)

                elif isinstance(message, UserMessage):
                    content = getattr(message, 'message', None)
                    if content:
                        msg_content = getattr(content, 'content', None)
                        if msg_content and verbose:
                            for block in msg_content:
                                block_type = getattr(block, 'type', None)
                                if block_type == 'tool_result':
                                    result_content = getattr(block, 'content', '')
                                    if isinstance(result_content, str) and result_content:
                                        preview = result_content[:300] + ("..." if len(result_content) > 300 else "")
                                        print_verbose("RESULT", preview, BLUE)

                elif isinstance(message, SystemMessage):
                    if verbose:
                        print_verbose("SYSTEM", str(message)[:200], CYAN)

                if isinstance(message, ResultMessage):
                    result_text = getattr(message, 'result', str(message))
                    session_id = getattr(message, 'session_id', '')
                    num_turns = getattr(message, 'num_turns', turn_count)
                    total_cost = getattr(message, 'total_cost_usd', None)
                    if verbose:
                        cost_str = f"${total_cost:.4f}" if total_cost else "N/A"
                        print_verbose("COMPLETE", f"Turns: {num_turns}, Cost: {cost_str}", GREEN)

        await asyncio.wait_for(run_query(), timeout=timeout_seconds)

        metrics = LLMMetrics(
            model=model,
            duration_ms=0,
            num_turns=num_turns,
            session_id=session_id,
            total_cost_usd=total_cost,
        )

        return AgentResult(
            status=TaskStatus.SUCCESS,
            summary="Task completed successfully",
            details=result_text,
            metrics=metrics,
            message_history=message_history,
        )

    except asyncio.TimeoutError:
        raise AgentError(f"Agent timed out after {timeout_seconds} seconds")
    except Exception as e:
        error_msg = str(e)
        if "max_turns" in error_msg.lower():
            raise MaxTurnsExceededError(f"Exceeded {max_turns} turns")
        if "server" in error_msg.lower() or "api" in error_msg.lower():
            raise ServerError(f"API error: {error_msg}")
        if "incomplete" in error_msg.lower():
            raise SessionIncompleteError(f"Session incomplete: {error_msg}")
        raise AgentError(f"Agent execution failed: {error_msg}")
```

### claude_agent.py

```python
#!/usr/bin/env python3
"""
Claude Agent entry point.

Usage:
    python claude_agent.py                    # Run with default task.md
    python claude_agent.py --task-file x.md   # Custom task file
    python claude_agent.py --output out.json  # Save output to file
    python claude_agent.py -v                 # Verbose output

Prerequisites:
    pip install -r requirements.txt
    Set ANTHROPIC_API_KEY in .env file
"""
import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import colorlog
from dotenv import load_dotenv

from src.agent import run_agent_with_timeout
from src.schemas import AgentResult, TaskStatus

# Load environment variables from .env file
load_dotenv()


def setup_logging(logs_dir: Path) -> logging.Logger:
    """Configure logging with console and file handlers."""
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Console handler with colors
    console_handler = colorlog.StreamHandler()
    console_handler.setFormatter(colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s [%(levelname)s]%(reset)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    ))

    # File handler for persistent logs
    log_file = logs_dir / f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    logger = colorlog.getLogger(__name__)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    return logger


def load_task_file(task_file: Path) -> str:
    if not task_file.exists():
        raise FileNotFoundError(f"Task file not found: {task_file}")
    return task_file.read_text(encoding="utf-8")


def save_message_history(
    result: AgentResult,
    output_dir: Path,
    session_id: str,
    logger: logging.Logger,
) -> None:
    """Save message history to JSONL file."""
    if not result.message_history:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"session_{session_id}.jsonl"
    with output_file.open("w", encoding="utf-8") as f:
        for msg in result.message_history:
            f.write(json.dumps(msg) + "\n")
    logger.info(f"Message history saved to: {output_file}")


def main() -> int:
    base_dir = Path(__file__).parent
    logs_dir = base_dir / "logs"
    output_dir = base_dir / "output"
    logger = setup_logging(logs_dir)

    parser = argparse.ArgumentParser(description="Execute Claude agent task")
    parser.add_argument(
        "--dir",
        default=".",
        help="Working directory (default: current)",
    )
    parser.add_argument(
        "--task-file",
        type=Path,
        default=Path("task.md"),
        help="Task file (default: task.md)",
    )
    parser.add_argument(
        "--output",
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--add-dir",
        action="append",
        default=[],
        help="Additional directories",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    try:
        task: str = load_task_file(args.task_file)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    logger.info(f"Task file: {args.task_file}")
    logger.info(f"Working directory: {args.dir}")

    try:
        result = asyncio.run(
            run_agent_with_timeout(
                task=task,
                working_dir=args.dir,
                additional_dirs=args.add_dir,
                verbose=args.verbose,
            )
        )

        # Save message history to JSONL
        session_id = (
            result.metrics.session_id
            if result.metrics
            else datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        save_message_history(result, output_dir, session_id, logger)

        # Output result JSON (excluding message_history to avoid duplication)
        output_json: str = result.model_dump_json(
            indent=2, exclude={"message_history"}
        )

        if args.output:
            Path(args.output).write_text(output_json, encoding="utf-8")
            logger.info(f"Result saved to: {args.output}")
        else:
            print(output_json)

        if result.metrics:
            logger.info(
                f"Turns: {result.metrics.num_turns}, "
                f"Cost: ${result.metrics.total_cost_usd:.4f}"
                if result.metrics.total_cost_usd
                else f"Turns: {result.metrics.num_turns}"
            )

        return 0 if result.status == TaskStatus.SUCCESS else 1

    except Exception as e:
        logger.error(f"Agent failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

---

## Auxiliary script

You may want to create additional python script as part of the agentic tools so it could use them instead of direct access to API or database. If so, add corresponding instructions to the agent promt (task.md file).

## Completion Output

After creating the agent, output only:

```
✓ Agent "{agent-name}" created at {agent-path}/
✓ Virtual environment created and dependencies installed

To run:
  # Copy .env.example to .env and set your API key
  cp .env.example .env
  # Edit .env and add your ANTHROPIC_API_KEY
  
  # Activate virtual environment
  source .venv/bin/activate      # Unix/macOS
  # .\.venv\Scripts\Activate.ps1  # Windows PowerShell
  
  # Run the agent
  python {agent-path}/claude_agent.py

Or press F5 in VS Code/Cursor (select .venv interpreter first).
```

---

## Available Tools

| Tool | Capability |
|------|------------|
| `Bash` | Execute shell commands |
| `Read` | Read file contents |
| `Write` | Create/overwrite files |
| `Edit` | Modify existing files |
| `Grep` | Search patterns in files |
| `Glob` | Find files by pattern |
| `WebFetch` | Fetch content from URLs |
| `Skill` | Execute custom skills |

---

## Configuration Options

Environment variables for customization:

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_MODEL` | `claude-sonnet-4-5` | Model alias |
| `AGENT_MAX_TURNS` | `50` | Maximum conversation turns |
| `AGENT_TIMEOUT_SECONDS` | `1800` | Execution timeout |
| `AGENT_MAX_BUDGET_USD` | `10.0` | Maximum API cost |
| `AGENT_PERMISSION_MODE` | `bypassPermissions` | Permission mode |

**Valid permission modes:**
- `default` - CLI prompts for dangerous tools
- `acceptEdits` - Auto-accept file edits
- `plan` - Planning mode, no execution
- `bypassPermissions` - Allow all tools (use with caution)

**Valid model aliases:**
- `claude-sonnet-4-5` - Fast, capable (recommended)
- `claude-opus-4` - Most capable

**SDK Options (`ClaudeAgentOptions`):**

| Parameter | Type | Description |
|-----------|------|-------------|
| `cwd` | `str` | Working directory for the agent |
| `add_dirs` | `list[str]` | **Additional directories the agent can access** (read/write). Use when agent needs files outside `cwd` — e.g., shared configs, data directories, or external resources |
| `model` | `str` | Model alias |
| `permission_mode` | `str` | Permission mode (see above) |
| `max_turns` | `int` | Maximum conversation turns |
| `max_budget_usd` | `float` | Maximum API cost |

---

## Troubleshooting

### Common Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| `ModuleNotFoundError: claude_agent_sdk` | Package not installed | Run `pip install claude-agent-sdk` |
| `Invalid API key` | API key missing or invalid | Check `.env` file has valid `ANTHROPIC_API_KEY` |
| `Command failed with exit code 1` | Various causes | Check logs, verify API key, check Claude CLI installation |
| `python3: command not found` | Python not in PATH | Install Python 3.10+ or fix PATH |
| `venv creation failed` | Missing python3-venv | `sudo apt install python3-venv` (Linux) |
| Agent timeout | Task too complex | Increase `AGENT_TIMEOUT_SECONDS` or simplify task |
| Logs directory growing | No cleanup | Manually clean `logs/` directory |
| Permission denied errors | Restrictive settings.local.json | Add required commands to `allow` list |

### Authentication

Set your API key in the `.env` file:

```bash
# Copy example and edit
cp .env.example .env

# Add your key to .env
ANTHROPIC_API_KEY=sk-ant-api03-xxxxx...
```

Verify the key works (from within existing venv):
```bash
curl -s https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-4-5-20250929","max_tokens":10,"messages":[{"role":"user","content":"Hi"}]}'
```

### Debugging Tips

1. **Check logs**: Review `logs/agent_{timestamp}.log` for detailed execution trace
2. **Verify API key**: Run `echo $ANTHROPIC_API_KEY | head -c 15` (should show `sk-ant-api...`)
3. **Test SDK**: Run `python -c "from claude_agent_sdk import query; print('OK')"`
4. **Check permissions**: Verify `.claude/settings.local.json` allows required operations
5. **Inspect message history**: Check `output/session_{session_id}.jsonl` for full conversation trace
6. **Test CLI**: Run `claude -p "hello"` to verify Claude CLI works

### Platform-Specific Notes

**macOS/Linux:**
- Use `source .venv/bin/activate`
- Set env vars with `export VAR=value`
- Load .env automatically with `python-dotenv`

**Windows:**
- Use `.\.venv\Scripts\Activate.ps1` (PowerShell) or `.venv\Scripts\activate.bat` (CMD)
- Set env vars with `$env:VAR="value"` (PowerShell) or `set VAR=value` (CMD)
- Path separators: use `\` in paths or raw strings

**Docker:**
- Mount `.env` file: `-v $(pwd)/.env:/app/.env`
- Or set env directly: `-e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY`
