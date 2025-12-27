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
5. Customize `src/prompts/system.j2` for agent's role
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
│   │   ├── prompts/
│   │   │   ├── system.j2
│   │   │   └── user.j2
│   │   ├── schemas.py
│   │   └── exceptions.py
│   ├── logs/
│   ├── task.md
│   └── claude_agent.py
```

---

## File Templates

### requirements.txt

```txt
claude-agent-sdk
jinja2
pydantic
pydantic-settings
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
# AGENT_PERMISSION_MODE=acceptEdits
```

### .gitignore

```gitignore
# Agent logs and output
logs/
output.json
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

# Agent artifacts
fetch_hn_stories.sh
stories_data.json
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
```

### src/agent.py

```python
"""Core agent implementation."""
import asyncio
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    query,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolUseBlock,
)
from jinja2 import Environment, FileSystemLoader

from .exceptions import (
    AgentError,
    MaxTurnsExceededError,
    ServerError,
)
from .schemas import AgentResult, LLMMetrics, TaskStatus

# Configuration (from environment with sensible defaults)
MAX_TURNS: int = int(os.getenv("AGENT_MAX_TURNS", "50"))
TIMEOUT_SECONDS: int = int(os.getenv("AGENT_TIMEOUT_SECONDS", "1800"))
MAX_BUDGET_USD: float = float(os.getenv("AGENT_MAX_BUDGET_USD", "10.0"))

# Model aliases: "claude-sonnet-4-5", "claude-opus-4"
DEFAULT_MODEL: str = os.getenv("AGENT_MODEL", "claude-sonnet-4-5")

# Permission modes: "default", "acceptEdits", "plan", "bypassPermissions"
DEFAULT_PERMISSION_MODE: str = os.getenv("AGENT_PERMISSION_MODE", "acceptEdits")

# Paths
SRC_DIR: Path = Path(__file__).parent
PROMPTS_DIR: Path = SRC_DIR / "prompts"
LOGS_DIR: Path = SRC_DIR.parent / "logs"


def _setup_jinja() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(PROMPTS_DIR)),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _render_prompt(template_name: str, **kwargs: Any) -> str:
    env: Environment = _setup_jinja()
    template = env.get_template(template_name)
    return template.render(**kwargs)


def _parse_output_json(working_dir: str) -> dict[str, Any]:
    output_path: Path = Path(working_dir) / "output.json"
    if output_path.exists():
        try:
            return json.loads(output_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _create_result(
    output_data: dict[str, Any],
    metrics: Optional[LLMMetrics] = None,
    error: Optional[str] = None,
) -> AgentResult:
    status_str: str = output_data.get("status", "FAILED")
    try:
        status = TaskStatus(status_str)
    except ValueError:
        status = TaskStatus.FAILED

    return AgentResult(
        status=status,
        summary=output_data.get("summary", "No summary provided"),
        details=output_data.get("details"),
        data=output_data.get("data", {}),
        metrics=metrics,
        error=error,
    )


def _extract_text_from_message(message: Any) -> str:
    """Extract text content from a message object."""
    if isinstance(message, AssistantMessage):
        texts: list[str] = []
        for block in message.content:
            if isinstance(block, TextBlock):
                texts.append(block.text)
            elif isinstance(block, ToolUseBlock):
                texts.append(f"[Tool: {block.name}]")
        return "\n".join(texts)
    if isinstance(message, ResultMessage):
        cost: float = message.total_cost_usd or 0.0
        return f"[Result] Cost: ${cost:.4f}, Turns: {message.num_turns}"
    if isinstance(message, SystemMessage):
        return f"[System: {message.subtype}]"
    return str(message)


async def run_agent(
    task: str,
    working_dir: str = ".",
    additional_dirs: Optional[list[str]] = None,
    parameters: Optional[dict[str, Any]] = None,
) -> AgentResult:
    """Run the agent with the given task."""
    session_id: str = str(uuid.uuid4())[:8]
    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir: Path = LOGS_DIR / f"{timestamp}_{session_id}"
    log_dir.mkdir(parents=True, exist_ok=True)

    working_path: Path = Path(working_dir).resolve()
    output_file: str = str(working_path / "output.json")

    system_prompt: str = _render_prompt(
        "system.j2",
        **(parameters or {}),
    )

    user_prompt: str = _render_prompt(
        "user.j2",
        working_dir=str(working_path),
        output_file=output_file,
        task=task,
        **(parameters or {}),
    )

    # Combine system prompt with user prompt for the query
    full_prompt: str = f"{system_prompt}\n\n---\n\n{user_prompt}"

    # Configure agent options
    # Note: Uses Claude CLI authentication (run `claude login` first)
    options = ClaudeAgentOptions(
        model=DEFAULT_MODEL,
        permission_mode=DEFAULT_PERMISSION_MODE,
        cwd=working_path,
        max_turns=MAX_TURNS,
        max_budget_usd=MAX_BUDGET_USD,
        allowed_tools=["Bash", "Read", "Write", "Edit", "Grep", "Glob", "WebFetch"],
    )

    start_time: datetime = datetime.now()
    full_response: list[str] = []
    num_turns: int = 0
    total_cost: float = 0.0
    claude_session_id: str = ""

    try:
        async for message in query(prompt=full_prompt, options=options):
            text_content: str = _extract_text_from_message(message)

            if isinstance(message, AssistantMessage):
                full_response.append(text_content)
                print(f"Assistant: {text_content[:200]}...")
            elif isinstance(message, ResultMessage):
                total_cost = message.total_cost_usd or 0.0
                num_turns = message.num_turns
                claude_session_id = message.session_id
                print(f"Completed: {text_content}")
            elif isinstance(message, SystemMessage):
                print(f"System: {message.subtype}")

        duration_ms: int = int((datetime.now() - start_time).total_seconds() * 1000)

        metrics = LLMMetrics(
            model=DEFAULT_MODEL,
            duration_ms=duration_ms,
            num_turns=num_turns,
            session_id=claude_session_id or session_id,
            total_cost_usd=total_cost if total_cost > 0 else None,
        )

        output_data: dict[str, Any] = _parse_output_json(str(working_path))

        log_file: Path = log_dir / "agent.log"
        log_file.write_text("\n".join(full_response), encoding="utf-8")

        output_log: Path = log_dir / "output.json"
        output_log.write_text(json.dumps(output_data, indent=2), encoding="utf-8")

        return _create_result(output_data, metrics)

    except asyncio.TimeoutError as e:
        raise MaxTurnsExceededError(f"Agent timed out after {TIMEOUT_SECONDS}s") from e
    except Exception as e:
        error_msg: str = str(e)
        if "server" in error_msg.lower() or "api" in error_msg.lower():
            raise ServerError(f"API error: {error_msg}") from e
        raise AgentError(f"Agent failed: {error_msg}") from e


async def run_agent_with_timeout(
    task: str,
    working_dir: str = ".",
    additional_dirs: Optional[list[str]] = None,
    parameters: Optional[dict[str, Any]] = None,
) -> AgentResult:
    return await asyncio.wait_for(
        run_agent(task, working_dir, additional_dirs, parameters),
        timeout=TIMEOUT_SECONDS,
    )
```

### src/prompts/system.j2

```jinja2
**Role**
You are an intelligent automation agent.
Your goal is to complete the assigned task accurately and efficiently.

**Available Tools**
- **Bash**: Execute shell commands (curl, wget, git, jq, etc.)
- **Read**: Read file contents
- **Write**: Create or overwrite files
- **Edit**: Modify existing files
- **Grep**: Search for patterns in files
- **Glob**: Find files by pattern
- **WebFetch**: Fetch content from URLs

**Rules**
- Execute deterministically; avoid unnecessary actions
- Make minimal changes to achieve the goal
- Verify results before reporting completion
- generate and use python scripts when it would be more efficient and effective for a sequence of steps or API calls instead of executing a series of steps one by one from agentic loop. 

**Workflow**
1. **Understand**: Analyze the task requirements
2. **Execute**: Perform each step carefully
3. **Verify**: Confirm the task is complete
4. **Report**: Create output.json with results

**Output Requirement**
Create `output.json`:
```json
{
  "status": "SUCCESS" | "PARTIAL" | "FAILED",
  "summary": "Brief description of outcome",
  "data": { ... }
}
```
```

### src/prompts/user.j2

```jinja2
**Working Directory**: {{ working_dir }}
**Output File**: {{ output_file }}

**Task**:
{{ task }}

Execute this task following your workflow. Create output.json when complete.
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

Prerequisites:
    pip install -r requirements.txt
    Set ANTHROPIC_API_KEY in .env file
"""
import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

import colorlog
from dotenv import load_dotenv

from src.agent import run_agent_with_timeout
from src.schemas import TaskStatus

# Load environment variables from .env file
load_dotenv()


def setup_logging() -> logging.Logger:
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
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
    logger = colorlog.getLogger(__name__)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def load_task_file(task_file: Path) -> str:
    if not task_file.exists():
        raise FileNotFoundError(f"Task file not found: {task_file}")
    return task_file.read_text(encoding="utf-8")


def main() -> int:
    logger = setup_logging()

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
    args = parser.parse_args()

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
            )
        )
        output: str = result.model_dump_json(indent=2)

        if args.output:
            Path(args.output).write_text(output, encoding="utf-8")
            logger.info(f"Result saved to: {args.output}")
        else:
            print(output)

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
| `AGENT_PERMISSION_MODE` | `acceptEdits` | Permission mode |

**Valid permission modes:**
- `default` - CLI prompts for dangerous tools
- `acceptEdits` - Auto-accept file edits
- `plan` - Planning mode, no execution
- `bypassPermissions` - Allow all tools (use with caution)

**Valid model aliases:**
- `claude-sonnet-4-5` - Fast, capable (recommended)
- `claude-opus-4` - Most capable

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

1. **Check logs**: Review `logs/{session-id}/agent.log` for detailed execution trace
2. **Verify API key**: Run `echo $ANTHROPIC_API_KEY | head -c 15` (should show `sk-ant-api...`)
3. **Test SDK**: Run `python -c "from claude_agent_sdk import query; print('OK')"`
4. **Check permissions**: Verify `.claude/settings.local.json` allows required operations
5. **Inspect output**: Check `logs/{session-id}/output.json` for agent's final result
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
