---
name: create-claude-agent
description: Create production-ready Claude agent projects using the Claude Code SDK.
---

# Create Claude Agent

## Instructions

1. Ask user about agent's purpose and required tools (Bash, Read, Write, Edit, Grep, Skill)
2. **Detect working folder** (see below)
3. **Check for naming conflicts** (existing agent with same name)
4. Create project structure using templates
5. Customize `src/prompts/system.j2` for agent's role
6. Customize `.claude/settings.local.json` permissions based on required tools
7. Create `.venv` and install dependencies
8. Initialize git repository if not exists
9. Output concise confirmation with next steps (no .md explanation files)

---

## Prerequisites

Before creating an agent, verify:

```bash
# Check Python version (requires 3.10+)
python3 --version | grep -E "3\.(1[0-9]|[2-9][0-9])" || echo "ERROR: Python 3.10+ required"

# Verify pip is available
python3 -m pip --version || echo "ERROR: pip not available"
```

---

## Working Folder Detection

Before creating files, detect the project structure:

**Step 1: Check if folder is empty**
```bash
[ -z "$(ls -A . 2>/dev/null)" ] && echo "empty" || echo "not empty"
```

**Step 2: Detect existing structure**
```bash
ls -d */ 2>/dev/null | head -5
ls .vscode/launch.json requirements.txt pyproject.toml 2>/dev/null
```

**Step 3: Determine agent location**

| Condition | Agent Location | Project Root |
|-----------|---------------|--------------|
| Empty folder | `./src/`, `./logs/`, `./claude_agent.py` | Current dir |
| Has `src/` | `src/agents/{agent-name}/` | Current dir |
| Has `agents/` | `agents/{agent-name}/` | Current dir |
| Has `scripts/` | `scripts/{agent-name}/` | Current dir |
| Other | `./{agent-name}/` | Current dir |

**Step 4: Check for naming conflicts**
```bash
# If agent location already exists, prompt for action
if [ -d "{agent-location}" ]; then
    echo "Directory {agent-location} already exists. Options:"
    echo "  1. Overwrite existing"
    echo "  2. Choose different name"
    echo "  3. Abort"
fi
```

**Step 5: Place project-level files at root**
- `.venv/` → project root
- `.vscode/launch.json` → project root (merge if exists)
- `requirements.txt` → project root (merge if exists)
- `.gitignore` → project root (merge if exists)
- `.env.example` → project root (create if not exists)

---

## Merging Existing Files

### Merging launch.json

When `.vscode/launch.json` exists, append new configuration to `configurations` array:

```bash
# Read existing configurations, add new one, write back
# Use jq if available, otherwise Python:
python3 -c "
import json
from pathlib import Path

launch_file = Path('.vscode/launch.json')
config = json.loads(launch_file.read_text())
new_config = {
    'name': 'Run {agent-name}',
    'type': 'debugpy',
    'request': 'launch',
    # ... rest of config
}
# Avoid duplicates by name
existing_names = [c['name'] for c in config.get('configurations', [])]
if new_config['name'] not in existing_names:
    config['configurations'].append(new_config)
    launch_file.write_text(json.dumps(config, indent=2))
"
```

### Merging requirements.txt

When `requirements.txt` exists, append only missing packages:

```bash
# Required packages for claude-agent
AGENT_DEPS="claude-code-sdk jinja2 pydantic pydantic-settings python-dotenv colorlog"

for pkg in $AGENT_DEPS; do
    grep -qi "^${pkg}" requirements.txt || echo "$pkg" >> requirements.txt
done
```

### Merging .gitignore

Append only missing patterns:

```bash
PATTERNS="logs/ .env __pycache__/ *.pyc .venv/"
for pattern in $PATTERNS; do
    grep -qxF "$pattern" .gitignore 2>/dev/null || echo "$pattern" >> .gitignore
done
```

---

## Setup Commands

After creating files, run:

```bash
# Create venv if not exists (with error handling)
if [ ! -d .venv ]; then
    python3 -m venv .venv || {
        echo "ERROR: Failed to create virtual environment"
        echo "Ensure python3-venv is installed: sudo apt install python3-venv"
        exit 1
    }
fi

# Activate (cross-platform)
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
elif [ -f .venv/Scripts/activate ]; then
    source .venv/Scripts/activate
else
    echo "ERROR: Could not find venv activation script"
    exit 1
fi

# Upgrade pip first (avoid dependency resolver issues)
pip install --upgrade pip

# Install dependencies with error handling
pip install -r requirements.txt || {
    echo "ERROR: Failed to install dependencies"
    echo "Check network connection and requirements.txt syntax"
    exit 1
}
```

### Initialize Git Repository

```bash
# Initialize git if not already a repo
if [ ! -d .git ]; then
    git init
    git add .gitignore
    echo "Git repository initialized"
fi
```

---

## Project Structure

```
{project-root}/
├── .venv/                          # Virtual environment
├── .vscode/
│   └── launch.json                 # Debug config (uses .venv)
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
claude-code-sdk>=0.1.0
jinja2>=3.1.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
python-dotenv>=1.0.0
colorlog>=6.0.0
```

> **Note**: Pin major versions to avoid breaking changes. Update periodically.

### .env.example

```env
# Required: Your Anthropic API key
ANTHROPIC_API_KEY=your-api-key-here

# Optional: Model selection (default: claude-sonnet-4-5-20250929)
# ANTHROPIC_MODEL=claude-sonnet-4-5-20250929

# Optional: Agent configuration
# AGENT_MAX_TURNS=100
# AGENT_TIMEOUT_SECONDS=1800
# AGENT_ENABLE_SKILLS=false
```

### .gitignore

```gitignore
# Agent logs and output
logs/
output.json

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
      "Bash(python:*)"
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

Use `{agent-path}` relative to project root. Uses cross-platform Python path:

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
        "PYTHONUNBUFFERED": "1"
      },
      "cwd": "${workspaceFolder}/{agent-path}",
      "envFile": "${workspaceFolder}/.env"
    }
  ]
}
```

> **Cross-platform note**: Using `${command:python.interpreterPath}` instead of hardcoded `.venv/bin/python` ensures compatibility with Windows (`.venv\Scripts\python.exe`) and Unix systems. Requires Python extension in VS Code/Cursor.

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
    """Configuration error (e.g., missing API key)."""
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
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from claude_code_sdk import ClaudeCodeOptions, ClaudeCodeClient, ResultMessage
from jinja2 import Environment, FileSystemLoader

from .exceptions import AgentError, MaxTurnsExceededError, ServerError, SessionIncompleteError
from .schemas import AgentResult, LLMMetrics, TaskStatus

# Configuration (from environment with sensible defaults)
MODEL: str = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
ALLOWED_TOOLS: list[str] = os.environ.get("AGENT_ALLOWED_TOOLS", "Bash,Read,Write,Edit,Grep").split(",")
MAX_TURNS: int = int(os.environ.get("AGENT_MAX_TURNS", "100"))
TIMEOUT_SECONDS: int = int(os.environ.get("AGENT_TIMEOUT_SECONDS", "1800"))
ENABLE_SKILLS: bool = os.environ.get("AGENT_ENABLE_SKILLS", "false").lower() == "true"
MAX_LOG_FILES: int = int(os.environ.get("AGENT_MAX_LOG_FILES", "50"))

# Paths
PROJECT_ROOT: Path = Path(__file__).parent.parent
PROMPTS_DIR: Path = Path(__file__).parent / "prompts"
LOGS_DIR: Path = PROJECT_ROOT / "logs"

_jinja_env: Environment = Environment(
    loader=FileSystemLoader(PROMPTS_DIR),
    trim_blocks=True,
    lstrip_blocks=True,
)


def _generate_session_id() -> str:
    ts: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{uuid.uuid4().hex[:8]}"


def _cleanup_old_logs() -> None:
    """Remove oldest log directories if exceeding MAX_LOG_FILES."""
    if not LOGS_DIR.exists():
        return
    log_dirs: list[Path] = sorted(
        [d for d in LOGS_DIR.iterdir() if d.is_dir()],
        key=lambda x: x.stat().st_mtime
    )
    while len(log_dirs) > MAX_LOG_FILES:
        oldest: Path = log_dirs.pop(0)
        for f in oldest.iterdir():
            f.unlink()
        oldest.rmdir()


def _setup_session(session_id: str) -> Path:
    _cleanup_old_logs()
    session_dir: Path = LOGS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def _build_options(session_dir: Path, working_dir: str, additional_dirs: list[str]) -> ClaudeCodeOptions:
    tools: list[str] = [t.strip() for t in ALLOWED_TOOLS if t.strip()]
    if ENABLE_SKILLS and "Skill" not in tools:
        tools.append("Skill")
    return ClaudeCodeOptions(
        system_prompt=_jinja_env.get_template("system.j2").render(),
        model=MODEL,
        max_turns=MAX_TURNS,
        allowed_tools=tools,
        cwd=str(session_dir),
        add_dirs=[working_dir] + additional_dirs,
        setting_sources=["project"] if ENABLE_SKILLS else [],
    )


def _validate_response(response: Optional[ResultMessage]) -> None:
    if response is None:
        raise SessionIncompleteError("Session did not complete")
    if response.is_error:
        raise ServerError(f"API error: {response.subtype}")
    if response.subtype == "error_max_turns":
        raise MaxTurnsExceededError(f"Exceeded {MAX_TURNS} turns")


def _parse_output(session_dir: Path) -> dict[str, Any]:
    output_file: Path = session_dir / "output.json"
    if output_file.exists():
        try:
            return json.loads(output_file.read_text())
        except json.JSONDecodeError:
            pass
    return {"status": "SUCCESS", "summary": "Completed"}


async def run_agent(
    task: str,
    working_dir: str,
    additional_dirs: Optional[list[str]] = None,
    parameters: Optional[dict[str, Any]] = None,
) -> AgentResult:
    session_id: str = _generate_session_id()
    session_dir: Path = _setup_session(session_id)
    additional_dirs = additional_dirs or []
    parameters = parameters or {}

    user_prompt: str = _jinja_env.get_template("user.j2").render(
        task=task,
        working_dir=working_dir,
        output_file=str(session_dir / "output.json"),
        **parameters,
    )

    options = _build_options(session_dir, working_dir, additional_dirs)
    log_file: Path = session_dir / "agent.log"
    result: Optional[ResultMessage] = None

    try:
        async with ClaudeCodeClient(options=options) as client:
            await client.query(user_prompt)
            with log_file.open("w", encoding="utf-8") as f:
                async for message in client.receive_response():
                    f.write(json.dumps(asdict(message)) + "\n")
                    if isinstance(message, ResultMessage):
                        result = message

        _validate_response(result)
        output = _parse_output(session_dir)
        return AgentResult(
            status=TaskStatus(output.get("status", "SUCCESS")),
            summary=output.get("summary", "Completed"),
            details=output.get("details"),
            data=output.get("data", {}),
            metrics=LLMMetrics(
                model=MODEL,
                duration_ms=result.duration_ms,
                num_turns=result.num_turns,
                session_id=result.session_id,
                total_cost_usd=result.total_cost_usd,
            ),
        )
    except AgentError:
        raise
    except asyncio.TimeoutError:
        raise AgentError(f"Timed out after {TIMEOUT_SECONDS}s")
    except Exception as e:
        return AgentResult(status=TaskStatus.ERROR, summary=f"Failed: {type(e).__name__}", error=str(e))


async def run_agent_with_timeout(
    task: str,
    working_dir: str,
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
- **Bash**: Execute shell commands (curl, wget, git, etc.)
- **Read**: Read file contents
- **Write**: Create or overwrite files
- **Edit**: Modify existing files
- **Grep**: Search for patterns in files

**Rules**
- Execute deterministically; avoid unnecessary actions
- Make minimal changes to achieve the goal
- Verify results before reporting completion

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
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

import colorlog
from dotenv import load_dotenv

from src.agent import run_agent_with_timeout
from src.exceptions import ConfigurationError
from src.schemas import TaskStatus

load_dotenv()


def setup_logging() -> colorlog.Logger:
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s [%(levelname)s]%(reset)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={"DEBUG": "cyan", "INFO": "green", "WARNING": "yellow", "ERROR": "red", "CRITICAL": "red,bg_white"},
    ))
    logger = colorlog.getLogger(__name__)
    logger.addHandler(handler)
    logger.setLevel(colorlog.INFO)
    return logger


def validate_api_key() -> None:
    api_key: Optional[str] = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ConfigurationError(
            "ANTHROPIC_API_KEY not found.\n"
            "Set it via .env file or: export ANTHROPIC_API_KEY=your-api-key"
        )
    if api_key == "your-api-key":
        raise ConfigurationError("ANTHROPIC_API_KEY is set to placeholder. Please use your actual API key.")


def load_task_file(task_file: Path) -> str:
    if not task_file.exists():
        raise FileNotFoundError(f"Task file not found: {task_file}")
    return task_file.read_text(encoding="utf-8")


def main() -> int:
    logger = setup_logging()

    parser = argparse.ArgumentParser(description="Execute Claude agent task")
    parser.add_argument("--dir", default=".", help="Working directory (default: current)")
    parser.add_argument("--task-file", type=Path, default=Path("task.md"), help="Task file (default: task.md)")
    parser.add_argument("--output", help="Output file path (default: stdout)")
    parser.add_argument("--add-dir", action="append", default=[], help="Additional directories")
    args = parser.parse_args()

    try:
        validate_api_key()
    except ConfigurationError as e:
        logger.error(str(e))
        return 1

    try:
        task: str = load_task_file(args.task_file)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    logger.info(f"Task file: {args.task_file}")
    logger.info(f"Working directory: {args.dir}")

    try:
        result = asyncio.run(run_agent_with_timeout(task=task, working_dir=args.dir, additional_dirs=args.add_dir))
        output: str = result.model_dump_json(indent=2)

        if args.output:
            Path(args.output).write_text(output)
            logger.info(f"Result saved to: {args.output}")
        else:
            print(output)

        if result.metrics:
            logger.info(f"Turns: {result.metrics.num_turns}, Cost: ${result.metrics.total_cost_usd:.4f}")

        return 0 if result.status == TaskStatus.SUCCESS else 1

    except Exception as e:
        logger.error(f"Agent failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

---

## Completion Output

After creating the agent, output only:

```
✓ Agent "{agent-name}" created at {agent-path}/

To run:
  # Unix/macOS
  source .venv/bin/activate
  
  # Windows (PowerShell)
  .\.venv\Scripts\Activate.ps1
  
  # Set API key
  export ANTHROPIC_API_KEY=your-key  # Unix
  $env:ANTHROPIC_API_KEY="your-key"  # Windows
  
  # Run
  python {agent-path}/claude_agent.py

Or press F5 in VS Code/Cursor (configure Python interpreter first).
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
| `Skill` | Execute custom skills |

---

## Troubleshooting

### Common Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| `ModuleNotFoundError: claude_code_sdk` | Package not installed | Run `pip install claude-code-sdk` |
| `ANTHROPIC_API_KEY not found` | Missing environment variable | Set via `.env` or `export ANTHROPIC_API_KEY=...` |
| `python3: command not found` | Python not in PATH | Install Python 3.10+ or fix PATH |
| `venv creation failed` | Missing python3-venv | `sudo apt install python3-venv` (Linux) |
| Agent timeout | Task too complex | Increase `AGENT_TIMEOUT_SECONDS` or simplify task |
| Logs directory growing | No cleanup | Set `AGENT_MAX_LOG_FILES` environment variable |
| Permission denied errors | Restrictive settings.local.json | Add required commands to `allow` list |

### Debugging Tips

1. **Check logs**: Review `logs/{session-id}/agent.log` for detailed execution trace
2. **Verify API key**: Run `echo $ANTHROPIC_API_KEY | head -c 10` (should show `sk-ant-...`)
3. **Test SDK**: Run `python -c "from claude_code_sdk import ClaudeCodeClient; print('OK')"`
4. **Check permissions**: Verify `.claude/settings.local.json` allows required operations
5. **Inspect output**: Check `logs/{session-id}/output.json` for agent's final result

### Platform-Specific Notes

**macOS/Linux:**
- Use `source .venv/bin/activate`
- Set env vars with `export VAR=value`

**Windows:**
- Use `.\.venv\Scripts\Activate.ps1` (PowerShell) or `.venv\Scripts\activate.bat` (CMD)
- Set env vars with `$env:VAR="value"` (PowerShell) or `set VAR=value` (CMD)
- Path separators: use `\` in paths or raw strings

**Docker:**
- Mount `.env` file: `-v $(pwd)/.env:/app/.env`
- Set env directly: `-e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY`
