---
name: create-claude-agent
description: Create production-ready Claude agent projects using the Claude Code SDK. Scaffolds complete project structure with customizable system prompts, tool configurations, and execution framework.
---

# Create Claude Agent

## Instructions

When this skill is invoked, follow these steps to create a complete Claude agent project:

### Step 1: Understand Requirements
- Ask the user about their agent's purpose (e.g., code analysis, automation, modification)
- Determine which tools the agent needs: Bash, Read, Write, Edit, Grep, Skill
- Identify if custom skills are required
- Confirm the target directory for project creation

### Step 2: Create Project Structure
Create the following directory structure:

```
{project-name}/
├── .claude/
│   ├── settings.local.json
│   └── skills/
├── src/
│   ├── __init__.py
│   ├── agent.py
│   ├── prompts/
│   │   ├── system.j2
│   │   └── user.j2
│   ├── schemas.py
│   └── exceptions.py
├── logs/
├── data/
├── .env
├── .gitignore
├── requirements.txt
└── main.py
```

### Step 3: Generate Core Files
Create all required files using the templates provided in the Reference Templates section below:

1. **requirements.txt** - Python dependencies
2. **.gitignore** - Ignore logs, data, env files
3. **.env** - API key placeholder
4. **src/exceptions.py** - Error handling classes
5. **src/schemas.py** - Data models (customize based on agent type)
6. **src/agent.py** - Core agent logic (customize tools, limits, model)
7. **src/prompts/system.j2** - System prompt (customize for agent's role)
8. **src/prompts/user.j2** - User prompt template
9. **.claude/settings.local.json** - Tool permissions
10. **main.py** - Entry point with CLI

### Step 4: Customize System Prompt
Based on the agent's purpose, create a tailored system prompt that includes:
- **Role**: What the agent is and its specialization
- **Available Tools**: Which tools and their allowed uses
- **Rules**: Constraints, boundaries, and quality requirements
- **Workflow**: Step-by-step process the agent follows
- **Output Requirement**: JSON format for results

### Step 5: Configure Agent Settings
In `src/agent.py`, set:
- `MODEL`: Claude model to use (default: claude-sonnet-4-5-20250929)
- `ALLOWED_TOOLS`: List of enabled tools
- `MAX_TURNS`: Maximum conversation turns
- `TIMEOUT_SECONDS`: Execution timeout
- `ENABLE_SKILLS`: Whether to load custom skills

### Step 6: Set Up Permissions
In `.claude/settings.local.json`, configure:
- Allowed bash commands (git, find, grep, etc.)
- Denied commands (rm -rf, sudo, etc.)

### Step 7: Create Custom Skills (Optional)
If needed, create skills in `.claude/skills/{skill-name}/`:
- SKILL.md with documentation
- scripts/run.sh with implementation

### Step 8: Provide Setup Instructions
Guide the user to:
1. Install dependencies: `pip install -r requirements.txt`
2. Set API key: `export ANTHROPIC_API_KEY="..."`
3. Run: `python main.py --task "..." --dir /path/to/target`

---

## Examples

### Example 1: Code Analysis Agent

**User Request**: "Create an agent that analyzes Python codebases for security issues"

**Agent Configuration**:
- **Tools**: Bash, Read, Grep (read-only)
- **Max Turns**: 50
- **Timeout**: 600 seconds

**System Prompt** (`src/prompts/system.j2`):
```jinja2
**Role**
You are a security analysis agent specialized in identifying vulnerabilities in Python code.
Your goal is to analyze codebases and produce structured security findings.

**Available Tools**
- **Bash**: For git commands and file listing only
- **Read**: For reading source files
- **Grep**: For searching security-sensitive patterns

**Rules**
- Never modify any files in the target repository
- Always cite specific files and line numbers for findings
- Focus on OWASP Top 10 vulnerabilities
- Provide severity ratings (CRITICAL, HIGH, MEDIUM, LOW)

**Workflow**
1. **Explore**: Identify Python files using bash/grep
2. **Search**: Find security-sensitive patterns (SQL, auth, crypto)
3. **Analyze**: Read flagged files and assess vulnerabilities
4. **Report**: Create output.json with detailed findings

**Output Requirement**
Create `output.json`:
```json
{
  "status": "SUCCESS",
  "summary": "Security analysis completed",
  "data": {
    "files_analyzed": 42,
    "findings": [
      {
        "file": "src/auth.py",
        "line": 15,
        "severity": "HIGH",
        "issue": "Hardcoded credentials detected",
        "recommendation": "Use environment variables"
      }
    ]
  }
}
```
```

**Usage**:
```bash
python main.py --task "Analyze for security vulnerabilities" --dir ./myapp
```

---

### Example 2: Code Refactoring Agent

**User Request**: "Create an agent that refactors code according to style guidelines"

**Agent Configuration**:
- **Tools**: Bash, Read, Write, Edit, Grep (full access)
- **Max Turns**: 100
- **Timeout**: 1800 seconds

**System Prompt** (`src/prompts/system.j2`):
```jinja2
**Role**
You are a code refactoring agent specialized in improving code quality and consistency.
Your goal is to apply refactoring patterns while preserving functionality.

**Available Tools**
- **Bash**: For git commands, running tests, and linters
- **Read**: For reading source files
- **Write**: For creating new files
- **Edit**: For modifying existing files
- **Grep**: For finding patterns to refactor

**Rules**
- Make minimal changes to achieve the refactoring goal
- Preserve all existing functionality
- Follow the project's existing style conventions
- Run tests after changes if available
- Stage and commit changes with descriptive messages
- Never modify files outside the specified scope

**Workflow**
1. **Understand**: Read target files and identify refactoring opportunities
2. **Plan**: Document what changes are needed and why
3. **Implement**: Apply refactorings using Edit tool
4. **Verify**: Run tests and check git diff
5. **Commit**: Stage changes with clear commit message
6. **Report**: Create output.json with summary

**Output Requirement**
Create `output.json`:
```json
{
  "status": "SUCCESS",
  "summary": "Refactored 5 files to follow PEP 8",
  "data": {
    "commit_hash": "abc123def",
    "files_changed": ["src/main.py", "src/utils.py"],
    "changes_applied": [
      "Renamed variables to snake_case",
      "Added type hints",
      "Extracted functions > 50 lines"
    ],
    "tests_passed": true
  }
}
```
```

**Usage**:
```bash
python main.py --task "Refactor to follow PEP 8 style guide" --dir ./myproject
```

---

### Example 3: Documentation Generator Agent

**User Request**: "Create an agent that generates API documentation from code"

**Agent Configuration**:
- **Tools**: Bash, Read, Write, Grep
- **Max Turns**: 75
- **Timeout**: 900 seconds

**System Prompt** (`src/prompts/system.j2`):
```jinja2
**Role**
You are a documentation generation agent specialized in creating comprehensive API documentation.
Your goal is to analyze code and produce well-structured documentation.

**Available Tools**
- **Bash**: For file operations
- **Read**: For reading source files
- **Write**: For creating documentation files
- **Grep**: For finding functions, classes, and docstrings

**Rules**
- Extract documentation from code comments and type hints
- Generate markdown format documentation
- Include function signatures, parameters, return types, and examples
- Organize by module and class hierarchy
- Never modify source code files
- Create documentation in a docs/ directory

**Workflow**
1. **Discover**: Find all API-relevant files (controllers, services)
2. **Extract**: Read files and parse function/class definitions
3. **Generate**: Create markdown documentation for each module
4. **Index**: Build a main README with navigation
5. **Report**: Create output.json

**Output Requirement**
Create `output.json`:
```json
{
  "status": "SUCCESS",
  "summary": "Generated API documentation for 12 modules",
  "data": {
    "docs_created": [
      "docs/api/auth.md",
      "docs/api/users.md",
      "docs/README.md"
    ],
    "endpoints_documented": 45,
    "coverage": "95%"
  }
}
```
```

**Usage**:
```bash
python main.py --task "Generate API documentation" --dir ./api-server --output docs-result.json
```

---

## Reference Templates

Below are the complete file templates to use when creating agent projects.

### Quick Start Guide

```bash
# 1. Create project
mkdir my-agent && cd my-agent
mkdir -p src logs data .claude/skills
touch src/__init__.py

# 2. Install dependencies
pip install claude-agent-sdk==0.1.18 jinja2>=3.1.0 pydantic>=2.0.0

# 3. Set API key
export ANTHROPIC_API_KEY="your-api-key"

# 4. Copy files from this template (see Complete Files section)

# 5. Run
python main.py --task "Your task description" --dir /path/to/target
```

---

### Project Structure

```
my-agent/
├── .claude/
│   ├── settings.local.json       # Tool permissions (define once)
│   └── skills/                   # Custom skills (if needed)
│       └── {skill-name}/
│           ├── SKILL.md
│           └── scripts/
│               └── run.sh
├── src/
│   ├── __init__.py
│   ├── agent.py                  # Core agent logic
│   ├── prompts/
│   │   ├── system.j2             # System prompt (customize for your domain)
│   │   └── user.j2               # User prompt template
│   ├── schemas.py                # Data models
│   └── exceptions.py             # Error types
├── logs/                         # Session logs (gitignored)
├── data/                         # Working data (gitignored)
├── .env                          # Environment variables (gitignored)
├── .gitignore
├── requirements.txt
└── main.py                       # Entry point
```

#### .gitignore

```gitignore
logs/
data/
.env
__pycache__/
*.pyc
.venv/
```

---

### Core Agent Implementation

The agent consists of these components:

| File | Purpose | Customize? |
|------|---------|------------|
| `src/agent.py` | Agent execution logic | Rarely |
| `src/prompts/system.j2` | Agent behavior definition | **Yes - this defines your agent** |
| `src/prompts/user.j2` | Task input format | Sometimes |
| `src/schemas.py` | Output data models | As needed |
| `.claude/settings.local.json` | Tool permissions | Once at setup |

#### What You Must Define in Code

When creating your agent, you define these in `src/agent.py`:

```python
# 1. Which tools your agent can use
ALLOWED_TOOLS = ["Bash", "Read", "Write", "Edit", "Grep"]

# 2. Execution limits
MAX_TURNS = 100
TIMEOUT_SECONDS = 1800

# 3. Model
MODEL = "claude-sonnet-4-5-20250929"
```

Available built-in tools:

| Tool | Capability |
|------|------------|
| `Bash` | Execute shell commands |
| `Read` | Read file contents |
| `Write` | Create/overwrite files |
| `Edit` | Modify existing files |
| `Grep` | Search patterns in files |
| `Skill` | Execute custom skills (requires `.claude/skills/`) |

---

### System Prompt Template

The system prompt defines your agent's behavior. This is the primary customization point.

#### Template Structure

```jinja2
**Role**
You are a [describe your agent's role and specialization].
Your goal is to [describe the primary objective].

**Available Tools**
- **Bash**: For [describe allowed uses]
- **Read**: For [describe allowed uses]
- **Write**: For [describe allowed uses]
- **Edit**: For [describe allowed uses]
- **Grep**: For [describe allowed uses]

**Rules**
- [Rule 1: What the agent must do]
- [Rule 2: What the agent must not do]
- [Rule 3: Constraints and boundaries]
- [Rule 4: Quality requirements]

**Workflow**
1. **[Step Name]**: [Description of what to do]
2. **[Step Name]**: [Description of what to do]
3. **[Step Name]**: [Description of what to do]
4. **Report**: Create output.json with results

**Output Requirement**
Create a file named `output.json` in the working directory:
```json
{
  "status": "SUCCESS" | "PARTIAL" | "FAILED",
  "summary": "Brief description of outcome",
  "details": "Extended explanation if needed",
  "data": { "custom": "fields" }
}
```
```

#### Example: Code Analysis Agent

```jinja2
**Role**
You are a code analysis agent specialized in identifying patterns and issues in codebases.
Your goal is to analyze code and produce structured findings.

**Available Tools**
- **Bash**: For git commands and file listing
- **Read**: For reading source files
- **Grep**: For searching patterns

**Rules**
- Never modify any files in the target repository
- Always cite specific files and line numbers
- Focus only on the requested analysis type
- Be precise with metrics and counts

**Workflow**
1. **Explore**: Understand the codebase structure
2. **Search**: Find relevant patterns using grep
3. **Analyze**: Read and examine matched files
4. **Report**: Create output.json with findings

**Output Requirement**
Create `output.json`:
```json
{
  "status": "SUCCESS",
  "summary": "Analysis completed",
  "data": {
    "files_analyzed": 42,
    "findings": [
      {"file": "src/auth.py", "line": 15, "issue": "..."}
    ]
  }
}
```
```

#### Example: Code Modification Agent

```jinja2
**Role**
You are a code modification agent specialized in implementing changes to codebases.
Your goal is to make precise, minimal changes according to specifications.

**Available Tools**
- **Bash**: For git commands and running tests
- **Read**: For reading source files
- **Write**: For creating new files
- **Edit**: For modifying existing files
- **Grep**: For searching code patterns

**Rules**
- Make minimal changes to achieve the goal
- Follow existing code style and conventions
- Stage and commit changes with descriptive messages
- Never modify unrelated files
- Run tests after changes if available

**Workflow**
1. **Understand**: Read relevant files and understand the task
2. **Plan**: Identify what needs to change
3. **Implement**: Make the changes using Edit/Write
4. **Verify**: Check changes with git diff
5. **Commit**: Stage and commit with clear message
6. **Report**: Create output.json

**Output Requirement**
Create `output.json`:
```json
{
  "status": "SUCCESS",
  "summary": "Implemented [description]",
  "data": {
    "commit_hash": "abc123",
    "files_changed": ["file1.py", "file2.py"]
  }
}
```
```

---

### Custom Skills

Skills are reusable procedures the agent can invoke. Use them for complex, multi-step operations.

#### When to Use Skills

- Operations that require specific command sequences
- Tasks that need environment variable configuration
- Reusable workflows across multiple tasks

#### Skill Structure

```
.claude/skills/{skill-name}/
├── SKILL.md              # Documentation (required)
└── scripts/
    └── run.sh            # Main script (required)
```

#### SKILL.md Template

```markdown
---
name: {skill-name}
description: One-line description.
---

# {Skill Name}

What this skill does.

## Usage

```bash
PARAM_ONE="value" bash scripts/run.sh
```

## Parameters

| Variable | Required | Description |
|----------|----------|-------------|
| `PARAM_ONE` | Yes | Description |
| `PARAM_TWO` | No | Description (default: X) |

## Output

What the skill produces.
```

#### scripts/run.sh Template

```bash
#!/usr/bin/env bash
set -euo pipefail

PARAM_ONE="${PARAM_ONE:-}"
PARAM_TWO="${PARAM_TWO:-default}"

if [[ -z "$PARAM_ONE" ]]; then
    echo "Error: PARAM_ONE required" >&2
    exit 1
fi

# Your implementation
echo "Running with $PARAM_ONE"
```

#### Enabling Skills in Agent

In `src/agent.py`, add `"Skill"` to allowed tools and enable settings:

```python
ALLOWED_TOOLS = ["Skill", "Bash", "Read", "Write", "Edit", "Grep"]
ENABLE_SKILLS = True  # Loads from .claude/skills/
```

---

### Customization Points

#### 1. Define Your Agent's Purpose

Edit `src/prompts/system.j2` to define:
- What role the agent plays
- What tools it can use and how
- Rules and constraints
- Workflow steps
- Output format

#### 2. Configure Tools

Edit `src/agent.py`:

```python
# Read-only agent (analysis, review)
ALLOWED_TOOLS = ["Bash", "Read", "Grep"]

# Full access agent (coding, automation)
ALLOWED_TOOLS = ["Bash", "Read", "Write", "Edit", "Grep"]

# With custom skills
ALLOWED_TOOLS = ["Skill", "Bash", "Read", "Write", "Edit", "Grep"]
```

#### 3. Set Execution Limits

Edit `src/agent.py`:

```python
# Simple tasks
MAX_TURNS = 30
TIMEOUT_SECONDS = 300  # 5 minutes

# Complex tasks
MAX_TURNS = 150
TIMEOUT_SECONDS = 3600  # 1 hour
```

#### 4. Configure Permissions

Edit `.claude/settings.local.json`:

```json
{
  "permissions": {
    "allow": [
      "Bash(git:*)",
      "Bash(python:*)"
    ],
    "deny": [
      "Bash(rm -rf:*)",
      "Bash(sudo:*)"
    ]
  }
}
```

#### 5. Define Output Schema

Edit `src/schemas.py` to match your agent's output:

```python
class AgentResult(BaseModel):
    status: TaskStatus
    summary: str
    # Add your custom fields
    findings: list[Finding] = Field(default_factory=list)
    metrics: dict[str, int] = Field(default_factory=dict)
```

---

### Complete Files

#### requirements.txt

```txt
claude-agent-sdk==0.1.18
jinja2>=3.1.0
pydantic>=2.0.0
python-dotenv>=1.0.0
```

#### .env

```bash
ANTHROPIC_API_KEY=your-api-key
```

#### .claude/settings.local.json

```json
{
  "permissions": {
    "allow": [
      "Bash(git:*)",
      "Bash(find:*)",
      "Bash(grep:*)",
      "Bash(ls:*)",
      "Bash(cat:*)",
      "Bash(head:*)",
      "Bash(tail:*)",
      "Bash(wc:*)",
      "Bash(python:*)"
    ],
    "deny": [
      "Bash(rm -rf:*)",
      "Bash(sudo:*)"
    ]
  }
}
```

#### src/exceptions.py

```python
"""Agent exceptions."""


class AgentError(Exception):
    """Base exception."""
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

#### src/schemas.py

```python
"""Data models."""
from datetime import datetime
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
    """
    Agent execution result.
    
    Customize this class to match your agent's output.
    """
    status: TaskStatus
    summary: str
    details: Optional[str] = None
    data: dict[str, Any] = Field(default_factory=dict)
    metrics: Optional[LLMMetrics] = None
    error: Optional[str] = None
```

#### src/agent.py

```python
"""
Core agent implementation.

Customize the constants at the top for your specific agent.
"""
import asyncio
import json
import os
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
)
from jinja2 import Environment, FileSystemLoader

from .exceptions import (
    AgentError,
    MaxTurnsExceededError,
    ServerError,
    SessionIncompleteError,
)
from .schemas import AgentResult, LLMMetrics, TaskStatus


# =============================================================================
# CONFIGURATION - Customize these for your agent
# =============================================================================

# Model to use
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")

# Tools the agent can use
# Options: "Bash", "Read", "Write", "Edit", "Grep", "Skill"
ALLOWED_TOOLS = ["Bash", "Read", "Write", "Edit", "Grep"]

# Execution limits
MAX_TURNS = 100
TIMEOUT_SECONDS = 1800  # 30 minutes

# Enable custom skills from .claude/skills/
ENABLE_SKILLS = False

# =============================================================================
# IMPLEMENTATION - Modify only if needed
# =============================================================================

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
PROMPTS_DIR = Path(__file__).parent / "prompts"
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data"

# Jinja environment
_jinja_env = Environment(
    loader=FileSystemLoader(PROMPTS_DIR),
    trim_blocks=True,
    lstrip_blocks=True,
)


def _generate_session_id() -> str:
    """Generate unique session ID."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:8]
    return f"{ts}_{uid}"


def _setup_session(session_id: str) -> Path:
    """Create session directory."""
    session_dir = LOGS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def _build_options(
    session_dir: Path,
    working_dir: str,
    additional_dirs: list[str],
) -> ClaudeAgentOptions:
    """Build agent options."""
    tools = list(ALLOWED_TOOLS)
    if ENABLE_SKILLS and "Skill" not in tools:
        tools.append("Skill")

    add_dirs = [working_dir] + additional_dirs

    return ClaudeAgentOptions(
        system_prompt=_jinja_env.get_template("system.j2").render(),
        model=MODEL,
        max_turns=MAX_TURNS,
        allowed_tools=tools,
        cwd=str(session_dir),
        add_dirs=add_dirs,
        setting_sources=["project"] if ENABLE_SKILLS else [],
    )


def _validate_response(response: Optional[ResultMessage]) -> None:
    """Validate agent response."""
    if response is None:
        raise SessionIncompleteError("Session did not complete")
    if response.is_error:
        raise ServerError(f"API error: {response.subtype}")
    if response.subtype == "error_max_turns":
        raise MaxTurnsExceededError(f"Exceeded {MAX_TURNS} turns")


def _parse_output(session_dir: Path) -> dict:
    """Parse agent's output.json."""
    output_file = session_dir / "output.json"
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
    parameters: Optional[dict] = None,
) -> AgentResult:
    """
    Execute the agent.

    Args:
        task: Task description for the agent
        working_dir: Directory the agent operates on
        additional_dirs: Extra directories agent can access
        parameters: Additional parameters passed to prompt

    Returns:
        AgentResult with execution outcome
    """
    session_id = _generate_session_id()
    session_dir = _setup_session(session_id)
    additional_dirs = additional_dirs or []
    parameters = parameters or {}

    # Build prompt
    user_prompt = _jinja_env.get_template("user.j2").render(
        task=task,
        working_dir=working_dir,
        output_file=str(session_dir / "output.json"),
        **parameters,
    )

    options = _build_options(session_dir, working_dir, additional_dirs)
    log_file = session_dir / "agent.log"
    result: Optional[ResultMessage] = None

    try:
        async with ClaudeSDKClient(options=options) as client:
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
        return AgentResult(
            status=TaskStatus.ERROR,
            summary=f"Failed: {type(e).__name__}",
            error=str(e),
        )


async def run_agent_with_timeout(
    task: str,
    working_dir: str,
    additional_dirs: Optional[list[str]] = None,
    parameters: Optional[dict] = None,
) -> AgentResult:
    """Execute agent with timeout."""
    return await asyncio.wait_for(
        run_agent(task, working_dir, additional_dirs, parameters),
        timeout=TIMEOUT_SECONDS,
    )
```

#### src/prompts/system.j2

```jinja2
**Role**
You are an intelligent automation agent.
Your goal is to complete the assigned task accurately and efficiently.

**Available Tools**
- **Bash**: Execute shell commands (git, file operations, etc.)
- **Read**: Read file contents
- **Write**: Create or overwrite files
- **Edit**: Modify existing files
- **Grep**: Search for patterns in files

**Rules**
- Execute deterministically; avoid unnecessary actions
- Make minimal changes to achieve the goal
- Verify results before reporting completion
- Never execute destructive commands without explicit instruction

**Workflow**
1. **Understand**: Analyze the task requirements
2. **Plan**: Determine the minimal steps needed
3. **Execute**: Perform each step carefully
4. **Verify**: Confirm the task is complete
5. **Report**: Create output.json with results

**Output Requirement**
Create a file named `output.json` in the current working directory:
```json
{
  "status": "SUCCESS" | "PARTIAL" | "FAILED",
  "summary": "Brief description of outcome",
  "details": "Extended explanation if needed",
  "data": { "key": "value" }
}
```
```

#### src/prompts/user.j2

```jinja2
**Working Directory**: {{ working_dir }}
**Output File**: {{ output_file }}

**Task**:
{{ task }}

Execute this task following your workflow. Create output.json when complete.
```

#### main.py

```python
#!/usr/bin/env python3
"""
Agent entry point.

Usage:
    python main.py --task "Your task" --dir /path/to/target
    python main.py --task "Analyze code" --dir ./myproject --output result.json
"""
import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.agent import run_agent_with_timeout
from src.schemas import TaskStatus

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute agent task")
    parser.add_argument("--task", required=True, help="Task description")
    parser.add_argument("--dir", required=True, help="Working directory")
    parser.add_argument("--output", help="Output file (default: stdout)")
    parser.add_argument("--add-dir", action="append", default=[], 
                        help="Additional directories")
    args = parser.parse_args()

    logger.info(f"Task: {args.task}")
    logger.info(f"Directory: {args.dir}")

    try:
        result = asyncio.run(run_agent_with_timeout(
            task=args.task,
            working_dir=args.dir,
            additional_dirs=args.add_dir,
        ))

        output = result.model_dump_json(indent=2)

        if args.output:
            Path(args.output).write_text(output)
            logger.info(f"Result: {args.output}")
        else:
            print(output)

        if result.metrics:
            logger.info(
                f"Turns: {result.metrics.num_turns}, "
                f"Cost: ${result.metrics.total_cost_usd:.4f}"
            )

        return 0 if result.status == TaskStatus.SUCCESS else 1

    except Exception as e:
        logger.error(f"Failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

---

### Execution Examples

#### Basic Usage

```bash
python main.py --task "List all TODO comments in the codebase" --dir ./myproject
```

#### With Output File

```bash
python main.py \
    --task "Find security issues in authentication code" \
    --dir ./myproject \
    --output findings.json
```

#### With Additional Directories

```bash
python main.py \
    --task "Compare implementation with reference" \
    --dir ./myproject \
    --add-dir ./reference-docs \
    --add-dir ./specs
```

#### Check Logs

```bash
# Find latest session
ls -lt logs/ | head -5

# View agent trace
cat logs/{session_id}/agent.log | head -100

# View output
cat logs/{session_id}/output.json
```

---

### API Reference

#### ClaudeAgentOptions

| Parameter | Type | Description |
|-----------|------|-------------|
| `system_prompt` | `str` | Agent behavior definition |
| `model` | `str` | Model identifier |
| `max_turns` | `int` | Maximum turns |
| `allowed_tools` | `list[str]` | Enabled tools |
| `cwd` | `str` | Working directory |
| `add_dirs` | `list[str]` | Additional accessible directories |
| `setting_sources` | `list[str]` | `["project"]` to load skills |

#### ResultMessage

| Property | Type | Description |
|----------|------|-------------|
| `session_id` | `str` | Session ID |
| `duration_ms` | `int` | Execution time |
| `num_turns` | `int` | Turns used |
| `total_cost_usd` | `float` | API cost |
| `is_error` | `bool` | Error flag |
| `subtype` | `str` | Error type |

#### Available Tools

| Tool | Capability |
|------|------------|
| `Bash` | Shell commands |
| `Read` | Read files |
| `Write` | Create files |
| `Edit` | Modify files |
| `Grep` | Search patterns |
| `Skill` | Custom skills |
