# Claude Code Skills Library

A curated collection of production-ready skills for Claude Code create by Extractum Team.

## About

This repository contains reusable skills that extend Claude Code's capabilities. Skills are structured procedures that Claude can invoke to perform complex, multi-step operations efficiently.

**Author**: Gregory Zemskov  
**Organization**: [Extractum](https://extractum.io)  
**License**: MIT

## Repository Structure

```
SKILLS/
├── general-purpose/       # Universal skills applicable to any project
│   └── create-claude-agent/
├── domain-specific/       # Specialized skills for specific domains
└── extractum-private/     # Private/proprietary skills (not included)
```

## Available Skills

### General Purpose

#### [create-claude-agent](./general-purpose/create-claude-agent/)
Create production-ready Claude agent projects using the Claude Code SDK. Scaffolds complete project structure with customizable system prompts, tool configurations, and execution framework.

**Use Cases**:
- Build autonomous code analysis agents
- Create code refactoring automation
- Generate documentation agents
- Custom task automation

**Features**:
- Complete project scaffolding
- Customizable system prompts
- Tool permission management
- Built-in error handling
- Session logging and metrics
- Extensible skill framework

### Domain Specific

*Coming soon - skills tailored for specific industries and use cases.*

## What are Claude Code Skills?

Skills in Claude Code are structured knowledge documents that:

1. **Define Capabilities**: Describe what the skill does and when to use it
2. **Provide Instructions**: Give Claude step-by-step guidance
3. **Include Examples**: Show concrete usage scenarios
4. **Can Execute Scripts**: Optionally include bash scripts for complex operations

### Skill Structure

Each skill follows this format:

```markdown
---
name: skill-name
description: Brief description of what this skill does
---

# Skill Name

## Instructions
Step-by-step guidance for Claude when this skill is active.

## Examples
Concrete examples with inputs and expected outputs.

## (Optional) Reference Templates
Additional resources, code templates, configuration files.
```

### Skills with Scripts

Skills can include executable scripts:

```
skill-name/
├── SKILL.md              # Documentation
└── scripts/
    └── run.sh            # Main executable script
```

## How to Use

### 1. In Claude Code (Cursor IDE)

Place skills in your project's `.claude/skills/` directory:

```bash
# For project-specific use
cp -r general-purpose/create-claude-agent .claude/skills/

# Claude can now invoke this skill
```

### 2. As a Reference Library

Clone this repository to access skill templates and examples:

```bash
git clone https://github.com/extractum-io/claude-skills.git
cd claude-skills
```

Browse and copy skills as needed for your projects.

## Resources

### Claude Code
- [Claude Code Skills](https://code.claude.com/docs/en/skills)
- [Claude Code Documentation](https://docs.anthropic.com)
- [Agent SDK Repository](https://github.com/anthropics/claude-code)

### Community
- **Issues**: Report bugs or request features via GitHub Issues
- **Discussions**: Share ideas and ask questions in GitHub Discussions
- **Contributing**: See [Contributing](#contributing) section above

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Maintained by**: [Extractum](https://extractum.io)  
**Contact**: For enterprise support and custom skill development - reach out ot us, info@extractum.io
