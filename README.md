# Claude Code Skills Library

A curated collection of production-ready skills, agents, and automation tools for Claude Code, created by the Extractum Team.

## About

This repository contains reusable skills and resources that extend Claude Code's capabilities. Skills are structured procedures that Claude can invoke to perform complex, multi-step operations — from DevOps automation and agent scaffolding to domain-specific research and analysis workflows.

The collection is organized into categories (general-purpose, domain-specific, DevOps, computer-use, agents, and more) and is continuously evolving. Browse the directories to discover what's available.

## What Are Claude Code Skills?

Skills are structured knowledge documents that give Claude step-by-step guidance for performing specific tasks. They can include documentation, reference templates, executable scripts, and hook configurations. Each skill follows a standard format with frontmatter metadata (`name`, `description`) and markdown instructions.

Skills may also ship with supporting scripts, command definitions, or configuration files — whatever is needed to get the job done.

## How to Use

### In Your Project

Copy any skill into your project's `.claude/skills/` directory. Claude Code will automatically detect and use it:

```bash
cp -r <skill-directory> /path/to/your/project/.claude/skills/
```

### As a Reference

Clone the repo and browse for skill templates, patterns, and examples to adapt to your own workflows:

```bash
git clone https://github.com/extractum-io/claude-skills.git
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on adding or improving skills.

## License

MIT License — see [LICENSE](LICENSE) for details.

---

**Maintained by**: [Extractum](https://extractum.io)  
**Author**: Gregory Zemskov  
**Contact**: info@extractum.io
