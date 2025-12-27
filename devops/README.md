# DevOps Skills

This directory contains skills related to DevOps workflows, automation, and infrastructure management.

## Available Skills

### [gh-repo-create](./gh-repo-create/)

**Purpose**: Automate the complete workflow of creating a GitHub repository and configuring SSH deployment keys.

**What it does**:
- Creates a new GitHub repository using GitHub CLI
- Generates dedicated SSH deployment keys (Ed25519)
- Configures custom SSH host in `~/.ssh/config`
- Adds deploy keys to GitHub with read/write permissions
- Sets up git remotes with proper SSH URLs
- Verifies the setup is ready for code push

**When to use**:
- Starting a new project that needs to be on GitHub
- Converting a local project to a GitHub repository
- Setting up secure SSH-based authentication for a repository
- Isolating SSH credentials per project

**Requirements**:
- GitHub CLI (`gh`) installed and authenticated
- Git installed
- OpenSSH (pre-installed on most systems)

**Platforms**: macOS, Linux, Unix

**Quick Start**:
```bash
cd /path/to/your/project
bash /path/to/SKILLS/devops/gh-repo-create/scripts/run.sh
```

---

## Skill Development Guidelines

When contributing new DevOps skills to this directory, ensure they:

1. **Are Cross-Platform**: Work on macOS, Linux, and Unix systems
2. **Are Well-Documented**: Include comprehensive SKILL.md with examples
3. **Are Automated**: Minimize manual intervention required
4. **Are Secure**: Follow security best practices
5. **Are Idempotent**: Can be run multiple times safely
6. **Include Verification**: Validate the setup is correct
7. **Have Error Handling**: Provide clear error messages and recovery steps

### Recommended Structure

```
skill-name/
├── SKILL.md              # Comprehensive documentation
├── README.md             # Quick reference guide
└── scripts/
    └── run.sh            # Main executable script
```

### Documentation Template

Each skill should include:
- **Purpose**: What problem it solves
- **Prerequisites**: Required tools and authentication
- **Instructions**: Step-by-step process
- **Examples**: Real-world usage scenarios
- **Verification**: How to validate success
- **Troubleshooting**: Common issues and solutions
- **Security Considerations**: Important security notes

## Contributing

To add a new DevOps skill:

1. Create a new directory under `devops/`
2. Follow the skill structure guidelines
3. Include comprehensive documentation
4. Test on multiple platforms
5. Add entry to this README

## Related Skills

- **General Purpose Skills**: [../general-purpose/](../general-purpose/)
- **Domain Specific Skills**: [../domain-specific/](../domain-specific/)

## Resources

- [GitHub CLI Documentation](https://cli.github.com/manual/)
- [Git Documentation](https://git-scm.com/doc)
- [SSH Keys Guide](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)
- [DevOps Best Practices](https://www.atlassian.com/devops/what-is-devops)

