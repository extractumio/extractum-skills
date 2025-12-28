# Contributing to Claude Code Skills Library

Thank you for your interest in contributing! This document provides guidelines for contributing skills to this repository.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Skill Submission Process](#skill-submission-process)
4. [Skill Quality Standards](#skill-quality-standards)
5. [Testing Guidelines](#testing-guidelines)
6. [Documentation Requirements](#documentation-requirements)
7. [Review Process](#review-process)

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment. We expect all contributors to:

- Use welcoming and inclusive language
- Respect differing viewpoints and experiences
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

### Prerequisites

1. **Fork** this repository to your GitHub account
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/claude-skills.git
   cd claude-skills
   ```
3. **Create a branch** for your contribution:
   ```bash
   git checkout -b feature/your-skill-name
   ```

### Development Environment

Ensure you have:
- Claude Code (Cursor IDE) for testing
- Basic understanding of skill structure
- Any domain-specific tools required for your skill

## Skill Submission Process

### Step 1: Choose the Right Category

- **General Purpose**: Universal skills applicable to any project
  - Example: code formatting, file organization, template generation
  
- **Domain Specific**: Specialized skills for specific technologies
  - Example: React component creation, Kubernetes deployment

### Step 2: Create Skill Structure

```bash
# For general-purpose skills
mkdir -p general-purpose/your-skill-name

# For domain-specific skills
mkdir -p domain-specific/your-skill-name

# Add required files
touch general-purpose/your-skill-name/SKILL.md
```

If your skill needs scripts:

```bash
mkdir -p general-purpose/your-skill-name/scripts
touch general-purpose/your-skill-name/scripts/run.sh
chmod +x general-purpose/your-skill-name/scripts/run.sh
```

### Step 3: Write SKILL.md

Use this template:

```markdown
---
name: your-skill-name
description: Brief one-sentence description of what this skill does
---

# Your Skill Name

## Instructions

When this skill is invoked, follow these steps:

### Step 1: [First Action]
- Clear guidance on what to do
- Specific tools or commands to use

### Step 2: [Second Action]
- Next step in the process
- Expected outcomes

### Step 3: [Final Action]
- How to conclude
- What to output or create

### Key Considerations
- Important constraints or rules
- Error handling guidance
- Edge cases to consider

## Examples

### Example 1: [Basic Usage]

**Scenario**: Describe when this example applies

**Input**:
```
Show what the input looks like
```

**Actions**:
1. First action taken
2. Second action taken
3. Final action

**Expected Output**:
```
Show expected result
```

### Example 2: [Advanced Usage]

**Scenario**: More complex use case

[Follow same structure as Example 1]

## Reference Templates

(Optional) Include any code templates, configuration files, or additional resources
```

### Step 4: Create Scripts (If Needed)

If your skill includes bash scripts:

```bash
#!/usr/bin/env bash
set -euo pipefail

# Script: scripts/run.sh
# Description: Brief description of what this script does

# Parameters with defaults
PARAM_NAME="${PARAM_NAME:-default_value}"

# Validation
if [[ -z "$REQUIRED_PARAM" ]]; then
    echo "Error: REQUIRED_PARAM must be set" >&2
    exit 1
fi

# Main logic
main() {
    echo "Executing skill..."
    # Your implementation here
}

# Execute
main "$@"
```

### Step 5: Test Your Skill

1. **Manual Testing**:
   ```bash
   # Copy to a test project
   cp -r general-purpose/your-skill-name /path/to/test-project/.claude/skills/
   
   # Test in Claude Code
   # Ask Claude to use your skill
   ```

2. **Script Testing** (if applicable):
   ```bash
   # Test scripts independently
   cd general-purpose/your-skill-name
   PARAM_NAME="test-value" bash scripts/run.sh
   ```

3. **Edge Cases**:
   - Test with missing parameters
   - Test with invalid inputs
   - Test in different project structures

### Step 6: Update Documentation

1. Add your skill to the appropriate README:
   - `general-purpose/README.md` or
   - `domain-specific/README.md`

2. Follow the existing format:
   ```markdown
   ### [your-skill-name](./your-skill-name/)
   
   **Description**: Brief description
   
   **When to Use**:
   - Use case 1
   - Use case 2
   
   **Key Features**:
   - Feature 1
   - Feature 2
   ```

### Step 7: Commit and Push

```bash
git add .
git commit -m "Add [your-skill-name] skill: brief description"
git push origin feature/your-skill-name
```

### Step 8: Create Pull Request

1. Go to your fork on GitHub
2. Click "New Pull Request"
3. Fill in the PR template:
   - **Title**: `Add [your-skill-name] skill`
   - **Description**: What the skill does and why it's useful
   - **Use Cases**: Specific examples of when to use it
   - **Testing**: How you tested it
   - **Dependencies**: Any requirements

## Skill Quality Standards

### Required Elements

- ✅ **Frontmatter**: Valid YAML with `name` and `description`
- ✅ **Instructions**: Clear, step-by-step guidance
- ✅ **Examples**: At least one concrete example with input/output
- ✅ **Testing**: Skill has been tested in Claude Code
- ✅ **Documentation**: README updated with new skill

### Best Practices

#### Instructions Should Be:
- **Specific**: Use exact commands, tools, or actions
- **Sequential**: Numbered steps in logical order
- **Complete**: Include error handling and edge cases
- **Actionable**: Claude should be able to execute without ambiguity

#### Examples Should Include:
- **Context**: When this example applies
- **Input**: What the user provides
- **Process**: Steps taken by Claude
- **Output**: Expected result

#### Scripts Should:
- **Validate Input**: Check for required parameters
- **Handle Errors**: Use `set -euo pipefail`
- **Be Idempotent**: Safe to run multiple times
- **Document Parameters**: Clear variable names and validation

### Anti-Patterns to Avoid

❌ **Vague Instructions**: "Do something with the files"  
✅ **Specific Instructions**: "Read all .py files in src/ directory"

❌ **Missing Examples**: Only theory, no concrete usage  
✅ **Concrete Examples**: Show actual inputs and outputs

❌ **Untested Skills**: Submit without testing  
✅ **Tested Skills**: Verified in real Claude Code environment

❌ **Hardcoded Values**: Scripts with embedded paths/values  
✅ **Parameterized**: Use environment variables

## Testing Guidelines

### Manual Testing Checklist

- [ ] Skill loads without errors in `.claude/skills/`
- [ ] Instructions are clear and unambiguous
- [ ] Claude can follow instructions independently
- [ ] Examples match actual output when executed
- [ ] Scripts run successfully with valid inputs
- [ ] Scripts fail gracefully with invalid inputs
- [ ] No hardcoded paths or credentials
- [ ] Works in different project structures

### Test Cases to Cover

1. **Happy Path**: Normal usage with valid inputs
2. **Missing Parameters**: Skill handles missing required params
3. **Invalid Input**: Skill validates and rejects bad input
4. **Edge Cases**: Empty directories, special characters, etc.
5. **Idempotency**: Running twice produces same result

## Documentation Requirements

### SKILL.md Must Include:

1. **Frontmatter**:
   ```yaml
   ---
   name: skill-name
   description: One-sentence description
   ---
   ```

2. **Title**: `# Skill Name`

3. **Instructions Section**: `## Instructions`
   - Step-by-step guidance
   - Key considerations
   - Error handling

4. **Examples Section**: `## Examples`
   - At least one complete example
   - Input, process, and output
   - Multiple examples for complex skills

5. **Optional Sections**:
   - `## Reference Templates`: Code templates, configs
   - `## Parameters`: For skills with scripts
   - `## Dependencies`: Required tools or libraries
   - `## Troubleshooting`: Common issues and solutions

### README Updates

Update the appropriate category README:

```markdown
### [skill-name](./skill-name/)

**Description**: One-sentence description

**When to Use**:
- Use case 1
- Use case 2

**Key Features**:
- Feature 1
- Feature 2
```

## Review Process

### What Happens After You Submit

1. **Automated Checks**: Validation of file structure and formatting
2. **Manual Review**: Maintainers will:
   - Test the skill in Claude Code
   - Review documentation clarity
   - Check code quality (for scripts)
   - Verify examples work as described
3. **Feedback**: You may receive comments or change requests
4. **Approval**: Once approved, your contribution will be merged

### Review Criteria

Reviewers will evaluate:

- **Functionality**: Does the skill work as described?
- **Clarity**: Can Claude execute instructions independently?
- **Quality**: Does it follow best practices?
- **Documentation**: Is it well-documented with examples?
- **Testing**: Has it been adequately tested?
- **Originality**: Does it add value to the library?

### Response Time

- Initial review: Within 5-7 business days
- Follow-up reviews: Within 2-3 business days
- Urgent fixes: Within 24-48 hours

## Questions or Issues?

- **Questions**: Open a GitHub Discussion
- **Bugs**: Open a GitHub Issue with `bug` label
- **Feature Requests**: Open a GitHub Issue with `enhancement` label
- **Security**: Email security@extractum.io (do not open public issue)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to the Claude Code Skills Library! 🎉

Your contributions help the entire community build better AI-assisted development workflows.
