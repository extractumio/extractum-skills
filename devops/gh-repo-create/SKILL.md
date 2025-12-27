---
name: gh-repo-create
description: Create a GitHub repository, configure SSH deployment keys, and set up git remote for the current folder. Works on macOS, Linux, and Unix systems.
---

# GitHub Repository Creation & Setup

## Security Note

⚠️ **Important**: This skill generates SSH keys **without passphrases** for convenience. This is suitable for:
- Personal projects on encrypted systems
- Development workstations with full disk encryption enabled

**Not recommended for**: Shared systems, production servers, or CI/CD pipelines.

For enhanced security, consider adding a passphrase when prompted during key generation, or use GitHub Personal Access Tokens (PAT) for automated systems.

## Instructions

When this skill is invoked, follow these steps to initialize a git repository, create it on GitHub, configure SSH deployment keys, and prepare for pushing code:

### Step 1: Get Repository Name from User

- **Ask the user** for the desired GitHub repository name
- Validate the repository name (no spaces, valid GitHub naming conventions)
- Store the repository name for use in subsequent steps
- Example prompt: "What would you like to name the GitHub repository?"

### Step 2: Verify GitHub CLI Installation & Authentication

Before proceeding, ensure the GitHub CLI (`gh`) is properly installed and authenticated:

1. **Check if `gh` exists**:
   ```bash
   which gh || command -v gh
   ```
   - If not found, inform the user to install GitHub CLI:
     - macOS: `brew install gh`
     - Linux: See https://github.com/cli/cli#installation
     - Unix: Use package manager or binary installation

2. **Check authentication status**:
   ```bash
   gh auth status
   ```
   - If not authenticated, guide user to run:
     ```bash
     gh auth login
     ```
   - Verify the output shows authenticated status with proper scopes

3. **Verify required permissions**:
   - Ensure the authentication includes `repo` scope for creating repositories
   - If scope is missing, re-authenticate with proper permissions

### Step 3: Create Repository on GitHub

Create the repository using GitHub CLI:

1. **Determine repository visibility** (ask user: public or private):
   ```bash
   # For public repository
   gh repo create <repo-name> --public --source=. --remote=origin
   
   # For private repository
   gh repo create <repo-name> --private --source=. --remote=origin
   ```

2. **Add repository description** (optional, ask user):
   ```bash
   gh repo create <repo-name> --public --description "Repository description" --source=. --remote=origin
   ```

3. **Verify repository creation**:
   ```bash
   gh repo view <owner>/<repo-name>
   ```
   - Confirm the repository exists on GitHub
   - Note the repository owner (username or organization)

### Step 4: Generate SSH Deployment Keys

Generate dedicated SSH key pair for this repository:

1. **Create SSH directory if it doesn't exist**:
   ```bash
   mkdir -p ~/.ssh
   chmod 700 ~/.ssh
   ```

2. **Generate SSH key pair**:
   ```bash
   ssh-keygen -t ed25519 -C "deploy-key-<repo-name>" -f ~/.ssh/deploy_<repo-name> -N ""
   ```
   - Use Ed25519 algorithm (modern, secure, fast)
   - No passphrase (`-N ""`) for automated deployments
   - Key files: `~/.ssh/deploy_<repo-name>` (private) and `~/.ssh/deploy_<repo-name>.pub` (public)

3. **Set proper permissions**:
   ```bash
   chmod 600 ~/.ssh/deploy_<repo-name>
   chmod 644 ~/.ssh/deploy_<repo-name>.pub
   ```

4. **Verify key generation**:
   ```bash
   ls -la ~/.ssh/deploy_<repo-name>*
   ssh-keygen -l -f ~/.ssh/deploy_<repo-name>.pub
   ```

### Step 5: Configure SSH Host in ~/.ssh/config

Add a custom SSH host configuration for this repository:

1. **Create or update ~/.ssh/config**:
   ```bash
   # Create if doesn't exist
   touch ~/.ssh/config
   chmod 600 ~/.ssh/config
   ```

2. **Add host configuration**:
   ```
   # GitHub Deploy Key for <repo-name>
   Host github.com-<repo-name>
       HostName github.com
       User git
       IdentityFile ~/.ssh/deploy_<repo-name>
       IdentitiesOnly yes
       AddKeysToAgent yes
   ```

3. **Append to config file safely**:
   ```bash
   cat >> ~/.ssh/config << 'EOF'
   
   # GitHub Deploy Key for <repo-name>
   Host github.com-<repo-name>
       HostName github.com
       User git
       IdentityFile ~/.ssh/deploy_<repo-name>
       IdentitiesOnly yes
       AddKeysToAgent yes
   EOF
   ```

4. **Verify configuration**:
   ```bash
   cat ~/.ssh/config | grep -A 5 "github.com-<repo-name>"
   ```

### Step 6: Deploy SSH Key to GitHub Repository

Add the public key to the GitHub repository as a deploy key with write permissions:

1. **Read the public key**:
   ```bash
   DEPLOY_KEY=$(cat ~/.ssh/deploy_<repo-name>.pub)
   ```

2. **Add deploy key via GitHub CLI**:
   ```bash
   gh repo deploy-key add ~/.ssh/deploy_<repo-name>.pub \
       --title "Deploy key for <repo-name> (no expiration)" \
       --allow-write \
       --repo <owner>/<repo-name>
   ```
   - `--allow-write`: Enables read/write operations (required for pushing)
   - `--title`: Descriptive name for the key
   - No expiration by default (deploy keys don't expire unless revoked)

3. **Verify deploy key installation**:
   ```bash
   gh repo deploy-key list --repo <owner>/<repo-name>
   ```
   - Confirm the key appears in the list
   - Check that "Allow write access" is enabled

4. **Test SSH connection**:
   ```bash
   ssh -T github.com-<repo-name>
   ```
   - Expected output: "Hi <username>! You've successfully authenticated..."

### Step 7: Configure Git Remote URLs

Set up git remotes to use the custom SSH host:

1. **Initialize git repository** (if not already done):
   ```bash
   git init
   ```

2. **Check existing remotes**:
   ```bash
   git remote -v
   ```

3. **Update origin remote to use custom SSH host**:
   ```bash
   # Remove existing origin if present
   git remote remove origin 2>/dev/null || true
   
   # Add new origin with custom SSH host
   git remote add origin git@github.com-<repo-name>:<owner>/<repo-name>.git
   ```

4. **Set upstream tracking**:
   ```bash
   # Will be used after first push
   git branch -M main
   ```

5. **Verify remote configuration**:
   ```bash
   git remote -v
   git remote show origin
   ```

### Step 8: Verify Setup (Non-Intrusive Check)

Perform validation checks without making changes to ensure everything is ready:

1. **Check git status**:
   ```bash
   git status
   ```
   - Verify git is initialized
   - Check current branch (should be `main` or `master`)

2. **Verify SSH key is accessible**:
   ```bash
   test -f ~/.ssh/deploy_<repo-name> && echo "✓ Private key exists"
   test -f ~/.ssh/deploy_<repo-name>.pub && echo "✓ Public key exists"
   ```

3. **Test SSH authentication** (non-intrusive):
   ```bash
   ssh -T github.com-<repo-name> 2>&1 | grep -q "successfully authenticated" && echo "✓ SSH authentication works"
   ```

4. **Verify remote configuration**:
   ```bash
   git remote get-url origin | grep -q "github.com-<repo-name>" && echo "✓ Remote URL configured correctly"
   ```

5. **Check deploy key on GitHub**:
   ```bash
   gh repo deploy-key list --repo <owner>/<repo-name> | grep -q "Deploy key" && echo "✓ Deploy key is installed"
   ```

6. **Test repository accessibility** (fetch without pulling):
   ```bash
   git ls-remote origin 2>&1 | grep -q "HEAD" && echo "✓ Repository is accessible"
   ```

7. **Summary report**:
   ```
   ✓ Git repository initialized
   ✓ GitHub repository created: https://github.com/<owner>/<repo-name>
   ✓ SSH deployment key generated: ~/.ssh/deploy_<repo-name>
   ✓ SSH config updated: github.com-<repo-name>
   ✓ Deploy key added to repository (read/write, no expiration)
   ✓ Git remote configured: git@github.com-<repo-name>:<owner>/<repo-name>.git
   ✓ SSH authentication successful
   
   Ready to commit and push! Next steps:
   1. git add .
   2. git commit -m "Initial commit"
   3. git push -u origin main
   ```

---

## Examples

### Example 1: Create Public Repository for New Project

**Scenario**: You have a local project folder and want to create a new public GitHub repository.

**Execution**:

```bash
# User is in project directory: ~/projects/my-awesome-app
cd ~/projects/my-awesome-app

# User invokes the skill
# Skill prompts: "What would you like to name the GitHub repository?"
# User responds: "my-awesome-app"

# Skill prompts: "Should this be a public or private repository?"
# User responds: "public"
```

**Actions Performed**:

1. Checks `gh` is installed and authenticated ✓
2. Creates repository: `gh repo create my-awesome-app --public --source=. --remote=origin`
3. Generates SSH keys: `~/.ssh/deploy_my-awesome-app` and `~/.ssh/deploy_my-awesome-app.pub`
4. Updates `~/.ssh/config` with host alias `github.com-my-awesome-app`
5. Adds deploy key to GitHub: `gh repo deploy-key add ... --allow-write`
6. Configures git remote: `git@github.com-my-awesome-app:username/my-awesome-app.git`
7. Verifies setup with non-intrusive checks

**Output**:

```
✓ GitHub CLI authenticated
✓ Git repository initialized
✓ GitHub repository created: https://github.com/username/my-awesome-app
✓ SSH deployment key generated: ~/.ssh/deploy_my-awesome-app
✓ SSH config updated: github.com-my-awesome-app
✓ Deploy key added to repository (read/write, no expiration)
✓ Git remote configured
✓ SSH authentication successful

Ready to push! Run:
  git add .
  git commit -m "Initial commit"
  git push -u origin main
```

---

### Example 2: Create Private Repository with Description

**Scenario**: You want to create a private repository with a custom description.

**Execution**:

```bash
cd ~/projects/secret-project

# Skill prompts for:
# - Repository name: "secret-project"
# - Visibility: "private"
# - Description: "Confidential project for internal use"
```

**Actions Performed**:

1. Creates private repository with description
2. Generates dedicated SSH keys for this project
3. Configures custom SSH host in `~/.ssh/config`
4. Deploys key with write access (no expiration)
5. Sets up git remote with custom SSH host
6. Validates all configurations

**Result**:

- Private repository created on GitHub
- SSH keys isolated per project
- Git remote ready for secure push operations
- User can commit and push code securely

---

### Example 3: Setup Repository in Organization

**Scenario**: You want to create a repository under a GitHub organization.

**Execution**:

```bash
cd ~/projects/company-project

# Skill prompts:
# - Repository name: "company-project"
# - Owner: User selects organization "my-company" instead of personal account
# - Visibility: "private"
```

**Actions Performed**:

1. Creates repository under organization: `gh repo create my-company/company-project --private`
2. Generates SSH keys: `~/.ssh/deploy_company-project`
3. Configures SSH host: `github.com-company-project`
4. Adds deploy key to `my-company/company-project`
5. Sets git remote: `git@github.com-company-project:my-company/company-project.git`
6. Verifies organization repository access

**Benefits**:

- Works seamlessly with personal accounts and organizations
- Deploy keys work regardless of account type
- SSH configuration isolates credentials per project

---

## Implementation Reference

Below is a complete bash script that implements this skill. This can be used as a reference or executed directly:

```bash
#!/usr/bin/env bash
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${YELLOW}→${NC} $1"
}

# Step 1: Get repository name from user
read -p "Enter the GitHub repository name: " REPO_NAME
if [[ -z "$REPO_NAME" ]]; then
    print_error "Repository name cannot be empty"
    exit 1
fi

# Validate repository name
if [[ ! "$REPO_NAME" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    print_error "Invalid repository name. Use only letters, numbers, hyphens, and underscores."
    exit 1
fi

print_success "Repository name: $REPO_NAME"

# Step 2: Verify GitHub CLI
print_info "Checking GitHub CLI installation..."

if ! command -v gh &> /dev/null; then
    print_error "GitHub CLI (gh) is not installed"
    echo "Install it from: https://github.com/cli/cli#installation"
    exit 1
fi

print_success "GitHub CLI found: $(gh --version | head -1)"

print_info "Checking authentication status..."
if ! gh auth status &> /dev/null; then
    print_error "Not authenticated with GitHub CLI"
    echo "Run: gh auth login"
    exit 1
fi

print_success "Authenticated with GitHub"

# Get GitHub username
GH_USERNAME=$(gh api user -q .login)
print_info "GitHub username: $GH_USERNAME"

# Step 3: Create repository
read -p "Should this be a public or private repository? (public/private): " VISIBILITY
if [[ "$VISIBILITY" != "public" && "$VISIBILITY" != "private" ]]; then
    VISIBILITY="public"
fi

read -p "Enter repository description (optional): " REPO_DESC

print_info "Creating GitHub repository..."

if [[ -n "$REPO_DESC" ]]; then
    gh repo create "$REPO_NAME" "--$VISIBILITY" --description "$REPO_DESC" --source=. --remote=origin
else
    gh repo create "$REPO_NAME" "--$VISIBILITY" --source=. --remote=origin
fi

print_success "Repository created: https://github.com/$GH_USERNAME/$REPO_NAME"

# Step 4: Generate SSH deployment keys
print_info "Generating SSH deployment keys..."

mkdir -p ~/.ssh
chmod 700 ~/.ssh

SSH_KEY_PATH="$HOME/.ssh/deploy_$REPO_NAME"

if [[ -f "$SSH_KEY_PATH" ]]; then
    print_error "SSH key already exists: $SSH_KEY_PATH"
    read -p "Overwrite? (yes/no): " OVERWRITE
    if [[ "$OVERWRITE" != "yes" ]]; then
        print_info "Using existing SSH key"
    else
        rm -f "$SSH_KEY_PATH" "$SSH_KEY_PATH.pub"
        ssh-keygen -t ed25519 -C "deploy-key-$REPO_NAME" -f "$SSH_KEY_PATH" -N ""
        print_success "New SSH key generated"
    fi
else
    ssh-keygen -t ed25519 -C "deploy-key-$REPO_NAME" -f "$SSH_KEY_PATH" -N ""
    print_success "SSH key generated"
fi

chmod 600 "$SSH_KEY_PATH"
chmod 644 "$SSH_KEY_PATH.pub"

print_success "Private key: $SSH_KEY_PATH"
print_success "Public key: $SSH_KEY_PATH.pub"

# Step 5: Configure SSH host
print_info "Configuring SSH host in ~/.ssh/config..."

SSH_CONFIG="$HOME/.ssh/config"
touch "$SSH_CONFIG"
chmod 600 "$SSH_CONFIG"

HOST_ALIAS="github.com-$REPO_NAME"

# Check if host already exists
if grep -q "Host $HOST_ALIAS" "$SSH_CONFIG"; then
    print_info "Host $HOST_ALIAS already exists in SSH config"
else
    cat >> "$SSH_CONFIG" << EOF

# GitHub Deploy Key for $REPO_NAME
Host $HOST_ALIAS
    HostName github.com
    User git
    IdentityFile $SSH_KEY_PATH
    IdentitiesOnly yes
    AddKeysToAgent yes
EOF
    print_success "SSH host configured: $HOST_ALIAS"
fi

# Step 6: Deploy SSH key to GitHub
print_info "Adding deploy key to GitHub repository..."

gh repo deploy-key add "$SSH_KEY_PATH.pub" \
    --title "Deploy key for $REPO_NAME (no expiration)" \
    --allow-write \
    --repo "$GH_USERNAME/$REPO_NAME"

print_success "Deploy key added with read/write access"

# Test SSH connection
print_info "Testing SSH connection..."
if ssh -T "$HOST_ALIAS" 2>&1 | grep -q "successfully authenticated"; then
    print_success "SSH authentication successful"
else
    print_error "SSH authentication test inconclusive (this may be normal)"
fi

# Step 7: Configure git remote
print_info "Configuring git remote..."

# Initialize git if needed
if [[ ! -d .git ]]; then
    git init
    print_success "Git repository initialized"
fi

# Remove existing origin if present
git remote remove origin 2>/dev/null || true

# Add new origin with custom SSH host
REMOTE_URL="git@$HOST_ALIAS:$GH_USERNAME/$REPO_NAME.git"
git remote add origin "$REMOTE_URL"
print_success "Git remote added: $REMOTE_URL"

# Set main branch
git branch -M main 2>/dev/null || true

# Step 8: Verify setup
print_info "Verifying setup..."

# Check git status
if git status &> /dev/null; then
    print_success "Git repository initialized"
fi

# Check SSH keys
if [[ -f "$SSH_KEY_PATH" ]]; then
    print_success "Private key exists: $SSH_KEY_PATH"
fi

if [[ -f "$SSH_KEY_PATH.pub" ]]; then
    print_success "Public key exists: $SSH_KEY_PATH.pub"
fi

# Check remote URL
if git remote get-url origin | grep -q "$HOST_ALIAS"; then
    print_success "Remote URL configured correctly"
fi

# Check deploy key
if gh repo deploy-key list --repo "$GH_USERNAME/$REPO_NAME" | grep -q "Deploy key"; then
    print_success "Deploy key is installed on GitHub"
fi

# Test repository accessibility
if git ls-remote origin &> /dev/null; then
    print_success "Repository is accessible"
fi

# Final summary
echo ""
echo "========================================="
echo "  Setup Complete!"
echo "========================================="
echo ""
echo "Repository: https://github.com/$GH_USERNAME/$REPO_NAME"
echo "SSH Key: $SSH_KEY_PATH"
echo "SSH Host: $HOST_ALIAS"
echo "Git Remote: $REMOTE_URL"
echo ""
echo "Next steps:"
echo "  1. git add ."
echo "  2. git commit -m \"Initial commit\""
echo "  3. git push -u origin main"
echo ""

