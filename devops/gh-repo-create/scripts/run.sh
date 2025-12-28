#!/usr/bin/env bash
set -euo pipefail

# GitHub Repository Creation & Setup Script
# Works on macOS, Linux, and Unix systems
# Usage: bash run.sh

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

print_header() {
    echo -e "${BLUE}=========================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}=========================================${NC}"
    echo ""
}

# Step 1: Get repository name from user
print_header "Step 1: Repository Information"

read -p "Enter the GitHub repository name: " REPO_NAME
if [[ -z "$REPO_NAME" ]]; then
    print_error "Repository name cannot be empty"
    exit 1
fi

# Validate repository name (alphanumeric, hyphens, underscores)
if [[ ! "$REPO_NAME" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    print_error "Invalid repository name. Use only letters, numbers, hyphens, and underscores."
    exit 1
fi

print_success "Repository name: $REPO_NAME"

# Step 2: Verify GitHub CLI
print_header "Step 2: GitHub CLI Verification"

print_info "Checking GitHub CLI installation..."

if ! command -v gh &> /dev/null; then
    print_error "GitHub CLI (gh) is not installed"
    echo ""
    echo "Install instructions:"
    echo "  macOS:  brew install gh"
    echo "  Linux:  https://github.com/cli/cli/blob/trunk/docs/install_linux.md"
    echo "  Other:  https://github.com/cli/cli#installation"
    exit 1
fi

print_success "GitHub CLI found: $(gh --version | head -1)"

print_info "Checking authentication status..."
if ! gh auth status &> /dev/null; then
    print_error "Not authenticated with GitHub CLI"
    echo ""
    echo "Please authenticate first:"
    echo "  gh auth login"
    exit 1
fi

print_success "Authenticated with GitHub"

# Get GitHub username
GH_USERNAME=$(gh api user -q .login)
print_success "GitHub username: $GH_USERNAME"

# Step 3: Create repository
print_header "Step 3: Create GitHub Repository"

read -p "Should this be a public or private repository? (public/private) [public]: " VISIBILITY
VISIBILITY=${VISIBILITY:-public}

if [[ "$VISIBILITY" != "public" && "$VISIBILITY" != "private" ]]; then
    print_info "Invalid choice, defaulting to public"
    VISIBILITY="public"
fi

read -p "Enter repository description (optional): " REPO_DESC

print_info "Creating GitHub repository..."

# Create repository
if [[ -n "$REPO_DESC" ]]; then
    if gh repo create "$REPO_NAME" "--$VISIBILITY" --description "$REPO_DESC" --source=. --remote=origin 2>&1; then
        print_success "Repository created with description"
    else
        print_error "Failed to create repository"
        exit 1
    fi
else
    if gh repo create "$REPO_NAME" "--$VISIBILITY" --source=. --remote=origin 2>&1; then
        print_success "Repository created"
    else
        print_error "Failed to create repository"
        exit 1
    fi
fi

print_success "Repository URL: https://github.com/$GH_USERNAME/$REPO_NAME"

# Step 4: Generate SSH deployment keys
print_header "Step 4: Generate SSH Deployment Keys"

print_info "Setting up SSH directory..."
mkdir -p ~/.ssh
chmod 700 ~/.ssh

SSH_KEY_PATH="$HOME/.ssh/deploy_$REPO_NAME"

if [[ -f "$SSH_KEY_PATH" ]]; then
    print_error "SSH key already exists: $SSH_KEY_PATH"
    read -p "Overwrite existing key? (yes/no) [no]: " OVERWRITE
    OVERWRITE=${OVERWRITE:-no}
    
    if [[ "$OVERWRITE" == "yes" ]]; then
        rm -f "$SSH_KEY_PATH" "$SSH_KEY_PATH.pub"
        ssh-keygen -t ed25519 -C "deploy-key-$REPO_NAME" -f "$SSH_KEY_PATH" -N ""
        print_success "New SSH key generated"
    else
        print_info "Using existing SSH key"
    fi
else
    print_info "Generating SSH key pair..."
    ssh-keygen -t ed25519 -C "deploy-key-$REPO_NAME" -f "$SSH_KEY_PATH" -N ""
    print_success "SSH key generated"
fi

chmod 600 "$SSH_KEY_PATH"
chmod 644 "$SSH_KEY_PATH.pub"

print_success "Private key: $SSH_KEY_PATH"
print_success "Public key: $SSH_KEY_PATH.pub"

# Show key fingerprint
KEY_FINGERPRINT=$(ssh-keygen -l -f "$SSH_KEY_PATH.pub" | awk '{print $2}')
print_info "Key fingerprint: $KEY_FINGERPRINT"

# Step 5: Configure SSH host
print_header "Step 5: Configure SSH Host"

SSH_CONFIG="$HOME/.ssh/config"
touch "$SSH_CONFIG"
chmod 600 "$SSH_CONFIG"

HOST_ALIAS="github.com-$REPO_NAME"

print_info "Configuring SSH host in ~/.ssh/config..."

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
print_header "Step 6: Deploy SSH Key to GitHub"

print_info "Adding deploy key to GitHub repository..."

if gh repo deploy-key add "$SSH_KEY_PATH.pub" \
    --title "Deploy key for $REPO_NAME (no expiration)" \
    --allow-write \
    --repo "$GH_USERNAME/$REPO_NAME" 2>&1; then
    print_success "Deploy key added with read/write access"
else
    print_error "Failed to add deploy key"
    exit 1
fi

# Verify deploy key installation
print_info "Verifying deploy key installation..."
if gh repo deploy-key list --repo "$GH_USERNAME/$REPO_NAME" | grep -q "Deploy key"; then
    print_success "Deploy key verified on GitHub"
fi

# Test SSH connection
print_info "Testing SSH connection..."
SSH_TEST_OUTPUT=$(ssh -T "$HOST_ALIAS" 2>&1 || true)
if echo "$SSH_TEST_OUTPUT" | grep -q "successfully authenticated"; then
    print_success "SSH authentication successful"
else
    print_info "SSH test output: $SSH_TEST_OUTPUT"
fi

# Step 7: Configure git remote
print_header "Step 7: Configure Git Remote"

print_info "Configuring git remote..."

# Initialize git if needed
if [[ ! -d .git ]]; then
    git init
    print_success "Git repository initialized"
else
    print_info "Git repository already initialized"
fi

# Remove existing origin if present
if git remote | grep -q "^origin$"; then
    git remote remove origin
    print_info "Removed existing origin remote"
fi

# Add new origin with custom SSH host
REMOTE_URL="git@$HOST_ALIAS:$GH_USERNAME/$REPO_NAME.git"
git remote add origin "$REMOTE_URL"
print_success "Git remote added: $REMOTE_URL"

# Set main branch
if git branch 2>/dev/null | grep -q "main"; then
    print_info "Already on main branch"
else
    git branch -M main 2>/dev/null || git checkout -b main 2>/dev/null || true
    print_success "Set default branch to main"
fi

# Step 8: Verify setup
print_header "Step 8: Verify Setup"

CHECKS_PASSED=0
CHECKS_TOTAL=6

print_info "Running verification checks..."

# Check 1: Git status
if git status &> /dev/null; then
    print_success "Git repository is initialized"
    ((CHECKS_PASSED++))
else
    print_error "Git repository check failed"
fi

# Check 2: SSH private key
if [[ -f "$SSH_KEY_PATH" ]]; then
    print_success "Private key exists: $SSH_KEY_PATH"
    ((CHECKS_PASSED++))
else
    print_error "Private key not found"
fi

# Check 3: SSH public key
if [[ -f "$SSH_KEY_PATH.pub" ]]; then
    print_success "Public key exists: $SSH_KEY_PATH.pub"
    ((CHECKS_PASSED++))
else
    print_error "Public key not found"
fi

# Check 4: Remote URL
if git remote get-url origin | grep -q "$HOST_ALIAS"; then
    print_success "Remote URL configured correctly"
    ((CHECKS_PASSED++))
else
    print_error "Remote URL check failed"
fi

# Check 5: Deploy key on GitHub
if gh repo deploy-key list --repo "$GH_USERNAME/$REPO_NAME" 2>/dev/null | grep -q "Deploy key"; then
    print_success "Deploy key is installed on GitHub"
    ((CHECKS_PASSED++))
else
    print_error "Deploy key verification failed"
fi

# Check 6: Repository accessibility
if git ls-remote origin &> /dev/null; then
    print_success "Repository is accessible via git"
    ((CHECKS_PASSED++))
else
    print_error "Repository accessibility check failed"
fi

# Final summary
print_header "Setup Complete!"

echo "Checks passed: $CHECKS_PASSED/$CHECKS_TOTAL"
echo ""

if [[ $CHECKS_PASSED -eq $CHECKS_TOTAL ]]; then
    print_success "All checks passed!"
else
    print_error "Some checks failed. Please review the output above."
fi

echo ""
echo "Repository Details:"
echo "  URL:        https://github.com/$GH_USERNAME/$REPO_NAME"
echo "  SSH Key:    $SSH_KEY_PATH"
echo "  SSH Host:   $HOST_ALIAS"
echo "  Git Remote: $REMOTE_URL"
echo "  Visibility: $VISIBILITY"
echo ""
echo "Next steps to push your code:"
echo "  1. git add ."
echo "  2. git commit -m \"Initial commit\""
echo "  3. git push -u origin main"
echo ""

exit 0
