#!/bin/bash
# Script to set up git hooks

# Set up colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Setting up git hooks...${NC}"

# Create hooks directory if it doesn't exist
mkdir -p .git/hooks

# Create pre-commit hook
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Stash any changes to test reports and coverage files
echo "Stashing any changes to test reports and coverage files..."
git stash push -q --keep-index -- reports/ htmlcov/

# Run tests before committing
echo "Running tests before commit..."
./run_tests.sh

# Store the exit code
EXIT_CODE=$?

# Restore stashed changes
echo "Restoring stashed changes..."
git stash pop -q || true

# Deactivate virtual environment
deactivate

# If tests fail, prevent commit
if [ $EXIT_CODE -ne 0 ]; then
    echo "Tests failed. Commit aborted."
    exit 1
fi

# If tests pass, allow commit
echo "Tests passed. Proceeding with commit."
exit 0
EOF

# Make pre-commit hook executable
chmod +x .git/hooks/pre-commit

echo -e "${GREEN}Git hooks set up successfully!${NC}"
echo -e "${BLUE}Pre-commit hook will run tests before each commit.${NC}" 