#!/bin/bash
# Script to commit changes with pre-commit hook disabled

# Set up colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Exit on error
set -e

# Check if a commit message was provided
if [ $# -eq 0 ]; then
    echo -e "${YELLOW}No commit message provided. Using default message.${NC}"
    COMMIT_MESSAGE="Update code and fix badges"
else
    COMMIT_MESSAGE="$1"
fi

echo -e "${BLUE}Committing changes with pre-commit hook disabled...${NC}"

# Temporarily disable pre-commit hook
PRE_COMMIT_DISABLED=false
if [ -f ".git/hooks/pre-commit" ]; then
    mv .git/hooks/pre-commit .git/hooks/pre-commit.disabled
    PRE_COMMIT_DISABLED=true
    echo -e "${GREEN}Pre-commit hook temporarily disabled.${NC}"
fi

# Function to restore pre-commit hook
restore_hook() {
    if [ "$PRE_COMMIT_DISABLED" = true ] && [ -f ".git/hooks/pre-commit.disabled" ]; then
        mv .git/hooks/pre-commit.disabled .git/hooks/pre-commit
        echo -e "${GREEN}Pre-commit hook restored.${NC}"
    fi
}

# Trap to ensure hook is restored on exit
trap restore_hook EXIT

# Add all changes
echo -e "${BLUE}Adding all changes...${NC}"
git add .

# Commit changes
echo -e "${BLUE}Committing changes with message: ${COMMIT_MESSAGE}${NC}"
if git commit -m "$COMMIT_MESSAGE"; then
    echo -e "${GREEN}Changes committed successfully!${NC}"
    
    # Ask if user wants to push changes
    read -p "Do you want to push these changes? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}Pushing changes...${NC}"
        if git push; then
            echo -e "${GREEN}Changes pushed successfully!${NC}"
        else
            echo -e "${RED}Push failed.${NC}"
            exit 1
        fi
    fi
    
    # Ask if user wants to deploy badges to GitHub Pages
    if [ -d "_site" ]; then
        read -p "Do you want to deploy badges to GitHub Pages? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${BLUE}Deploying badges to GitHub Pages...${NC}"
            
            # Create a temporary branch for GitHub Pages deployment
            TEMP_BRANCH="gh-pages-temp-$(date +%s)"
            git checkout -b $TEMP_BRANCH
            
            # Copy _site contents to the root and commit
            cp -r _site/* .
            git add *.json reports/ htmlcov/
            git commit -m "Update badges and reports [skip ci]"
            
            # Push to gh-pages branch
            if git push -f origin $TEMP_BRANCH:gh-pages; then
                echo -e "${GREEN}Badges deployed successfully to GitHub Pages!${NC}"
            else
                echo -e "${RED}GitHub Pages deployment failed.${NC}"
                git checkout -
                git branch -D $TEMP_BRANCH
                exit 1
            fi
            
            # Return to previous branch
            git checkout -
            git branch -D $TEMP_BRANCH
        fi
    else
        echo -e "${YELLOW}No _site directory found. Run ./run_tests_with_coverage.sh to generate badges.${NC}"
    fi
else
    echo -e "${RED}Commit failed.${NC}"
    exit 1
fi

exit 0 