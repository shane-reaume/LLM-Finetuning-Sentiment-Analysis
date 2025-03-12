#!/bin/bash
# Script to deploy badges and reports to GitHub Pages without committing to main

# Set up colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Exit on error
set -e

# Parse command line options
SKIP_TESTS=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --skip-tests) SKIP_TESTS=true ;;
        --help) 
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --skip-tests    Skip running tests and just deploy existing _site directory"
            echo "  --help          Show this help message"
            exit 0
            ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

echo -e "${BLUE}=== Deploying to GitHub Pages ===${NC}"

# Run tests if not skipped
if [ "$SKIP_TESTS" = false ]; then
    echo -e "${BLUE}Running tests to generate coverage and badges...${NC}"
    ./run_tests_with_coverage.sh
    if [ $? -ne 0 ]; then
        echo -e "${RED}Tests failed. Deployment aborted.${NC}"
        exit 1
    fi
fi

# Check if _site directory exists
if [ ! -d "_site" ]; then
    echo -e "${RED}Error: _site directory not found.${NC}"
    echo -e "${YELLOW}Run ./run_tests_with_coverage.sh first to generate badges and reports.${NC}"
    exit 1
fi

# Get current branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo -e "${BLUE}Current branch: ${CURRENT_BRANCH}${NC}"

# Create a temporary branch for GitHub Pages deployment
TEMP_BRANCH="gh-pages-temp-$(date +%s)"
echo -e "${BLUE}Creating temporary branch: ${TEMP_BRANCH}${NC}"
git checkout -b $TEMP_BRANCH

# Copy _site contents to the root
echo -e "${BLUE}Copying files from _site directory...${NC}"
cp -r _site/* .

# Explicitly create .nojekyll file
echo -e "${BLUE}Creating .nojekyll file...${NC}"
touch .nojekyll

# Add all necessary files
echo -e "${BLUE}Adding files to git...${NC}"
git add -f *.json .nojekyll
git add -f reports/ || true
git add -f htmlcov/ || true

# Commit changes
echo -e "${BLUE}Committing changes...${NC}"
git commit -m "Update badges and reports [skip ci]"

# Push to gh-pages branch
echo -e "${BLUE}Pushing to gh-pages branch...${NC}"
if git push -f origin HEAD:gh-pages; then
    echo -e "${GREEN}Successfully deployed to GitHub Pages!${NC}"
else
    echo -e "${RED}Failed to push to GitHub Pages.${NC}"
    git checkout $CURRENT_BRANCH
    git branch -D $TEMP_BRANCH
    exit 1
fi

# Return to original branch
echo -e "${BLUE}Returning to original branch: ${CURRENT_BRANCH}${NC}"
git checkout $CURRENT_BRANCH
git branch -D $TEMP_BRANCH

echo -e "${GREEN}=== Deployment Complete ===${NC}"
echo -e "${BLUE}Badges and reports are now available at:${NC}"
echo -e "${YELLOW}https://shane-reaume.github.io/LLM-Finetuning-Sentiment-Analysis/${NC}"

exit 0 