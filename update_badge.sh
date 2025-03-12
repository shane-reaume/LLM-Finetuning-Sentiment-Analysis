#!/bin/bash
# Script to manually update the badge on GitHub Pages

# Set up colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Exit on error
set -e

# Check if an accuracy percentage was provided
if [ $# -eq 0 ]; then
    echo -e "${YELLOW}No accuracy percentage provided. Using default value of 55%.${NC}"
    ACCURACY=55
else
    ACCURACY=$1
fi

# Validate accuracy is a number between 0 and 100
if ! [[ "$ACCURACY" =~ ^[0-9]+$ ]] || [ "$ACCURACY" -lt 0 ] || [ "$ACCURACY" -gt 100 ]; then
    echo -e "${RED}Error: Accuracy must be a number between 0 and 100.${NC}"
    exit 1
fi

echo -e "${BLUE}Updating badge with accuracy: ${ACCURACY}%${NC}"

# Create a temporary directory
TEMP_DIR=$(mktemp -d)
echo -e "${BLUE}Created temporary directory: ${TEMP_DIR}${NC}"

# Function to clean up temporary directory
cleanup() {
    echo -e "${BLUE}Cleaning up...${NC}"
    rm -rf "$TEMP_DIR"
}

# Trap to ensure cleanup on exit
trap cleanup EXIT

# Clone the gh-pages branch
echo -e "${BLUE}Cloning gh-pages branch...${NC}"
git clone --single-branch --branch gh-pages https://github.com/shane-reaume/LLM-Finetuning-Sentiment-Analysis.git "$TEMP_DIR"

# Change to the temporary directory
cd "$TEMP_DIR"

# Determine color based on accuracy
if [ "$ACCURACY" -ge 90 ]; then
    COLOR="brightgreen"
elif [ "$ACCURACY" -ge 80 ]; then
    COLOR="green"
elif [ "$ACCURACY" -ge 70 ]; then
    COLOR="yellowgreen"
elif [ "$ACCURACY" -ge 60 ]; then
    COLOR="yellow"
else
    COLOR="red"
fi

# Create badge data
echo -e "${BLUE}Creating badge data...${NC}"
cat > challenge-tests-badge.json << EOF
{
  "schemaVersion": 1,
  "label": "challenge tests",
  "message": "${ACCURACY}%",
  "color": "${COLOR}"
}
EOF

# Commit and push changes
echo -e "${BLUE}Committing and pushing changes...${NC}"
git config user.name "Badge Update Script"
git config user.email "noreply@example.com"
git add challenge-tests-badge.json
git commit -m "Update challenge tests badge to ${ACCURACY}%"

# Ask for GitHub credentials
echo -e "${YELLOW}You will need to enter your GitHub credentials to push the changes.${NC}"
git push

echo -e "${GREEN}Badge updated successfully!${NC}"
echo -e "${BLUE}The badge should now show ${ACCURACY}% accuracy.${NC}"
echo -e "${BLUE}You can view it at: https://shane-reaume.github.io/LLM-Finetuning-Sentiment-Analysis/challenge-tests-badge.json${NC}"

exit 0 