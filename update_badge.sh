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

# Create artifacts directory if it doesn't exist
ARTIFACTS_DIR="_site"
mkdir -p "$ARTIFACTS_DIR"
echo -e "${BLUE}Using artifacts directory: ${ARTIFACTS_DIR}${NC}"

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
cat > "${ARTIFACTS_DIR}/challenge-tests-badge.json" << EOF
{
  "schemaVersion": 1,
  "label": "challenge tests",
  "message": "${ACCURACY}%",
  "color": "${COLOR}"
}
EOF

# Create reports directory if it doesn't exist
mkdir -p "${ARTIFACTS_DIR}/reports"

# Copy any existing reports if they exist
if [ -d "reports" ]; then
    cp -r reports/* "${ARTIFACTS_DIR}/reports/"
fi

echo -e "${GREEN}Badge updated successfully!${NC}"
echo -e "${BLUE}The badge should now show ${ACCURACY}% accuracy.${NC}"
echo -e "${BLUE}You can view it at: https://shane-reaume.github.io/LLM-Finetuning-Sentiment-Analysis/challenge-tests-badge.json${NC}"
echo -e "${YELLOW}Note: You'll need to deploy the contents of the ${ARTIFACTS_DIR} directory to GitHub Pages for the badge to be visible.${NC}"

exit 0 