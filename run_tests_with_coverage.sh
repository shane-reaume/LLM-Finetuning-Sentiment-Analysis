#!/bin/bash

# Set up colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Exit on error
set -e

echo -e "${BLUE}=== Running tests with coverage ===${NC}"

# Create necessary directories
mkdir -p reports htmlcov

# Run pytest with coverage
echo -e "${YELLOW}Running pytest with coverage...${NC}"
python -m pytest tests/ --cov=src --cov-report=xml --cov-report=html --cov-report=term

# Generate challenge test results
echo -e "${YELLOW}Running challenge tests...${NC}"
python -m src.model.sentiment_challenge_test --model_dir="models/sentiment" --output="reports/challenge_test_results.json"

# Calculate and display challenge test accuracy
echo -e "${YELLOW}Challenge test results:${NC}"
ACCURACY=$(python -c "import json; data=json.load(open('reports/challenge_test_results.json')); print(round(data['summary']['accuracy'] * 100))")
echo -e "${GREEN}Challenge test accuracy: ${ACCURACY}%${NC}"

# Calculate and display coverage
echo -e "${YELLOW}Coverage results:${NC}"
COVERAGE=$(python -c "import xml.etree.ElementTree as ET; tree = ET.parse('coverage.xml'); root = tree.getroot(); print(round(float(root.attrib['line-rate']) * 100))")
echo -e "${GREEN}Coverage: ${COVERAGE}%${NC}"

# Generate coverage badge
echo -e "${YELLOW}Generating coverage badge...${NC}"
mkdir -p badges
cat > badges/coverage-badge.json << EOF
{
  "schemaVersion": 1,
  "label": "coverage",
  "message": "${COVERAGE}%",
  "color": "$([ $COVERAGE -ge 80 ] && echo 'brightgreen' || [ $COVERAGE -ge 70 ] && echo 'green' || [ $COVERAGE -ge 60 ] && echo 'yellowgreen' || [ $COVERAGE -ge 50 ] && echo 'yellow' || echo 'red')"
}
EOF

# Generate challenge tests badge
echo -e "${YELLOW}Generating challenge tests badge...${NC}"
cat > badges/challenge-tests-badge.json << EOF
{
  "schemaVersion": 1,
  "label": "challenge tests",
  "message": "${ACCURACY}%",
  "color": "$([ $ACCURACY -ge 90 ] && echo 'brightgreen' || [ $ACCURACY -ge 80 ] && echo 'green' || [ $ACCURACY -ge 70 ] && echo 'yellowgreen' || [ $ACCURACY -ge 60 ] && echo 'yellow' || echo 'red')"
}
EOF

echo -e "${BLUE}=== Test Summary ===${NC}"
echo -e "${GREEN}Challenge Test Accuracy: ${ACCURACY}%${NC}"
echo -e "${GREEN}Code Coverage: ${COVERAGE}%${NC}"
echo -e "${BLUE}=== Test reports generated ===${NC}"
echo -e "HTML coverage report: ${YELLOW}htmlcov/index.html${NC}"
echo -e "Challenge test results: ${YELLOW}reports/challenge_test_results.json${NC}"
echo -e "Badge files: ${YELLOW}badges/coverage-badge.json${NC} and ${YELLOW}badges/challenge-tests-badge.json${NC}"

# Suggest next steps
echo -e "${BLUE}=== Next Steps ===${NC}"
echo -e "1. ${YELLOW}View the HTML coverage report:${NC} open htmlcov/index.html"
echo -e "2. ${YELLOW}Improve test coverage:${NC} Add more tests for files with low coverage"
echo -e "3. ${YELLOW}Update badges:${NC} Run ./update_badge.sh ${ACCURACY} to update the GitHub Pages badge" 