#!/bin/bash
# Script to run tests and generate coverage reports

# Set up colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}Warning: Not running in a virtual environment. Some commands may fail.${NC}"
    echo -e "${YELLOW}Consider activating the virtual environment with: source venv/bin/activate${NC}"
fi

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}Error: pytest is not installed or not in PATH.${NC}"
    echo -e "${YELLOW}Try running: pip install -r requirements.txt${NC}"
    exit 1
fi

# Create reports directory if it doesn't exist
mkdir -p reports/coverage

echo -e "${BLUE}Running unit tests...${NC}"
pytest tests/test_sentiment_model.py -v
TEST_RESULT=$?

echo -e "${BLUE}Generating coverage report...${NC}"
pytest --cov=src --cov-report=html --cov-report=term
COV_RESULT=$?

# Check if model directory exists
if [ ! -d "models/sentiment" ]; then
    echo -e "${YELLOW}Warning: Model directory 'models/sentiment' not found. Challenge tests may fail.${NC}"
    echo -e "${YELLOW}You may need to train the model first with: python -m src.model.sentiment_train${NC}"
fi

echo -e "${BLUE}Running challenge tests...${NC}"
python -m src.model.sentiment_challenge_test --model_dir models/sentiment
CHALLENGE_RESULT=$?

# Copy coverage report to reports directory
if [ -d "htmlcov" ]; then
    cp -r htmlcov/* reports/coverage/
    echo -e "${GREEN}Coverage report copied to reports/coverage/${NC}"
else
    echo -e "${YELLOW}Warning: htmlcov directory not found. Coverage report may not have been generated.${NC}"
fi

# Print summary
echo -e "\n${BLUE}Test Summary:${NC}"
if [ $TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ Unit tests passed${NC}"
else
    echo -e "${RED}✗ Unit tests failed${NC}"
fi

if [ $COV_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ Coverage report generated${NC}"
    echo -e "${BLUE}Coverage report available at:${NC} reports/coverage/index.html"
else
    echo -e "${RED}✗ Coverage report generation failed${NC}"
fi

if [ $CHALLENGE_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ Challenge tests completed${NC}"
    echo -e "${BLUE}Challenge test results available at:${NC} reports/challenge_test_results.json"
else
    echo -e "${RED}✗ Challenge tests failed${NC}"
fi

# Exit with error if any test failed
if [ $TEST_RESULT -ne 0 ] || [ $COV_RESULT -ne 0 ] || [ $CHALLENGE_RESULT -ne 0 ]; then
    exit 1
fi

echo -e "\n${GREEN}All tests passed!${NC}"
exit 0 