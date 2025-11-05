#!/bin/bash
# Script to load/create conda environment and run all unit tests

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Environment name from environment.yml
ENV_NAME="rco_devel"
ENV_FILE="../environment.yml"

# Test files to run
TEST_FILES=(
    "test_coordinate_conversion.py"
    "test_molecular_system.py"
    "test_ring_closing_force_field.py"
    "test_ring_closure_optimizer.py"
    "test_server.py"
    "test_iotools.py"
)

echo -e "${GREEN}=== Running All Unit Tests ===${NC}"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda not found in PATH${NC}"
    echo "Please ensure conda is installed and in your PATH"
    exit 1
fi

# Initialize conda for bash shell
eval "$(conda shell.bash hook)"

# Check if environment exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo -e "${GREEN}Found existing conda environment: ${ENV_NAME}${NC}"
    echo "Activating environment..."
    conda activate "$ENV_NAME"
else
    echo -e "${YELLOW}Environment ${ENV_NAME} not found${NC}"
    
    if [ ! -f "$ENV_FILE" ]; then
        echo -e "${RED}Error: ${ENV_FILE} not found${NC}"
        exit 1
    fi
    
    echo "Creating environment from ${ENV_FILE}..."
    conda env create -f "$ENV_FILE"
    echo -e "${GREEN}Environment created successfully${NC}"
    echo "Activating environment..."
    conda activate "$ENV_NAME"
fi

# Verify activation
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]; then
    echo -e "${RED}Error: Failed to activate environment ${ENV_NAME}${NC}"
    exit 1
fi

echo -e "${GREEN}Environment activated: ${CONDA_DEFAULT_ENV}${NC}"
echo ""

# Verify Python and required packages
echo "Checking Python installation..."
python --version || {
    echo -e "${RED}Error: Python not available in environment${NC}"
    exit 1
}

echo "Checking required packages..."
python -c "import numpy; import unittest; import openmm" 2>/dev/null || {
    echo -e "${YELLOW}Warning: Some packages may be missing, but continuing...${NC}"
}

echo ""
echo -e "${BLUE}=== Running Unit Tests ===${NC}"
echo ""

# Track results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
FAILED_FILES=()

# Run each test file
for test_file in "${TEST_FILES[@]}"; do
    if [ ! -f "$test_file" ]; then
        echo -e "${YELLOW}Warning: Test file ${test_file} not found, skipping...${NC}"
        continue
    fi
    
    echo -e "${BLUE}Running: ${test_file}${NC}"
    echo "----------------------------------------"
    
    # Run the test
    if python "$test_file"; then
        echo -e "${GREEN}✓ ${test_file} passed${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo -e "${RED}✗ ${test_file} failed${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        FAILED_FILES+=("$test_file")
    fi
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo ""
done

# Print summary
echo -e "${BLUE}=== Test Summary ===${NC}"
echo -e "Total test files: ${TOTAL_TESTS}"
echo -e "${GREEN}Passed: ${PASSED_TESTS}${NC}"
if [ $FAILED_TESTS -gt 0 ]; then
    echo -e "${RED}Failed: ${FAILED_TESTS}${NC}"
    echo -e "${RED}Failed files:${NC}"
    for file in "${FAILED_FILES[@]}"; do
        echo -e "  ${RED}- ${file}${NC}"
    done
else
    echo -e "${GREEN}Failed: 0${NC}"
fi
echo ""

# Exit with appropriate code
if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}=== All tests passed! ===${NC}"
    exit 0
else
    echo -e "${RED}=== Some tests failed ===${NC}"
    exit 1
fi
