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
    "test_analytical_distance.py"
    "test_coordinate_conversion.py"
    "test_zmatrix.py"
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
echo -e "${BLUE}=== Running Unit Tests with pytest ===${NC}"
echo ""

# Change to project root directory (parent of test directory)
cd "$SCRIPT_DIR/.."

# Check if pytest is available
if ! python -m pytest --version &> /dev/null; then
    echo -e "${RED}Error: pytest not found${NC}"
    echo "Please install pytest: conda install pytest"
    exit 1
fi

# Run all tests with pytest (consistent with GitHub workflow)
echo -e "${BLUE}Running pytest on test directory...${NC}"
echo "----------------------------------------"

if python -m pytest test/ -v --tb=short --color=yes; then
    echo ""
    echo -e "${GREEN}=== All tests passed! ===${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}=== Some tests failed ===${NC}"
    exit 1
fi
