#!/bin/bash
# Script to load/create conda environment and run all examples

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

# Example files and directories to run (excluding plot_energy_terms and api_usage_example.py)
EXAMPLES=(
    "acyclic_bond_formation"
    "cyclic_bond_formation"
    "policyclic_bond_formation"
    "simple_minimization"
    "server_usage"
)

echo -e "${GREEN}=== Running All Examples ===${NC}"
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
python -c "import numpy; import openmm" 2>/dev/null || {
    echo -e "${YELLOW}Warning: Some packages may be missing, but continuing...${NC}"
}

echo ""
echo -e "${BLUE}=== Running Examples ===${NC}"
echo ""

# Track results
TOTAL_EXAMPLES=0
PASSED_EXAMPLES=0
FAILED_EXAMPLES=0
FAILED_LIST=()

# Run each example
for example in "${EXAMPLES[@]}"; do
    TOTAL_EXAMPLES=$((TOTAL_EXAMPLES + 1))
    
    echo -e "${BLUE}Running: ${example}${NC}"
    echo "----------------------------------------"
    
    # Check if it's a Python file or a directory
    if [ -f "$example" ]; then
        # It's a Python file
        if python "$example"; then
            echo -e "${GREEN}✓ ${example} completed successfully${NC}"
            PASSED_EXAMPLES=$((PASSED_EXAMPLES + 1))
        else
            echo -e "${RED}✗ ${example} failed${NC}"
            FAILED_EXAMPLES=$((FAILED_EXAMPLES + 1))
            FAILED_LIST+=("$example")
        fi
    elif [ -d "$example" ]; then
        # It's a directory - look for run.sh
        run_script="${example}/run.sh"
        if [ -f "$run_script" ]; then
            # Change to the example directory and run the script
            if (cd "$example" && bash run.sh); then
                echo -e "${GREEN}✓ ${example} completed successfully${NC}"
                PASSED_EXAMPLES=$((PASSED_EXAMPLES + 1))
            else
                echo -e "${RED}✗ ${example} failed${NC}"
                FAILED_EXAMPLES=$((FAILED_EXAMPLES + 1))
                FAILED_LIST+=("$example")
            fi
        else
            echo -e "${YELLOW}Warning: ${run_script} not found, skipping...${NC}"
            FAILED_EXAMPLES=$((FAILED_EXAMPLES + 1))
            FAILED_LIST+=("$example (no run.sh)")
        fi
    else
        echo -e "${YELLOW}Warning: ${example} not found, skipping...${NC}"
        FAILED_EXAMPLES=$((FAILED_EXAMPLES + 1))
        FAILED_LIST+=("$example (not found)")
    fi
    
    echo ""
done

# Print summary
echo -e "${BLUE}=== Example Run Summary ===${NC}"
echo -e "Total examples: ${TOTAL_EXAMPLES}"
echo -e "${GREEN}Completed successfully: ${PASSED_EXAMPLES}${NC}"
if [ $FAILED_EXAMPLES -gt 0 ]; then
    echo -e "${RED}Failed: ${FAILED_EXAMPLES}${NC}"
    echo -e "${RED}Failed examples:${NC}"
    for item in "${FAILED_LIST[@]}"; do
        echo -e "  ${RED}- ${item}${NC}"
    done
else
    echo -e "${GREEN}Failed: 0${NC}"
fi
echo ""

# Exit with appropriate code
if [ $FAILED_EXAMPLES -eq 0 ]; then
    echo -e "${GREEN}=== All examples completed successfully! ===${NC}"
    exit 0
else
    echo -e "${RED}=== Some examples failed ===${NC}"
    exit 1
fi

