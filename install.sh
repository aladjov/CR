#!/bin/bash

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Customer Retention Framework Setup${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Use uv's default virtual environment
VENV_NAME=".venv"
VENV_PATH="$SCRIPT_DIR/$VENV_NAME"

# Check if uv is installed
echo -e "${BLUE}[1/5]${NC} Checking for uv..."
if ! command -v uv &> /dev/null; then
    echo -e "${RED}Error: uv is not installed.${NC}"
    echo -e "${YELLOW}Please install uv first:${NC}"
    echo -e "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo -e "  or visit: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi
echo -e "${GREEN}✓ uv is installed${NC}"
echo ""

# Check if virtual environment exists
echo -e "${BLUE}[2/5]${NC} Checking virtual environment..."
if [ -d "$VENV_PATH" ]; then
    echo -e "${YELLOW}Virtual environment '$VENV_NAME' already exists at: $VENV_PATH${NC}"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Removing existing virtual environment...${NC}"
        rm -rf "$VENV_PATH"
        uv venv "$VENV_NAME"
        echo -e "${GREEN}✓ Virtual environment created: $VENV_PATH${NC}"
    else
        echo -e "${BLUE}Using existing virtual environment${NC}"
    fi
else
    echo -e "${BLUE}Creating virtual environment '$VENV_NAME'...${NC}"
    uv venv "$VENV_NAME"
    echo -e "${GREEN}✓ Virtual environment created: $VENV_PATH${NC}"
fi
echo ""

# Activate virtual environment
echo -e "${BLUE}[3/5]${NC} Activating virtual environment..."
source "$VENV_PATH/bin/activate"
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Install dependencies and package (including SHAP based on architecture)
echo -e "${BLUE}[4/5]${NC} Installing package with all dependencies..."
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    echo -e "${BLUE}Detected Apple Silicon - installing with ml-shap${NC}"
    uv pip install -e ".[dev,ml-shap]"
elif [ "$ARCH" = "x86_64" ] && [ "$(uname)" = "Darwin" ]; then
    echo -e "${BLUE}Detected Intel Mac - installing with ml-shap-intel${NC}"
    uv pip install -e ".[dev,ml-shap-intel]"
else
    echo -e "${BLUE}Detected Linux - installing with ml-shap${NC}"
    uv pip install -e ".[dev,ml-shap]"
fi
echo -e "${GREEN}✓ Package and dependencies installed${NC}"
echo ""

# Verify pytest installation
echo -e "${BLUE}[5/5]${NC} Verifying installation..."
if ! python -m pytest --version &> /dev/null; then
    echo -e "${RED}Error: pytest not found after installation${NC}"
    exit 1
fi
PYTEST_VERSION=$(python -m pytest --version | head -n 1)
echo -e "${GREEN}✓ $PYTEST_VERSION${NC}"

if python -c "import customer_retention" 2>/dev/null; then
    echo -e "${GREEN}✓ Package imports successfully${NC}"
else
    echo -e "${RED}Error: Package import failed${NC}"
    exit 1
fi
echo ""

# Print success message and next steps
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo ""
echo -e "1. Activate the virtual environment:"
echo -e "   ${YELLOW}source $VENV_PATH/bin/activate${NC}"
echo ""
echo -e "2. Run all tests:"
echo -e "   ${YELLOW}python -m pytest tests/${NC}"
echo ""
echo -e "3. Run tests with coverage:"
echo -e "   ${YELLOW}python -m pytest tests/ --cov=src/customer_retention --cov-report=term-missing${NC}"
echo ""
echo -e "4. Run specific test file:"
echo -e "   ${YELLOW}python -m pytest tests/profiling/test_quality_checks.py -v${NC}"
echo ""
echo -e "5. Deactivate virtual environment when done:"
echo -e "   ${YELLOW}deactivate${NC}"
echo ""
echo -e "${BLUE}Environment details:${NC}"
echo -e "  Virtual env: $VENV_PATH"
echo -e "  Python: $(python --version)"
echo -e "  Pytest: $PYTEST_VERSION"
echo ""
echo -e "${GREEN}✓ SHAP installed for model interpretation${NC}"
echo -e "   ${BLUE}Note: On Databricks ML runtime, shap is pre-installed${NC}"
echo ""
