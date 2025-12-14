#!/bin/bash
# setup.sh - Initial environment setup for Isotope Mixing Model

set -e  # Exit on error

echo "========================================"
echo "BayesianMCMC - Setup Script"
echo "========================================"
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "Found $PYTHON_VERSION"

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "WARNING: pip not found, attempting to install..."
    python3 -m ensurepip --upgrade
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created successfully!"
else
    echo "Virtual environment already exists, skipping creation..."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "========================================"
echo "Setup completed successfully!"
echo "========================================"
echo ""
echo "To activate the environment in future sessions, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the example:"
echo "  bash scripts/run_example.sh"