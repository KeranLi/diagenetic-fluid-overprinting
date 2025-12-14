#!/bin/bash
# run_example.sh - Run the basic usage example

set -e  # Exit on error

echo "========================================"
echo "Water-Rock Reaction Model - Run Example Script"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Please run setup first:"
    echo "  bash scripts/setup.sh"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if example script exists
if [ ! -f "examples/basic_usage.py" ]; then
    echo "ERROR: Example script not found at examples/basic_usage.py"
    exit 1
fi

# Run the example
echo "Running basic usage example..."
echo "This will generate PNG files and CSV exports for three fluid types."
echo "This may take a few minutes depending on the number of iterations."
echo ""

python3 examples/basic_usage.py

echo ""
echo "========================================"
echo "Example completed successfully!"
echo "========================================"
echo ""
echo "Generated files:"
echo "  - fluid_comparison.png"
echo "  - *_results_*.png"
echo "  - *_results_*.csv"