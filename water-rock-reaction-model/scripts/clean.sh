#!/bin/bash
# clean.sh - Clean generated files and temporary data

set -e  # Exit on error

echo "========================================"
echo "Water-Rock Reaction Model - Clean Script"
echo "========================================"
echo ""

# Files to clean
files_to_remove=(
    "*.png"
    "*.svg"
    "*.pdf"
    "*.csv"
    "*.pyc"
    "**/__pycache__/"
    ".pytest_cache/"
    "*.log"
)

echo "Removing generated files..."

for pattern in "${files_to_remove[@]}"; do
    if compgen -G "$pattern" > /dev/null; then
        rm -rf $pattern
        echo "  Removed: $pattern"
    fi
done

# Ask about virtual environment
read -p "Do you want to remove the virtual environment? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -d "venv" ]; then
        rm -rf venv
        echo "Virtual environment removed."
    fi
fi

echo ""
echo "Clean completed!"