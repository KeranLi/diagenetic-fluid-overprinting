#!/bin/bash
# run_mcmc.sh - Run only the MCMC inversion example

set -e

echo "========================================"
echo "BayesianMCMC - MCMC Only Script"
echo "========================================"
echo ""

# Check virtual environment
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Run: bash scripts/setup.sh"
    exit 1
fi

source venv/bin/activate

# Create temporary script for MCMC only
cat > /tmp/run_mcmc_only.py << 'EOF'
import sys
sys.path.append('.')
from examples.basic_usage import run_mcmc_inversion

if __name__ == "__main__":
    print("Running MCMC inversion only...")
    run_mcmc_inversion()
    print("MCMC completed! Check mcmc_traces.png")
EOF

python3 /tmp/run_mcmc_only.py

# Clean up
rm /tmp/run_mcmc_only.py