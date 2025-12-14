"""
Basic usage example for the Isotope Mixing Model.
This script demonstrates how to initialize the model, run simulations, and visualize results.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isotope_mixing_model import IsotopeMixingModel
from isotope_mixing_model.config import (
    ENDMEMBER_RANGES, MINERAL_CONCENTRATIONS, 
    OBSERVED_DATA, DATA_UNCERTAINTIES, INITIAL_PROPORTIONS
)

# Set plot style
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['font.size'] = 10


def run_basic_simulation():
    """Run a basic endmember mixing simulation."""
    
    # Initialize the model
    model = IsotopeMixingModel(
        endmembers=['Calcite1', 'Calcite2', 'Dolomite1', 'Dolomite2'],
        isotopes=['Ca', 'Sr', 'C', 'O'],
        endmember_ranges=ENDMEMBER_RANGES,
        mineral_concentrations=MINERAL_CONCENTRATIONS
    )
    
    print("Running Monte Carlo mixture simulation...")
    
    # Run Monte Carlo simulation to find valid mixtures
    valid_f1, valid_f2, valid_f3, valid_f4 = model.simulate_mixture(
        observed_data=OBSERVED_DATA,
        error_threshold=0.01,
        num_simulations=10000
    )
    
    print(f"Found {len(valid_f1)} valid mixtures")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    axes[0, 0].hist(valid_f1, bins=30, alpha=0.7, label='Calcite1')
    axes[0, 0].set_title('Calcite1 Proportions')
    axes[0, 0].set_xlabel('Mixing Fraction')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    
    axes[0, 1].hist(valid_f2, bins=30, alpha=0.7, label='Calcite2', color='orange')
    axes[0, 1].set_title('Calcite2 Proportions')
    axes[0, 1].set_xlabel('Mixing Fraction')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    
    axes[1, 0].hist(valid_f3, bins=30, alpha=0.7, label='Dolomite1', color='green')
    axes[1, 0].set_title('Dolomite1 Proportions')
    axes[1, 0].set_xlabel('Mixing Fraction')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    
    axes[1, 1].hist(valid_f4, bins=30, alpha=0.7, label='Dolomite2', color='red')
    axes[1, 1].set_title('Dolomite2 Proportions')
    axes[1, 1].set_xlabel('Mixing Fraction')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('mixing_proportions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Calcite1:  mean={np.mean(valid_f1):.3f}, std={np.std(valid_f1):.3f}")
    print(f"Calcite2:  mean={np.mean(valid_f2):.3f}, std={np.std(valid_f2):.3f}")
    print(f"Dolomite1: mean={np.mean(valid_f3):.3f}, std={np.std(valid_f3):.3f}")
    print(f"Dolomite2: mean={np.mean(valid_f4):.3f}, std={np.std(valid_f4):.3f}")


def run_mcmc_inversion():
    """Run MCMC inversion for parameter estimation."""
    
    # Initialize the model
    model = IsotopeMixingModel(
        endmembers=['Calcite1', 'Calcite2', 'Dolomite1', 'Dolomite2'],
        isotopes=['Ca', 'Sr', 'C', 'O'],
        endmember_ranges=ENDMEMBER_RANGES,
        mineral_concentrations=MINERAL_CONCENTRATIONS
    )
    
    # Generate synthetic endmember values (means of ranges)
    endmember_values = np.array([
        [np.mean(ENDMEMBER_RANGES[em][iso]) for iso in ['Ca', 'Sr', 'C', 'O']]
        for em in ['Calcite1', 'Calcite2', 'Dolomite1', 'Dolomite2']
    ])
    
    # Initial parameters (zeros mean no correction)
    initial_params = np.zeros(model.n_endmembers - 1 + model.n_endmembers * model.n_isotopes)
    
    print("\nRunning MCMC inversion...")
    print(f"Target acceptance rate: 0.234")
    
    # Run MCMC
    chain = model.run_mcmc_sampling(
        observed_data=OBSERVED_DATA,
        data_uncertainties=DATA_UNCERTAINTIES,
        initial_proportions=INITIAL_PROPORTIONS,
        endmember_values=endmember_values,
        n_samples=5000,
        n_burn_in=1000,
        initial_params=initial_params
    )
    
    # Plot parameter traces
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    axes = axes.flatten()
    
    for i in range(min(len(chain[0]), 9)):
        axes[i].plot(chain[:, i], alpha=0.7)
        axes[i].set_title(f'Parameter {i}')
        axes[i].set_xlabel('Iteration')
        axes[i].set_ylabel('Value')
    
    plt.tight_layout()
    plt.savefig('mcmc_traces.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print final parameter statistics
    print("\nMCMC Results (post-burn-in):")
    post_burnin_chain = chain[1000:]
    for i in range(len(initial_params)):
        mean_val = np.mean(post_burnin_chain[:, i])
        std_val = np.std(post_burnin_chain[:, i])
        print(f"Parameter {i}: mean={mean_val:.4f}, std={std_val:.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Isotope Mixing Model - Basic Usage Example")
    print("=" * 60)
    
    # Run basic simulation
    run_basic_simulation()
    
    # Run MCMC inversion
    run_mcmc_inversion()
    
    print("\nDone! Check the generated PNG files for results.")