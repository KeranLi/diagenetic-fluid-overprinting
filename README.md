# Diagenetic Fluid Overprinting on Micrite Stable Isotopes: Implications for Paleo-elevation Reconstruction

## 🎯 Overview

This repository provides the complete computational framework for quantifying diagenetic fluid overprinting effects on micritic carbonate stable isotopes, developed for *Geochimica et Cosmochimica Acta*. The code implements two coupled numerical models:

1. **BayesianMCMC**: Bayesian inverse modeling of no more than four-component mixing (calcite1, calcite2, dolomite1, dolomite2 or others) using Ca, Sr, C, and O isotope and others systematics.
2. **WaterRockReactionModel**: Forward Monte Carlo simulation of oxygen isotope exchange during fluid-rock interaction, incorporating temperature-dependent fractionation and calcite-to-dolomite transformation kinetics.

Together, these tools enable rigorous assessment of post-depositional alteration effects on micrite δ¹⁸O signatures, critical for robust paleo-elevation reconstructions.

---

## 📊 Scientific Context

Micritic carbonates are widely used paleo-elevation proxies, but their δ¹⁸O values are susceptible to diagenetic overprinting by meteoric, marine, and hydrothermal fluids. This computational framework addresses two fundamental uncertainties:

- **Mixing Ambiguity**: Deconvolution of primary vs. diagenetic endmember contributions
- **Kinetic Overprinting**: Quantification of isotopic re-equilibration during fluid-rock interaction

The models employ Bayesian MCMC methods and mass-balance principles to propagate uncertainties through the diagenetic system.

---

## 🛠 Model Descriptions

### 1. Isotope Mixing Model

**Mathematical Framework:**
- **Forward Model**: `δ_mixed = Σ(f_i × (δ_i + ε_i))` where f_i are proportion-corrected mixing fractions
- **Bayesian Inversion**: Metropolis-Hastings MCMC with adaptive step sizes (target acceptance: 23.4%)
- **Parameter Space**: `θ = [α_1...α_n-1, ε_1,1...ε_n,m]` (proportion corrections + isotopic offsets)
- **Objective Function**: `χ² = Σ[(δ_obs - δ_calc)/σ_obs]²`

**Key Features:**
- Concentration-weighted mixing for Ca and Sr isotopes
- Uncertainty quantification via posterior parameter distributions
- Support for 4 endmembers × 4 isotope systems

### 2. Water-Rock Reaction Model

**Mathematical Framework:**
- **Fractionation Factor**: `1000·lnα = (A·10⁶/T²) + B`
- **Mass Balance**: `δ'_rock = [(M_rock - ΔM)·δ_rock + ΔM·δ_eq] / M_rock`
- **Dolomitization Kinetics**: `T_threshold = 30°C` with rate factor `k = 0.1`
- **Monte Carlo Sampling**: 10⁴ iterations with random W/R ratios (0.01–100) and fluid compositions

**Key Features:**
- VPDB-VSMOW scale conversions (precise: δ_vsmow = 1.03092·δ_vpdb + 30.92)
- Temperature-dependent mineral transformations
- Stochastic sampling of parameter space

---

## 💻 Installation

### Quick Start (Linux, macOS may also works, window may spend time on the environment)
```bash
# Clone repository
git clone https://github.com/KeranLi/diagenetic-fluid-overprinting.git
cd diagenetic-fluid-overprinting

# Setup environment and dependencies
bash scripts/setup.sh

# Run complete analysis suite
bash scripts/run_all.sh

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Run examples
python3 examples/combined_analysis.py

from BayesianMCMC import IsotopeMixingModel

# Initialize with geological endmembers
model = IsotopeMixingModel(
    endmembers=['Calcite1', 'Dolomite1', 'Dolomite2'],
    isotopes=['Ca', 'Sr', 'C', 'O'],
    endmember_ranges=ENDMEMBER_RANGES,
    mineral_concentrations=MINERAL_CONCENTRATIONS
)

# Run Bayesian inversion
results, mcmc_chain = model.invert_observed_data(
    observed_data=[0.0, 0.0002, 0.0, 0.0],
    uncertainties=[0.1, 0.0002, 0.0, 0.0]
)

# Plot posterior distributions
model.plot_posterior(mcmc_chain)

from water_rock_reaction_model import WaterRockReactionModel

# Simulate meteoric diagenesis
model = WaterRockReactionModel(
    fluid_type='lake_water',
    iterations=5000,
    mineral_type='calcite'
)

# Run temperature-path simulation
results, histories = model.run_model(
    temperature_min=20,
    temperature_max=70,
    steps=10
)

# Generate publication figures
fig = model.plot_results(
    results, 
    histories, 
    save_figure=True,
    filename_prefix="micrite_diagenesis"
)

# Export data
model.export_results(results, histories, "lake_water_results")

Python >= 3.8
numpy >= 1.21.0
scipy >= 1.7.0
pandas >= 1.3.0
matplotlib >= 3.5.0
tqdm >= 4.60.0

🛡️ License
This project is licensed under the MIT License