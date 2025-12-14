"""
Example configuration data for the isotope mixing model.
Users should modify these values based on their specific geological context.
"""
import numpy as np

# Example endmember isotopic ranges (min, max values)
# Units: delta notation (‰) or absolute ratios depending on isotope system
ENDMEMBER_RANGES = {
    'Calcite1': {
        'Ca': (-0.0, -0.0),   # δ44/40Ca
        'Sr': (-0.0, -0.0), # 87Sr/86Sr
        'C': (-0.0, -0.0),    # δ13C
        'O': (-0.0, -0.0)    # δ18O
    },
    'Calcite2': {
        'Ca': (-0.0, -0.0),
        'Sr': (-0.0, -0.0),
        'C': (-0.0, -0.0),
        'O': (-0.0, -0.0)
    },
    'Dolomite1': {
        'Ca': (-0.0, -0.0),
        'Sr': (-0.0, -0.0),
        'C': (-0.0, -0.0),
        'O': (-0.0, -0.0)
    },
    'Dolomite2': {
        'Ca': (-0.0, -0.0),
        'Sr': (-0.0, -0.0),
        'C': (-0.0, -0.0),
        'O': (-0.0, 0.0)
    }
}

# Mineral concentrations (weight % or molar concentrations)
MINERAL_CONCENTRATIONS = {
    'Calcite1': {'Ca': 40.0, 'Sr': 0.02, 'C': 12.0, 'O': 48.0},
    'Calcite2': {'Ca': 39.5, 'Sr': 0.03, 'C': 12.0, 'O': 48.0},
    'Dolomite1': {'Ca': 21.7, 'Sr': 0.01, 'C': 13.0, 'O': 52.0},
    'Dolomite2': {'Ca': 21.5, 'Sr': 0.015, 'C': 13.0, 'O': 52.0}
}

# Example observed data for a mixed sample
OBSERVED_DATA = np.array([0.8, 0.7095, 1.5, -2.5])  # [Ca, Sr, C, O]

# Data uncertainties (1-sigma)
DATA_UNCERTAINTIES = np.array([0.1, 0.0002, 0.2, 0.3])

# Initial mixing proportions (must sum to 1)
INITIAL_PROPORTIONS = np.array([0.25, 0.25, 0.25, 0.25])