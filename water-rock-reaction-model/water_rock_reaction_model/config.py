"""
Default configuration parameters for the Water-Rock Reaction Model.
Modify these values to match your geological setting.
"""

# Default fluid isotopic ranges (δ18O, VSMOW)
# Format: (min_value, max_value)
DEFAULT_FLUID_RANGES = {
    'seawater': (-2.0, 5.0),
    'lake_water': (-15.0, 0.0),
    'hydrothermal': (5.0, 15.0)
}

# Mineral properties
# mass: molecular weight (g/mol)
# O_atoms: number of oxygen atoms per formula unit
MINERAL_PROPERTIES = {
    'calcite': {'mass': 100.09, 'O_atoms': 3},
    'dolomite': {'mass': 184.40, 'O_atoms': 6}
}

# Dolomitization parameters
DOLOMITIZATION_THRESHOLD = 30  # Temperature threshold in °C
DOLOMITIZATION_RATE = 0.1      # Reaction progress factor

# Rock parameters
ROCK_MASS = 100.0  # Default rock mass (g)

# Water-rock ratio range
W_R_RATIO_RANGE = {
    'min': 0.01,  # Minimum W/R ratio (log10 scale)
    'max': 100.0  # Maximum W/R ratio (log10 scale)
}

# Isotopic conversion factors
# VPDB to VSMOW conversion: d18O_vsmow = d18O_vpdb * A + B
VPDB_VSMOW_CONVERSION = {'A': 1.03092, 'B': 30.92}