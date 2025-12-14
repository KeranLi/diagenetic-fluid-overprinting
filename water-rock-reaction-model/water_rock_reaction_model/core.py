"""
Complete Water-Rock Reaction Model for simulating oxygen isotope exchange during fluid-rock interactions.
Supports multiple fluid types, mineral transformations (calcite→dolomite), and Monte Carlo uncertainty analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any


class WaterRockReactionModel:
    """
    Water-rock reaction model for simulating oxygen isotope evolution during fluid-rock interactions.
    
    Features:
    - Three fluid types: seawater, lake water, hydrothermal fluid
    - Mineral transformation: calcite to dolomite
    - Temperature-dependent fractionation factors
    - Mass balance calculations
    - Monte Carlo uncertainty quantification
    """
    
    def __init__(self, 
                 fluid_type: str = 'seawater',
                 iterations: int = 10000,
                 mineral_type: str = 'calcite',
                 initial_fluid_range: Optional[Tuple[float, float]] = None):
        """
        Initialize the water-rock reaction model.
        
        Args:
            fluid_type: Type of fluid ('seawater', 'lake_water', 'hydrothermal')
            iterations: Number of Monte Carlo simulations
            mineral_type: Initial mineral type ('calcite' or 'dolomite')
            initial_fluid_range: Custom initial fluid δ18O range (min, max). If None, uses defaults.
        
        Raises:
            ValueError: If fluid_type or mineral_type is not supported
        """
        # Core model attributes
        self.fluid_type = fluid_type
        self.iterations = iterations
        self.mineral_type = mineral_type
        
        # Set fluid isotopic ranges based on type
        if initial_fluid_range is not None:
            self.initial_fluid_d18O_range = initial_fluid_range
        else:
            if fluid_type == 'seawater':
                self.initial_fluid_d18O_range = (-2.0, 5.0)
            elif fluid_type == 'lake_water':
                self.initial_fluid_d18O_range = (-15.0, 0.0)
            elif fluid_type == 'hydrothermal':
                self.initial_fluid_d18O_range = (5.0, 15.0)
            else:
                raise ValueError(f"Unsupported fluid_type: '{fluid_type}'")
        
        # Initial rock composition (Tibetan lake calcite)
        self.initial_rock_d18O = -2.0
        
        # Mineral parameters
        if mineral_type == 'calcite':
            self.mineral_mass = 100.09
            self.O_atoms_per_mineral = 3
        elif mineral_type == 'dolomite':
            self.mineral_mass = 184.40
            self.O_atoms_per_mineral = 6
        else:
            raise ValueError(f"Unsupported mineral_type: '{mineral_type}'")
        
        # Simulation parameters
        self.rock_mass = 100.0
        self.min_w_r_ratio = 0.01
        self.max_w_r_ratio = 100.0
        
        # Dolomitization parameters
        self.dolomitization_threshold = 30
        self.dolomitization_rate = 0.1
    
    def fractionation_factor(self, temperature: float, 
                             current_mineral_type: Optional[str] = None) -> float:
        """
        Calculate mineral-water oxygen isotope fractionation factor (alpha).
        
        Args:
            temperature: Temperature in °C
            current_mineral_type: Override the current mineral type. If None, uses self.mineral_type.
        
        Returns:
            Fractionation factor alpha
        """
        if current_mineral_type is None:
            current_mineral_type = self.mineral_type
            
        T_k = temperature + 273.15
        
        # Fractionation coefficients for different minerals
        if current_mineral_type == 'calcite':
            A, B = 2.78, -2.89
        else:  # dolomite
            A, B = 3.14, -1.5
        
        thousand_ln_alpha = (A * 1e6) / (T_k**2) + B
        return np.exp(thousand_ln_alpha / 1000.0)
    
    def convert_vpdb_to_vsmow(self, d18O_vpdb: float) -> float:
        """Convert δ18O from VPDB to VSMOW scale."""
        return d18O_vpdb * 1.03092 + 30.92
    
    def convert_vsmow_to_vpdb(self, d18O_vsmow: float) -> float:
        """Convert δ18O from VSMOW to VPDB scale."""
        return (d18O_vsmow - 30.92) / 1.03092
    
    def calculate_rock_d18O(self, fluid_d18O: float, temperature: float,
                           current_mineral_type: Optional[str] = None) -> float:
        """
        Calculate equilibrium rock δ18O value at given temperature and fluid composition.
        
        Args:
            fluid_d18O: Fluid δ18O value (VSMOW)
            temperature: Temperature in °C
            current_mineral_type: Override mineral type. If None, uses self.mineral_type.
        
        Returns:
            Equilibrium rock δ18O value (VSMOW)
        """
        if current_mineral_type is None:
            current_mineral_type = self.mineral_type
            
        alpha = self.fractionation_factor(temperature, current_mineral_type)
        return alpha * (fluid_d18O + 1000) - 1000
    
    def check_dolomitization(self, temperature: float, 
                            current_mineral_type: str) -> Tuple[str, float]:
        """
        Check if dolomitization occurs based on temperature.
        
        Args:
            temperature: Current temperature in °C
            current_mineral_type: Current mineral type
        
        Returns:
            Tuple of (new_mineral_type, conversion_factor)
        """
        if current_mineral_type == 'dolomite':
            return 'dolomite', 1.0
        
        if temperature >= self.dolomitization_threshold:
            return 'dolomite', self.dolomitization_rate
        else:
            return 'calcite', 1.0
    
    def mass_balance_exchange(self, rock_d18O_current: float, 
                             fluid_d18O_current: float,
                             w_r_ratio: float, temperature: float,
                             current_mineral_type: str) -> Tuple[float, float, str]:
        """
        Perform mass balance calculation for isotopic exchange between rock and fluid.
        
        Args:
            rock_d18O_current: Current rock δ18O (VSMOW)
            fluid_d18O_current: Current fluid δ18O (VSMOW)
            w_r_ratio: Water-rock mass ratio
            temperature: Temperature in °C
            current_mineral_type: Current mineral type
        
        Returns:
            Tuple of (new_rock_d18O, new_fluid_d18O, new_mineral_type)
        """
        new_mineral_type, conversion_factor = self.check_dolomitization(
            temperature, current_mineral_type
        )
        
        # Equilibrium values
        rock_d18O_eq = self.calculate_rock_d18O(
            fluid_d18O_current, temperature, new_mineral_type
        )
        alpha = self.fractionation_factor(temperature, new_mineral_type)
        fluid_d18O_eq = ((rock_d18O_current + 1000) / alpha) - 1000
        
        # Mass calculations
        O_mass = 16.0
        rock_oxygen_mass = self.rock_mass * (self.O_atoms_per_mineral * O_mass) / self.mineral_mass
        fluid_mass = self.rock_mass * w_r_ratio
        fluid_oxygen_mass = fluid_mass * O_mass / 18.0
        
        # Exchange amounts (10% of available mass)
        exchange_rock = rock_oxygen_mass * conversion_factor * 0.1
        exchange_fluid = fluid_oxygen_mass * conversion_factor * 0.1
        
        total_rock_O = rock_oxygen_mass
        total_fluid_O = fluid_oxygen_mass
        
        # Mass balance mixing
        new_rock_d18O = ((total_rock_O - exchange_rock) * rock_d18O_current + 
                         exchange_rock * rock_d18O_eq) / total_rock_O
        
        new_fluid_d18O = ((total_fluid_O - exchange_fluid) * fluid_d18O_current + 
                          exchange_fluid * fluid_d18O_eq) / total_fluid_O
        
        return new_rock_d18O, new_fluid_d18O, new_mineral_type
    
    def run_single_simulation(self, temperature_min: float, temperature_max: float,
                             steps: int = 50, 
                             log_w_r_ratio: Optional[float] = None) -> Dict[str, Any]:
        """
        Run a single water-rock reaction simulation along a temperature path.
        
        Args:
            temperature_min: Starting temperature in °C
            temperature_max: Ending temperature in °C
            steps: Number of temperature steps
            log_w_r_ratio: Log10 of water-rock ratio. If None, randomly samples.
        
        Returns:
            Dictionary containing simulation results and histories
        """
        if log_w_r_ratio is None:
            log_w_r_ratio = np.random.uniform(
                np.log10(self.min_w_r_ratio), 
                np.log10(self.max_w_r_ratio)
            )
        w_r_ratio = 10**log_w_r_ratio
        
        # Initialize fluid composition
        fluid_d18O = np.random.uniform(
            self.initial_fluid_d18O_range[0], 
            self.initial_fluid_d18O_range[1]
        )
        
        # Temperature path
        temperatures = np.linspace(temperature_min, temperature_max, steps)
        rock_d18O = self.convert_vpdb_to_vsmow(self.initial_rock_d18O)
        current_mineral_type = self.mineral_type
        
        # History tracking
        rock_d18O_history = []
        fluid_d18O_history = []
        mineral_history = []
        
        # Step through temperature evolution
        for temp in temperatures:
            rock_d18O, fluid_d18O, current_mineral_type = self.mass_balance_exchange(
                rock_d18O, fluid_d18O, w_r_ratio, temp, current_mineral_type
            )
            
            rock_d18O_history.append(self.convert_vsmow_to_vpdb(rock_d18O))
            fluid_d18O_history.append(fluid_d18O)
            mineral_history.append(current_mineral_type)
        
        return {
            'temperatures': temperatures,
            'final_rock_d18O': rock_d18O_history[-1],
            'final_fluid_d18O': fluid_d18O_history[-1],
            'w_r_ratio': w_r_ratio,
            'log_w_r_ratio': log_w_r_ratio,
            'rock_d18O_history': rock_d18O_history,
            'fluid_d18O_history': fluid_d18O_history,
            'mineral_history': mineral_history
        }
    
    def run_model(self, temperature_min: float, temperature_max: float, 
                  steps: int = 50) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Run complete Monte Carlo simulation with multiple iterations.
        
        Args:
            temperature_min: Starting temperature in °C
            temperature_max: Ending temperature in °C
            steps: Number of temperature steps
        
        Returns:
            Tuple of (results_dataframe, all_histories_list)
        """
        results_list = []
        all_histories = []
        
        print(f"Running {self.iterations} iterations for {self.fluid_type} "
              f"({self.mineral_type})...")
        
        for i in tqdm(range(self.iterations), desc="Simulating"):
            result = self.run_single_simulation(temperature_min, temperature_max, steps)
            
            results_list.append({
                'iteration': i,
                'final_rock_d18O': result['final_rock_d18O'],
                'final_fluid_d18O': result['final_fluid_d18O'],
                'w_r_ratio': result['w_r_ratio'],
                'log_w_r_ratio': result['log_w_r_ratio'],
                'initial_fluid_d18O': result['fluid_d18O_history'][0],
                'final_mineral': result['mineral_history'][-1]
            })
            
            all_histories.append(result)
        
        results_df = pd.DataFrame(results_list)
        return results_df, all_histories
    
    def plot_results(self, results_df: pd.DataFrame, all_histories: List[Dict],
                     n_plot: int = 100, save_figure: bool = False,
                     filename_prefix: str = "model_results",
                     dpi: int = 300, file_format: str = "svg") -> plt.Figure:
        """
        Generate professional visualization of simulation results.
        
        Args:
            results_df: DataFrame with simulation results
            all_histories: List of full simulation histories
            n_plot: Number of individual paths to plot
            save_figure: Whether to save the figure to file
            filename_prefix: Prefix for saved file name
            dpi: Resolution for saved figure
            file_format: File format ('png', 'pdf', 'svg')
        
        Returns:
            Matplotlib figure object
        """
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=100)
        
        # Main title
        fig.suptitle(f'Water-Rock Reaction Model Results\n{self.mineral_type.capitalize()} + {self.fluid_type.replace("_", " ").title()}',
                     fontsize=16, fontweight='bold', y=0.995)
        
        # Color schemes
        colors = plt.cm.viridis(np.linspace(0, 1, n_plot))
        cmap_fluid = 'plasma'
        cmap_wr = 'coolwarm'
        
        # Plot 1: Final rock δ18O distribution
        rock_data = results_df['final_rock_d18O']
        data_range = rock_data.max() - rock_data.min()
        bins = min(50, max(10, int(len(rock_data) / 10))) if data_range > 0.01 else 10
        
        axes[0, 0].hist(rock_data, bins=bins, alpha=0.7, edgecolor='black', color='steelblue')
        axes[0, 0].axvline(rock_data.median(), color='crimson', linestyle='--', linewidth=2,
                          label=f"Median: {rock_data.median():.2f}‰")
        axes[0, 0].axvline(rock_data.mean(), color='darkorange', linestyle=':', linewidth=2,
                          label=f"Mean: {rock_data.mean():.2f}‰")
        axes[0, 0].set_xlabel('$\delta^{18}O_{VPDB}$ (‰)', fontsize=11)
        axes[0, 0].set_ylabel('Frequency', fontsize=11)
        axes[0, 0].set_title(f'Final Rock $\delta^{18}O$ Distribution\n{self.mineral_type.capitalize()}',
                            fontsize=12, fontweight='bold')
        axes[0, 0].legend(fontsize=9)
        axes[0, 0].grid(True, alpha=0.3, linestyle='--')
        
        # Plot 2: Initial fluid δ18O distribution
        init_fluid_data = results_df['initial_fluid_d18O']
        axes[0, 1].hist(init_fluid_data, bins=bins, alpha=0.7, edgecolor='black', color='darkcyan')
        axes[0, 1].axvline(init_fluid_data.median(), color='crimson', linestyle='--', linewidth=2,
                          label=f"Median: {init_fluid_data.median():.2f}‰")
        axes[0, 1].set_xlabel('$\delta^{18}O_{VSMOW}$ (‰)', fontsize=11)
        axes[0, 1].set_ylabel('Frequency', fontsize=11)
        axes[0, 1].set_title(f'Initial Fluid $\delta^{18}O$ Distribution\n{self.fluid_type.replace("_", " ").title()}',
                            fontsize=12, fontweight='bold')
        axes[0, 1].legend(fontsize=9)
        axes[0, 1].grid(True, alpha=0.3, linestyle='--')
        
        # Plot 3: Water-rock ratio distribution
        wr_data = results_df['log_w_r_ratio']
        wr_range = wr_data.max() - wr_data.min()
        bins_wr = min(50, max(10, int(len(wr_data) / 10))) if wr_range > 0.01 else 10
        
        axes[0, 2].hist(wr_data, bins=bins_wr, alpha=0.7, edgecolor='black', color='darkgreen')
        axes[0, 2].axvline(wr_data.median(), color='crimson', linestyle='--', linewidth=2,
                          label=f"Median: {wr_data.median():.2f}")
        axes[0, 2].set_xlabel('log$_{10}$(W/R ratio)', fontsize=11)
        axes[0, 2].set_ylabel('Frequency', fontsize=11)
        axes[0, 2].set_title(f'Water-Rock Ratio Distribution\nlog$_{10}$(W/R)',
                            fontsize=12, fontweight='bold')
        axes[0, 2].legend(fontsize=9)
        axes[0, 2].grid(True, alpha=0.3, linestyle='--')
        
        # Plot 4: Temperature evolution paths
        axes[1, 0].set_prop_cycle('color', colors)
        for i in range(min(n_plot, len(all_histories))):
            axes[1, 0].plot(all_histories[i]['temperatures'],
                            all_histories[i]['rock_d18O_history'],
                            alpha=0.3, linewidth=0.8)
        
        axes[1, 0].set_xlabel('Temperature (°C)', fontsize=11)
        axes[1, 0].set_ylabel('$\delta^{18}O_{VPDB}$ (‰)', fontsize=11)
        axes[1, 0].set_title(f'Rock $\delta^{18}O$ Evolution Paths (n={min(n_plot, len(all_histories))})',
                            fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.4, linestyle='--')
        
        # Plot 5: Final rock δ18O vs W/R ratio
        scatter1 = axes[1, 1].scatter(results_df['log_w_r_ratio'],
                                     results_df['final_rock_d18O'],
                                     alpha=0.6, s=15, c=results_df['initial_fluid_d18O'],
                                     cmap=cmap_fluid, edgecolors='k', linewidths=0.3)
        axes[1, 1].set_xlabel('log$_{10}$(W/R ratio)', fontsize=11)
        axes[1, 1].set_ylabel('Final Rock $\delta^{18}O_{VPDB}$ (‰)', fontsize=11)
        axes[1, 1].set_title('Final Rock $\delta^{18}O$ vs W/R Ratio',
                            fontsize=12, fontweight='bold')
        cbar1 = plt.colorbar(scatter1, ax=axes[1, 1],
                            label='Initial Fluid $\delta^{18}O_{VSMOW}$ (‰)')
        cbar1.ax.tick_params(labelsize=9)
        axes[1, 1].grid(True, alpha=0.3, linestyle='--')
        
        # Plot 6: Initial fluid vs final rock
        scatter2 = axes[1, 2].scatter(results_df['initial_fluid_d18O'],
                                     results_df['final_rock_d18O'],
                                     alpha=0.6, s=15, c=results_df['log_w_r_ratio'],
                                     cmap=cmap_wr, edgecolors='k', linewidths=0.3)
        axes[1, 2].set_xlabel('Initial Fluid $\delta^{18}O_{VSMOW}$ (‰)', fontsize=11)
        axes[1, 2].set_ylabel('Final Rock $\delta^{18}O_{VPDB}$ (‰)', fontsize=11)
        axes[1, 2].set_title('Initial Fluid vs Final Rock $\delta^{18}O$',
                            fontsize=12, fontweight='bold')
        cbar2 = plt.colorbar(scatter2, ax=axes[1, 2],
                            label='log$_{10}$(W/R ratio)')
        cbar2.ax.tick_params(labelsize=9)
        axes[1, 2].grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        # Add statistics text box
        stats_text = f"""Model Parameters:
Mineral: {self.mineral_type.capitalize()}
Fluid: {self.fluid_type.replace("_", " ").title()}
Iterations: {self.iterations}

Final Rock δ18O:
Mean: {rock_data.mean():.2f}‰
Median: {rock_data.median():.2f}‰
Std: {rock_data.std():.2f}‰
Range: {rock_data.min():.2f} to {rock_data.max():.2f}‰
"""
        
        fig.text(0.02, 0.02, stats_text, fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.5),
                 verticalalignment='bottom', horizontalalignment='left')
        
        # Save figure
        if save_figure:
            filename = f"{filename_prefix}_{self.mineral_type}_{self.fluid_type}.{file_format}"
            plt.savefig(filename, dpi=dpi, bbox_inches='tight',
                       transparent=False, facecolor='white')
            print(f"\nFigure saved as: {filename}")
        
        return fig
    
    def export_results(self, results_df: pd.DataFrame, all_histories: List[Dict],
                      filename_prefix: str):
        """
        Export simulation results to CSV files.
        
        Args:
            results_df: DataFrame with simulation results
            all_histories: List of full simulation histories
            filename_prefix: Prefix for output file names
        """
        # Summary statistics
        summary = results_df.describe()
        summary.to_csv(f"{filename_prefix}_summary.csv")
        results_df.to_csv(f"{filename_prefix}_detailed.csv", index=False)
        
        # Export representative paths
        data_dict = {}
        for i, hist in enumerate(all_histories[:10]):
            data_dict[f'path_{i}_temp'] = hist['temperatures']
            data_dict[f'path_{i}_rock'] = hist['rock_d18O_history']
            data_dict[f'path_{i}_fluid'] = hist['fluid_d18O_history']
            data_dict[f'path_{i}_mineral'] = hist['mineral_history']
        
        example_histories = pd.DataFrame(data_dict)
        example_histories.to_csv(f"{filename_prefix}_paths.csv", index=False)
        print(f"Results exported to {filename_prefix}_*.csv")