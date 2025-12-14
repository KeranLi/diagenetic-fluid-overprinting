"""
Core Isotope Mixing Model implementation for geochemical endmember analysis.
"""

import numpy as np
from scipy.optimize import minimize, Bounds
from typing import Dict, List, Tuple, Optional


class IsotopeMixingModel:
    """
    Bayesian isotope mixing model for quantifying proportions of geochemical endmembers.
    
    This class implements forward modeling, MCMC-based inversion, and Monte Carlo
    simulation for multi-isotope, multi-endmember mixing problems.
    """

    def __init__(self, 
                 endmembers: List[str],
                 isotopes: List[str],
                 endmember_ranges: Dict[str, Dict[str, Tuple[float, float]]],
                 mineral_concentrations: Dict[str, Dict[str, float]],
                 endmember_values: Optional[np.ndarray] = None):
        """
        Initialize the isotope mixing model.
        
        Args:
            endmembers: List of endmember names (e.g., ['Calcite1', 'Calcite2', 'Dolomite1', 'Dolomite2'])
            isotopes: List of isotope names (e.g., ['Ca', 'Sr', 'C', 'O'])
            endmember_ranges: Dictionary mapping each endmember to its isotopic value ranges
            mineral_concentrations: Dictionary mapping each endmember to element concentrations
            endmember_values: Optional array of mean isotopic values for each endmember
        """
        self.endmembers = endmembers
        self.isotopes = isotopes
        self.endmember_ranges = endmember_ranges
        self.mineral_concentrations = mineral_concentrations
        self.endmember_values = endmember_values
        
        self.n_endmembers = len(endmembers)
        self.n_isotopes = len(isotopes)
        
        # Initialize step size for MCMC (can be adapted during sampling)
        self.step_size = 0.1

    def generate_endmember_samples(self, num_samples: int) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Generate random isotopic samples for each endmember within specified ranges.
        
        Args:
            num_samples: Number of samples to generate per endmember
            
        Returns:
            Dictionary with endmembers as keys and isotopic samples as values
        """
        samples = {}
        for endmember in self.endmembers:
            endmember_samples = {}
            for isotope in self.isotopes:
                # Get isotopic range for current endmember
                value_range = self.endmember_ranges[endmember][isotope]
                endmember_samples[isotope] = np.random.uniform(
                    value_range[0], value_range[1], num_samples
                )
            samples[endmember] = endmember_samples
        return samples

    def forward_model(self, 
                      params: np.ndarray, 
                      proportions: np.ndarray,
                      endmember_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward model: Calculate mixed isotopic values from proportions.
        
        Args:
            params: Parameter array containing proportion corrections and isotopic offsets
            proportions: Initial mixing proportions (should sum to 1)
            endmember_values: Array of endmember isotopic values
            
        Returns:
            Tuple of (normalized_mixing_proportions, mixed_isotopic_values)
        """
        # Proportion correction parameters
        alpha = params[:self.n_endmembers - 1]
        
        # Isotopic correction parameters
        iso_corr = params[self.n_endmembers - 1:].reshape(
            self.n_endmembers, self.n_isotopes
        )
        
        # Apply exponential correction to proportions and normalize
        corrected_proportions = proportions[:-1] * np.exp(alpha)
        total = np.sum(corrected_proportions)
        mixing_ratio = np.append(
            corrected_proportions / total, 
            1 - np.sum(corrected_proportions / total)
        )
        
        # Calculate mixed isotopic values with corrections
        mixed_values = np.zeros(self.n_isotopes)
        for i in range(self.n_endmembers):
            mixed_values += mixing_ratio[i] * (endmember_values[i] + iso_corr[i])
            
        return mixing_ratio, mixed_values

    def calculate_concentration_ratios(self, 
                                       proportions: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """
        Calculate concentration-weighted ratios for each element.
        
        Args:
            proportions: Dictionary of endmember proportions
            
        Returns:
            Dictionary with element-specific contribution ratios
        """
        total_concentration = {
            element: sum(
                proportions[endmember] * self.mineral_concentrations[endmember][element]
                for endmember in self.endmembers
            )
            for element in ['Ca', 'Sr']
        }
        
        concentration_ratios = {
            element: {
                endmember: (
                    proportions[endmember] * self.mineral_concentrations[endmember][element] / 
                    total_concentration[element]
                )
                for endmember in self.endmembers
            }
            for element in ['Ca', 'Sr']
        }
        
        return concentration_ratios

    def objective_function(self, 
                           params: np.ndarray,
                           observed_data: np.ndarray,
                           data_uncertainties: np.ndarray,
                           initial_proportions: np.ndarray,
                           endmember_values: np.ndarray) -> float:
        """
        Objective function: Minimize weighted sum of squared residuals.
        
        Args:
            params: Parameter array to optimize
            observed_data: Measured isotopic values
            data_uncertainties: 1-sigma uncertainties for each isotope
            initial_proportions: Initial mixing proportions
            endmember_values: Endmember isotopic compositions
            
        Returns:
            Chi-squared value (negative log-likelihood)
        """
        _, calc_values = self.forward_model(params, initial_proportions, endmember_values)
        residuals = (observed_data - calc_values) / data_uncertainties
        return np.sum(residuals ** 2)

    def run_mcmc_sampling(self,
                          observed_data: np.ndarray,
                          data_uncertainties: np.ndarray,
                          initial_proportions: np.ndarray,
                          endmember_values: np.ndarray,
                          n_samples: int = 10000,
                          n_burn_in: int = 1000,
                          adaptive_interval: int = 100,
                          initial_params: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Markov Chain Monte Carlo (MCMC) sampling for Bayesian inversion.
        
        Implements Metropolis-Hastings algorithm with adaptive step size.
        
        Args:
            observed_data: Measured isotopic values
            data_uncertainties: Data uncertainties (1-sigma)
            initial_proportions: Initial mixing proportions
            endmember_values: Endmember isotopic compositions
            n_samples: Total number of MCMC samples
            n_burn_in: Number of burn-in iterations
            adaptive_interval: Interval for step size adaptation
            initial_params: Initial parameter values
            
        Returns:
            MCMC chain of parameters
        """
        n_params = self.n_endmembers - 1 + self.n_endmembers * self.n_isotopes
        
        if initial_params is None:
            initial_params = np.zeros(n_params)
        
        # Parameter bounds (constrain corrections to reasonable ranges)
        bounds = Bounds(
            lb=[-0.5] * n_params,
            ub=[0.5] * n_params
        )
        
        # Initialize chain
        chain = np.zeros((n_samples, n_params))
        
        # MCMC tuning parameters
        target_acceptance_rate = 0.234  # Optimal for multivariate cases
        accepted = 0
        
        current_params = initial_params.copy()
        current_likelihood = self.objective_function(
            current_params, observed_data, data_uncertainties, 
            initial_proportions, endmember_values
        )
        
        for i in range(n_samples):
            # Propose new parameters
            proposed_params = current_params + np.random.normal(
                0, self.step_size, n_params
            )
            # Apply boundary constraints
            proposed_params = np.clip(proposed_params, bounds.lb, bounds.ub)
            
            # Calculate acceptance probability
            proposed_likelihood = self.objective_function(
                proposed_params, observed_data, data_uncertainties,
                initial_proportions, endmember_values
            )
            acceptance_prob = np.exp(-(proposed_likelihood - current_likelihood))
            
            # Accept/reject step
            if acceptance_prob > np.random.rand():
                current_params = proposed_params.copy()
                current_likelihood = proposed_likelihood
                accepted += 1
                
            chain[i] = current_params
            
            # Adapt step size after burn-in
            if i > n_burn_in and i % adaptive_interval == 0:
                current_acceptance_rate = accepted / (i - n_burn_in)
                if current_acceptance_rate < target_acceptance_rate - 0.1:
                    self.step_size *= 0.9  # Decrease step size
                elif current_acceptance_rate > target_acceptance_rate + 0.1:
                    self.step_size *= 1.1  # Increase step size
                    
        return chain

    def simulate_mixture(self,
                         observed_data: np.ndarray,
                         error_threshold: float,
                         num_simulations: int) -> Tuple[list, list, list, list]:
        """
        Monte Carlo simulation of endmember mixing.
        
        Randomly samples proportions and endmember compositions to find valid mixtures
        that match observed data within specified error threshold.
        
        Args:
            observed_data: Target isotopic values
            error_threshold: Acceptable error threshold for valid mixtures
            num_simulations: Number of Monte Carlo trials
            
        Returns:
            Tuple of valid proportion lists for each endmember
        """
        valid_f1, valid_f2, valid_f3, valid_f4 = [], [], [], []
        
        for _ in range(num_simulations):
            # Sample random proportions from Dirichlet distribution
            proportions = np.random.dirichlet([1, 1, 1, 1])
            f1, f2, f3, f4 = proportions
            
            # Ensure non-negative proportions (Dirichlet guarantees this)
            if f1 < 0:
                continue
                
            # Calculate total concentrations for normalization
            total_concentration = {
                'Ca': sum(
                    p * self.mineral_concentrations[endmember]['Ca']
                    for p, endmember in zip([f1, f2, f3, f4], self.endmembers)
                ),
                'Sr': sum(
                    p * self.mineral_concentrations[endmember]['Sr']
                    for p, endmember in zip([f1, f2, f3, f4], self.endmembers)
                )
            }
            
            # Sample random endmember compositions and calculate mixture
            mixed_values = np.zeros(self.n_isotopes)
            for j, isotope in enumerate(self.isotopes):
                # Randomly sample isotopic value for each endmember
                em_values = []
                for endmember in self.endmembers:
                    iso_range = self.endmember_ranges[endmember][isotope]
                    em_values.append(np.random.uniform(iso_range[0], iso_range[1]))
                
                # Apply concentration-weighted or theoretical mixing
                if isotope in ['Ca', 'Sr']:
                    # Concentration-normalized mixing
                    for k, (p, endmember) in enumerate(zip([f1, f2, f3, f4], self.endmembers)):
                        conc_factor = self.mineral_concentrations[endmember][isotope]
                        mixed_values[j] += p * em_values[k] * conc_factor / total_concentration[isotope]
                else:
                    # Theoretical ratio mixing
                    for k, (p, endmember) in enumerate(zip([f1, f2, f3, f4], self.endmembers)):
                        mixed_values[j] += p * em_values[k] * self.mineral_concentrations[endmember][isotope]
            
            # Calculate total relative error
            total_error = 0
            for i in range(self.n_isotopes):
                # Avoid division by zero
                denom = self.observed_data[i] + 1e-10 if hasattr(self, 'observed_data') else observed_data[i] + 1e-10
                error = abs(mixed_values[i] - observed_data[i]) / denom
                total_error += error ** 2
                
            # Store valid mixtures
            if total_error < error_threshold:
                valid_f1.append(f1)
                valid_f2.append(f2)
                valid_f3.append(f3)
                valid_f4.append(f4)
                
        return valid_f1, valid_f2, valid_f3, valid_f4