#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monte Carlo Simulation Runner for Piston Leak Lab
================================================

Runs ensemble simulations of coupled ODE+ABM systems to explore
parameter space and generate statistical distributions of outcomes.
"""

import os
import sys
import yaml
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

# Import models
try:
    from models.core_ode import PistonLeakODE, PistonLeakParams
    from models.abm import PistonLeakABM, ABMParams, BeliefState
except ImportError:
    # For direct execution from sims directory
    sys.path.append(os.path.join(parent_dir, 'models'))
    from core_ode import PistonLeakODE, PistonLeakParams
    from abm import PistonLeakABM, ABMParams, BeliefState

# Import visualization module
from .visualization import (
    plot_trust_trajectories,
    plot_phase_space,
    plot_rp_ratio,
    plot_collapse_heatmap,
    plot_abm_evolution,
    create_interactive_dashboard
)


class MonteCarloRunner:
    """
    Monte Carlo ensemble simulation runner for Piston Leak models.
    
    Manages configuration, parameter sampling, parallel execution,
    and result aggregation/visualization.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the Monte Carlo runner.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Set up random number generator with seed
        self.seed = self.config['ensemble'].get('random_seed', 42)
        self.rng = np.random.RandomState(self.seed)
        
        # Initialize results storage
        self.results = []
        self.summary = {}
    
    def _load_config(self) -> Dict:
        """
        Load the YAML configuration file.
        
        Returns:
            Dictionary with configuration parameters
        """
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required sections
        required_sections = ['meta', 'ensemble', 'time', 'parameters', 'output']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")
        
        return config
    
    def _create_ode_params(self, run_idx: int) -> PistonLeakParams:
        """
        Create ODE parameters for a specific run.
        
        For ensemble runs, this can sample from parameter distributions
        or apply perturbations to baseline values.
        
        Args:
            run_idx: Run index in the ensemble
            
        Returns:
            PistonLeakParams object with parameters for this run
        """
        param_config = self.config['parameters']
        
        # Start with baseline values
        params = PistonLeakParams(
            alpha1=param_config.get('alpha1', 0.6),
            beta1=param_config.get('beta1', 0.3),
            gamma1=param_config.get('gamma1', 0.2),
            alpha2=param_config.get('alpha2', 0.5),
            beta2=param_config.get('beta2', 0.25),
            k_policy=param_config.get('k_policy', 0.8),
            delta=param_config.get('delta', 0.15),
            noise_sigma=param_config.get('noise_sigma', 0.1),
            leak_frequency=param_config.get('leak_frequency', 0.05),
            epsilon_overlap=param_config.get('epsilon_overlap', 0.15),
            trust_slope_threshold=param_config.get('trust_slope_threshold', -0.05),
            entropy_critical=param_config.get('entropy_critical', 1.2)
        )
        
        # Apply run-specific perturbations if configured
        perturb = self.config['ensemble'].get('perturb_parameters', False)
        if perturb:
            perturb_scale = self.config['ensemble'].get('perturb_scale', 0.1)
            
            # Apply random perturbations to all parameters
            params.alpha1 *= (1 + self.rng.normal(0, perturb_scale))
            params.beta1 *= (1 + self.rng.normal(0, perturb_scale))
            params.gamma1 *= (1 + self.rng.normal(0, perturb_scale))
            params.alpha2 *= (1 + self.rng.normal(0, perturb_scale))
            params.beta2 *= (1 + self.rng.normal(0, perturb_scale))
            params.k_policy *= (1 + self.rng.normal(0, perturb_scale))
            params.delta *= (1 + self.rng.normal(0, perturb_scale))
            
            # Ensure parameters remain valid after perturbation
            params.alpha1 = max(0.01, params.alpha1)
            params.beta1 = max(0.01, params.beta1)
            params.gamma1 = max(0.01, params.gamma1)
            params.alpha2 = max(0.01, params.alpha2)
            params.beta2 = max(0.01, params.beta2)
            params.k_policy = max(0.01, params.k_policy)
            params.delta = max(0.01, params.delta)
        
        return params
    
    def _create_abm_params(self, run_idx: int) -> ABMParams:
        """
        Create ABM parameters for a specific run.
        
        Args:
            run_idx: Run index in the ensemble
            
        Returns:
            ABMParams object with parameters for this run
        """
        param_config = self.config['parameters']
        network_config = param_config.get('network', {})
        influence_config = param_config.get('influence_field', {})
        
        # Create ABM parameters
        abm_params = ABMParams(
            network_type=network_config.get('type', 'watts_strogatz'),
            n_agents=network_config.get('n_agents', 3000),
            k=network_config.get('k', 6),
            p_rewire=network_config.get('p_rewire', 0.2),
            
            believer_weight=influence_config.get('believer_weight', 1.0),
            skeptic_weight=influence_config.get('skeptic_weight', 1.2),
            agnostic_weight=influence_config.get('agnostic_weight', 0.8),
            
            init_believer_prob=param_config.get('init_believer_prob', 0.6),
            init_skeptic_prob=param_config.get('init_skeptic_prob', 0.2),
            init_agnostic_prob=param_config.get('init_agnostic_prob', 0.2)
        )
        
        return abm_params
    
    def _create_transparency_function(self, run_idx: int):
        """
        Create a transparency function for this run.
        
        Args:
            run_idx: Run index in the ensemble
            
        Returns:
            Function R(t, y) that returns transparency level
        """
        param_config = self.config['parameters']
        transparency_type = param_config.get('transparency_type', 'constant')
        
        if transparency_type == 'constant':
            # Constant transparency
            R_value = param_config.get('transparency_value', 0.1)
            return lambda t, y: R_value
            
        elif transparency_type == 'adaptive':
            # Transparency increases with entropy
            base_R = param_config.get('base_transparency', 0.1)
            adapt_scale = param_config.get('transparency_adaptivity', 0.2)
            return lambda t, y: base_R + adapt_scale * y[1]**2
            
        elif transparency_type == 'delayed':
            # Transparency kicks in after a threshold time
            delay_time = param_config.get('transparency_delay', 100)
            before_R = param_config.get('before_transparency', 0.05)
            after_R = param_config.get('after_transparency', 0.3)
            return lambda t, y: after_R if t > delay_time else before_R
            
        elif transparency_type == 'reactive':
            # Transparency increases when trust falls below threshold
            trust_threshold = param_config.get('trust_threshold', 0.4)
            low_trust_R = param_config.get('low_trust_transparency', 0.3)
            high_trust_R = param_config.get('high_trust_transparency', 0.1)
            return lambda t, y: low_trust_R if y[0] < trust_threshold else high_trust_R
            
        else:
            # Default to constant transparency
            return lambda t, y: 0.1
    
    def _run_single_simulation(self, run_idx: int) -> Dict:
        """
        Run a single simulation with the given parameters.
        
        Args:
            run_idx: Run index in the ensemble
            
        Returns:
            Dictionary with simulation results
        """
        # Create parameters for this run
        ode_params = self._create_ode_params(run_idx)
        abm_params = self._create_abm_params(run_idx)
        transparency_func = self._create_transparency_function(run_idx)
        
        # Set up ODE model
        run_seed = self.seed + run_idx  # Unique seed for each run
        ode_model = PistonLeakODE(
            params=ode_params,
            seed=run_seed
        )
        ode_model.set_transparency_function(transparency_func)
        
        # Set up ABM model
        abm_model = PistonLeakABM(
            params=abm_params,
            ode_model=ode_model,
            seed=run_seed
        )
        
        # Run simulation
        t_max = self.config['time']['t_max']
        dt = self.config['time']['dt']
        
        # Run ODE simulation
        ode_results = ode_model.simulate((0, t_max), dt=dt)
        
        # Only run ABM if enabled
        abm_enabled = self.config['ensemble'].get('run_abm', True)
        if abm_enabled:
            abm_results = abm_model.simulate(int(t_max / dt) + 1, ode_results)
        else:
            abm_results = None
        
        # Combine results
        results = {
            'run_idx': run_idx,
            'ode_params': {
                'alpha1': ode_params.alpha1,
                'beta1': ode_params.beta1,
                'gamma1': ode_params.gamma1,
                'alpha2': ode_params.alpha2,
                'beta2': ode_params.beta2,
                'k_policy': ode_params.k_policy,
                'delta': ode_params.delta,
                'noise_sigma': ode_params.noise_sigma,
                'leak_frequency': ode_params.leak_frequency
            },
            'ode_results': ode_results,
            'abm_results': abm_results,
            'collapse': ode_results['collapse'],
            'collapse_time': ode_results['collapse_time']
        }
        
        return results
    
    def run_ensemble(self):
        """
        Run the full ensemble of Monte Carlo simulations.
        """
        n_runs = self.config['ensemble']['runs']
        print(f"Running {n_runs} Monte Carlo simulations...")
        
        # Run simulations
        for i in range(n_runs):
            print(f"Run {i+1}/{n_runs}...")
            result = self._run_single_simulation(i)
            self.results.append(result)
        
        # Compute summary statistics
        self._compute_summary()
        
        # Save results if configured
        if self.config['output'].get('save_summary', True):
            self._save_results()
        
        # Generate plots if configured
        if self.config['output'].get('plots', True):
            self._generate_plots()
    
    def _compute_summary(self):
        """
        Compute summary statistics across all runs.
        """
        n_runs = len(self.results)
        n_collapse = sum(1 for r in self.results if r['collapse'])
        
        # Calculate collapse probability
        collapse_prob = n_collapse / n_runs if n_runs > 0 else 0
        
        # Calculate mean collapse time (for runs that collapsed)
        collapse_times = [r['collapse_time'] for r in self.results if r['collapse']]
        mean_collapse_time = np.mean(collapse_times) if collapse_times else None
        
        # Calculate terminal trust statistics
        final_trust = [r['ode_results']['trust'][-1] for r in self.results]
        mean_trust = np.mean(final_trust)
        std_trust = np.std(final_trust)
        
        # Calculate terminal entropy statistics
        final_entropy = [r['ode_results']['entropy'][-1] for r in self.results]
        mean_entropy = np.mean(final_entropy)
        std_entropy = np.std(final_entropy)
        
        # Identify recovery vs. collapse basins
        trust_threshold = 0.4  # Threshold for high-trust recovery basin
        recovery_indices = [i for i, r in enumerate(self.results) 
                           if r['ode_results']['trust'][-1] > trust_threshold]
        collapse_indices = [i for i, r in enumerate(self.results) 
                          if r['ode_results']['trust'][-1] <= trust_threshold]
        
        # Calculate R/P ratio statistics
        rp_ratios = []
        for r in self.results:
            # Use mean transparency as R
            R_values = []
            for t_idx, t in enumerate(r['ode_results']['times']):
                y = np.array([
                    r['ode_results']['trust'][t_idx],
                    r['ode_results']['entropy'][t_idx],
                    r['ode_results']['pressure'][t_idx]
                ])
                R = self._create_transparency_function(r['run_idx'])(t, y)
                R_values.append(R)
            
            # Calculate R/P ratio
            P_values = r['ode_results']['pressure']
            rp_ratio = np.mean(R_values) / np.mean(P_values) if np.mean(P_values) > 0 else float('inf')
            rp_ratios.append(rp_ratio)
        
        mean_rp_ratio = np.mean(rp_ratios)
        
        # Store summary
        self.summary = {
            'n_runs': n_runs,
            'n_collapse': n_collapse,
            'collapse_probability': collapse_prob,
            'mean_collapse_time': mean_collapse_time,
            'mean_final_trust': mean_trust,
            'std_final_trust': std_trust,
            'mean_final_entropy': mean_entropy,
            'std_final_entropy': std_entropy,
            'recovery_basin_size': len(recovery_indices) / n_runs if n_runs > 0 else 0,
            'collapse_basin_size': len(collapse_indices) / n_runs if n_runs > 0 else 0,
            'mean_rp_ratio': mean_rp_ratio,
            'critical_rp_ratio': self._estimate_critical_rp_ratio()
        }
        
        print("Summary statistics:")
        for k, v in self.summary.items():
            print(f"  {k}: {v}")
    
    def _estimate_critical_rp_ratio(self) -> float:
        """
        Estimate the critical R/P ratio separating recovery and collapse basins.
        
        Returns:
            Estimated critical R/P ratio
        """
        # Calculate R/P ratio for each run
        rp_ratios = []
        recovery = []
        
        for r in self.results:
            # Use mean transparency as R
            R_values = []
            for t_idx, t in enumerate(r['ode_results']['times']):
                y = np.array([
                    r['ode_results']['trust'][t_idx],
                    r['ode_results']['entropy'][t_idx],
                    r['ode_results']['pressure'][t_idx]
                ])
                R = self._create_transparency_function(r['run_idx'])(t, y)
                R_values.append(R)
            
            # Calculate R/P ratio
            P_values = r['ode_results']['pressure']
            rp_ratio = np.mean(R_values) / np.mean(P_values) if np.mean(P_values) > 0 else float('inf')
            rp_ratios.append(rp_ratio)
            
            # Determine if this run recovered (high final trust)
            trust_threshold = 0.4
            recovery.append(r['ode_results']['trust'][-1] > trust_threshold)
        
        # If we don't have both recovery and collapse cases, return None
        if not (any(recovery) and any(not r for r in recovery)):
            return None
        
        # Sort R/P ratios and find boundary
        sorted_pairs = sorted(zip(rp_ratios, recovery), key=lambda x: x[0])
        
        # Find the boundary between recovery and collapse
        prev_rp = None
        prev_recovery = None
        
        for rp, rec in sorted_pairs:
            if prev_recovery is not None and prev_recovery != rec:
                # We found a boundary, return the midpoint
                return (prev_rp + rp) / 2
            
            prev_rp = rp
            prev_recovery = rec
        
        # If we get here, no clear boundary was found
        return None
    
    def _save_results(self):
        """
        Save simulation results to disk.
        """
        output_path = self.config['output'].get('path', 'results/')
        os.makedirs(output_path, exist_ok=True)
        
        # Save timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save summary statistics
        summary_path = os.path.join(output_path, f"summary_{timestamp}.json")
        with open(summary_path, 'w') as f:
            json.dump(self.summary, f, indent=2, default=str)
        
        # Save detailed timeseries if requested
        save_timeseries = self.config['output'].get('save_timeseries', False)
        if save_timeseries:
            # Convert to DataFrame for easy CSV export
            for i, result in enumerate(self.results):
                # ODE timeseries
                ode_data = {
                    'time': result['ode_results']['times'],
                    'trust': result['ode_results']['trust'],
                    'entropy': result['ode_results']['entropy'],
                    'pressure': result['ode_results']['pressure'],
                    'fcc': result['ode_results']['metrics']['fcc'],
                    'rsd': result['ode_results']['metrics']['rsd'],
                    'overlap': result['ode_results']['metrics']['overlap']
                }
                ode_df = pd.DataFrame(ode_data)
                ode_path = os.path.join(output_path, f"ode_run{i}_{timestamp}.csv")
                ode_df.to_csv(ode_path, index=False)
                
                # ABM timeseries if available
                if result['abm_results'] is not None:
                    abm_data = {
                        'time': result['abm_results']['times'],
                        'believer_fraction': result['abm_results']['state_fractions'][BeliefState.BELIEVER],
                        'skeptic_fraction': result['abm_results']['state_fractions'][BeliefState.SKEPTIC],
                        'agnostic_fraction': result['abm_results']['state_fractions'][BeliefState.AGNOSTIC],
                        'network_entropy': result['abm_results']['entropy']
                    }
                    abm_df = pd.DataFrame(abm_data)
                    abm_path = os.path.join(output_path, f"abm_run{i}_{timestamp}.csv")
                    abm_df.to_csv(abm_path, index=False)
        
        print(f"Results saved to {output_path}")
    
    def _generate_plots(self):
        """
        Generate visualizations of simulation results.
        """
        output_path = self.config['output'].get('path', 'results/')
        os.makedirs(output_path, exist_ok=True)
        
        # Save timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate plots
        plot_trust_trajectories(self.results, output_path, timestamp)
        plot_phase_space(self.results, output_path, timestamp)
        plot_rp_ratio(self.results, self._create_transparency_function, output_path, timestamp)
        
        # Only create heatmap if we have enough runs
        if len(self.results) >= 100:
            plot_collapse_heatmap(self.results, output_path, timestamp)
        
        # ABM evolution plot if ABM was run
        if self.results[0]['abm_results'] is not None:
            plot_abm_evolution(self.results, output_path, timestamp)
        
        # Interactive dashboard if enabled
        if self.config['output'].get('interactive_dashboard', False):
            create_interactive_dashboard(
                self.results, 
                self.summary, 
                output_path, 
                timestamp
            )
        
        print(f"Plots saved to {output_path}")

# Main entry point
def main():
    """Main entry point for command-line execution."""
    parser = argparse.ArgumentParser(description='Run Piston Leak Monte Carlo simulations')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--n', type=int, help='Override number of runs')
    args = parser.parse_args()
    
    # Load config and potentially override number of runs
    mc_runner = MonteCarloRunner(args.config)
    if args.n is not None:
        mc_runner.config['ensemble']['runs'] = args.n
    
    # Run simulations
    mc_runner.run_ensemble()

if __name__ == "__main__":
    main()
