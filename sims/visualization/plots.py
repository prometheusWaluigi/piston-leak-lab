#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Static plotting functions for Piston Leak Lab
============================================

Generates static plots for Monte Carlo simulation results,
including trajectory plots, phase space diagrams, and parameter
sensitivity analyses.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Callable, Optional, Union, Any
from enum import Enum

# Make enum class available for BeliefState
class BeliefState(Enum):
    """Possible belief states of agents in the network."""
    BELIEVER = 1   # Believes institutional narrative
    SKEPTIC = 2    # Rejects institutional narrative
    AGNOSTIC = 3   # Undecided/uncertain


def plot_trust_trajectories(results: List[Dict], output_path: str, timestamp: str):
    """
    Plot trust trajectories for all runs, colored by collapse.
    
    Args:
        results: List of simulation results
        output_path: Directory to save the plot
        timestamp: Timestamp for unique filename
    """
    plt.figure(figsize=(10, 6))
    
    for result in results:
        times = result['ode_results']['times']
        trust = result['ode_results']['trust']
        
        if result['collapse']:
            # Collapsed trajectories in red
            plt.plot(times, trust, 'r-', alpha=0.3)
            if result['collapse_time'] is not None:
                # Mark collapse point
                collapse_idx = np.abs(times - result['collapse_time']).argmin()
                plt.plot(times[collapse_idx], trust[collapse_idx], 'ro', alpha=0.3)
        else:
            # Non-collapsed trajectories in blue
            plt.plot(times, trust, 'b-', alpha=0.3)
    
    # Add critical trust threshold
    plt.axhline(y=0.4, color='k', linestyle='--', alpha=0.7)
    
    plt.title('Trust Trajectories')
    plt.xlabel('Time (media cycles)')
    plt.ylabel('Trust (T)')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(['Collapse Trajectories', 'Recovery Trajectories', 'Critical Trust Threshold'])
    
    # Save plot
    plot_path = os.path.join(output_path, f"trust_trajectories_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_phase_space(results: List[Dict], output_path: str, timestamp: str):
    """
    Plot phase space (Trust vs Entropy) with coloring by collapse.
    
    Args:
        results: List of simulation results
        output_path: Directory to save the plot
        timestamp: Timestamp for unique filename
    """
    plt.figure(figsize=(10, 8))
    
    for result in results:
        trust = result['ode_results']['trust']
        entropy = result['ode_results']['entropy']
        
        if result['collapse']:
            # Collapsed trajectories in red
            plt.plot(entropy, trust, 'r-', alpha=0.3)
            # Mark final state
            plt.plot(entropy[-1], trust[-1], 'ro', alpha=0.5)
        else:
            # Non-collapsed trajectories in blue
            plt.plot(entropy, trust, 'b-', alpha=0.3)
            # Mark final state
            plt.plot(entropy[-1], trust[-1], 'bo', alpha=0.5)
    
    # Add critical thresholds
    plt.axhline(y=0.4, color='k', linestyle='--', alpha=0.7)
    plt.axvline(x=1.2, color='k', linestyle=':', alpha=0.7)
    
    plt.title('Phase Space: Trust vs Narrative Entropy')
    plt.xlabel('Narrative Entropy (N)')
    plt.ylabel('Trust (T)')
    plt.grid(True, alpha=0.3)
    plt.legend(['Collapse Trajectories', 'Recovery Trajectories', 
                'Critical Trust Threshold', 'Critical Entropy Threshold'])
    
    # Save plot
    plot_path = os.path.join(output_path, f"phase_space_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_rp_ratio(results: List[Dict], transparency_func: Callable, 
                 output_path: str, timestamp: str):
    """
    Plot R/P ratio vs final trust to visualize basin boundary.
    
    Args:
        results: List of simulation results
        transparency_func: Function to create transparency function for a run
        output_path: Directory to save the plot
        timestamp: Timestamp for unique filename
    """
    plt.figure(figsize=(10, 6))
    
    # Calculate R/P ratios
    rp_ratios = []
    final_trust = []
    collapse_status = []
    
    for result in results:
        # Calculate mean transparency (R)
        R_values = []
        for t_idx, t in enumerate(result['ode_results']['times']):
            y = np.array([
                result['ode_results']['trust'][t_idx],
                result['ode_results']['entropy'][t_idx],
                result['ode_results']['pressure'][t_idx]
            ])
            R = transparency_func(result['run_idx'])(t, y)
            R_values.append(R)
        
        # Calculate mean pressure (P)
        P_values = result['ode_results']['pressure']
        mean_P = np.mean(P_values)
        
        # Calculate R/P ratio
        rp_ratio = np.mean(R_values) / mean_P if mean_P > 0 else 10.0  # Cap at 10 for visualization
        rp_ratio = min(rp_ratio, 10.0)  # Cap at 10 for visualization
        
        rp_ratios.append(rp_ratio)
        final_trust.append(result['ode_results']['trust'][-1])
        collapse_status.append(result['collapse'])
    
    # Plot points
    collapse_points = [i for i, c in enumerate(collapse_status) if c]
    recovery_points = [i for i, c in enumerate(collapse_status) if not c]
    
    if collapse_points:
        plt.scatter(
            [rp_ratios[i] for i in collapse_points],
            [final_trust[i] for i in collapse_points],
            c='r', alpha=0.7, label='Collapse'
        )
    
    if recovery_points:
        plt.scatter(
            [rp_ratios[i] for i in recovery_points],
            [final_trust[i] for i in recovery_points],
            c='b', alpha=0.7, label='Recovery'
        )
    
    # Fit logistic regression to find boundary
    if collapse_points and recovery_points:
        try:
            from sklearn.linear_model import LogisticRegression
            
            X = np.array(rp_ratios).reshape(-1, 1)
            y = np.array([not c for c in collapse_status])  # 1 = recovery, 0 = collapse
            
            model = LogisticRegression()
            model.fit(X, y)
            
            # Plot decision boundary
            x_boundary = np.linspace(min(rp_ratios), max(rp_ratios), 100)
            y_boundary = model.predict_proba(x_boundary.reshape(-1, 1))[:, 1]
            
            plt.plot(x_boundary, y_boundary, 'k--', label='Decision Boundary')
            
            # Mark critical R/P value (at p=0.5)
            critical_rp = -model.intercept_[0] / model.coef_[0][0]
            if 0 < critical_rp < 10:
                plt.axvline(x=critical_rp, color='k', linestyle=':', alpha=0.7,
                           label=f'Critical R/P ≈ {critical_rp:.2f}')
        except (ImportError, ValueError) as e:
            # If sklearn not available or fitting fails, skip this part
            pass
    
    plt.title('Transparency/Pressure Ratio vs Final Trust')
    plt.xlabel('Mean R/P Ratio')
    plt.ylabel('Final Trust (T)')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save plot
    plot_path = os.path.join(output_path, f"rp_ratio_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_collapse_heatmap(results: List[Dict], output_path: str, timestamp: str):
    """
    Plot collapse probability heatmap as function of parameters.
    
    Args:
        results: List of simulation results
        output_path: Directory to save the plot
        timestamp: Timestamp for unique filename
    """
    # Extract parameters of interest
    alpha1_values = [r['ode_params']['alpha1'] for r in results]
    k_policy_values = [r['ode_params']['k_policy'] for r in results]
    collapse_status = [int(r['collapse']) for r in results]
    
    # Create 2D histogram
    hist, xedges, yedges = np.histogram2d(
        alpha1_values,
        k_policy_values,
        bins=10,
        weights=collapse_status
    )
    
    # Normalize by count
    count, _, _ = np.histogram2d(
        alpha1_values,
        k_policy_values,
        bins=[xedges, yedges]
    )
    
    # Avoid division by zero
    count = np.maximum(count, 1)
    hist = hist / count
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(xedges, yedges, hist.T, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(label='Collapse Probability')
    
    plt.title('Collapse Probability by Parameter Values')
    plt.xlabel('Trust Decay (α₁)')
    plt.ylabel('Policy Aggression (k_policy)')
    
    # Save plot
    plot_path = os.path.join(output_path, f"collapse_heatmap_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_abm_evolution(results: List[Dict], output_path: str, timestamp: str):
    """
    Plot ABM belief state evolution over time.
    
    Args:
        results: List of simulation results
        output_path: Directory to save the plot
        timestamp: Timestamp for unique filename
    """
    # Choose a representative run (median final trust)
    final_trust = [r['ode_results']['trust'][-1] for r in results]
    median_idx = np.argsort(final_trust)[len(final_trust) // 2]
    result = results[median_idx]
    
    if result['abm_results'] is None:
        return
    
    # Extract state fractions
    times = result['abm_results']['times']
    believer_frac = result['abm_results']['state_fractions'][BeliefState.BELIEVER]
    skeptic_frac = result['abm_results']['state_fractions'][BeliefState.SKEPTIC]
    agnostic_frac = result['abm_results']['state_fractions'][BeliefState.AGNOSTIC]
    
    # Plot evolution
    plt.figure(figsize=(10, 6))
    plt.stackplot(
        times,
        believer_frac,
        skeptic_frac,
        agnostic_frac,
        labels=['Believer', 'Skeptic', 'Agnostic'],
        colors=['blue', 'red', 'gray'],
        alpha=0.7
    )
    
    # Add overlay of Trust, Narrative Entropy
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    ax2.plot(result['ode_results']['times'], result['ode_results']['trust'], 'k-', label='Trust (T)')
    ax2.plot(result['ode_results']['times'], result['ode_results']['entropy'], 'k--', label='Entropy (N)')
    
    # Mark collapse point if applicable
    if result['collapse'] and result['collapse_time'] is not None:
        collapse_idx = np.abs(times - result['collapse_time']).argmin()
        plt.axvline(x=times[collapse_idx], color='red', linestyle='--', alpha=0.7, 
                   label=f'Collapse at t={result["collapse_time"]:.1f}')
    
    ax1.set_xlabel('Time (media cycles)')
    ax1.set_ylabel('Population Fraction')
    ax2.set_ylabel('T, N Values')
    
    ax1.set_ylim(0, 1)
    ax2.set_ylim(0, 2)
    
    plt.title('Agent Belief State Evolution')
    
    # Create combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Save plot
    plot_path = os.path.join(output_path, f"abm_evolution_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_attractor_metrics(results: List[Dict], output_path: str, timestamp: str):
    """
    Plot attractor metrics (FCC, RSD, overlap) over time.
    
    Args:
        results: List of simulation results
        output_path: Directory to save the plot
        timestamp: Timestamp for unique filename
    """
    # Choose representative runs - one collapse, one recovery
    collapse_indices = [i for i, r in enumerate(results) if r['collapse']]
    recovery_indices = [i for i, r in enumerate(results) if not r['collapse']]
    
    if not collapse_indices or not recovery_indices:
        return
    
    collapse_run = results[collapse_indices[0]]
    recovery_run = results[recovery_indices[0]]
    
    # Create two subplots - one for each run
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    
    # Plot collapse run
    times = collapse_run['ode_results']['times']
    fcc = collapse_run['ode_results']['metrics']['fcc']
    rsd = collapse_run['ode_results']['metrics']['rsd']
    overlap = collapse_run['ode_results']['metrics']['overlap']
    trust = collapse_run['ode_results']['trust']
    
    ax1.plot(times, fcc, 'b-', label='FCC', alpha=0.7)
    ax1.plot(times, rsd, 'r-', label='RSD', alpha=0.7)
    ax1.plot(times, overlap, 'g-', label='FCC×RSD ε-band', alpha=0.7)
    
    # Mark collapse point if applicable
    if collapse_run['collapse_time'] is not None:
        collapse_idx = np.abs(times - collapse_run['collapse_time']).argmin()
        ax1.axvline(x=times[collapse_idx], color='red', linestyle='--', alpha=0.7, 
                  label=f'Collapse at t={collapse_run["collapse_time"]:.1f}')
    
    # Add critical overlap threshold
    ax1.axhline(y=0.15, color='k', linestyle=':', alpha=0.7, label='ε-threshold')
    
    ax1.set_title('Attractor Metrics (Collapse Trajectory)')
    ax1.set_ylabel('Metric Value')
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot trust on twin axis
    ax1_twin = ax1.twinx()
    ax1_twin.plot(times, trust, 'k-', label='Trust', alpha=0.5)
    ax1_twin.set_ylabel('Trust (T)')
    ax1_twin.set_ylim(0, 1)
    
    # Plot recovery run
    times = recovery_run['ode_results']['times']
    fcc = recovery_run['ode_results']['metrics']['fcc']
    rsd = recovery_run['ode_results']['metrics']['rsd']
    overlap = recovery_run['ode_results']['metrics']['overlap']
    trust = recovery_run['ode_results']['trust']
    
    ax2.plot(times, fcc, 'b-', label='FCC', alpha=0.7)
    ax2.plot(times, rsd, 'r-', label='RSD', alpha=0.7)
    ax2.plot(times, overlap, 'g-', label='FCC×RSD ε-band', alpha=0.7)
    
    # Add critical overlap threshold
    ax2.axhline(y=0.15, color='k', linestyle=':', alpha=0.7, label='ε-threshold')
    
    ax2.set_title('Attractor Metrics (Recovery Trajectory)')
    ax2.set_xlabel('Time (media cycles)')
    ax2.set_ylabel('Metric Value')
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot trust on twin axis
    ax2_twin = ax2.twinx()
    ax2_twin.plot(times, trust, 'k-', label='Trust', alpha=0.5)
    ax2_twin.set_ylabel('Trust (T)')
    ax2_twin.set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_path, f"attractor_metrics_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_parameter_sensitivity(results: List[Dict], output_path: str, timestamp: str):
    """
    Plot parameter sensitivity analysis for key parameters.
    
    Args:
        results: List of simulation results
        output_path: Directory to save the plot
        timestamp: Timestamp for unique filename
    """
    # Extract parameters and outcome measures
    params = {
        'alpha1': [],
        'beta1': [],
        'k_policy': [],
        'delta': [],
        'noise_sigma': []
    }
    
    final_trust = []
    final_entropy = []
    collapse_status = []
    
    for result in results:
        for param in params:
            params[param].append(result['ode_params'][param])
        
        final_trust.append(result['ode_results']['trust'][-1])
        final_entropy.append(result['ode_results']['entropy'][-1])
        collapse_status.append(result['collapse'])
    
    # Create scatterplot matrix
    # For each parameter, create scatter plot vs final trust
    fig, axes = plt.subplots(len(params), 1, figsize=(10, 3*len(params)))
    
    for i, (param_name, param_values) in enumerate(params.items()):
        ax = axes[i]
        
        for status, label, color in [(True, 'Collapse', 'r'), (False, 'Recovery', 'b')]:
            indices = [j for j, s in enumerate(collapse_status) if s == status]
            if indices:
                ax.scatter(
                    [param_values[j] for j in indices],
                    [final_trust[j] for j in indices],
                    c=color, alpha=0.7, label=label
                )
        
        ax.set_xlabel(param_name)
        ax.set_ylabel('Final Trust')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_path, f"parameter_sensitivity_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
