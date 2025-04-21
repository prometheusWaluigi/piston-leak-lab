#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive Dashboard for Piston Leak Lab
=======================================

Generates interactive Plotly-based dashboards for exploring
Monte Carlo simulation results and parameter spaces.
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Union, Any


def create_interactive_dashboard(results: List[Dict], summary: Dict, 
                                output_path: str, timestamp: str):
    """
    Create an interactive HTML dashboard for exploring simulation results.
    
    Args:
        results: List of simulation results
        summary: Summary statistics
        output_path: Directory to save the dashboard
        timestamp: Timestamp for unique filename
    """
    # Convert results to DataFrames for easier manipulation
    ode_data = []
    for i, result in enumerate(results):
        # Basic run metadata
        run_data = {
            'run_idx': i,
            'collapse': result['collapse'],
            'collapse_time': result['collapse_time']
        }
        
        # Add ODE parameters
        for param, value in result['ode_params'].items():
            run_data[f'param_{param}'] = value
        
        # Add timeseries data
        n_steps = len(result['ode_results']['times'])
        for t_idx in range(n_steps):
            step_data = run_data.copy()
            step_data.update({
                'time': result['ode_results']['times'][t_idx],
                'trust': result['ode_results']['trust'][t_idx],
                'entropy': result['ode_results']['entropy'][t_idx],
                'pressure': result['ode_results']['pressure'][t_idx],
                'fcc': result['ode_results']['metrics']['fcc'][t_idx],
                'rsd': result['ode_results']['metrics']['rsd'][t_idx],
                'overlap': result['ode_results']['metrics']['overlap'][t_idx]
            })
            ode_data.append(step_data)
    
    # Convert to DataFrame
    ode_df = pd.DataFrame(ode_data)
    
    # Create dashboard
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Trust Trajectories', 
            'Phase Space (T vs N)',
            'Parameter Sensitivity', 
            'Attractor Metrics',
            'Final State Distribution',
            'Summary Statistics'
        ),
        specs=[
            [{'type': 'scatter'}, {'type': 'scatter'}],
            [{'type': 'scatter'}, {'type': 'scatter'}],
            [{'type': 'scatter'}, {'type': 'table'}]
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.05
    )
    
    # Plot 1: Trust Trajectories
    for i, result in enumerate(results):
        color = 'rgba(255, 0, 0, 0.3)' if result['collapse'] else 'rgba(0, 0, 255, 0.3)'
        name = f"Run {i} ({'Collapse' if result['collapse'] else 'Recovery'})"
        
        fig.add_trace(
            go.Scatter(
                x=result['ode_results']['times'],
                y=result['ode_results']['trust'],
                mode='lines',
                line=dict(color=color),
                name=name,
                hovertemplate='Time: %{x:.1f}<br>Trust: %{y:.3f}'
            ),
            row=1, col=1
        )
    
    # Add critical threshold line
    fig.add_trace(
        go.Scatter(
            x=[0, results[0]['ode_results']['times'][-1]],
            y=[0.4, 0.4],
            mode='lines',
            line=dict(color='black', dash='dash'),
            name='Critical Trust'
        ),
        row=1, col=1
    )
    
    # Plot 2: Phase Space
    for i, result in enumerate(results):
        color = 'rgba(255, 0, 0, 0.3)' if result['collapse'] else 'rgba(0, 0, 255, 0.3)'
        name = f"Run {i} ({'Collapse' if result['collapse'] else 'Recovery'})"
        
        fig.add_trace(
            go.Scatter(
                x=result['ode_results']['entropy'],
                y=result['ode_results']['trust'],
                mode='lines',
                line=dict(color=color),
                name=name,
                hovertemplate='Entropy: %{x:.3f}<br>Trust: %{y:.3f}'
            ),
            row=1, col=2
        )
    
    # Add critical thresholds
    fig.add_trace(
        go.Scatter(
            x=[0, 3],
            y=[0.4, 0.4],
            mode='lines',
            line=dict(color='black', dash='dash'),
            name='Critical Trust'
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=[1.2, 1.2],
            y=[0, 1],
            mode='lines',
            line=dict(color='black', dash='dot'),
            name='Critical Entropy'
        ),
        row=1, col=2
    )
    
    # Plot 3: Parameter Sensitivity (α₁ vs final trust)
    fig.add_trace(
        go.Scatter(
            x=[r['ode_params']['alpha1'] for r in results],
            y=[r['ode_results']['trust'][-1] for r in results],
            mode='markers',
            marker=dict(
                color=['red' if r['collapse'] else 'blue' for r in results],
                size=10,
                opacity=0.7
            ),
            name='α₁ vs Final Trust',
            hovertemplate='α₁: %{x:.3f}<br>Final Trust: %{y:.3f}'
        ),
        row=2, col=1
    )
    
    # Plot 4: Attractor Metrics for representative run
    # Choose median trust run
    final_trust = [r['ode_results']['trust'][-1] for r in results]
    median_idx = np.argsort(final_trust)[len(final_trust) // 2]
    rep_result = results[median_idx]
    
    fig.add_trace(
        go.Scatter(
            x=rep_result['ode_results']['times'],
            y=rep_result['ode_results']['metrics']['fcc'],
            mode='lines',
            line=dict(color='blue'),
            name='FCC'
        ),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=rep_result['ode_results']['times'],
            y=rep_result['ode_results']['metrics']['rsd'],
            mode='lines',
            line=dict(color='red'),
            name='RSD'
        ),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=rep_result['ode_results']['times'],
            y=rep_result['ode_results']['metrics']['overlap'],
            mode='lines',
            line=dict(color='green'),
            name='FCC×RSD Overlap'
        ),
        row=2, col=2
    )
    
    # Add critical threshold
    fig.add_trace(
        go.Scatter(
            x=[0, rep_result['ode_results']['times'][-1]],
            y=[0.15, 0.15],
            mode='lines',
            line=dict(color='black', dash='dot'),
            name='ε-threshold'
        ),
        row=2, col=2
    )
    
    # Plot 5: Final state distribution
    final_trust_bins = np.linspace(0, 1, 11)
    hist_data = np.histogram([r['ode_results']['trust'][-1] for r in results], bins=final_trust_bins)
    
    fig.add_trace(
        go.Bar(
            x=[(final_trust_bins[i] + final_trust_bins[i+1])/2 for i in range(len(final_trust_bins)-1)],
            y=hist_data[0],
            marker=dict(color='purple'),
            name='Final Trust Distribution'
        ),
        row=3, col=1
    )
    
    # Plot 6: Summary statistics table
    summary_table = {
        'Metric': [
            'Total Runs',
            'Collapse Events',
            'Collapse Probability',
            'Mean Collapse Time',
            'Mean Final Trust',
            'Mean Final Entropy',
            'Recovery Basin Size',
            'Critical R/P Ratio'
        ],
        'Value': [
            f"{summary['n_runs']}",
            f"{summary['n_collapse']}",
            f"{summary['collapse_probability']:.3f}",
            f"{summary['mean_collapse_time']:.1f}" if summary['mean_collapse_time'] else "N/A",
            f"{summary['mean_final_trust']:.3f}",
            f"{summary['mean_final_entropy']:.3f}",
            f"{summary['recovery_basin_size']:.3f}",
            f"{summary['critical_rp_ratio']:.3f}" if summary['critical_rp_ratio'] else "N/A"
        ]
    }
    
    fig.add_trace(
        go.Table(
            header=dict(
                values=list(summary_table.keys()),
                fill_color='paleturquoise',
                align='left'
            ),
            cells=dict(
                values=[summary_table[k] for k in summary_table.keys()],
                fill_color='lavender',
                align='left'
            )
        ),
        row=3, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text="Piston Leak Lab — Monte Carlo Simulation Results",
        height=1200,
        width=1600,
        showlegend=False,
        template='plotly_white'
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Time (media cycles)", row=1, col=1)
    fig.update_yaxes(title_text="Trust (T)", row=1, col=1)
    
    fig.update_xaxes(title_text="Narrative Entropy (N)", row=1, col=2)
    fig.update_yaxes(title_text="Trust (T)", row=1, col=2)
    
    fig.update_xaxes(title_text="α₁ (Trust Decay Rate)", row=2, col=1)
    fig.update_yaxes(title_text="Final Trust", row=2, col=1)
    
    fig.update_xaxes(title_text="Time (media cycles)", row=2, col=2)
    fig.update_yaxes(title_text="Metric Value", row=2, col=2)
    
    fig.update_xaxes(title_text="Final Trust", row=3, col=1)
    fig.update_yaxes(title_text="Count", row=3, col=1)
    
    # Add annotations
    fig.add_annotation(
        x=0.5, y=1.1,
        xref="paper", yref="paper",
        text="Piston Leak Symbolic Dynamical-Systems Model",
        showarrow=False,
        font=dict(size=20)
    )
    
    fig.add_annotation(
        x=0.5, y=-0.15,
        xref="paper", yref="paper",
        text="May your entropy gradients be ever in your favor.",
        showarrow=False,
        font=dict(
            size=14,
            italic=True
        )
    )
    
    # Save dashboard
    dashboard_path = os.path.join(output_path, f"dashboard_{timestamp}.html")
    fig.write_html(dashboard_path)
    
    print(f"Interactive dashboard saved to {dashboard_path}")


def create_parameter_explorer(results: List[Dict], output_path: str, timestamp: str):
    """
    Create a specialized interactive parameter explorer dashboard.
    
    Args:
        results: List of simulation results
        output_path: Directory to save the dashboard
        timestamp: Timestamp for unique filename
    """
    # Extract parameters and results for all runs
    param_data = []
    
    for i, result in enumerate(results):
        run_data = {
            'run_idx': i,
            'collapse': result['collapse'],
            'collapse_time': result['collapse_time'],
            'final_trust': result['ode_results']['trust'][-1],
            'final_entropy': result['ode_results']['entropy'][-1],
            'final_pressure': result['ode_results']['pressure'][-1],
        }
        
        # Add parameters
        for param, value in result['ode_params'].items():
            run_data[param] = value
        
        param_data.append(run_data)
    
    # Convert to DataFrame
    param_df = pd.DataFrame(param_data)
    
    # Define key parameters to explore
    key_params = ['alpha1', 'beta1', 'gamma1', 'alpha2', 'beta2', 'k_policy', 'delta']
    outcome_vars = ['final_trust', 'final_entropy', 'final_pressure', 'collapse']
    
    # Create scatterplot matrix
    dimensions = [
        dict(
            label=param,
            values=param_df[param]
        )
        for param in key_params
    ] + [
        dict(
            label="Final Trust",
            values=param_df['final_trust']
        ),
        dict(
            label="Final Entropy",
            values=param_df['final_entropy']
        )
    ]
    
    fig = go.Figure(data=go.Splom(
        dimensions=dimensions,
        marker=dict(
            color=param_df['collapse'].map({True: 'red', False: 'blue'}),
            size=7,
            colorscale=[[0, 'blue'], [1, 'red']],
            line_color='white',
            line_width=0.5,
            opacity=0.7
        ),
        diagonal_visible=False,
        showupperhalf=False,
        name="Parameter Explorer",
        text=[f"Run {i}<br>{'Collapse' if c else 'Recovery'}" 
              for i, c in enumerate(param_df['collapse'])]
    ))
    
    # Update layout
    fig.update_layout(
        title="Parameter Space Explorer",
        width=1200,
        height=1000,
        dragmode='select',
        hovermode='closest'
    )
    
    # Add annotation explaining color scheme
    fig.add_annotation(
        x=0.5, y=1.05,
        xref="paper", yref="paper",
        text="Blue = Recovery Trajectory, Red = Collapse Trajectory",
        showarrow=False,
        font=dict(size=12)
    )
    
    # Save dashboard
    explorer_path = os.path.join(output_path, f"param_explorer_{timestamp}.html")
    fig.write_html(explorer_path)
    
    print(f"Parameter explorer saved to {explorer_path}")


def create_temporal_dashboard(results: List[Dict], output_path: str, timestamp: str):
    """
    Create a specialized dashboard for exploring temporal evolution.
    
    Args:
        results: List of simulation results
        output_path: Directory to save the dashboard
        timestamp: Timestamp for unique filename
    """
    # Choose representative runs - one collapse, one recovery
    collapse_runs = [r for r in results if r['collapse']]
    recovery_runs = [r for r in results if not r['collapse']]
    
    if not collapse_runs or not recovery_runs:
        print("Warning: Both collapse and recovery examples needed for temporal dashboard")
        return
    
    # Choose middle examples from each category
    collapse_run = collapse_runs[len(collapse_runs) // 2]
    recovery_run = recovery_runs[len(recovery_runs) // 2]
    
    # Create figure with multiple plots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Collapse Trajectory', 
            'Recovery Trajectory',
            'Attractor Metrics (Collapse)', 
            'Attractor Metrics (Recovery)'
        ),
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # Plot TNP variables for collapse case
    fig.add_trace(
        go.Scatter(
            x=collapse_run['ode_results']['times'],
            y=collapse_run['ode_results']['trust'],
            mode='lines',
            line=dict(color='blue', width=2),
            name='Trust'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=collapse_run['ode_results']['times'],
            y=collapse_run['ode_results']['entropy'],
            mode='lines',
            line=dict(color='red', width=2),
            name='Entropy'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=collapse_run['ode_results']['times'],
            y=collapse_run['ode_results']['pressure'],
            mode='lines',
            line=dict(color='green', width=2),
            name='Pressure'
        ),
        row=1, col=1
    )
    
    # Mark collapse point if applicable
    if collapse_run['collapse_time'] is not None:
        collapse_idx = np.abs(collapse_run['ode_results']['times'] - collapse_run['collapse_time']).argmin()
        
        fig.add_vline(
            x=collapse_run['ode_results']['times'][collapse_idx],
            line=dict(color='red', width=2, dash='dash'),
            row=1, col=1
        )
        
        fig.add_annotation(
            x=collapse_run['ode_results']['times'][collapse_idx],
            y=1.1,
            text=f"Collapse detected (t={collapse_run['collapse_time']:.1f})",
            showarrow=True,
            arrowhead=1,
            row=1, col=1
        )
    
    # Plot TNP variables for recovery case
    fig.add_trace(
        go.Scatter(
            x=recovery_run['ode_results']['times'],
            y=recovery_run['ode_results']['trust'],
            mode='lines',
            line=dict(color='blue', width=2),
            name='Trust'
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=recovery_run['ode_results']['times'],
            y=recovery_run['ode_results']['entropy'],
            mode='lines',
            line=dict(color='red', width=2),
            name='Entropy'
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=recovery_run['ode_results']['times'],
            y=recovery_run['ode_results']['pressure'],
            mode='lines',
            line=dict(color='green', width=2),
            name='Pressure'
        ),
        row=1, col=2
    )
    
    # Plot attractor metrics for collapse case
    fig.add_trace(
        go.Scatter(
            x=collapse_run['ode_results']['times'],
            y=collapse_run['ode_results']['metrics']['fcc'],
            mode='lines',
            line=dict(color='purple', width=2),
            name='FCC'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=collapse_run['ode_results']['times'],
            y=collapse_run['ode_results']['metrics']['rsd'],
            mode='lines',
            line=dict(color='orange', width=2),
            name='RSD'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=collapse_run['ode_results']['times'],
            y=collapse_run['ode_results']['metrics']['overlap'],
            mode='lines',
            line=dict(color='cyan', width=2),
            name='FCC×RSD Overlap'
        ),
        row=2, col=1
    )
    
    # Add critical threshold
    fig.add_trace(
        go.Scatter(
            x=[0, collapse_run['ode_results']['times'][-1]],
            y=[0.15, 0.15],
            mode='lines',
            line=dict(color='black', dash='dot'),
            name='ε-threshold'
        ),
        row=2, col=1
    )
    
    # Plot attractor metrics for recovery case
    fig.add_trace(
        go.Scatter(
            x=recovery_run['ode_results']['times'],
            y=recovery_run['ode_results']['metrics']['fcc'],
            mode='lines',
            line=dict(color='purple', width=2),
            name='FCC'
        ),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=recovery_run['ode_results']['times'],
            y=recovery_run['ode_results']['metrics']['rsd'],
            mode='lines',
            line=dict(color='orange', width=2),
            name='RSD'
        ),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=recovery_run['ode_results']['times'],
            y=recovery_run['ode_results']['metrics']['overlap'],
            mode='lines',
            line=dict(color='cyan', width=2),
            name='FCC×RSD Overlap'
        ),
        row=2, col=2
    )
    
    # Add critical threshold
    fig.add_trace(
        go.Scatter(
            x=[0, recovery_run['ode_results']['times'][-1]],
            y=[0.15, 0.15],
            mode='lines',
            line=dict(color='black', dash='dot'),
            name='ε-threshold'
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title="Temporal Evolution Comparison: Collapse vs Recovery",
        height=1000,
        width=1400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.5
        )
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Time (media cycles)", row=1, col=1)
    fig.update_yaxes(title_text="Variable Value", row=1, col=1)
    
    fig.update_xaxes(title_text="Time (media cycles)", row=1, col=2)
    fig.update_yaxes(title_text="Variable Value", row=1, col=2)
    
    fig.update_xaxes(title_text="Time (media cycles)", row=2, col=1)
    fig.update_yaxes(title_text="Metric Value", row=2, col=1)
    
    fig.update_xaxes(title_text="Time (media cycles)", row=2, col=2)
    fig.update_yaxes(title_text="Metric Value", row=2, col=2)
    
    # Save dashboard
    temporal_path = os.path.join(output_path, f"temporal_evolution_{timestamp}.html")
    fig.write_html(temporal_path)
    
    print(f"Temporal evolution dashboard saved to {temporal_path}")
