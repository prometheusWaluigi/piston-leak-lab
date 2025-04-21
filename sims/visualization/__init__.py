#!/usr/bin/env python3

"""
Visualization module for Piston Leak Lab
=======================================

Provides plotting functions for simulation results and interactive
dashboards for exploring parameter spaces.
"""

from .plots import (
    plot_trust_trajectories,
    plot_phase_space,
    plot_rp_ratio,
    plot_collapse_heatmap,
    plot_abm_evolution,
    plot_attractor_metrics,
    plot_parameter_sensitivity
)

from .dashboard import (
    create_interactive_dashboard,
    create_parameter_explorer,
    create_temporal_dashboard
)

__all__ = [
    'plot_trust_trajectories',
    'plot_phase_space',
    'plot_rp_ratio',
    'plot_collapse_heatmap',
    'plot_abm_evolution',
    'plot_attractor_metrics',
    'plot_parameter_sensitivity',
    'create_interactive_dashboard',
    'create_parameter_explorer',
    'create_temporal_dashboard'
]
