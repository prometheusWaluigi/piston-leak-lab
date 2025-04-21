#!/usr/bin/env python3

"""
Tests for the ABM model components.
"""

import sys
import os
import pytest
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.abm import PistonLeakABM, ABMParams, BeliefState
from models.core_ode import PistonLeakODE


def test_abm_initialization():
    """Test that ABM model initializes correctly."""
    model = PistonLeakABM()
    assert model is not None
    assert model.params is not None
    assert isinstance(model.params, ABMParams)


def test_network_generation():
    """Test that social network is generated correctly."""
    # Use a small network for testing
    params = ABMParams(
        n_agents=100,
        k=4,
        p_rewire=0.1
    )
    model = PistonLeakABM(params=params)
    
    # Check network properties
    assert len(model.network.nodes()) == 100
    assert len(model.network.edges()) > 0
    
    # Check that states are initialized
    assert len(model.states) == 100
    
    # Check that all states are valid
    for state in model.states.values():
        assert state in BeliefState


def test_belief_distribution():
    """Test that belief state distribution is computed correctly."""
    params = ABMParams(
        n_agents=100,
        init_believer_prob=0.6,
        init_skeptic_prob=0.3,
        init_agnostic_prob=0.1
    )
    model = PistonLeakABM(params=params)
    
    # Get distribution
    dist = model.get_state_distribution()
    
    # Check that all states are present
    assert BeliefState.BELIEVER in dist
    assert BeliefState.SKEPTIC in dist
    assert BeliefState.AGNOSTIC in dist
    
    # Check that probabilities sum to 1
    assert abs(sum(dist.values()) - 1.0) < 1e-10
    
    # Check that distribution is approximately correct
    # (allowing for random variation)
    assert 0.5 <= dist[BeliefState.BELIEVER] <= 0.7
    assert 0.2 <= dist[BeliefState.SKEPTIC] <= 0.4
    assert 0.05 <= dist[BeliefState.AGNOSTIC] <= 0.15


def test_abm_update():
    """Test that ABM state updates work."""
    params = ABMParams(
        n_agents=100,
        k=4
    )
    model = PistonLeakABM(params=params)
    
    # Get initial distribution
    initial_dist = model.get_state_distribution()
    
    # Perform an update with a TNP state
    tnp_state = (0.7, 0.3, 0.2)  # T, N, P
    new_counts = model.update(tnp_state)
    
    # Check that counts are returned
    assert BeliefState.BELIEVER in new_counts
    assert BeliefState.SKEPTIC in new_counts
    assert BeliefState.AGNOSTIC in new_counts
    
    # Check that state counts were updated
    assert len(model.state_counts[BeliefState.BELIEVER]) == 1
    assert len(model.state_counts[BeliefState.SKEPTIC]) == 1
    assert len(model.state_counts[BeliefState.AGNOSTIC]) == 1
    
    # Check that history was updated
    assert len(model.history) == 1


def test_abm_simulation_with_ode():
    """Test that ABM simulation with ODE coupling works."""
    # Create ODE model
    ode_model = PistonLeakODE()
    
    # Create ABM with smaller network for testing
    abm_params = ABMParams(n_agents=100)
    abm_model = PistonLeakABM(params=abm_params, ode_model=ode_model)
    
    # Run ODE simulation first
    ode_results = ode_model.simulate((0, 5), dt=1.0)
    
    # Run ABM simulation with ODE results
    abm_results = abm_model.simulate(6, ode_results)
    
    # Check that results contain expected keys
    assert 'times' in abm_results
    assert 'state_counts' in abm_results
    assert 'state_fractions' in abm_results
    assert 'entropy' in abm_results
    assert 'clustering' in abm_results
    assert 'final_states' in abm_results
    
    # Check that state fractions are tracked for all belief states
    assert BeliefState.BELIEVER in abm_results['state_fractions']
    assert BeliefState.SKEPTIC in abm_results['state_fractions']
    assert BeliefState.AGNOSTIC in abm_results['state_fractions']
    
    # Check array lengths
    assert len(abm_results['times']) == 6
    assert len(abm_results['state_fractions'][BeliefState.BELIEVER]) == 6
    assert len(abm_results['entropy']) == 6
    
    # Check that final states are recorded
    assert len(abm_results['final_states']) == 100
