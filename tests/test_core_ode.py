#!/usr/bin/env python3

"""
Tests for the ODE model components.
"""

import sys
import os
import pytest
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.core_ode import PistonLeakODE, PistonLeakParams


def test_ode_initialization():
    """Test that ODE model initializes correctly."""
    model = PistonLeakODE()
    assert model is not None
    assert model.params is not None
    assert isinstance(model.params, PistonLeakParams)


def test_ode_simulation():
    """Test that ODE simulation runs without errors."""
    model = PistonLeakODE()
    results = model.simulate((0, 10), dt=1.0)
    
    # Check that results contain expected keys
    assert 'times' in results
    assert 'trust' in results
    assert 'entropy' in results
    assert 'pressure' in results
    assert 'metrics' in results
    assert 'collapse' in results
    
    # Check array lengths
    assert len(results['times']) == 11  # 0 to 10 inclusive
    assert len(results['trust']) == 11
    assert len(results['entropy']) == 11
    assert len(results['pressure']) == 11
    
    # Check that metrics are computed
    assert 'fcc' in results['metrics']
    assert 'rsd' in results['metrics']
    assert 'overlap' in results['metrics']
    
    # Check variable ranges
    assert np.all(results['trust'] >= 0) and np.all(results['trust'] <= 1)
    assert np.all(results['entropy'] >= 0)
    assert np.all(results['pressure'] >= 0)


def test_custom_params():
    """Test that custom parameters are properly applied."""
    params = PistonLeakParams(
        alpha1=0.3,
        beta1=0.2,
        gamma1=0.1,
    )
    model = PistonLeakODE(params=params)
    
    assert model.params.alpha1 == 0.3
    assert model.params.beta1 == 0.2
    assert model.params.gamma1 == 0.1


def test_transparency_function():
    """Test that custom transparency functions work."""
    model = PistonLeakODE()
    
    # Simple function that returns constant value
    def constant_transparency(t, y):
        return 0.5
    
    model.set_transparency_function(constant_transparency)
    
    # Run simulation with custom transparency
    results = model.simulate((0, 5), dt=1.0)
    
    # Should run without errors
    assert results is not None
