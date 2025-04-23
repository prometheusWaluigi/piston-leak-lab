#!/usr/bin/env python3
"""
Transparency Functions for Piston Leak Lab
==========================================

Implements various transparency functions R(t,y) for institutional
disclosure dynamics in the Trust-Narrative-Pressure system.
"""

import numpy as np
import math
from typing import Callable, Dict, Any


def create_transparency_function(config: Dict[str, Any]) -> Callable[[float, np.ndarray], float]:
    """
    Create a transparency function based on configuration parameters.
    
    Args:
        config: Dictionary with transparency function parameters
        
    Returns:
        Function R(t, y) that calculates transparency value
    """
    transparency_type = config.get('transparency_type', 'constant')
    
    if transparency_type == 'constant':
        # Constant transparency level
        R_value = config.get('transparency_value', 0.1)
        return lambda t, y: R_value
        
    elif transparency_type == 'adaptive':
        # Transparency increases with entropy
        base_R = config.get('base_transparency', 0.1)
        adapt_scale = config.get('transparency_adaptivity', 0.2)
        # Cap maximum transparency at 1.0 to prevent numerical issues
        return lambda t, y: min(1.0, base_R + adapt_scale * y[1]**2)
        
    elif transparency_type == 'delayed':
        # Transparency kicks in after a threshold time
        delay_time = config.get('transparency_delay', 100)
        before_R = config.get('before_transparency', 0.05)
        after_R = config.get('after_transparency', 0.3)
        return lambda t, y: after_R if t > delay_time else before_R
        
    elif transparency_type == 'reactive':
        # Transparency increases when trust falls below threshold
        trust_threshold = config.get('trust_threshold', 0.4)
        low_trust_R = config.get('low_trust_transparency', 0.3)
        high_trust_R = config.get('high_trust_transparency', 0.1)
        return lambda t, y: low_trust_R if y[0] < trust_threshold else high_trust_R
        
    elif transparency_type == 'cyclic':
        # Oscillating transparency to model periodic disclosure
        cycle_period = config.get('cycle_period', 30)
        min_R = config.get('min_transparency', 0.05)
        max_R = config.get('max_transparency', 0.3) 
        amplitude = (max_R - min_R) / 2
        offset = min_R + amplitude
        # Sinusoidal oscillation
        return lambda t, y: offset + amplitude * math.sin(2 * math.pi * t / cycle_period)
        
    elif transparency_type == 'threshold':
        # Threshold-based transparency that jumps at critical entropy
        entropy_threshold = config.get('entropy_threshold', 0.5)
        low_entropy_R = config.get('low_entropy_transparency', 0.05)
        high_entropy_R = config.get('high_entropy_transparency', 0.3)
        return lambda t, y: high_entropy_R if y[1] > entropy_threshold else low_entropy_R
        
    elif transparency_type == 'sigmoid':
        # Smooth sigmoid transition based on entropy
        entropy_mid = config.get('entropy_midpoint', 0.5)
        steepness = config.get('sigmoid_steepness', 10.0)
        min_R = config.get('min_transparency', 0.05)
        max_R = config.get('max_transparency', 0.3)
        
        def sigmoid_transparency(t, y):
            entropy = y[1]
            sigmoid_val = 1.0 / (1.0 + math.exp(-steepness * (entropy - entropy_mid)))
            return min_R + (max_R - min_R) * sigmoid_val
            
        return sigmoid_transparency
        
    elif transparency_type == 'trust_proportional':
        # Transparency proportional to trust (high trust enables more transparency)
        base_R = config.get('base_transparency', 0.05)
        trust_factor = config.get('trust_factor', 0.3)
        return lambda t, y: base_R + trust_factor * y[0]
        
    elif transparency_type == 'entropy_delayed':
        # Hybrid: delayed transparency that also depends on entropy level
        delay_time = config.get('transparency_delay', 100)
        base_R = config.get('base_transparency', 0.05)
        entropy_factor = config.get('entropy_factor', 0.2)
        
        def entropy_delayed_transparency(t, y):
            if t <= delay_time:
                return base_R
            else:
                # After delay, scale with entropy
                entropy = y[1]
                return min(1.0, base_R + entropy_factor * entropy)
                
        return entropy_delayed_transparency
        
    elif transparency_type == 'burst':
        # Periodic transparency bursts followed by silence
        burst_period = config.get('burst_period', 90)  # Days between bursts
        burst_duration = config.get('burst_duration', 10)  # Burst length in days
        low_R = config.get('low_transparency', 0.05)
        high_R = config.get('high_transparency', 0.3)
        
        def burst_transparency(t, y):
            # Calculate where we are in the cycle
            cycle_position = t % burst_period
            return high_R if cycle_position < burst_duration else low_R
            
        return burst_transparency
        
    else:
        # Default to constant transparency if unknown type
        print(f"Warning: Unknown transparency type '{transparency_type}', using constant 0.1")
        return lambda t, y: 0.1


# Example usage
if __name__ == "__main__":
    # Test with a few example configurations
    configs = [
        {"transparency_type": "constant", "transparency_value": 0.2},
        {"transparency_type": "adaptive", "base_transparency": 0.1, "transparency_adaptivity": 0.3},
        {"transparency_type": "cyclic", "cycle_period": 30, "min_transparency": 0.05, "max_transparency": 0.4},
        {"transparency_type": "threshold", "entropy_threshold": 0.6, "low_entropy_transparency": 0.1, "high_entropy_transparency": 0.5}
    ]
    
    # Create and test each function
    for i, config in enumerate(configs):
        print(f"\nTesting transparency function {i+1}: {config['transparency_type']}")
        
        # Create the function
        R_func = create_transparency_function(config)
        
        # Test with a few different states
        test_states = [
            (10.0, np.array([0.8, 0.1, 0.1])),  # High trust, low entropy
            (50.0, np.array([0.5, 0.5, 0.2])),  # Medium trust, medium entropy
            (120.0, np.array([0.2, 0.9, 0.4]))  # Low trust, high entropy
        ]
        
        for t, y in test_states:
            R_value = R_func(t, y)
            print(f"  t={t:.1f}, T={y[0]:.1f}, N={y[1]:.1f}, P={y[2]:.1f} → R={R_value:.3f}")
