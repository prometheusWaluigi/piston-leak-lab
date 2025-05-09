#!/usr/bin/env python3
"""
Coupled ODE Core for Piston Leak Lab
====================================

Implements the Trust-Narrative-Pressure dynamical system
modeling institutional narrative collapse.

See paper §4.1 for mathematical formulation.
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Callable


@dataclass
class PistonLeakParams:
    """Parameters for the Piston Leak ODE system."""
    # Trust decay/growth parameters
    alpha1: float = 0.6  # Trust decay due to entropy
    beta1: float = 0.3   # Trust boost from transparency
    gamma1: float = 0.2  # Trust penalty from suppression
    
    # Narrative entropy parameters
    alpha2: float = 0.5  # Entropy growth from suppression
    beta2: float = 0.25  # Entropy dampening from trust
    
    # Suppression dynamics
    k_policy: float = 0.8  # Aggressiveness of narrative control
    delta: float = 0.15    # Suppression decay rate
    
    # Noise model
    noise_sigma: float = 0.1    # Std-dev of exogenous leak shocks
    leak_frequency: float = 0.05  # Probability of a leak per timestep
    
    # Attractor mode metrics
    fcc_scale: float = 1.0  # Fractured Character Continuity
    ccd_scale: float = 1.0  # Causal Chain Discontinuity
    sdmm_scale: float = 1.0  # Semantic Doubling & Memetic Mirrorplay
    rsd_scale: float = 1.0  # Recursive Suppression Disclosure
    mst_scale: float = 1.0  # Myth of the Single Timeline
    
    # Collapse thresholds
    epsilon_overlap: float = 0.15  # ε-band for FCC x RSD overlap
    trust_slope_threshold: float = -0.05  # Critical dT/dt
    entropy_critical: float = 1.2  # Critical entropy threshold
    
    def __post_init__(self):
        """Validate parameter constraints."""
        assert self.alpha1 > 0, "Trust decay rate must be positive"
        assert self.beta1 > 0, "Transparency boost must be positive"
        assert self.gamma1 > 0, "Suppression penalty must be positive"
        assert self.alpha2 > 0, "Entropy growth rate must be positive"
        assert self.beta2 > 0, "Entropy dampening must be positive"
        assert self.k_policy > 0, "Policy aggression must be positive"
        assert self.delta > 0, "Suppression decay must be positive"
        assert 0 <= self.leak_frequency <= 1, "Leak frequency must be a probability"


class PistonLeakODE:
    """
    Core ODE implementation of the Piston Leak dynamical system.
    
    Models the coupled dynamics of Trust (T), Narrative Entropy (N),
    and Suppression Pressure (P) in institutional narrative networks.
    """
    
    def __init__(self, 
                 params: PistonLeakParams = None, 
                 initial_state: tuple[float, float, float] = (0.8, 0.2, 0.1),
                 seed: int = 42):
        """
        Initialize the ODE system.
        
        Args:
            params: Model parameters
            initial_state: Starting values for (T, N, P)
            seed: Random seed for noise generation
        """
        self.params = params or PistonLeakParams()
        self.initial_state = initial_state
        self.rng = np.random.RandomState(seed)
        
        # Transparency function (could be time-dependent or state-dependent)
        self._R = lambda t, y: 0.1  # Default minimal transparency
        
        # Storage for simulation results
        self.times = None
        self.trajectories = None
        self.attractor_metrics = None
        self.collapse_detected = False
    
    def set_transparency_function(self, r_func: Callable[[float, np.ndarray], float]):  # noqa: N803
        """
        Set the transparency function R(t, y).
        
        Args:
            r_func: Function taking time t and state vector y, returning transparency value
        """
        self._R = r_func
    
    def _noise_process(self, t: float) -> float:
        """
        Generate exogenous leak noise.
        
        Poisson-like process for discrete leak events.
        
        Args:
            t: Current time
            
        Returns:
            Noise value
        """
        if self.rng.random() < self.params.leak_frequency:
            return self.rng.normal(0, self.params.noise_sigma)
        return 0.0

    def _fractured_character_continuity(self, t: float, y: np.ndarray) -> float:
        """
        Calculate Fractured Character Continuity (FCC) metric.
        
        Measures the degree to which a narrative node carries
        contradictory epistemic roles.
        
        Args:
            t: Current time
            y: Current state vector [T, N, P]
            
        Returns:
            FCC metric value
        """
        T, N, P = y  # noqa: N806
        # Higher entropy and suppression increase FCC
        return self.params.fcc_scale * (N * P) / (1 + T)
    
    def _recursive_suppression_disclosure(self, t: float, y: np.ndarray) -> float:
        """
        Calculate Recursive Suppression Disclosure (RSD) metric.
        
        Measures how suppression efforts amplify belief in the suppressed claim.
        
        Args:
            t: Current time
            y: Current state vector [T, N, P]
            
        Returns:
            RSD metric value
        """
        T, N, P = y  # noqa: N806
        # RSD grows with suppression and entropy, diminishes with trust
        return self.params.rsd_scale * (P**2 * N) / (1 + T**2)

    def _policy_response(self, n: float, t: float) -> float:
        """
        Institutional policy response function g(N, T).
        
        Models how institutions increase suppression when
        entropy rises and trust falls.
        
        Args:
            n: Current narrative entropy
            t: Current trust level
            
        Returns:
            Policy response intensity
        """
        # Suppression increases with entropy and decreases with trust
        return self.params.k_policy * n / (0.1 + t)
    
    def _system(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Core ODE system function.
        
        Implements the coupled differential equations for T, N, P.
        
        Args:
            t: Current time
            y: Current state vector [T, N, P]
            
        Returns:
            Derivatives [dT/dt, dN/dt, dP/dt]
        """
        T, N, P = y  # noqa: N806
        R = self._R(t, y)  # noqa: N806  # Current transparency level
        
        # Trust dynamics (§4.1, eq 1)
        dT = -self.params.alpha1 * N + self.params.beta1 * R - self.params.gamma1 * P  # noqa: N806
        
        # Add bounds checking to prevent trust from exceeding [0,1] range
        # If trust is at bounds and derivative would push it out of bounds, zero the derivative
        if (T <= 0 and dT < 0) or (T >= 1 and dT > 0):
            dT = 0
        
        # Narrative entropy dynamics (§4.1, eq 2)
        noise = self._noise_process(t)
        dN = self.params.alpha2 * P + noise - self.params.beta2 * T  # noqa: N806
        
        # Add bounds checking to prevent entropy from going out of reasonable bounds
        # If entropy is at lower bound and derivative would push it down, zero the derivative
        if N <= 0 and dN < 0:
            dN = 0
        # Add damping term to prevent runaway entropy growth
        elif N > 10.0:  # If entropy is getting very large
            dN = dN - 0.1 * N  # Add additional damping proportional to entropy
        # Hard ceiling on entropy derivative to prevent numerical instability
        dN = np.clip(dN, -100.0, 100.0)
        
        # Suppression pressure dynamics (§4.1, eq 3)
        # Limit policy response to prevent numeric overflow
        policy = min(self._policy_response(min(N, 100.0), max(0.01, T)), 100.0)
        dP = policy - self.params.delta * P  # noqa: N806
        
        # Add bounds checking to prevent pressure from becoming negative
        # If pressure is at lower bound and derivative would push it down, zero the derivative
        if P <= 0 and dP < 0:
            dP = 0
        # Add damping term to prevent runaway pressure growth
        elif P > 10.0:  # If pressure is getting very large
            dP = dP - 0.1 * P  # Add additional damping proportional to pressure
        # Hard ceiling on pressure derivative to prevent numerical instability
        dP = np.clip(dP, -100.0, 100.0)
        
        return np.array([dT, dN, dP])
    
    def check_collapse(self, t: float, y: np.ndarray) -> bool:
        """
        Check if narrative collapse criteria are met.
        
        Implements the Phase-I Collapse criterion from §4.3.
        
        Args:
            t: Current time
            y: Current state vector [T, N, P]
            
        Returns:
            True if collapse detected, False otherwise
        """
        T, N, P = y  # noqa: N806
        
        # Calculate attractor metrics
        fcc = self._fractured_character_continuity(t, y)
        rsd = self._recursive_suppression_disclosure(t, y)
        
        # Calculate trust slope
        dt = 0.01  # Small delta for numerical derivative
        y_next = y + self._system(t, y) * dt
        dT_dt = (y_next[0] - T) / dt
        
        # Check collapse conditions (§4.3)
        overlap = min(fcc, rsd)  # Intersection size
        
        return (
            dT_dt < self.params.trust_slope_threshold and
            N > self.params.entropy_critical and
            overlap > self.params.epsilon_overlap
        )
    
    def simulate(self, t_span: tuple[float, float], dt: float = 1.0) -> dict:
        """
        Simulate the ODE system over a time span.
        
        Args:
            t_span: (t_start, t_end) simulation timespan
            dt: Time step for output
            
        Returns:
            Dictionary with simulation results
        """
        t_eval = np.arange(t_span[0], t_span[1] + dt, dt)
        
        # Solve the ODE system with more stable integrator
        try:
            solution = solve_ivp(
                fun=self._system,
                t_span=t_span,
                y0=self.initial_state,
                t_eval=t_eval,
                method='Radau',  # More stable for stiff systems
                atol=1e-6,       # Absolute tolerance
                rtol=1e-3        # Relative tolerance
            )
        except Exception as e:
            print(f"Warning: ODE solver failed with error: {e}")
            print("Falling back to less accurate but more stable solver...")
            
            # Fall back to more stable method if the first one fails
            solution = solve_ivp(
                fun=self._system,
                t_span=t_span,
                y0=self.initial_state,
                t_eval=t_eval,
                method='LSODA',  # Very stable solver
                atol=1e-3,       # Relaxed tolerances
                rtol=1e-2
            )
        
        self.times = solution.t
        self.trajectories = solution.y
        
        # Apply bounds to ensure values stay within valid ranges
        # This is a safety check in case the ODE solver steps exceeded bounds
        self.trajectories[0] = np.clip(self.trajectories[0], 0.0, 1.0)      # Trust in [0,1]
        self.trajectories[1] = np.clip(self.trajectories[1], 0.0, 1000.0)   # Bound entropy to prevent overflow
        self.trajectories[2] = np.clip(self.trajectories[2], 0.0, 1000.0)   # Bound pressure to prevent overflow
        
        # Calculate attractor metrics for each timestep
        n_steps = len(self.times)
        fcc_values = np.zeros(n_steps)
        rsd_values = np.zeros(n_steps)
        
        for i in range(n_steps):
            t = self.times[i]
            y = self.trajectories[:, i]
            fcc_values[i] = self._fractured_character_continuity(t, y)
            rsd_values[i] = self._recursive_suppression_disclosure(t, y)
            
            # Check for collapse
            if not self.collapse_detected and self.check_collapse(t, y):
                self.collapse_detected = True
                self.collapse_time = t
        
        self.attractor_metrics = {
            'fcc': fcc_values,
            'rsd': rsd_values,
            'overlap': np.minimum(fcc_values, rsd_values)
        }
        
        return {
            'times': self.times,
            'trust': self.trajectories[0],
            'entropy': self.trajectories[1],
            'pressure': self.trajectories[2],
            'metrics': self.attractor_metrics,
            'collapse': self.collapse_detected,
            'collapse_time': getattr(self, 'collapse_time', None)
        }


# Example usage
if __name__ == "__main__":
    # Custom transparency function - increases with time
    def adaptive_transparency(t, y):
        T, N, P = y  # noqa: N806
        # Transparency increases when entropy is high
        return 0.1 + 0.2 * N ** 2
    
    # Initialize model with default parameters
    model = PistonLeakODE()
    model.set_transparency_function(adaptive_transparency)
    
    # Run simulation for 365 days
    results = model.simulate((0, 365), dt=1.0)
    
    print(f"Final trust: {results['trust'][-1]:.3f}")
    print(f"Final entropy: {results['entropy'][-1]:.3f}")
    print(f"Final pressure: {results['pressure'][-1]:.3f}")
    
    if results['collapse']:
        print(f"Narrative collapse detected at t = {results['collapse_time']:.1f}")
    else:
        print("No narrative collapse detected in simulation timeframe.")
