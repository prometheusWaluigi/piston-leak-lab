# Core ODE Model API

The core ODE module implements the Trust-Narrative-Pressure dynamical system that models institutional narrative collapse.

## PistonLeakParams

```python
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
    epsilon_overlap: float = 0.15  # ε-band for FCC×RSD overlap
    trust_slope_threshold: float = -0.05  # Critical dT/dt
    entropy_critical: float = 1.2  # Critical entropy threshold
```

This class defines the parameters for the ODE system. The parameters control:

- How quickly trust decays due to entropy (`alpha1`)
- How much transparency boosts trust (`beta1`)
- How much suppression erodes trust (`gamma1`)
- The growth of narrative entropy from suppression (`alpha2`)
- The dampening of entropy from trust (`beta2`)
- The policy response aggressiveness (`k_policy`)
- The decay rate of suppression (`delta`)
- The noise model for exogenous leaks
- The scaling factors for attractor metrics
- The thresholds for narrative collapse

## PistonLeakODE

```python
class PistonLeakODE:
    """
    Core ODE implementation of the Piston Leak dynamical system.
    
    Models the coupled dynamics of Trust (T), Narrative Entropy (N),
    and Suppression Pressure (P) in institutional narrative networks.
    """
    
    def __init__(self, 
                 params: PistonLeakParams = None, 
                 initial_state: Tuple[float, float, float] = (0.8, 0.2, 0.1),
                 seed: int = 42):
        """
        Initialize the ODE system.
        
        Args:
            params: Model parameters
            initial_state: Starting values for (T, N, P)
            seed: Random seed for noise generation
        """
```

This class implements the core ODE system. The key methods include:

### Setting Transparency Function

```python
def set_transparency_function(self, R_func: Callable[[float, np.ndarray], float]):
    """
    Set the transparency function R(t, y).
    
    Args:
        R_func: Function taking time t and state vector y, returning transparency value
    """
```

### Simulation

```python
def simulate(self, t_span: Tuple[float, float], dt: float = 1.0) -> Dict:
    """
    Simulate the ODE system over a time span.
    
    Args:
        t_span: (t_start, t_end) simulation timespan
        dt: Time step for output
        
    Returns:
        Dictionary with simulation results
    """
```

The `simulate` method runs the ODE integration and returns a dictionary with:

- `times`: Timepoints
- `trust`: Trust values at each timepoint
- `entropy`: Narrative entropy at each timepoint
- `pressure`: Suppression pressure at each timepoint
- `metrics`: Dictionary with attractor metrics (`fcc`, `rsd`, `overlap`)
- `collapse`: Boolean indicating if narrative collapse was detected
- `collapse_time`: Time at which collapse occurred (if any)

### Collapse Detection

```python
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
```

Collapse is detected when:

1. Trust is declining faster than the critical slope threshold
2. Narrative entropy exceeds the critical threshold
3. The overlap between FCC and RSD exceeds the epsilon band

## Example Usage

```python
# Create a custom transparency function
def adaptive_transparency(t, y):
    T, N, P = y
    return 0.1 + 0.2 * N  # Transparency increases with entropy

# Initialize the model
model = PistonLeakODE()
model.set_transparency_function(adaptive_transparency)

# Run simulation
results = model.simulate((0, 365), dt=1.0)

# Check for collapse
if results['collapse']:
    print(f"Narrative collapse detected at t = {results['collapse_time']:.1f}")
else:
    print("No collapse detected")
```
