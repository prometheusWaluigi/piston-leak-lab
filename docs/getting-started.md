# Getting Started with Piston Leak Lab

This guide will walk you through setting up and running your first simulation with the Piston Leak Lab framework.

## Installation

### Prerequisites

- Python 3.11 or higher
- Poetry 1.8 or higher (recommended)
- Git LFS (optional, for pulling dataset files)

### Setup Process

```bash
# Clone the repository
git clone https://github.com/gaslit420/piston-leak-lab.git
cd piston-leak-lab

# Install with Poetry (recommended)
poetry install

# Activate the environment
poetry shell
```

### Optional Accelerators

For high-performance simulations, you can install the oneAPI extras:

```bash
poetry install -E oneapi
```

This enables Intel GPU and optimized CPU acceleration for large-scale simulations.

## Running Your First Simulation

The framework ships with a baseline configuration ready to run:

```bash
# Run with default parameters
run-mc --config sims/baseline.yml

# Run with reduced ensemble size (faster)
run-mc --config sims/baseline.yml --n 100

# Override output directory
run-mc --config sims/baseline.yml --out my_results
```

## Understanding the Output

After running a simulation, you'll find several types of output in the results directory:

- **Summary statistics** (JSON): Collapse probabilities, critical ratios, etc.
- **Time-series data** (CSV): If enabled, detailed trajectories for all runs
- **Visualizations** (PNG): Trust trajectories, phase space plots, attractor metrics
- **Interactive dashboard** (HTML): If enabled, an exploratory interface

Example summary output:

```json
{
  "n_runs": 500,
  "n_collapse": 237,
  "collapse_probability": 0.474,
  "mean_collapse_time": 143.2,
  "mean_final_trust": 0.423,
  "mean_final_entropy": 1.187,
  "recovery_basin_size": 0.526,
  "critical_rp_ratio": 1.74
}
```

## Configuration Options

Simulations are controlled via YAML configuration files. The key parameters include:

### Ensemble Settings

```yaml
ensemble:
  runs: 500                # Number of Monte Carlo runs
  random_seed: 42          # Random seed for reproducibility
  run_abm: true            # Enable agent-based modeling layer
  perturb_parameters: true # Randomize parameters slightly each run
```

### Dynamic Parameters

```yaml
parameters:
  alpha1: 0.6              # Trust decay vs entropy
  beta1: 0.3               # Transparency restoration boost
  gamma1: 0.2              # Suppression penalty on trust
  
  transparency_type: "adaptive"  # Transparency policy
  base_transparency: 0.1
  transparency_adaptivity: 0.2
```

See the [Configuration Reference](configuration.md) for the complete list of options.

## Next Steps

- Try modifying parameters to explore different outcomes
- Implement custom transparency functions to test intervention strategies
- Run parameter sweeps to identify critical thresholds
- Visualize results using the provided tools
