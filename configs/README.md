# Piston Leak Lab - Simulation Configurations

This directory contains YAML configuration files for various simulation scenarios in the Piston Leak Lab system.

## Overview

The Piston Leak Lab simulates the dynamics of institutional narrative collapse through a coupled ODE and Agent-Based Model system. The ODE system models Trust (T), Narrative Entropy (N), and Suppression Pressure (P) dynamics, while the ABM simulates belief state evolution in a social network.

## Configuration Categories

The configurations are organized into the following categories:

### 1. Transparency Function Variations

Testing different institutional disclosure strategies:

- `transparency_adaptive.yml` - Transparency increases with entropy
- `transparency_delayed.yml` - Transparency increases after a time threshold
- `transparency_reactive.yml` - Transparency increases when trust falls below threshold
- `transparency_cyclic.yml` - Oscillating transparency (periodic disclosure)
- `transparency_threshold.yml` - Transparency jumps at critical entropy values

### 2. Network Topology Variations

Testing different social network structures:

- `network_scale_free.yml` - Scale-free network (Barabasi-Albert)
- `network_random.yml` - Random network (Erdos-Renyi)

### 3. Parameter Sensitivity

Sweeping key parameters to identify critical thresholds:

- `sensitivity_trust_decay.yml` - Trust decay rate (α₁)
- `sensitivity_transparency_boost.yml` - Transparency restoration boost (β₁)
- `sensitivity_narrative_control.yml` - Aggressiveness of narrative control (k_policy)

### 4. Noise Model Variations

Testing different leak patterns:

- `noise_leak_frequency.yml` - Probability of leak per timestep
- `noise_magnitude.yml` - Standard deviation of leak shocks

### 5. Initial Condition Variations

Testing sensitivity to starting conditions:

- `initial_preCollapse.yml` - Low trust, high entropy state
- `initial_polarized.yml` - Highly polarized belief distribution

### 6. Combined Scenarios

Special cases combining multiple parameter changes:

- `combined_high_pressure.yml` - High narrative control pressure with high belief skepticism

## Running Simulations

Use the batch script to run all simulations:

```bash
python run_batch_sims.py
```

Or run a specific configuration:

```bash
python sims/run_mc.py --config configs/transparency_adaptive.yml
```

## Results

Simulation results will be stored in the `results/` directory, organized by configuration name. Each run produces:

- Summary statistics in JSON format
- Time series data in CSV format (if save_timeseries: true)
- Visualization plots (if plots: true)
