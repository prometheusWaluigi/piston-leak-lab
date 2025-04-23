# Piston Leak Lab Database Tools

This repository contains a suite of tools for processing, analyzing, and visualizing the simulation results from the Piston Leak Lab project.

## Overview

The Piston Leak Lab project investigates how institutional narratives fracture under semantic overload using COVID-19 as a case study. It combines symbolic dynamical-systems modeling (ODE + agent-based) with narrative topology analysis.

These tools help you:

1. Import all simulation results into a DuckDB database
2. Run comprehensive analyses and generate visualizations
3. Interactively query the database and create custom visualizations

## Scripts

### 1. `duckdb_import.py`

This script processes all simulation results from the Piston Leak Lab and loads them into a DuckDB database.

```bash
python duckdb_import.py
```

The script will:
- Scan all result directories and extract metadata
- Process all ABM run files (agent fractions, network entropy, etc.)
- Process all ODE run files (trust, entropy, pressure dynamics)
- Extract summary metrics from JSON files
- Create a DuckDB database with tables and indexes
- Set up useful views for analysis

### 2. `analyze_results.py`

Performs comprehensive analysis on the database and generates visualizations.

```bash
python analyze_results.py
```

Analyses include:
- Scenario comparison (collapse probability, basin stability, etc.)
- Believer fraction evolution across scenarios
- Trust-entropy-pressure dynamics
- Stability metrics correlation
- Population fraction trajectories
- Interactive metrics dashboard

Outputs various CSV files, images, and interactive HTML visualizations.

### 3. `query_tool.py`

An interactive command-line tool for exploring the database and generating custom visualizations.

```bash
python query_tool.py
```

Features:
- Browse database tables and schemas
- Run custom SQL queries
- Compare scenarios across metrics
- Analyze time series data
- Run predefined custom queries
- Generate interactive visualizations

## Database Structure

The database (`piston_leak_results.duckdb`) contains the following tables:

- **scenarios**: Metadata about each simulation scenario
- **abm_runs**: Agent-based model simulation results
- **ode_runs**: ODE simulation results
- **summary_metrics**: Aggregated metrics from summary JSON files

And the following views:

- **abm_final_states**: Final state of each ABM run
- **ode_final_states**: Final state of each ODE run
- **scenario_comparison**: Comparative overview of scenarios

## Example Queries

Here are some example queries you can run with the query tool:

```sql
-- Compare final believer fractions across scenarios
SELECT 
    scenario_name, 
    AVG(final_believer_fraction) as mean_final_bf
FROM abm_final_states
GROUP BY scenario_name
ORDER BY mean_final_bf DESC;

-- Find scenarios where trust collapses
SELECT 
    scenario_name,
    COUNT(*) as collapse_count,
    COUNT(*) * 100.0 / COUNT(DISTINCT run_id) as collapse_percentage
FROM ode_runs
WHERE trust < 0.2 AND time > 50
GROUP BY scenario_name
HAVING collapse_percentage > 0
ORDER BY collapse_percentage DESC;

-- Track pressure-trust relationship
SELECT 
    scenario_name,
    time,
    AVG(pressure) as mean_pressure,
    AVG(trust) as mean_trust
FROM ode_runs
WHERE scenario_name IN ('baseline', 'leak_storm', 'skeptic_surge')
GROUP BY scenario_name, time
ORDER BY scenario_name, time;
```

## Requirements

- Python 3.11+
- DuckDB
- pandas
- matplotlib
- seaborn
- plotly

Install requirements with:

```bash
pip install duckdb pandas matplotlib seaborn plotly
```

## Workflow

1. Run `duckdb_import.py` to create the database
2. Run `analyze_results.py` to generate comprehensive analyses
3. Use `query_tool.py` for interactive exploration

## Output Directory Structure

- `analysis/`: Contains output from the analysis script
- `query_results/`: Contains output from the query tool

## Notes

- These tools assume the simulation results are in `d:\dev\piston-leak-lab\results\`
- Modify path variables in the scripts if your directory structure differs
- The database is created at `d:\dev\piston-leak-lab\piston_leak_results.duckdb`