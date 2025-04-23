#!/usr/bin/env python3
"""
PistonLeakLab - DuckDB Batch Query Script

This script runs a series of predefined queries against the PistonLeakLab
DuckDB database and saves the results to CSV files. Useful for 
batch processing or scheduled reporting.

Usage:
    python batch_queries.py
"""

import os
import duckdb
import pandas as pd

# Set paths
DB_PATH = r"d:\dev\piston-leak-lab\piston_leak_results.duckdb"
OUTPUT_PATH = r"d:\dev\piston-leak-lab\reports"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Connect to database
conn = duckdb.connect(DB_PATH)

# Define queries
QUERIES = {
    "scenario_summary.csv": """
        -- Overview of all scenarios
        SELECT 
            s.scenario_name,
            s.run_timestamp,
            s.abm_run_count,
            s.ode_run_count,
            m.n_runs,
            m.n_collapse,
            m.collapse_probability,
            m.mean_final_trust,
            m.mean_final_entropy,
            m.recovery_basin_size,
            m.collapse_basin_size,
            m.mean_rp_ratio,
            m.critical_rp_ratio
        FROM scenarios s
        JOIN summary_metrics m ON s.scenario_name = m.scenario_name
        ORDER BY m.recovery_basin_size DESC, m.mean_final_trust DESC
    """,
    
    "belief_dynamics.csv": """
        -- Final belief fractions across scenarios
        WITH final_state AS (
            SELECT 
                scenario_name, 
                run_id,
                LAST_VALUE(believer_fraction) OVER (
                    PARTITION BY scenario_name, run_id 
                    ORDER BY time 
                    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                ) AS final_believer,
                LAST_VALUE(skeptic_fraction) OVER (
                    PARTITION BY scenario_name, run_id 
                    ORDER BY time 
                    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                ) AS final_skeptic,
                LAST_VALUE(agnostic_fraction) OVER (
                    PARTITION BY scenario_name, run_id 
                    ORDER BY time 
                    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                ) AS final_agnostic
            FROM abm_runs
        ),
        scenario_stats AS (
            SELECT 
                scenario_name,
                AVG(final_believer) as avg_final_believer,
                STDDEV(final_believer) as std_final_believer,
                AVG(final_skeptic) as avg_final_skeptic,
                STDDEV(final_skeptic) as std_final_skeptic,
                AVG(final_agnostic) as avg_final_agnostic,
                STDDEV(final_agnostic) as std_final_agnostic
            FROM final_state
            GROUP BY scenario_name
        )
        SELECT * FROM scenario_stats
        ORDER BY avg_final_believer DESC
    """,
    
    "trust_collapse_timing.csv": """
        -- Trust collapse timing analysis
        WITH collapse_events AS (
            SELECT 
                scenario_name,
                run_id,
                MIN(time) as collapse_time
            FROM ode_runs
            WHERE trust < 0.3  -- Trust threshold for "collapse"
            GROUP BY scenario_name, run_id
        ),
        scenario_stats AS (
            SELECT 
                scenario_name,
                COUNT(*) as collapse_count,
                AVG(collapse_time) as avg_collapse_time,
                MIN(collapse_time) as min_collapse_time,
                MAX(collapse_time) as max_collapse_time,
                STDDEV(collapse_time) as std_collapse_time
            FROM collapse_events
            GROUP BY scenario_name
        )
        SELECT 
            s.scenario_name,
            COALESCE(stats.collapse_count, 0) as collapse_count,
            (COALESCE(stats.collapse_count, 0) * 100.0 / s.ode_run_count) as collapse_percentage,
            COALESCE(stats.avg_collapse_time, -1) as avg_collapse_time,
            COALESCE(stats.min_collapse_time, -1) as min_collapse_time,
            COALESCE(stats.max_collapse_time, -1) as max_collapse_time,
            COALESCE(stats.std_collapse_time, 0) as std_collapse_time
        FROM scenarios s
        LEFT JOIN scenario_stats stats ON s.scenario_name = stats.scenario_name
        ORDER BY collapse_percentage DESC
    """,
    
    "pressure_thresholds.csv": """
        -- Pressure threshold crossing analysis
        WITH threshold_events AS (
            SELECT 
                scenario_name,
                run_id,
                MIN(time) as threshold_time
            FROM ode_runs
            WHERE pressure > 0.5  -- Critical pressure threshold
            GROUP BY scenario_name, run_id
        ),
        scenario_stats AS (
            SELECT 
                scenario_name,
                COUNT(*) as threshold_count,
                AVG(threshold_time) as avg_threshold_time,
                MIN(threshold_time) as min_threshold_time,
                MAX(threshold_time) as max_threshold_time,
                STDDEV(threshold_time) as std_threshold_time
            FROM threshold_events
            GROUP BY scenario_name
        )
        SELECT 
            s.scenario_name,
            COALESCE(stats.threshold_count, 0) as threshold_count,
            (COALESCE(stats.threshold_count, 0) * 100.0 / s.ode_run_count) as threshold_percentage,
            COALESCE(stats.avg_threshold_time, -1) as avg_threshold_time,
            COALESCE(stats.min_threshold_time, -1) as min_threshold_time,
            COALESCE(stats.max_threshold_time, -1) as max_threshold_time,
            COALESCE(stats.std_threshold_time, 0) as std_threshold_time
        FROM scenarios s
        LEFT JOIN scenario_stats stats ON s.scenario_name = stats.scenario_name
        ORDER BY threshold_percentage DESC
    """,
    
    "time_series_believer.csv": """
        -- Time series of average believer fraction
        SELECT 
            scenario_name,
            time,
            AVG(believer_fraction) as avg_believer_fraction,
            STDDEV(believer_fraction) as std_believer_fraction,
            MIN(believer_fraction) as min_believer_fraction,
            MAX(believer_fraction) as max_believer_fraction
        FROM abm_runs
        GROUP BY scenario_name, time
        ORDER BY scenario_name, time
    """,
    
    "time_series_trust.csv": """
        -- Time series of average trust values
        SELECT 
            scenario_name,
            time,
            AVG(trust) as avg_trust,
            STDDEV(trust) as std_trust,
            MIN(trust) as min_trust,
            MAX(trust) as max_trust
        FROM ode_runs
        GROUP BY scenario_name, time
        ORDER BY scenario_name, time
    """,
    
    "time_series_pressure.csv": """
        -- Time series of average pressure values
        SELECT 
            scenario_name,
            time,
            AVG(pressure) as avg_pressure,
            STDDEV(pressure) as std_pressure,
            MIN(pressure) as min_pressure,
            MAX(pressure) as max_pressure
        FROM ode_runs
        GROUP BY scenario_name, time
        ORDER BY scenario_name, time
    """,
    
    "belief_change_magnitude.csv": """
        -- Magnitude of belief change between initial and final states
        WITH initial_state AS (
            SELECT 
                scenario_name, 
                run_id,
                FIRST_VALUE(believer_fraction) OVER (
                    PARTITION BY scenario_name, run_id 
                    ORDER BY time 
                    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                ) as initial_believer
            FROM abm_runs
        ),
        final_state AS (
            SELECT 
                scenario_name, 
                run_id,
                LAST_VALUE(believer_fraction) OVER (
                    PARTITION BY scenario_name, run_id 
                    ORDER BY time 
                    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                ) as final_believer
            FROM abm_runs
        ),
        run_changes AS (
            SELECT 
                i.scenario_name,
                i.run_id,
                i.initial_believer,
                f.final_believer,
                (f.final_believer - i.initial_believer) as belief_change,
                ABS(f.final_believer - i.initial_believer) as belief_change_magnitude
            FROM initial_state i
            JOIN final_state f ON i.scenario_name = f.scenario_name AND i.run_id = f.run_id
        )
        SELECT 
            scenario_name,
            AVG(initial_believer) as avg_initial_believer,
            AVG(final_believer) as avg_final_believer,
            AVG(belief_change) as avg_belief_change,
            AVG(belief_change_magnitude) as avg_belief_change_magnitude,
            MAX(belief_change_magnitude) as max_belief_change_magnitude
        FROM run_changes
        GROUP BY scenario_name
        ORDER BY avg_belief_change_magnitude DESC
    """,
    
    "network_entropy_stats.csv": """
        -- Network entropy statistics over time
        SELECT 
            scenario_name,
            time,
            AVG(network_entropy) as avg_network_entropy,
            STDDEV(network_entropy) as std_network_entropy,
            MIN(network_entropy) as min_network_entropy,
            MAX(network_entropy) as max_network_entropy
        FROM abm_runs
        GROUP BY scenario_name, time
        ORDER BY scenario_name, time
    """,
    
    "trust_pressure_correlation.csv": """
        -- Correlation between trust and pressure across scenarios
        WITH scenario_corr AS (
            SELECT 
                scenario_name,
                run_id,
                CORR(trust, pressure) as trust_pressure_corr
            FROM ode_runs
            GROUP BY scenario_name, run_id
        )
        SELECT 
            scenario_name,
            AVG(trust_pressure_corr) as avg_correlation,
            STDDEV(trust_pressure_corr) as std_correlation,
            MIN(trust_pressure_corr) as min_correlation,
            MAX(trust_pressure_corr) as max_correlation
        FROM scenario_corr
        GROUP BY scenario_name
        ORDER BY avg_correlation
    """
}

def run_queries():
    """Run all predefined queries and save results to CSV files"""
    print(f"Connected to database: {DB_PATH}")
    print(f"Running {len(QUERIES)} batch queries...")
    
    for filename, query in QUERIES.items():
        try:
            print(f"Running query for: {filename}")
            result = conn.execute(query).fetchdf()
            
            # Save to CSV
            output_file = os.path.join(OUTPUT_PATH, filename)
            result.to_csv(output_file, index=False)
            print(f"  - Saved to {output_file} ({len(result)} rows)")
            
        except Exception as e:
            print(f"  - Error running query for {filename}: {e}")
    
    print("Batch queries complete!")

if __name__ == "__main__":
    try:
        run_queries()
    finally:
        conn.close()