#!/usr/bin/env python3
"""
PistonLeakLab - DuckDB Data Import Script

This script processes all simulation results from the Piston Leak Lab project
and loads them into a DuckDB database for analysis.

Usage:
    python duckdb_import.py 

Creates a new database file called piston_leak_results.duckdb with tables:
- scenarios: metadata about each simulation scenario
- abm_runs: agent-based model simulation results
- ode_runs: ODE simulation results 
- summary_metrics: aggregated metrics from the summary JSON files
"""

import os
import re
import json
import glob
import pandas as pd
import duckdb
import datetime as dt

# Set the base path
BASE_PATH = r"d:\dev\piston-leak-lab\results"
OUTPUT_DB = r"d:\dev\piston-leak-lab\piston_leak_results.duckdb"

def timestamp_from_filename(filename):
    """Extract timestamp from a result filename"""
    match = re.search(r'_(\d{8}_\d{6})\.', filename)
    if match:
        timestamp_str = match.group(1)
        return dt.datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
    return None

def scenario_name_from_path(path):
    """Extract scenario name from the directory path"""
    return os.path.basename(path)

def process_directories():
    """Process all result directories and return metadata dataframe"""
    scenario_data = []
    
    for scenario_dir in glob.glob(os.path.join(BASE_PATH, "*")):
        if not os.path.isdir(scenario_dir):
            continue
            
        scenario_name = scenario_name_from_path(scenario_dir)
        
        # Find timestamp from the first file
        csv_files = glob.glob(os.path.join(scenario_dir, "*.csv"))
        if not csv_files:
            continue
            
        timestamp = timestamp_from_filename(csv_files[0])
        
        # Count the number of ABM and ODE runs
        abm_files = glob.glob(os.path.join(scenario_dir, "abm_run*.csv"))
        ode_files = glob.glob(os.path.join(scenario_dir, "ode_run*.csv"))
        
        scenario_data.append({
            "scenario_name": scenario_name,
            "run_timestamp": timestamp,
            "abm_run_count": len(abm_files),
            "ode_run_count": len(ode_files),
            "result_path": scenario_dir
        })
    
    return pd.DataFrame(scenario_data)

def process_abm_runs(scenarios_df):
    """Process all ABM run files and return a dataframe"""
    all_abm_data = []
    
    for _, row in scenarios_df.iterrows():
        scenario_dir = row['result_path']
        scenario_name = row['scenario_name']
        timestamp = row['run_timestamp']
        
        # Process each ABM file
        for abm_file in glob.glob(os.path.join(scenario_dir, "abm_run*.csv")):
            # Extract run ID from filename
            run_id = int(re.search(r'abm_run(\d+)_', abm_file).group(1))
            
            # Read the CSV data
            df = pd.read_csv(abm_file)
            
            # Add metadata columns
            df['scenario_name'] = scenario_name
            df['run_id'] = run_id
            df['run_timestamp'] = timestamp
            
            all_abm_data.append(df)
    
    if all_abm_data:
        return pd.concat(all_abm_data, ignore_index=True)
    return pd.DataFrame()

def process_ode_runs(scenarios_df):
    """Process all ODE run files and return a dataframe"""
    all_ode_data = []
    
    for _, row in scenarios_df.iterrows():
        scenario_dir = row['result_path']
        scenario_name = row['scenario_name']
        timestamp = row['run_timestamp']
        
        # Process each ODE file
        for ode_file in glob.glob(os.path.join(scenario_dir, "ode_run*.csv")):
            # Extract run ID from filename
            run_id = int(re.search(r'ode_run(\d+)_', ode_file).group(1))
            
            # Read the CSV data
            df = pd.read_csv(ode_file)
            
            # Add metadata columns
            df['scenario_name'] = scenario_name
            df['run_id'] = run_id
            df['run_timestamp'] = timestamp
            
            all_ode_data.append(df)
    
    if all_ode_data:
        return pd.concat(all_ode_data, ignore_index=True)
    return pd.DataFrame()

def process_summary_metrics(scenarios_df):
    """Process all summary JSON files and return a dataframe"""
    summary_data = []
    
    for _, row in scenarios_df.iterrows():
        scenario_dir = row['result_path']
        scenario_name = row['scenario_name']
        timestamp = row['run_timestamp']
        
        # Find summary json file
        summary_files = glob.glob(os.path.join(scenario_dir, "summary_*.json"))
        
        for summary_file in summary_files:
            with open(summary_file, 'r') as f:
                data = json.load(f)
                
                # Add metadata
                data['scenario_name'] = scenario_name
                data['run_timestamp'] = timestamp
                summary_data.append(data)
    
    return pd.DataFrame(summary_data)

def create_database():
    """Create the DuckDB database and tables"""
    # Connect to DuckDB
    conn = duckdb.connect(OUTPUT_DB)
    
    # Process all the data
    print("Scanning result directories...")
    scenarios_df = process_directories()
    
    print(f"Found {len(scenarios_df)} scenario directories")
    if len(scenarios_df) == 0:
        print(f"No scenarios found in {BASE_PATH}")
        return
    
    print("Processing ABM runs...")
    abm_df = process_abm_runs(scenarios_df)
    
    print("Processing ODE runs...")
    ode_df = process_ode_runs(scenarios_df)
    
    print("Processing summary metrics...")
    summary_df = process_summary_metrics(scenarios_df)
    
    # Create tables in DuckDB
    print("Creating database tables...")
    
    conn.execute("CREATE TABLE scenarios AS SELECT * FROM scenarios_df")
    
    if not abm_df.empty:
        conn.execute("CREATE TABLE abm_runs AS SELECT * FROM abm_df")
    else:
        print("Warning: No ABM run data found")
    
    if not ode_df.empty:
        conn.execute("CREATE TABLE ode_runs AS SELECT * FROM ode_df")
    else:
        print("Warning: No ODE run data found")
    
    if not summary_df.empty:
        conn.execute("CREATE TABLE summary_metrics AS SELECT * FROM summary_df")
    else:
        print("Warning: No summary metrics found")
    
    # Create indexes for faster queries
    print("Creating indexes...")
    conn.execute("CREATE INDEX idx_scenarios_name ON scenarios(scenario_name)")
    
    if not abm_df.empty:
        conn.execute("CREATE INDEX idx_abm_scenario ON abm_runs(scenario_name, run_id)")
    
    if not ode_df.empty:
        conn.execute("CREATE INDEX idx_ode_scenario ON ode_runs(scenario_name, run_id)")
    
    if not summary_df.empty:
        conn.execute("CREATE INDEX idx_summary_scenario ON summary_metrics(scenario_name)")
    
    # Create some useful views
    print("Creating views...")
    
    # View for last timepoint of each ABM run
    if not abm_df.empty:
        conn.execute("""
        CREATE VIEW abm_final_states AS
        SELECT scenario_name, run_id, run_timestamp, 
               MAX(time) as final_time,
               FIRST(believer_fraction ORDER BY time DESC) as final_believer_fraction,
               FIRST(skeptic_fraction ORDER BY time DESC) as final_skeptic_fraction,
               FIRST(agnostic_fraction ORDER BY time DESC) as final_agnostic_fraction,
               FIRST(network_entropy ORDER BY time DESC) as final_network_entropy
        FROM abm_runs
        GROUP BY scenario_name, run_id, run_timestamp
        """)
    
    # View for last timepoint of each ODE run
    if not ode_df.empty:
        conn.execute("""
        CREATE VIEW ode_final_states AS
        SELECT scenario_name, run_id, run_timestamp,
               MAX(time) as final_time,
               FIRST(trust ORDER BY time DESC) as final_trust,
               FIRST(entropy ORDER BY time DESC) as final_entropy,
               FIRST(pressure ORDER BY time DESC) as final_pressure
        FROM ode_runs
        GROUP BY scenario_name, run_id, run_timestamp
        """)
    
    # Scenario comparison view
    if not summary_df.empty:
        conn.execute("""
        CREATE VIEW scenario_comparison AS
        SELECT 
            scenario_name, 
            run_timestamp,
            n_runs,
            n_collapse,
            collapse_probability,
            mean_collapse_time,
            mean_final_trust,
            std_final_trust,
            mean_final_entropy,
            std_final_entropy,
            recovery_basin_size,
            collapse_basin_size,
            mean_rp_ratio,
            critical_rp_ratio
        FROM summary_metrics
        ORDER BY run_timestamp DESC
        """)
    
    # Done
    conn.close()
    print(f"Database created successfully: {OUTPUT_DB}")

if __name__ == "__main__":
    create_database()
    
    # Print example usage
    print("\nExample usage:")
    print("import duckdb")
    print("conn = duckdb.connect('piston_leak_results.duckdb')")
    print("conn.execute('SELECT * FROM scenarios').fetchall()")
    print("conn.execute('SELECT * FROM scenario_comparison').fetchdf()")
    print("conn.execute('SELECT scenario_name, AVG(final_trust) FROM ode_final_states GROUP BY scenario_name').fetchdf()")