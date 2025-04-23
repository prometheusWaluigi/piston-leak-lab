#!/usr/bin/env python3
"""
Batch Simulation Runner for Piston Leak Lab
===========================================

Runs multiple Monte Carlo ensembles with different parameter configurations.
"""

import os
import glob
import subprocess
import concurrent.futures
import time
import argparse
from pathlib import Path

def run_simulation(config_file, output_dir=None):
    """
    Run a single simulation with the given config file.
    
    Args:
        config_file: Path to the YAML config file
        output_dir: Override output directory (optional)
    
    Returns:
        True if successful, False otherwise
    """
    cmd = ["python", os.path.join("sims", "run_mc.py"), "--config", config_file]
    
    # Override output directory if specified
    if output_dir:
        cmd.extend(["--out", output_dir])
    
    print(f"Starting simulation with {config_file}")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"ERROR in {config_file}: {result.stderr}")
            with open(f"{config_file}_error.log", "w") as f:
                f.write(result.stderr)
            return False
        else:
            duration = time.time() - start_time
            print(f"Completed {config_file} in {duration:.1f} seconds")
            return True
    except Exception as e:
        print(f"EXCEPTION in {config_file}: {str(e)}")
        return False

def main():
    """Main entry point for batch simulation running."""
    parser = argparse.ArgumentParser(description='Run batch Piston Leak simulations')
    parser.add_argument('--configs', type=str, default="configs/*.yml", 
                        help='Glob pattern for config files')
    parser.add_argument('--max-workers', type=int, default=4,
                        help='Maximum number of parallel workers')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Override output directory for all simulations')
    args = parser.parse_args()
    
    # Find all YAML config files
    config_files = glob.glob(args.configs)
    
    if not config_files:
        print(f"No configuration files found matching {args.configs}")
        return
    
    print(f"Found {len(config_files)} configuration files:")
    for cf in config_files:
        print(f"  - {cf}")
    
    # Create output directory if specified and doesn't exist
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
    
    # Run simulations with thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(run_simulation, cf, args.out_dir) for cf in config_files]
        
        # Process results as they complete
        success_count = 0
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            if future.result():
                success_count += 1
            print(f"Progress: {i+1}/{len(config_files)} simulations processed")
    
    # Report results
    print(f"Completed {success_count}/{len(config_files)} simulations successfully")
    
    # Create consolidated results if all simulations succeeded
    if success_count == len(config_files):
        print("All simulations completed successfully!")
        # TODO: Add code here to consolidate results across simulations if needed


if __name__ == "__main__":
    main()
