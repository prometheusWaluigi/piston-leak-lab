#!/usr/bin/env python3
"""
PistonLeakLab - Complete Analysis Pipeline

This script runs the entire analysis pipeline:
1. Imports simulation results into DuckDB
2. Runs comprehensive analyses and generates visualizations
3. Runs batch queries to generate report data

Usage:
    python run_pipeline.py [--skip-import] [--skip-analysis] [--skip-reports]
"""

import os
import sys
import subprocess
import argparse
import time
from datetime import datetime

# Set paths
BASE_PATH = r"d:\dev\piston-leak-lab"
SCRIPTS_PATH = BASE_PATH
DB_PATH = os.path.join(BASE_PATH, "piston_leak_results.duckdb")
ANALYSIS_PATH = os.path.join(BASE_PATH, "analysis")
REPORTS_PATH = os.path.join(BASE_PATH, "reports")

def create_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs(ANALYSIS_PATH, exist_ok=True)
    os.makedirs(REPORTS_PATH, exist_ok=True)

def run_import():
    """Run the database import script"""
    script_path = os.path.join(SCRIPTS_PATH, "duckdb_import.py")
    
    print("\n" + "="*80)
    print("STEP 1: Importing simulation results to DuckDB")
    print("="*80)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:", result.stderr)
        
        print(f"Import completed successfully in {time.time() - start_time:.2f} seconds")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"Error running import script: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

def run_analysis():
    """Run the analysis script"""
    script_path = os.path.join(SCRIPTS_PATH, "analyze_results.py")
    
    print("\n" + "="*80)
    print("STEP 2: Running comprehensive analysis")
    print("="*80)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:", result.stderr)
        
        print(f"Analysis completed successfully in {time.time() - start_time:.2f} seconds")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"Error running analysis script: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

def run_batch_queries():
    """Run the batch queries script"""
    script_path = os.path.join(SCRIPTS_PATH, "batch_queries.py")
    
    print("\n" + "="*80)
    print("STEP 3: Running batch queries for reports")
    print("="*80)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:", result.stderr)
        
        print(f"Batch queries completed successfully in {time.time() - start_time:.2f} seconds")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"Error running batch queries script: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

def generate_summary_report():
    """Generate a summary report of the pipeline execution"""
    report_file = os.path.join(BASE_PATH, "pipeline_summary.txt")
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PISTON LEAK LAB - ANALYSIS PIPELINE SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Pipeline executed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Output directories:\n")
        f.write(f"- Database: {DB_PATH}\n")
        f.write(f"- Analysis: {ANALYSIS_PATH}\n")
        f.write(f"- Reports: {REPORTS_PATH}\n\n")
        
        # Database info if it exists
        if os.path.exists(DB_PATH):
            size_mb = os.path.getsize(DB_PATH) / (1024 * 1024)
            f.write(f"Database size: {size_mb:.2f} MB\n")
        else:
            f.write("Database not found\n")
        
        # Analysis files
        analysis_files = os.listdir(ANALYSIS_PATH) if os.path.exists(ANALYSIS_PATH) else []
        f.write(f"\nAnalysis files generated: {len(analysis_files)}\n")
        for file in sorted(analysis_files)[:10]:  # Show first 10 files
            f.write(f"- {file}\n")
        if len(analysis_files) > 10:
            f.write(f"- ... and {len(analysis_files) - 10} more files\n")
        
        # Report files
        report_files = os.listdir(REPORTS_PATH) if os.path.exists(REPORTS_PATH) else []
        f.write(f"\nReport files generated: {len(report_files)}\n")
        for file in sorted(report_files):
            f.write(f"- {file}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"\nSummary report generated: {report_file}")

def main():
    """Main entry point for the pipeline"""
    parser = argparse.ArgumentParser(description="Run the Piston Leak Lab analysis pipeline")
    parser.add_argument('--skip-import', action='store_true', help='Skip the import step')
    parser.add_argument('--skip-analysis', action='store_true', help='Skip the analysis step')
    parser.add_argument('--skip-reports', action='store_true', help='Skip the reports step')
    
    args = parser.parse_args()
    
    # Create directories
    create_directories()
    
    # Track overall start time
    overall_start = time.time()
    
    # Run pipeline steps
    import_success = True
    analysis_success = True
    reports_success = True
    
    if not args.skip_import:
        import_success = run_import()
    else:
        print("\nSkipping import step as requested")
    
    if import_success and not args.skip_analysis:
        analysis_success = run_analysis()
    elif args.skip_analysis:
        print("\nSkipping analysis step as requested")
    else:
        print("\nSkipping analysis step due to import failure")
    
    if import_success and not args.skip_reports:
        reports_success = run_batch_queries()
    elif args.skip_reports:
        print("\nSkipping reports step as requested")
    else:
        print("\nSkipping reports step due to previous failures")
    
    # Generate summary report
    generate_summary_report()
    
    # Print summary
    print("\n" + "="*80)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*80)
    print(f"Total execution time: {time.time() - overall_start:.2f} seconds")
    print(f"Import step: {'Skipped' if args.skip_import else 'Success' if import_success else 'Failed'}")
    print(f"Analysis step: {'Skipped' if args.skip_analysis or not import_success else 'Success' if analysis_success else 'Failed'}")
    print(f"Reports step: {'Skipped' if args.skip_reports or not import_success else 'Success' if reports_success else 'Failed'}")
    
    # Set exit code
    if not args.skip_import and not import_success:
        sys.exit(1)
    elif not args.skip_analysis and not analysis_success:
        sys.exit(2)
    elif not args.skip_reports and not reports_success:
        sys.exit(3)
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()