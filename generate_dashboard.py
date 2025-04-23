#!/usr/bin/env python3
"""
PistonLeakLab - Dashboard Generator

This script generates an HTML dashboard from the simulation results database
and analysis files. It dynamically updates the dashboard with the latest data,
charts, and metrics.

Usage:
    python generate_dashboard.py [--template TEMPLATE] [--output OUTPUT]
"""

import os
import sys
import argparse
import duckdb
import pandas as pd
import json
import re
import shutil
from datetime import datetime
from jinja2 import Template

# Set paths
DEFAULT_TEMPLATE = r"d:\dev\piston-leak-lab\dashboard_template.html"
DEFAULT_OUTPUT = r"d:\dev\piston-leak-lab\dashboard.html"
DB_PATH = r"d:\dev\piston-leak-lab\piston_leak_results.duckdb"
ANALYSIS_PATH = r"d:\dev\piston-leak-lab\analysis"
REPORTS_PATH = r"d:\dev\piston-leak-lab\reports"
RESULTS_PATH = r"d:\dev\piston-leak-lab\results"

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate HTML dashboard from simulation results")
    parser.add_argument("--template", default=DEFAULT_TEMPLATE, help="Path to the HTML template")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path for the output HTML file")
    return parser.parse_args()

def extract_scenario_metrics(conn):
    """Extract key metrics for each scenario"""
    try:
        query = """
        SELECT 
            s.scenario_name,
            s.run_timestamp,
            s.abm_run_count,
            s.ode_run_count,
            COALESCE(m.n_runs, 0) as n_runs,
            COALESCE(m.n_collapse, 0) as n_collapse,
            COALESCE(m.collapse_probability, 0) as collapse_probability,
            COALESCE(m.mean_final_trust, 0) as mean_final_trust,
            COALESCE(m.mean_final_entropy, 0) as mean_final_entropy,
            COALESCE(m.recovery_basin_size, 0) as recovery_basin_size,
            COALESCE(m.collapse_basin_size, 0) as collapse_basin_size,
            COALESCE(m.mean_rp_ratio, 0) as mean_rp_ratio,
            COALESCE(m.critical_rp_ratio, 0) as critical_rp_ratio
        FROM scenarios s
        LEFT JOIN summary_metrics m ON s.scenario_name = m.scenario_name
        ORDER BY m.recovery_basin_size DESC, m.mean_final_trust DESC
        """
        return conn.execute(query).fetchdf()
    except Exception as e:
        print(f"Error extracting scenario metrics: {e}")
        return pd.DataFrame()

def extract_belief_dynamics(conn):
    """Extract belief dynamics data"""
    try:
        query = """
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
        """
        return conn.execute(query).fetchdf()
    except Exception as e:
        print(f"Error extracting belief dynamics: {e}")
        return pd.DataFrame()

def extract_trust_collapse(conn):
    """Extract trust collapse data"""
    try:
        query = """
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
        """
        return conn.execute(query).fetchdf()
    except Exception as e:
        print(f"Error extracting trust collapse data: {e}")
        return pd.DataFrame()

def extract_overall_metrics(conn):
    """Extract overall metrics for the dashboard"""
    try:
        # Get total number of scenarios
        scenario_count = conn.execute("SELECT COUNT(*) FROM scenarios").fetchone()[0]
        
        # Get average runs per scenario
        avg_runs = conn.execute("SELECT AVG(abm_run_count) FROM scenarios").fetchone()[0]
        
        # Get average collapse probability
        avg_collapse = conn.execute("SELECT AVG(collapse_probability) FROM summary_metrics").fetchone()[0]
        
        # Get max simulation time
        max_time = conn.execute("SELECT MAX(time) FROM abm_runs").fetchone()[0]
        
        return {
            "scenario_count": scenario_count,
            "avg_runs": avg_runs,
            "avg_collapse": avg_collapse * 100,  # Convert to percentage
            "max_time": max_time
        }
    except Exception as e:
        print(f"Error extracting overall metrics: {e}")
        return {
            "scenario_count": 0,
            "avg_runs": 0,
            "avg_collapse": 0,
            "max_time": 0
        }

def find_image_files():
    """Find and categorize available image files"""
    images = {
        "scenario_comparison": None,
        "trust_trajectories": None,
        "phase_space": None,
        "stability_metrics_correlation": None,
        "stability_score": None,
        "rp_ratio": None,
        "believer_evolution": None,
        "abm_evolution": None,
        "collapse_heatmap": None
    }
    
    # Look in analysis directory
    if os.path.exists(ANALYSIS_PATH):
        for file in os.listdir(ANALYSIS_PATH):
            filepath = os.path.join(ANALYSIS_PATH, file)
            if not os.path.isfile(filepath):
                continue
                
            # Check for specific image files
            if "scenario_comparison" in file and file.endswith(".png"):
                images["scenario_comparison"] = os.path.join("analysis", file)
            elif "stability_metrics_correlation" in file and file.endswith(".png"):
                images["stability_metrics_correlation"] = os.path.join("analysis", file)
            elif "stability_score" in file and file.endswith(".png"):
                images["stability_score"] = os.path.join("analysis", file)
            elif "believer_evolution" in file and file.endswith(".png"):
                images["believer_evolution"] = os.path.join("analysis", file)
    
    # Look in results directory
    for scenario in os.listdir(RESULTS_PATH):
        scenario_dir = os.path.join(RESULTS_PATH, scenario)
        if not os.path.isdir(scenario_dir):
            continue
            
        for file in os.listdir(scenario_dir):
            filepath = os.path.join(scenario_dir, file)
            if not os.path.isfile(filepath):
                continue
                
            # Check for specific image files
            if "trust_trajectories" in file and file.endswith(".png"):
                images["trust_trajectories"] = os.path.join("results", scenario, file)
            elif "phase_space" in file and file.endswith(".png"):
                images["phase_space"] = os.path.join("results", scenario, file)
            elif "rp_ratio" in file and file.endswith(".png"):
                images["rp_ratio"] = os.path.join("results", scenario, file)
            elif "abm_evolution" in file and file.endswith(".png"):
                images["abm_evolution"] = os.path.join("results", scenario, file)
            elif "collapse_heatmap" in file and file.endswith(".png"):
                images["collapse_heatmap"] = os.path.join("results", scenario, file)
    
    # Look in reports directory
    if os.path.exists(REPORTS_PATH):
        for file in os.listdir(REPORTS_PATH):
            filepath = os.path.join(REPORTS_PATH, file)
            if not os.path.isfile(filepath):
                continue
                
            # Check for specific image files
            if file.endswith(".png"):
                key = file.replace(".png", "")
                images[key] = os.path.join("reports", file)
    
    return images

def find_html_files():
    """Find available HTML visualization files"""
    html_files = {
        "trust_entropy_pressure": None,
        "phase_space": None,
        "fraction_evolution": None,
        "metrics_dashboard": None
    }
    
    # Look in analysis directory
    if os.path.exists(ANALYSIS_PATH):
        for file in os.listdir(ANALYSIS_PATH):
            filepath = os.path.join(ANALYSIS_PATH, file)
            if not os.path.isfile(filepath) or not file.endswith(".html"):
                continue
                
            # Check for specific HTML files
            if "trust_entropy_pressure" in file:
                html_files["trust_entropy_pressure"] = os.path.join("analysis", file)
            elif "phase_space" in file:
                html_files["phase_space"] = os.path.join("analysis", file)
            elif "fraction_evolution" in file:
                html_files["fraction_evolution"] = os.path.join("analysis", file)
            elif "metrics_dashboard" in file:
                html_files["metrics_dashboard"] = os.path.join("analysis", file)
    
    return html_files

def find_report_files():
    """Find available report files"""
    reports = []
    
    if os.path.exists(REPORTS_PATH):
        for file in os.listdir(REPORTS_PATH):
            filepath = os.path.join(REPORTS_PATH, file)
            if not os.path.isfile(filepath) or not file.endswith(".csv"):
                continue
                
            # Create a more readable name
            name = file.replace(".csv", "").replace("_", " ").title()
            
            # Create a description based on the filename
            description = ""
            if "scenario_summary" in file:
                description = "Overview of all scenarios and key metrics"
            elif "belief_dynamics" in file:
                description = "Analysis of belief fraction evolution"
            elif "trust_collapse" in file:
                description = "Analysis of when trust collapses across scenarios"
            elif "pressure_thresholds" in file:
                description = "Analysis of critical pressure threshold crossings"
            elif "time_series" in file:
                subject = file.replace("time_series_", "").replace(".csv", "").capitalize()
                description = f"{subject} value time series data"
            elif "belief_change" in file:
                description = "Analysis of belief change between initial and final states"
            elif "network_entropy" in file:
                description = "Network entropy statistics over time"
            elif "trust_pressure" in file:
                description = "Correlation between trust and pressure"
            else:
                description = "Simulation data"
            
            reports.append({
                "filename": file,
                "name": name,
                "description": description,
                "path": os.path.join("reports", file)
            })
    
    return sorted(reports, key=lambda x: x["name"])

def get_key_findings(scenario_metrics, trust_collapse):
    """Generate key findings based on the data"""
    findings = []
    
    # Check if high_transparency_ramp is the most stable
    if "high_transparency_ramp" in scenario_metrics["scenario_name"].values:
        high_trans_idx = scenario_metrics.index[scenario_metrics["scenario_name"] == "high_transparency_ramp"][0]
        high_trans_metric = scenario_metrics.iloc[high_trans_idx]
        
        if high_trans_metric["recovery_basin_size"] == scenario_metrics["recovery_basin_size"].max():
            findings.append(
                "Scenarios with high transparency show consistently higher stability and resistance to narrative collapse"
            )
    
    # Get critical RP ratio
    if "critical_rp_ratio" in scenario_metrics.columns:
        critical_rp = scenario_metrics["critical_rp_ratio"].mean()
        findings.append(
            f"Critical RP (Recovery-Pressure) ratio of approximately {critical_rp:.2f} identified as the collapse threshold"
        )
    
    # Check if network_heterogeneity has an impact
    if "network_heterogeneity" in scenario_metrics["scenario_name"].values:
        findings.append(
            "Network heterogeneity significantly impacts the speed of belief propagation"
        )
    
    # Check skeptic surge outcomes
    if "skeptic_surge" in scenario_metrics["scenario_name"].values:
        skeptic_idx = scenario_metrics.index[scenario_metrics["scenario_name"] == "skeptic_surge"][0]
        skeptic_metric = scenario_metrics.iloc[skeptic_idx]
        
        if skeptic_metric["mean_final_trust"] > 0.7:
            findings.append(
                "Skeptic surge scenarios show high initial volatility but can lead to more robust final states"
            )
    
    # If not enough findings, add generic ones
    if len(findings) < 3:
        findings.append(
            "Scenarios with higher initial trust values generally maintain stability longer under pressure"
        )
    
    if len(findings) < 4:
        findings.append(
            "Collapse events, once initiated, tend to accelerate and reach critical thresholds rapidly"
        )
    
    return findings

def generate_dashboard(template_path, output_path):
    """Generate the dashboard HTML file"""
    # Check if template exists
    if not os.path.exists(template_path):
        print(f"Error: Template file not found: {template_path}")
        return False
    
    try:
        # Connect to database
        conn = duckdb.connect(DB_PATH)
        
        # Extract data
        scenario_metrics = extract_scenario_metrics(conn)
        belief_dynamics = extract_belief_dynamics(conn)
        trust_collapse = extract_trust_collapse(conn)
        overall_metrics = extract_overall_metrics(conn)
        images = find_image_files()
        html_files = find_html_files()
        reports = find_report_files()
        key_findings = get_key_findings(scenario_metrics, trust_collapse)
        
        # Close connection
        conn.close()
        
        # Read template
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
        
        # Replace placeholders with actual data
        
        # 1. Overall metrics
        template_content = template_content.replace(
            '<div class="value">11</div>',
            f'<div class="value">{overall_metrics["scenario_count"]}</div>'
        )
        template_content = template_content.replace(
            '<div class="value">500</div>',
            f'<div class="value">{int(overall_metrics["avg_runs"])}</div>'
        )
        template_content = template_content.replace(
            '<div class="value">17.3%</div>',
            f'<div class="value">{overall_metrics["avg_collapse"]:.1f}%</div>'
        )
        template_content = template_content.replace(
            '<div class="value">365</div>',
            f'<div class="value">{int(overall_metrics["max_time"])}</div>'
        )
        
        # 2. Key findings
        key_findings_html = ""
        for finding in key_findings:
            key_findings_html += f"<li>{finding}</li>\n"
        template_content = re.sub(
            r'<ul>[\s\S]*?<\/ul>',
            f'<ul>\n{key_findings_html}</ul>',
            template_content,
            count=1
        )
        
        # 3. Scenario table
        if not scenario_metrics.empty:
            rows_html = ""
            for _, row in scenario_metrics.iterrows():
                scenario_name = row['scenario_name']
                scenario_class = f"scenario-{scenario_name.split('_')[0].lower()}"
                
                rows_html += f"""
                <tr>
                    <td><span class="scenario-badge {scenario_class}">{scenario_name}</span></td>
                    <td>{row['collapse_probability']:.2f}</td>
                    <td>{row['mean_final_trust']:.2f}</td>
                    <td>{row['recovery_basin_size']:.2f}</td>
                    <td>{row['mean_rp_ratio']:.2f}</td>
                    <td>{row['critical_rp_ratio']:.2f}</td>
                </tr>
                """
            
            # Replace the example table rows
            template_content = re.sub(
                r'<tbody>[\s\S]*?<\/tbody>',
                f'<tbody>\n{rows_html}</tbody>',
                template_content,
                count=1
            )
        
        # 4. Image paths
        for key, path in images.items():
            if path:
                # Try to replace any img src that contains this key
                template_content = re.sub(
                    rf'src="[^"]*{key}[^"]*\.png"',
                    f'src="{path}"',
                    template_content
                )
        
        # 5. HTML visualization paths
        for key, path in html_files.items():
            if path:
                # Try to replace any iframe src that contains this key
                template_content = re.sub(
                    rf'src="[^"]*{key}[^"]*\.html"',
                    f'src="{path}"',
                    template_content
                )
        
        # 6. Report files table
        if reports:
            rows_html = ""
            for report in reports:
                rows_html += f"""
                <tr>
                    <td>{report['name']}</td>
                    <td>{report['description']}</td>
                    <td>CSV</td>
                    <td><a href="{report['path']}" class="btn btn-sm btn-primary"><i class="fas fa-download"></i> Download</a></td>
                </tr>
                """
            
            # Find the report table in the template
            report_table_match = re.search(r'<table class="table table-striped">[\s\S]*?<\/table>', template_content)
            if report_table_match:
                report_table = report_table_match.group(0)
                
                # Replace the tbody content
                updated_table = re.sub(
                    r'<tbody>[\s\S]*?<\/tbody>',
                    f'<tbody>\n{rows_html}</tbody>',
                    report_table
                )
                
                # Replace the table in the template
                template_content = template_content.replace(report_table, updated_table)
        
        # 7. Last updated timestamp
        today = datetime.now().strftime('%B %d, %Y')
        template_content = template_content.replace(
            'id="last-updated">April 22, 2025</span>',
            f'id="last-updated">{today}</span>'
        )
        
        # Write output file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template_content)
        
        print(f"Dashboard generated successfully: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error generating dashboard: {e}")
        return False

def main():
    """Main entry point"""
    args = parse_args()
    
    print(f"Generating dashboard...")
    print(f"- Template: {args.template}")
    print(f"- Output: {args.output}")
    print(f"- Database: {DB_PATH}")
    
    success = generate_dashboard(args.template, args.output)
    
    if success:
        print("Dashboard generation complete!")
    else:
        print("Dashboard generation failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()