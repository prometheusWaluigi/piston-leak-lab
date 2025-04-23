#!/usr/bin/env python3
"""
PistonLeakLab - DuckDB Analysis Script

This script performs analysis on the DuckDB database containing
the Piston Leak Lab simulation results. It generates various queries,
visualizations, and comparative analyses.

Usage:
    python analyze_results.py

Requirements:
    - DuckDB database created by the duckdb_import.py script
    - pandas, matplotlib, seaborn, plotly
"""

import os
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio

# Set the output path
OUTPUT_PATH = r"d:\dev\piston-leak-lab\analysis"
DB_PATH = r"d:\dev\piston-leak-lab\piston_leak_results.duckdb"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)

def connect_db():
    """Connect to the DuckDB database"""
    return duckdb.connect(DB_PATH)

def get_scenario_names(conn):
    """Get a list of all scenario names"""
    return conn.execute(
        "SELECT DISTINCT scenario_name FROM scenarios ORDER BY scenario_name"
    ).fetchnumpy()['scenario_name']

def scenario_comparison(conn):
    """Generate comparative analysis between scenarios"""
    # Get summary metrics for each scenario
    df = conn.execute("""
        SELECT * FROM scenario_comparison
    """).fetchdf()
    
    # Generate comparison plots
    plt.figure(figsize=(12, 8))
    
    # Trust vs. Collapse probability
    plt.subplot(2, 2, 1)
    sns.scatterplot(data=df, x='mean_final_trust', y='collapse_probability', 
                   hue='scenario_name', s=100)
    plt.title('Trust vs. Collapse Probability')
    plt.xlabel('Mean Final Trust')
    plt.ylabel('Collapse Probability')
    
    # Basin stability
    plt.subplot(2, 2, 2)
    df_melt = pd.melt(df, id_vars=['scenario_name'], 
                      value_vars=['recovery_basin_size', 'collapse_basin_size'],
                      var_name='Basin Type', value_name='Size')
    sns.barplot(data=df_melt, x='scenario_name', y='Size', hue='Basin Type')
    plt.title('Basin Stability Analysis')
    plt.xlabel('Scenario')
    plt.ylabel('Basin Size')
    plt.xticks(rotation=45)
    
    # RP Ratio
    plt.subplot(2, 2, 3)
    sns.barplot(data=df, x='scenario_name', y='mean_rp_ratio')
    plt.axhline(y=df['critical_rp_ratio'].mean(), color='r', linestyle='--', 
                label=f'Critical RP Ratio (~{df["critical_rp_ratio"].mean():.2f})')
    plt.title('Recovery-Pressure Ratio by Scenario')
    plt.xlabel('Scenario')
    plt.ylabel('Mean RP Ratio')
    plt.xticks(rotation=45)
    plt.legend()
    
    # Collapse time
    plt.subplot(2, 2, 4)
    sns.barplot(data=df, x='scenario_name', y='mean_collapse_time')
    plt.title('Mean Time to Collapse')
    plt.xlabel('Scenario')
    plt.ylabel('Time Steps')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'scenario_comparison.png'), dpi=300)
    
    # Generate a table with key metrics
    results_df = df[[
        'scenario_name', 'collapse_probability', 'mean_final_trust',
        'recovery_basin_size', 'mean_rp_ratio', 'mean_collapse_time'
    ]]
    
    # Save to CSV
    results_df.to_csv(os.path.join(OUTPUT_PATH, 'scenario_comparison.csv'), index=False)
    
    return results_df

def believer_evolution_analysis(conn):
    """Analyze the evolution of believer fractions across scenarios"""
    # Get believer fraction evolution for each scenario
    df = conn.execute("""
        WITH scenario_avg AS (
            SELECT 
                scenario_name,
                time,
                AVG(believer_fraction) as mean_believer,
                STDDEV(believer_fraction) as std_believer
            FROM abm_runs
            GROUP BY scenario_name, time
        )
        SELECT * FROM scenario_avg
        ORDER BY scenario_name, time
    """).fetchdf()
    
    # Create a plot for each scenario
    scenarios = df['scenario_name'].unique()
    
    # Determine number of rows and columns for subplots
    n_scenarios = len(scenarios)
    n_cols = min(3, n_scenarios)
    n_rows = (n_scenarios + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(15, n_rows * 4))
    
    for i, scenario in enumerate(scenarios):
        ax = plt.subplot(n_rows, n_cols, i + 1)
        
        # Get data for this scenario
        scenario_df = df[df['scenario_name'] == scenario]
        
        # Plot mean with standard deviation band
        ax.plot(scenario_df['time'], scenario_df['mean_believer'], 
                label='Mean Believer Fraction')
        ax.fill_between(
            scenario_df['time'],
            scenario_df['mean_believer'] - scenario_df['std_believer'],
            scenario_df['mean_believer'] + scenario_df['std_believer'],
            alpha=0.3, label='±1 Std Dev'
        )
        
        ax.set_title(f'Scenario: {scenario}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Believer Fraction')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'believer_evolution.png'), dpi=300)
    
    # Save the data
    df.to_csv(os.path.join(OUTPUT_PATH, 'believer_evolution.csv'), index=False)
    
    return df

def trust_entropy_pressure_analysis(conn):
    """Analyze the relationship between trust, entropy, and pressure"""
    # Get average ODE values for each scenario and timestep
    df = conn.execute("""
        WITH scenario_avg AS (
            SELECT 
                scenario_name,
                time,
                AVG(trust) as mean_trust,
                AVG(entropy) as mean_entropy,
                AVG(pressure) as mean_pressure
            FROM ode_runs
            GROUP BY scenario_name, time
        )
        SELECT * FROM scenario_avg
        ORDER BY scenario_name, time
    """).fetchdf()
    
    # Create interactive plotly visualizations
    scenarios = df['scenario_name'].unique()
    
    # Create a figure with subplots - one per scenario
    fig = make_subplots(
        rows=len(scenarios), cols=1,
        subplot_titles=[f"Scenario: {s}" for s in scenarios],
        shared_xaxes=True,
        vertical_spacing=0.05
    )
    
    for i, scenario in enumerate(scenarios):
        # Get data for this scenario
        scenario_df = df[df['scenario_name'] == scenario]
        
        # Add trust line
        fig.add_trace(
            go.Scatter(
                x=scenario_df['time'], 
                y=scenario_df['mean_trust'],
                mode='lines',
                name=f'{scenario} - Trust',
                line=dict(color='blue')
            ),
            row=i+1, col=1
        )
        
        # Add entropy line
        fig.add_trace(
            go.Scatter(
                x=scenario_df['time'], 
                y=scenario_df['mean_entropy'],
                mode='lines',
                name=f'{scenario} - Entropy',
                line=dict(color='red')
            ),
            row=i+1, col=1
        )
        
        # Add pressure line
        fig.add_trace(
            go.Scatter(
                x=scenario_df['time'], 
                y=scenario_df['mean_pressure'],
                mode='lines',
                name=f'{scenario} - Pressure',
                line=dict(color='green')
            ),
            row=i+1, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=300 * len(scenarios),
        width=1000,
        title_text="Trust, Entropy, and Pressure Evolution",
        showlegend=True
    )
    
    # Save to HTML
    pio.write_html(fig, os.path.join(OUTPUT_PATH, 'trust_entropy_pressure.html'))
    
    # Phase space analysis - trust vs entropy
    phase_fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=["Phase Space: Trust vs Entropy"],
    )
    
    for scenario in scenarios:
        # Get data for this scenario
        scenario_df = df[df['scenario_name'] == scenario]
        
        # Add phase space trajectory
        phase_fig.add_trace(
            go.Scatter(
                x=scenario_df['mean_entropy'], 
                y=scenario_df['mean_trust'],
                mode='lines+markers',
                name=f'{scenario}',
                hovertext=scenario_df['time'].apply(lambda t: f'Time: {t}')
            )
        )
    
    phase_fig.update_layout(
        height=600,
        width=800,
        title_text="Phase Space Analysis: Trust vs Entropy",
        xaxis_title="Entropy",
        yaxis_title="Trust"
    )
    
    # Save to HTML
    pio.write_html(phase_fig, os.path.join(OUTPUT_PATH, 'phase_space.html'))
    
    # Save the data
    df.to_csv(os.path.join(OUTPUT_PATH, 'trust_entropy_pressure.csv'), index=False)
    
    return df

def extract_stability_metrics(conn):
    """Extract and analyze stability metrics for each scenario"""
    # Get the key stability metrics
    df = conn.execute("""
        SELECT 
            scenario_name,
            collapse_probability,
            mean_rp_ratio,
            critical_rp_ratio,
            recovery_basin_size,
            collapse_basin_size,
            mean_collapse_time
        FROM summary_metrics
    """).fetchdf()
    
    # Create a correlation matrix figure
    corr_columns = [col for col in df.columns if col != 'scenario_name']
    corr_matrix = df[corr_columns].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Stability Metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'stability_metrics_correlation.png'), dpi=300)
    
    # Create a scatter plot matrix
    pd.plotting.scatter_matrix(df[corr_columns], figsize=(12, 12), diagonal='kde')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'stability_metrics_scatter_matrix.png'), dpi=300)
    
    # Save the metrics
    df.to_csv(os.path.join(OUTPUT_PATH, 'stability_metrics.csv'), index=False)
    
    # Calculate a stability score for each scenario
    # The score is a weighted average of several normalized metrics
    # Higher is more stable
    
    # Normalize values between 0 and 1
    df_norm = df.copy()
    
    # For collapse probability, lower is better
    df_norm['norm_collapse_prob'] = 1 - (df['collapse_probability'] - df['collapse_probability'].min()) / \
                                    (df['collapse_probability'].max() - df['collapse_probability'].min() + 1e-10)
    
    # For RP ratio, higher is better
    df_norm['norm_rp_ratio'] = (df['mean_rp_ratio'] - df['mean_rp_ratio'].min()) / \
                               (df['mean_rp_ratio'].max() - df['mean_rp_ratio'].min() + 1e-10)
    
    # For recovery basin size, higher is better
    df_norm['norm_recovery'] = (df['recovery_basin_size'] - df['recovery_basin_size'].min()) / \
                               (df['recovery_basin_size'].max() - df['recovery_basin_size'].min() + 1e-10)
    
    # For collapse time, higher is better
    df_norm['norm_collapse_time'] = (df['mean_collapse_time'] - df['mean_collapse_time'].min()) / \
                                    (df['mean_collapse_time'].max() - df['mean_collapse_time'].min() + 1e-10)
    
    # Compute stability score with weights
    weights = {
        'norm_collapse_prob': 0.35,
        'norm_rp_ratio': 0.25,
        'norm_recovery': 0.25,
        'norm_collapse_time': 0.15
    }
    
    df_norm['stability_score'] = sum(df_norm[metric] * weight for metric, weight in weights.items())
    
    # Sort by stability score
    df_norm = df_norm.sort_values('stability_score', ascending=False)
    
    # Create a bar chart of stability scores
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df_norm, x='scenario_name', y='stability_score')
    plt.title('Narrative Stability Score by Scenario')
    plt.xlabel('Scenario')
    plt.ylabel('Stability Score (higher is better)')
    plt.xticks(rotation=45)
    
    # Add value labels to bars
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{p.get_height():.2f}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'stability_score.png'), dpi=300)
    
    # Save the stability scores
    stability_df = df_norm[['scenario_name', 'stability_score'] + list(weights.keys())]
    stability_df.to_csv(os.path.join(OUTPUT_PATH, 'stability_scores.csv'), index=False)
    
    return stability_df

def fraction_trajectory_comparison(conn):
    """Compare the trajectories of believer, skeptic, and agnostic fractions"""
    # Get average fraction values for each scenario and timestep
    df = conn.execute("""
        WITH scenario_avg AS (
            SELECT 
                scenario_name,
                time,
                AVG(believer_fraction) as mean_believer,
                AVG(skeptic_fraction) as mean_skeptic,
                AVG(agnostic_fraction) as mean_agnostic
            FROM abm_runs
            GROUP BY scenario_name, time
        )
        SELECT * FROM scenario_avg
        ORDER BY scenario_name, time
    """).fetchdf()
    
    # Get list of scenarios
    scenarios = df['scenario_name'].unique()
    
    # Create an interactive plot
    fig = make_subplots(
        rows=len(scenarios), cols=1,
        subplot_titles=[f"Scenario: {s}" for s in scenarios],
        shared_xaxes=True,
        vertical_spacing=0.05
    )
    
    for i, scenario in enumerate(scenarios):
        # Get data for this scenario
        scenario_df = df[df['scenario_name'] == scenario]
        
        # Add believer line
        fig.add_trace(
            go.Scatter(
                x=scenario_df['time'], 
                y=scenario_df['mean_believer'],
                mode='lines',
                name=f'{scenario} - Believer',
                line=dict(color='blue')
            ),
            row=i+1, col=1
        )
        
        # Add skeptic line
        fig.add_trace(
            go.Scatter(
                x=scenario_df['time'], 
                y=scenario_df['mean_skeptic'],
                mode='lines',
                name=f'{scenario} - Skeptic',
                line=dict(color='red')
            ),
            row=i+1, col=1
        )
        
        # Add agnostic line
        fig.add_trace(
            go.Scatter(
                x=scenario_df['time'], 
                y=scenario_df['mean_agnostic'],
                mode='lines',
                name=f'{scenario} - Agnostic',
                line=dict(color='green')
            ),
            row=i+1, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=300 * len(scenarios),
        width=1000,
        title_text="Population Fraction Evolution by Belief Type",
        showlegend=True
    )
    
    # Save to HTML
    pio.write_html(fig, os.path.join(OUTPUT_PATH, 'fraction_evolution.html'))
    
    # Save the data
    df.to_csv(os.path.join(OUTPUT_PATH, 'fraction_evolution.csv'), index=False)
    
    return df

def create_metrics_dashboard(conn):
    """Create a comprehensive metrics dashboard for all scenarios"""
    # Get summary metrics
    summary_df = conn.execute("SELECT * FROM summary_metrics").fetchdf()
    
    # Create a plotly dashboard
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Collapse Probability by Scenario",
            "Recovery/Collapse Basin Size",
            "Mean RP Ratio vs Critical RP Ratio",
            "Mean Time to Collapse"
        ],
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "bar"}]
        ]
    )
    
    # Collapse probability
    fig.add_trace(
        go.Bar(
            x=summary_df['scenario_name'],
            y=summary_df['collapse_probability'],
            name="Collapse Probability"
        ),
        row=1, col=1
    )
    
    # Basin sizes
    scenarios = summary_df['scenario_name'].tolist()
    
    fig.add_trace(
        go.Bar(
            x=scenarios,
            y=summary_df['recovery_basin_size'],
            name="Recovery Basin"
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(
            x=scenarios,
            y=summary_df['collapse_basin_size'],
            name="Collapse Basin"
        ),
        row=1, col=2
    )
    
    # RP ratios
    fig.add_trace(
        go.Bar(
            x=scenarios,
            y=summary_df['mean_rp_ratio'],
            name="Mean RP Ratio"
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=scenarios,
            y=summary_df['critical_rp_ratio'],
            mode="lines+markers",
            name="Critical RP Ratio"
        ),
        row=2, col=1
    )
    
    # Collapse time
    fig.add_trace(
        go.Bar(
            x=scenarios,
            y=summary_df['mean_collapse_time'],
            name="Mean Time to Collapse"
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        width=1200,
        title_text="Piston Leak Lab - Scenario Metrics Dashboard",
        showlegend=True
    )
    
    # Update xaxis properties
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(tickangle=45, row=i, col=j)
    
    # Save to HTML
    pio.write_html(fig, os.path.join(OUTPUT_PATH, 'metrics_dashboard.html'))
    
    return summary_df

def main():
    """Main entry point for analysis"""
    print(f"Connecting to database: {DB_PATH}")
    conn = connect_db()
    
    print("Checking scenarios...")
    scenarios = get_scenario_names(conn)
    print(f"Found {len(scenarios)} scenarios: {', '.join(scenarios)}")
    
    print("\nRunning analyses...")
    
    print("1. Scenario comparison...")
    scenario_df = scenario_comparison(conn)
    print(f"  - Saved to {os.path.join(OUTPUT_PATH, 'scenario_comparison.png')}")
    print(f"  - Saved to {os.path.join(OUTPUT_PATH, 'scenario_comparison.csv')}")
    
    print("2. Believer evolution analysis...")
    believer_df = believer_evolution_analysis(conn)
    print(f"  - Saved to {os.path.join(OUTPUT_PATH, 'believer_evolution.png')}")
    print(f"  - Saved to {os.path.join(OUTPUT_PATH, 'believer_evolution.csv')}")
    
    print("3. Trust-entropy-pressure analysis...")
    tep_df = trust_entropy_pressure_analysis(conn)
    print(f"  - Saved to {os.path.join(OUTPUT_PATH, 'trust_entropy_pressure.html')}")
    print(f"  - Saved to {os.path.join(OUTPUT_PATH, 'phase_space.html')}")
    print(f"  - Saved to {os.path.join(OUTPUT_PATH, 'trust_entropy_pressure.csv')}")
    
    print("4. Stability metrics analysis...")
    stability_df = extract_stability_metrics(conn)
    print(f"  - Saved to {os.path.join(OUTPUT_PATH, 'stability_metrics_correlation.png')}")
    print(f"  - Saved to {os.path.join(OUTPUT_PATH, 'stability_metrics_scatter_matrix.png')}")
    print(f"  - Saved to {os.path.join(OUTPUT_PATH, 'stability_score.png')}")
    print(f"  - Saved to {os.path.join(OUTPUT_PATH, 'stability_metrics.csv')}")
    print(f"  - Saved to {os.path.join(OUTPUT_PATH, 'stability_scores.csv')}")
    
    print("5. Fraction trajectory comparison...")
    fraction_df = fraction_trajectory_comparison(conn)
    print(f"  - Saved to {os.path.join(OUTPUT_PATH, 'fraction_evolution.html')}")
    print(f"  - Saved to {os.path.join(OUTPUT_PATH, 'fraction_evolution.csv')}")
    
    print("6. Creating metrics dashboard...")
    metrics_df = create_metrics_dashboard(conn)
    print(f"  - Saved to {os.path.join(OUTPUT_PATH, 'metrics_dashboard.html')}")
    
    print("\nAll analyses complete!")
    print(f"Results saved to: {OUTPUT_PATH}")
    
    # Provide some insights
    most_stable = stability_df.iloc[0]['scenario_name']
    least_stable = stability_df.iloc[-1]['scenario_name']
    
    print("\nKey Insights:")
    print(f"- Most stable scenario: {most_stable}")
    print(f"- Least stable scenario: {least_stable}")
    
    mean_collapse = scenario_df['collapse_probability'].mean()
    print(f"- Mean collapse probability across scenarios: {mean_collapse:.2f}")
    
    # Get the scenario with highest believer fraction at the end
    final_believers = conn.execute("""
        WITH final_state AS (
            SELECT 
                scenario_name, 
                run_id,
                LAST_VALUE(believer_fraction) OVER (
                    PARTITION BY scenario_name, run_id 
                    ORDER BY time 
                    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                ) AS final_bf
            FROM abm_runs
        ),
        scenario_avg AS (
            SELECT 
                scenario_name,
                AVG(final_bf) as mean_final_bf
            FROM final_state
            GROUP BY scenario_name
        )
        SELECT scenario_name, mean_final_bf
        FROM scenario_avg
        ORDER BY mean_final_bf DESC
        LIMIT 1
    """).fetchone()
    
    print(f"- Scenario with highest final believer fraction: {final_believers[0]} ({final_believers[1]:.2f})")
    
    conn.close()

if __name__ == "__main__":
    main()