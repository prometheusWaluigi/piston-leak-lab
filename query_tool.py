#!/usr/bin/env python3
"""
PistonLeakLab - DuckDB Interactive Query Tool

This script provides an interactive interface for querying the 
Piston Leak Lab DuckDB database and generating visualizations.

Usage:
    python query_tool.py
"""

import os
import sys
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from IPython.display import display, HTML

# Path to the database
DB_PATH = r"d:\dev\piston-leak-lab\piston_leak_results.duckdb"

class PistonLeakQueryTool:
    """Interactive query tool for the Piston Leak Lab database"""
    
    def __init__(self):
        """Initialize the query tool"""
        self.conn = None
        self.last_result = None
        self.output_dir = r"d:\dev\piston-leak-lab\query_results"
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def connect(self):
        """Connect to the database"""
        if not os.path.exists(DB_PATH):
            print(f"Error: Database not found at {DB_PATH}")
            print("Please run the duckdb_import.py script first.")
            sys.exit(1)
        
        try:
            self.conn = duckdb.connect(DB_PATH)
            print(f"Connected to database: {DB_PATH}")
            return True
        except Exception as e:
            print(f"Error connecting to database: {e}")
            return False
    
    def list_tables(self):
        """List all tables in the database"""
        if not self.conn:
            print("Not connected to database. Call connect() first.")
            return
        
        tables = self.conn.execute("SHOW TABLES").fetchall()
        print("Available tables:")
        for i, (table,) in enumerate(tables):
            print(f"  {i+1}. {table}")
    
    def show_schema(self, table_name):
        """Show the schema for a table"""
        if not self.conn:
            print("Not connected to database. Call connect() first.")
            return
        
        try:
            schema = self.conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()
            print(f"Schema for table '{table_name}':")
            for col in schema:
                print(f"  {col[1]} ({col[2]})")
        except Exception as e:
            print(f"Error getting schema: {e}")
    
    def execute_query(self, query):
        """Execute a SQL query"""
        if not self.conn:
            print("Not connected to database. Call connect() first.")
            return None
        
        try:
            result = self.conn.execute(query).fetchdf()
            self.last_result = result
            print(f"Query returned {len(result)} rows.")
            if len(result) > 0:
                display(result.head(10))
                if len(result) > 10:
                    print("... (showing first 10 rows)")
            return result
        except Exception as e:
            print(f"Error executing query: {e}")
            return None
    
    def save_result(self, filename=None):
        """Save the last query result to a CSV file"""
        if self.last_result is None:
            print("No query result to save. Execute a query first.")
            return
        
        if filename is None:
            filename = "query_result.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        self.last_result.to_csv(filepath, index=False)
        print(f"Result saved to {filepath}")
    
    def plot_bar(self, x_col, y_col, title=None, color_col=None):
        """Create a bar chart from the last query result"""
        if self.last_result is None:
            print("No query result to plot. Execute a query first.")
            return
        
        if x_col not in self.last_result.columns or y_col not in self.last_result.columns:
            print(f"Columns {x_col} or {y_col} not found in result.")
            return
        
        if color_col and color_col not in self.last_result.columns:
            print(f"Color column {color_col} not found in result.")
            color_col = None
        
        if title is None:
            title = f"{y_col} by {x_col}"
        
        plt.figure(figsize=(12, 6))
        if color_col:
            ax = sns.barplot(data=self.last_result, x=x_col, y=y_col, hue=color_col)
        else:
            ax = sns.barplot(data=self.last_result, x=x_col, y=y_col)
        
        plt.title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the figure
        filepath = os.path.join(self.output_dir, f"bar_{x_col}_{y_col}.png")
        plt.savefig(filepath, dpi=300)
        print(f"Plot saved to {filepath}")
        plt.show()
    
    def plot_line(self, x_col, y_col, group_col=None, title=None):
        """Create a line chart from the last query result"""
        if self.last_result is None:
            print("No query result to plot. Execute a query first.")
            return
        
        if x_col not in self.last_result.columns or y_col not in self.last_result.columns:
            print(f"Columns {x_col} or {y_col} not found in result.")
            return
        
        if group_col and group_col not in self.last_result.columns:
            print(f"Group column {group_col} not found in result.")
            group_col = None
        
        if title is None:
            title = f"{y_col} over {x_col}"
        
        plt.figure(figsize=(12, 6))
        if group_col:
            ax = sns.lineplot(data=self.last_result, x=x_col, y=y_col, hue=group_col)
        else:
            ax = sns.lineplot(data=self.last_result, x=x_col, y=y_col)
        
        plt.title(title)
        plt.tight_layout()
        
        # Save the figure
        filepath = os.path.join(self.output_dir, f"line_{x_col}_{y_col}.png")
        plt.savefig(filepath, dpi=300)
        print(f"Plot saved to {filepath}")
        plt.show()
    
    def plot_interactive(self, x_col, y_col, color_col=None, size_col=None, title=None, 
                         plot_type='scatter'):
        """Create an interactive plotly visualization"""
        if self.last_result is None:
            print("No query result to plot. Execute a query first.")
            return
        
        if x_col not in self.last_result.columns or y_col not in self.last_result.columns:
            print(f"Columns {x_col} or {y_col} not found in result.")
            return
        
        if color_col and color_col not in self.last_result.columns:
            print(f"Color column {color_col} not found in result.")
            color_col = None
        
        if size_col and size_col not in self.last_result.columns:
            print(f"Size column {size_col} not found in result.")
            size_col = None
        
        if title is None:
            title = f"{y_col} vs {x_col}"
        
        if plot_type == 'scatter':
            fig = px.scatter(
                self.last_result, x=x_col, y=y_col, 
                color=color_col, size=size_col,
                title=title
            )
        elif plot_type == 'line':
            fig = px.line(
                self.last_result, x=x_col, y=y_col, 
                color=color_col,
                title=title
            )
        elif plot_type == 'bar':
            fig = px.bar(
                self.last_result, x=x_col, y=y_col,
                color=color_col,
                title=title
            )
        else:
            print(f"Unknown plot type: {plot_type}")
            return
        
        # Save the figure
        filepath = os.path.join(self.output_dir, f"interactive_{plot_type}_{x_col}_{y_col}.html")
        pio.write_html(fig, filepath)
        print(f"Interactive plot saved to {filepath}")
        
        # Display the plot
        fig.show()
    
    def compare_scenarios(self, metric_column, agg_func='AVG'):
        """Compare scenarios based on a specific metric"""
        if not self.conn:
            print("Not connected to database. Call connect() first.")
            return
        
        # Check if the column exists in the abm_runs table
        abm_schema = self.conn.execute("PRAGMA table_info('abm_runs')").fetchdf()['name'].tolist()
        
        if metric_column in abm_schema:
            # Query the ABM data
            query = f"""
            WITH final_state AS (
                SELECT 
                    scenario_name, 
                    run_id,
                    LAST_VALUE({metric_column}) OVER (
                        PARTITION BY scenario_name, run_id 
                        ORDER BY time 
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                    ) AS final_value
                FROM abm_runs
            ),
            scenario_avg AS (
                SELECT 
                    scenario_name,
                    {agg_func}(final_value) as {agg_func.lower()}_final_{metric_column}
                FROM final_state
                GROUP BY scenario_name
            )
            SELECT * FROM scenario_avg
            ORDER BY {agg_func.lower()}_final_{metric_column} DESC
            """
        else:
            # Check if the column exists in the ode_runs table
            ode_schema = self.conn.execute("PRAGMA table_info('ode_runs')").fetchdf()['name'].tolist()
            
            if metric_column in ode_schema:
                # Query the ODE data
                query = f"""
                WITH final_state AS (
                    SELECT 
                        scenario_name, 
                        run_id,
                        LAST_VALUE({metric_column}) OVER (
                            PARTITION BY scenario_name, run_id 
                            ORDER BY time 
                            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        ) AS final_value
                    FROM ode_runs
                ),
                scenario_avg AS (
                    SELECT 
                        scenario_name,
                        {agg_func}(final_value) as {agg_func.lower()}_final_{metric_column}
                    FROM final_state
                    GROUP BY scenario_name
                )
                SELECT * FROM scenario_avg
                ORDER BY {agg_func.lower()}_final_{metric_column} DESC
                """
            else:
                # Check if the column exists in the summary_metrics table
                summary_schema = self.conn.execute("PRAGMA table_info('summary_metrics')").fetchdf()['name'].tolist()
                
                if metric_column in summary_schema:
                    # Query the summary metrics data
                    query = f"""
                    SELECT 
                        scenario_name,
                        {metric_column}
                    FROM summary_metrics
                    ORDER BY {metric_column} DESC
                    """
                else:
                    print(f"Error: Column '{metric_column}' not found in any table.")
                    return
        
        # Execute the query
        result = self.execute_query(query)
        
        if result is not None and len(result) > 0:
            # Plot the result
            self.plot_bar(
                x_col='scenario_name', 
                y_col=result.columns[1], 
                title=f"Scenario Comparison - {metric_column}"
            )
            
            # Also create an interactive version
            self.plot_interactive(
                x_col='scenario_name',
                y_col=result.columns[1],
                plot_type='bar',
                title=f"Scenario Comparison - {metric_column}"
            )
    
    def analyze_time_series(self, metric_column, scenarios=None):
        """Analyze the time series evolution of a metric"""
        if not self.conn:
            print("Not connected to database. Call connect() first.")
            return
        
        # Check if the column exists in the abm_runs table
        abm_schema = self.conn.execute("PRAGMA table_info('abm_runs')").fetchdf()['name'].tolist()
        
        # Build scenario filter
        scenario_filter = ""
        if scenarios:
            scenario_list = "', '".join(scenarios)
            scenario_filter = f"WHERE scenario_name IN ('{scenario_list}')"
        
        if metric_column in abm_schema:
            # Query the ABM data
            query = f"""
            SELECT 
                scenario_name,
                time,
                AVG({metric_column}) as mean_{metric_column}
            FROM abm_runs
            {scenario_filter}
            GROUP BY scenario_name, time
            ORDER BY scenario_name, time
            """
        else:
            # Check if the column exists in the ode_runs table
            ode_schema = self.conn.execute("PRAGMA table_info('ode_runs')").fetchdf()['name'].tolist()
            
            if metric_column in ode_schema:
                # Query the ODE data
                query = f"""
                SELECT 
                    scenario_name,
                    time,
                    AVG({metric_column}) as mean_{metric_column}
                FROM ode_runs
                {scenario_filter}
                GROUP BY scenario_name, time
                ORDER BY scenario_name, time
                """
            else:
                print(f"Error: Column '{metric_column}' not found in abm_runs or ode_runs tables.")
                return
        
        # Execute the query
        result = self.execute_query(query)
        
        if result is not None and len(result) > 0:
            # Plot the result
            self.plot_line(
                x_col='time', 
                y_col=f'mean_{metric_column}', 
                group_col='scenario_name',
                title=f"Time Series - {metric_column}"
            )
            
            # Also create an interactive version
            self.plot_interactive(
                x_col='time',
                y_col=f'mean_{metric_column}',
                color_col='scenario_name',
                plot_type='line',
                title=f"Time Series - {metric_column}"
            )
    
    def custom_queries(self):
        """Present some useful custom queries"""
        queries = {
            "1": {
                "name": "Top scenarios by stability",
                "query": """
                SELECT 
                    scenario_name, 
                    mean_final_trust,
                    mean_rp_ratio,
                    recovery_basin_size
                FROM summary_metrics
                ORDER BY mean_final_trust DESC, recovery_basin_size DESC
                """
            },
            "2": {
                "name": "Final population state by scenario",
                "query": """
                WITH final_state AS (
                    SELECT 
                        scenario_name, 
                        AVG(LAST_VALUE(believer_fraction) OVER (
                            PARTITION BY scenario_name, run_id 
                            ORDER BY time 
                            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        )) as final_believer,
                        AVG(LAST_VALUE(skeptic_fraction) OVER (
                            PARTITION BY scenario_name, run_id 
                            ORDER BY time 
                            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        )) as final_skeptic,
                        AVG(LAST_VALUE(agnostic_fraction) OVER (
                            PARTITION BY scenario_name, run_id 
                            ORDER BY time 
                            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        )) as final_agnostic
                    FROM abm_runs
                    GROUP BY scenario_name
                )
                SELECT * FROM final_state
                ORDER BY final_believer DESC
                """
            },
            "3": {
                "name": "Recovery-Pressure Ratio vs Collapse Probability",
                "query": """
                SELECT 
                    scenario_name,
                    mean_rp_ratio,
                    collapse_probability
                FROM summary_metrics
                ORDER BY mean_rp_ratio DESC
                """
            },
            "4": {
                "name": "Most impactful scenarios (belief change)",
                "query": """
                WITH initial_state AS (
                    SELECT 
                        scenario_name, 
                        AVG(FIRST_VALUE(believer_fraction) OVER (
                            PARTITION BY scenario_name, run_id 
                            ORDER BY time 
                            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        )) as initial_believer
                    FROM abm_runs
                    GROUP BY scenario_name
                ),
                final_state AS (
                    SELECT 
                        scenario_name, 
                        AVG(LAST_VALUE(believer_fraction) OVER (
                            PARTITION BY scenario_name, run_id 
                            ORDER BY time 
                            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        )) as final_believer
                    FROM abm_runs
                    GROUP BY scenario_name
                )
                SELECT 
                    i.scenario_name,
                    i.initial_believer,
                    f.final_believer,
                    (f.final_believer - i.initial_believer) as belief_change,
                    ABS(f.final_believer - i.initial_believer) as belief_change_magnitude
                FROM initial_state i
                JOIN final_state f ON i.scenario_name = f.scenario_name
                ORDER BY belief_change_magnitude DESC
                """
            },
            "5": {
                "name": "Critical threshold crossing",
                "query": """
                WITH threshold_cross AS (
                    SELECT 
                        scenario_name,
                        run_id,
                        MIN(time) as threshold_time
                    FROM ode_runs
                    WHERE pressure > 0.5  -- Critical threshold
                    GROUP BY scenario_name, run_id
                ),
                scenario_avg AS (
                    SELECT 
                        scenario_name,
                        AVG(threshold_time) as mean_threshold_time,
                        STDDEV(threshold_time) as std_threshold_time,
                        COUNT(*) as threshold_cross_count
                    FROM threshold_cross
                    GROUP BY scenario_name
                )
                SELECT 
                    s.scenario_name,
                    COALESCE(t.mean_threshold_time, -1) as mean_threshold_time,
                    COALESCE(t.std_threshold_time, 0) as std_threshold_time,
                    COALESCE(t.threshold_cross_count, 0) as threshold_cross_count,
                    t.threshold_cross_count * 1.0 / s.abm_run_count as threshold_cross_ratio
                FROM scenarios s
                LEFT JOIN scenario_avg t ON s.scenario_name = t.scenario_name
                ORDER BY threshold_cross_ratio DESC
                """
            },
        }
        
        print("Available custom queries:")
        for key, query_info in queries.items():
            print(f"  {key}. {query_info['name']}")
        
        choice = input("Enter query number (or 'q' to quit): ")
        
        if choice.lower() == 'q':
            return
        
        if choice in queries:
            print(f"\nExecuting: {queries[choice]['name']}")
            result = self.execute_query(queries[choice]['query'])
            
            if result is not None and len(result) > 0:
                # For some specific queries, create visualizations
                if choice == "1":
                    self.plot_bar(
                        x_col='scenario_name', 
                        y_col='mean_final_trust', 
                        title="Scenarios by Mean Final Trust"
                    )
                elif choice == "2":
                    # Reshape for plotting
                    plot_df = pd.melt(
                        result, 
                        id_vars=['scenario_name'], 
                        value_vars=['final_believer', 'final_skeptic', 'final_agnostic'],
                        var_name='population_type', 
                        value_name='fraction'
                    )
                    
                    # Store for plotting
                    self.last_result = plot_df
                    
                    self.plot_bar(
                        x_col='scenario_name', 
                        y_col='fraction', 
                        color_col='population_type',
                        title="Final Population State by Scenario"
                    )
                elif choice == "3":
                    self.plot_interactive(
                        x_col='mean_rp_ratio',
                        y_col='collapse_probability',
                        color_col='scenario_name',
                        plot_type='scatter',
                        title="RP Ratio vs Collapse Probability"
                    )
                elif choice == "4":
                    self.plot_bar(
                        x_col='scenario_name', 
                        y_col='belief_change', 
                        title="Belief Change by Scenario"
                    )
        else:
            print(f"Invalid choice: {choice}")
    
    def run_interactive(self):
        """Run the interactive query tool"""
        if not self.connect():
            return
        
        while True:
            print("\n" + "="*40)
            print("Piston Leak Lab DuckDB Query Tool")
            print("="*40)
            print("1. List tables")
            print("2. Show table schema")
            print("3. Execute custom SQL query")
            print("4. Compare scenarios")
            print("5. Analyze time series")
            print("6. Run predefined custom queries")
            print("7. Exit")
            
            choice = input("\nEnter choice (1-7): ")
            
            if choice == "1":
                self.list_tables()
            
            elif choice == "2":
                table = input("Enter table name: ")
                self.show_schema(table)
            
            elif choice == "3":
                print("Enter SQL query (type 'done' on a new line when finished):")
                lines = []
                while True:
                    line = input()
                    if line.lower() == 'done':
                        break
                    lines.append(line)
                
                query = " ".join(lines)
                if query:
                    result = self.execute_query(query)
                    
                    if result is not None and len(result) > 0:
                        # Offer to save or visualize
                        viz = input("Save result to CSV? (y/n): ")
                        if viz.lower() == 'y':
                            filename = input("Enter filename (or press Enter for default): ")
                            if filename:
                                self.save_result(filename)
                            else:
                                self.save_result()
                        
                        # Ask if user wants to visualize
                        viz = input("Create visualization? (y/n): ")
                        if viz.lower() == 'y':
                            # Ask for column names
                            print("Available columns:", ", ".join(result.columns))
                            x_col = input("Enter x-axis column: ")
                            y_col = input("Enter y-axis column: ")
                            
                            if x_col in result.columns and y_col in result.columns:
                                plot_type = input("Plot type (bar/line/scatter): ")
                                if plot_type in ['bar', 'line', 'scatter']:
                                    color_col = input("Color column (optional, press Enter to skip): ")
                                    if color_col and color_col not in result.columns:
                                        print(f"Warning: Column '{color_col}' not found. Ignoring.")
                                        color_col = None
                                    
                                    self.plot_interactive(
                                        x_col=x_col,
                                        y_col=y_col,
                                        color_col=color_col if color_col else None,
                                        plot_type=plot_type
                                    )
                                else:
                                    print(f"Invalid plot type: {plot_type}")
                            else:
                                print("Invalid column names.")
            
            elif choice == "4":
                print("Available tables to extract columns from:")
                print("- abm_runs (e.g., believer_fraction, skeptic_fraction, network_entropy)")
                print("- ode_runs (e.g., trust, entropy, pressure)")
                print("- summary_metrics (e.g., collapse_probability, mean_rp_ratio)")
                
                metric = input("Enter metric column to compare across scenarios: ")
                agg = input("Enter aggregation function (AVG, MAX, MIN): ")
                
                if not agg or agg.upper() not in ['AVG', 'MAX', 'MIN']:
                    agg = 'AVG'
                
                self.compare_scenarios(metric, agg.upper())
            
            elif choice == "5":
                print("Available tables to extract columns from:")
                print("- abm_runs (e.g., believer_fraction, skeptic_fraction, network_entropy)")
                print("- ode_runs (e.g., trust, entropy, pressure)")
                
                metric = input("Enter metric column to analyze over time: ")
                
                # Get list of scenarios
                scenarios = self.conn.execute(
                    "SELECT DISTINCT scenario_name FROM scenarios ORDER BY scenario_name"
                ).fetchnumpy()['scenario_name']
                
                print("Available scenarios:")
                for i, scenario in enumerate(scenarios):
                    print(f"  {i+1}. {scenario}")
                
                scenario_input = input("Enter scenario numbers to include (comma-separated, or 'all'): ")
                
                selected_scenarios = None
                if scenario_input.lower() != 'all':
                    try:
                        indices = [int(idx.strip()) - 1 for idx in scenario_input.split(',')]
                        selected_scenarios = [scenarios[idx] for idx in indices if 0 <= idx < len(scenarios)]
                    except:
                        print("Invalid input. Using all scenarios.")
                
                self.analyze_time_series(metric, selected_scenarios)
            
            elif choice == "6":
                self.custom_queries()
            
            elif choice == "7":
                print("Exiting...")
                break
            
            else:
                print(f"Invalid choice: {choice}")
    
    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    tool = PistonLeakQueryTool()
    try:
        tool.run_interactive()
    except KeyboardInterrupt:
        print("\nOperation interrupted by user.")
    finally:
        tool.close()