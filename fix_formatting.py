#!/usr/bin/env python3
"""
Fix formatting issues in the Piston Leak Lab codebase.

This script applies Black formatting to all Python files in the project.
"""

import subprocess
from pathlib import Path


FILES_TO_FORMAT = [
    "fix_imports.py",
    "fix_linting.py",
    "fix_tests.py",
    "models/core_ode.py",
    "models/abm.py",
    "sims/run_mc.py",
    "sims/visualization/__init__.py",
    "sims/visualization/dashboard.py",
    "sims/visualization/plots.py",
    "tests/test_abm.py",
    "tests/test_core_ode.py"
]


def format_file(file_path):
    """Format a single file with Black."""
    try:
        print(f"Formatting {file_path}...")
        subprocess.run(["black", file_path], check=True)
        print(f"  ✓ Successfully formatted {file_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed to format {file_path}: {e}")
        return False
    except FileNotFoundError:
        print("  ✗ Black is not installed or not in PATH")
        print("    Install with: pip install black")
        return False


def format_all_files():
    """Format all Python files in the project."""
    success = True
    
    for file_path in FILES_TO_FORMAT:
        if not format_file(file_path):
            success = False
    
    return success


def main():
    """Main entry point for the script."""
    print("=== Piston Leak Lab Formatter ===")
    print("Applying Black formatting to all Python files...")
    
    if format_all_files():
        print("\n✅ All files formatted successfully!")
    else:
        print("\n⚠️ Some files could not be formatted.")
        print("Make sure Black is installed and all files exist.")


if __name__ == "__main__":
    main()
