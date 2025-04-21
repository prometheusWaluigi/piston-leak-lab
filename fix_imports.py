#!/usr/bin/env python3
"""
Fix Python import statements for modern type annotations.
This script helps transition from deprecated typing imports to modern syntax.
"""

import os
import re

PYTHON_FILES = [
    "models/core_ode.py",
    "models/abm.py",
    "sims/run_mc.py",
    "sims/visualization/dashboard.py",
    "sims/visualization/plots.py",
    "sims/visualization/__init__.py",
    "tests/test_abm.py",
    "tests/test_core_ode.py"
]

def fix_imports(file_path):
    """Fix imports and type annotations in a file."""
    print(f"Processing {file_path}...")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Remove UTF-8 encoding declaration
    content = re.sub(r'# -\*- coding: utf-8 -\*-', '', content)
    
    # Fix imports
    content = re.sub(r'from typing import Dict, List, Tuple, Optional, Union', 'from typing import Any', content)
    content = re.sub(r'from typing import Dict, List, Tuple', 'from typing import Any', content)
    content = re.sub(r'from typing import Callable', 'from collections.abc import Callable', content)
    
    # Fix type annotations
    content = re.sub(r'Dict\[([^]]+)\]', r'dict[\1]', content)
    content = re.sub(r'List\[([^]]+)\]', r'list[\1]', content)
    content = re.sub(r'Tuple\[([^]]+)\]', r'tuple[\1]', content)
    content = re.sub(r'Optional\[([^]]+)\]', r'None | \1', content)
    content = re.sub(r'Union\[([^,]+), None\]', r'None | \1', content)
    content = re.sub(r'Union\[([^,]+), ([^]]+)\]', r'\1 | \2', content)
    
    # Write updated content
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"  ✓ Fixed imports in {file_path}")

def main():
    """Process all Python files in the list."""
    print("Fixing imports for modern Python type annotations...")
    
    for file_path in PYTHON_FILES:
        if os.path.exists(file_path):
            fix_imports(file_path)
        else:
            print(f"  ⚠ File not found: {file_path}")
    
    print("\nDone! To fix remaining issues, run:")
    print("  poetry run black .")
    print("  poetry run ruff check --fix .")

if __name__ == "__main__":
    main()
