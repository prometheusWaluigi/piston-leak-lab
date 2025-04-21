#!/usr/bin/env python3
"""
Fix common linting issues in the Piston Leak Lab codebase.
"""

import re
from pathlib import Path

def fix_file(file_path):
    """Fix linting issues in a file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Remove UTF-8 encoding declaration
    content = re.sub(r'# -\*- coding: utf-8 -\*-', '', content)
    
    # Fix deprecated typing imports
    content = re.sub(r'from typing import (?:Dict|List|Tuple|Optional|Union)', 'from typing import Any', content)
    content = re.sub(r'Dict\[', 'dict[', content)
    content = re.sub(r'List\[', 'list[', content)
    content = re.sub(r'Tuple\[', 'tuple[', content)
    content = re.sub(r'Optional\[', 'None | ', content)
    content = re.sub(r'Union\[([^,]+), None\]', 'None | \\1', content)
    content = re.sub(r'Union\[([^,]+), ([^]]+)\]', '\\1 | \\2', content)
    
    # Fix other typing imports
    content = re.sub(r'from typing import Callable', 'from collections.abc import Callable', content)
    
    # Write back to file
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Fixed: {file_path}")

def main():
    """Process all Python files in the project."""
    for dirpath in ["models", "sims", "tests"]:
        for path in Path(dirpath).rglob("*.py"):
            fix_file(path)
    
    print("\nDone! Run 'poetry run ruff check --fix .' to fix remaining issues automatically.")

if __name__ == "__main__":
    main()
