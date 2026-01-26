#!/usr/bin/env python
"""Update exploration notebooks to use configurable experiments directory.

This script updates all exploration notebooks to import paths from
customer_retention.core.config.experiments instead of hardcoding them.
"""

import json
import re
from pathlib import Path


EXPERIMENTS_IMPORT = 'from customer_retention.core.config.experiments import FINDINGS_DIR, EXPERIMENTS_DIR, OUTPUT_DIR, setup_experiments_structure'


def update_notebook(notebook_path: Path) -> tuple[bool, list[str]]:
    """Update a single notebook to use configurable paths.

    Returns (modified, changes_made).
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    changes = []
    import_added = False

    for i, cell in enumerate(nb.get('cells', [])):
        if cell.get('cell_type') != 'code':
            continue

        source = cell.get('source', [])
        if isinstance(source, str):
            source = source.split('\n')
            source = [line + '\n' if j < len(source) - 1 else line for j, line in enumerate(source)]

        new_source = []
        cell_modified = False

        for line in source:
            original_line = line

            # Pattern 1: FINDINGS_DIR = Path("../experiments/findings")
            if re.match(r'^FINDINGS_DIR\s*=\s*Path\s*\(\s*["\']', line):
                # Replace with comment - import handles this
                line = '# FINDINGS_DIR imported from customer_retention.core.config.experiments\n'
                changes.append(f"Cell {i}: Replaced FINDINGS_DIR definition")
                cell_modified = True

            # Pattern 2: OUTPUT_DIR = Path("../experiments/findings")
            elif re.match(r'^OUTPUT_DIR\s*=\s*Path\s*\(\s*["\']\.\.\/experiments', line):
                line = '# OUTPUT_DIR imported from customer_retention.core.config.experiments\n'
                changes.append(f"Cell {i}: Replaced OUTPUT_DIR definition")
                cell_modified = True

            # Pattern 3: OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            elif re.match(r'^OUTPUT_DIR\.mkdir\s*\(', line):
                line = 'setup_experiments_structure()  # Creates all experiment directories\n'
                changes.append(f"Cell {i}: Replaced mkdir with setup_experiments_structure()")
                cell_modified = True

            # Pattern 4: output_dir="../experiments/findings" in function calls
            elif 'output_dir="../experiments/findings"' in line or "output_dir='../experiments/findings'" in line:
                line = re.sub(
                    r'output_dir\s*=\s*["\']\.\.\/experiments\/findings["\']',
                    'output_dir=str(FINDINGS_DIR)',
                    line
                )
                if line != original_line:
                    changes.append(f"Cell {i}: Updated output_dir argument")
                    cell_modified = True

            # Pattern 5: output_path = Path("../experiments/findings")
            elif re.match(r'^output_path\s*=\s*Path\s*\(\s*["\']\.\.\/experiments\/findings', line):
                line = 'output_path = FINDINGS_DIR\n'
                changes.append(f"Cell {i}: Updated output_path")
                cell_modified = True

            new_source.append(line)

        if cell_modified:
            cell['source'] = new_source

        # Add import to first cell with imports (if not already present)
        if not import_added and cell.get('cell_type') == 'code':
            source_text = ''.join(cell.get('source', []))
            has_imports = 'import ' in source_text or 'from ' in source_text
            has_experiments_import = 'from customer_retention.core.config.experiments' in source_text

            if has_imports and not has_experiments_import:
                source = cell.get('source', [])
                if isinstance(source, str):
                    source = source.split('\n')
                    source = [line + '\n' if j < len(source) - 1 else line for j, line in enumerate(source)]

                # Find last import line and add after it
                last_import_idx = -1
                for idx, line in enumerate(source):
                    if line.strip().startswith('import ') or line.strip().startswith('from '):
                        last_import_idx = idx

                if last_import_idx >= 0:
                    source.insert(last_import_idx + 1, EXPERIMENTS_IMPORT + '\n')
                    cell['source'] = source
                    import_added = True
                    changes.append(f"Cell {i}: Added experiments import")

    if changes:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)

    return bool(changes), changes


def main():
    notebooks_dir = Path(__file__).parent.parent / "exploration_notebooks"

    print("Updating exploration notebooks to use configurable experiments directory...")
    print(f"Directory: {notebooks_dir}")
    print(f"Import: {EXPERIMENTS_IMPORT}\n")

    notebooks = sorted(notebooks_dir.glob("*.ipynb"))
    total_changes = 0

    for nb_path in notebooks:
        modified, changes = update_notebook(nb_path)
        if modified:
            print(f"\n{nb_path.name}:")
            for change in changes:
                print(f"  - {change}")
            total_changes += len(changes)
        else:
            print(f"{nb_path.name}: No changes needed")

    print(f"\n{'='*60}")
    print(f"Total: {total_changes} changes across {len(notebooks)} notebooks")
    print("\nNotebooks now use:")
    print("  - FINDINGS_DIR from customer_retention.core.config.experiments")
    print("  - Override with: export CR_EXPERIMENTS_DIR=/your/path")


if __name__ == "__main__":
    main()
