#!/usr/bin/env python3
"""
Quick notebook validation script.

Usage:
    python scripts/notebooks/test_notebooks.py              # Run all notebooks
    python scripts/notebooks/test_notebooks.py 01 02 03     # Run specific notebooks (by number)
    python scripts/notebooks/test_notebooks.py --fast       # Skip slow notebooks
"""
import argparse
import sys
import tempfile
import time
from pathlib import Path

try:
    import papermill as pm
except ImportError:
    print("papermill not installed. Run: pip install papermill")
    sys.exit(1)


# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
NOTEBOOKS_DIR = PROJECT_ROOT / "exploration_notebooks"
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"
TEST_DATA_PATH = FIXTURES_DIR / "customer_retention_retail.csv"

# Notebook list with metadata (name, description, is_slow)
NOTEBOOKS = {
    "00": ("00_start_here.ipynb", "Start Here", False),
    "01": ("01_data_discovery.ipynb", "Data Discovery", False),
    "01a": ("01a_temporal_deep_dive.ipynb", "Temporal Deep Dive", True),  # slow
    "01b": ("01b_temporal_quality.ipynb", "Temporal Quality", False),
    "01c": ("01c_temporal_patterns.ipynb", "Temporal Patterns", True),  # slow
    "01d": ("01d_event_aggregation.ipynb", "Event Aggregation", False),
    "02": ("02_column_deep_dive.ipynb", "Column Deep Dive", False),
    "02a": ("02a_text_columns_deep_dive.ipynb", "Text Columns Deep Dive", False),
    "03": ("03_quality_assessment.ipynb", "Quality Assessment", False),
    "04": ("04_relationship_analysis.ipynb", "Relationship Analysis", False),
    "05": ("05_multi_dataset.ipynb", "Multi Dataset", False),
    "06": ("06_feature_opportunities.ipynb", "Feature Opportunities", False),
    "07": ("07_modeling_readiness.ipynb", "Modeling Readiness", False),
    "08": ("08_baseline_experiments.ipynb", "Baseline Experiments", True),  # slow
    "09": ("09_business_alignment.ipynb", "Business Alignment", False),
    "10": ("10_spec_generation.ipynb", "Spec Generation", True),  # slow
}


def run_notebook(notebook_path: Path, output_path: Path, params: dict = None) -> tuple[bool, str, float]:
    """Run a notebook and return (success, error_message, duration)."""
    start = time.time()
    try:
        pm.execute_notebook(
            str(notebook_path),
            str(output_path),
            parameters=params or {},
            cwd=str(NOTEBOOKS_DIR),
            kernel_name="python3",
        )
        return True, "", time.time() - start
    except pm.PapermillExecutionError as e:
        return False, f"Cell {e.cell_index}: {e.ename}: {e.evalue}", time.time() - start
    except Exception as e:
        return False, str(e), time.time() - start


def main():
    parser = argparse.ArgumentParser(description="Test exploration notebooks")
    parser.add_argument("notebooks", nargs="*", help="Notebook numbers to run (e.g., 01 02 03)")
    parser.add_argument("--fast", action="store_true", help="Skip slow notebooks")
    parser.add_argument("--keep-output", action="store_true", help="Keep output notebooks")
    args = parser.parse_args()

    # Determine which notebooks to run
    if args.notebooks:
        to_run = [n.zfill(2) for n in args.notebooks]
    else:
        to_run = list(NOTEBOOKS.keys())

    if args.fast:
        to_run = [n for n in to_run if not NOTEBOOKS[n][2]]

    # Check prerequisites
    if not TEST_DATA_PATH.exists():
        print(f"ERROR: Test data not found: {TEST_DATA_PATH}")
        sys.exit(1)

    # Create temp workspace
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        explorations_dir = workspace / "explorations"
        explorations_dir.mkdir()

        print(f"\n{'='*60}")
        print(f"NOTEBOOK VALIDATION")
        print(f"{'='*60}")
        print(f"Notebooks to test: {', '.join(to_run)}")
        print(f"Test data: {TEST_DATA_PATH}")
        print(f"Workspace: {workspace}")
        print(f"{'='*60}\n")

        results = []
        total_time = 0

        for nb_num in to_run:
            if nb_num not in NOTEBOOKS:
                print(f"  [SKIP] Unknown notebook: {nb_num}")
                continue

            filename, description, is_slow = NOTEBOOKS[nb_num]
            notebook_path = NOTEBOOKS_DIR / filename
            output_path = workspace / f"{filename.replace('.ipynb', '_output.ipynb')}"

            if not notebook_path.exists():
                print(f"  [SKIP] {nb_num}: {description} - not found")
                continue

            print(f"  [RUN]  {nb_num}: {description}...", end=" ", flush=True)

            # Set parameters for notebook 01
            params = {}
            if nb_num == "01":
                params = {
                    "DATA_PATH": str(TEST_DATA_PATH),
                    "OUTPUT_DIR": str(explorations_dir),
                }

            success, error, duration = run_notebook(notebook_path, output_path, params)
            total_time += duration

            if success:
                print(f"PASS ({duration:.1f}s)")
                results.append((nb_num, description, "PASS", ""))
            else:
                print(f"FAIL ({duration:.1f}s)")
                print(f"         Error: {error[:100]}...")
                results.append((nb_num, description, "FAIL", error))

        # Summary
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")

        passed = sum(1 for r in results if r[2] == "PASS")
        failed = sum(1 for r in results if r[2] == "FAIL")

        print(f"Total: {len(results)} | Passed: {passed} | Failed: {failed}")
        print(f"Time: {total_time:.1f}s")

        if failed > 0:
            print(f"\nFailed notebooks:")
            for nb_num, desc, status, error in results:
                if status == "FAIL":
                    print(f"  - {nb_num}: {desc}")
                    print(f"    {error[:200]}")
            sys.exit(1)
        else:
            print(f"\nAll notebooks passed!")
            sys.exit(0)


if __name__ == "__main__":
    main()
