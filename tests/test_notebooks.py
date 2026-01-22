"""
Notebook execution tests using papermill.

Run with: pytest tests/test_notebooks.py -v
Run single: pytest tests/test_notebooks.py::test_notebook_01_data_discovery -v
"""
from pathlib import Path

import papermill as pm
import pytest

# Test data paths
FIXTURES_DIR = Path(__file__).parent / "fixtures"
TEST_DATA_PATH = FIXTURES_DIR / "customer_retention_retail.csv"
TEST_TS_DATA_PATH = FIXTURES_DIR / "customer_transactions.csv"

# Notebooks directory
NOTEBOOKS_DIR = Path(__file__).parent.parent / "templates" / "notebooks" / "exploration"

# List of exploration notebooks in order (standard track)
EXPLORATION_NOTEBOOKS = [
    "01_data_discovery.ipynb",
    "02_column_deep_dive.ipynb",
    "03_quality_assessment.ipynb",
    "04_relationship_analysis.ipynb",
    "05_multi_dataset.ipynb",
    "06_feature_opportunities.ipynb",
    "07_modeling_readiness.ipynb",
    "08_baseline_experiments.ipynb",
    "09_business_alignment.ipynb",
    "10_spec_generation.ipynb",
]

# Time series track notebooks
TS_NOTEBOOKS = [
    "02TS_temporal_deep_dive.ipynb",
    "03TS_temporal_quality.ipynb",
    "04TS_temporal_patterns.ipynb",
]


@pytest.fixture(scope="module")
def notebook_workspace(tmp_path_factory):
    """Create a temporary workspace with explorations folder for notebooks."""
    workspace = tmp_path_factory.mktemp("notebook_test")
    explorations_dir = workspace / "explorations"
    explorations_dir.mkdir()
    return workspace, explorations_dir


@pytest.fixture(scope="module")
def run_notebook_01(notebook_workspace):
    """
    Run notebook 01 first to generate findings (entity-level data).
    This is a module-scoped fixture so it only runs once.
    """
    workspace, explorations_dir = notebook_workspace
    notebook_path = NOTEBOOKS_DIR / "01_data_discovery.ipynb"
    output_path = workspace / "01_output.ipynb"

    if not notebook_path.exists():
        pytest.skip(f"Notebook not found: {notebook_path}")

    if not TEST_DATA_PATH.exists():
        pytest.skip(f"Test data not found: {TEST_DATA_PATH}")

    # Run notebook 01 with parameters
    pm.execute_notebook(
        str(notebook_path),
        str(output_path),
        parameters={
            "DATA_PATH": str(TEST_DATA_PATH),
            "OUTPUT_DIR": str(explorations_dir),
        },
        cwd=str(NOTEBOOKS_DIR),
        kernel_name="python3",
    )

    # Verify findings were created
    findings_files = list(explorations_dir.glob("*_findings.yaml"))
    assert len(findings_files) > 0, "No findings file created by notebook 01"

    return explorations_dir


@pytest.fixture(scope="module")
def ts_workspace(tmp_path_factory):
    """Create a separate workspace for time series notebook tests."""
    workspace = tmp_path_factory.mktemp("ts_notebook_test")
    explorations_dir = workspace / "explorations"
    explorations_dir.mkdir()
    return workspace, explorations_dir


@pytest.fixture(scope="module")
def run_notebook_01_ts(ts_workspace):
    """
    Run notebook 01 with time series (event-level) data.
    This generates findings that the TS notebooks will use.
    """
    workspace, explorations_dir = ts_workspace
    notebook_path = NOTEBOOKS_DIR / "01_data_discovery.ipynb"
    output_path = workspace / "01_ts_output.ipynb"

    if not notebook_path.exists():
        pytest.skip(f"Notebook not found: {notebook_path}")

    if not TEST_TS_DATA_PATH.exists():
        pytest.skip(f"Time series test data not found: {TEST_TS_DATA_PATH}")

    # Run notebook 01 with time series data
    pm.execute_notebook(
        str(notebook_path),
        str(output_path),
        parameters={
            "DATA_PATH": str(TEST_TS_DATA_PATH),
            "OUTPUT_DIR": str(explorations_dir),
        },
        cwd=str(NOTEBOOKS_DIR),
        kernel_name="python3",
    )

    # Verify findings were created
    findings_files = list(explorations_dir.glob("*_findings.yaml"))
    assert len(findings_files) > 0, "No findings file created by notebook 01 for TS data"

    return explorations_dir


@pytest.fixture(scope="module")
def multi_dataset_workspace(tmp_path_factory):
    """Create workspace for multi-dataset notebook tests with both entity and event data."""
    workspace = tmp_path_factory.mktemp("multi_notebook_test")
    explorations_dir = workspace / "explorations"
    explorations_dir.mkdir()
    return workspace, explorations_dir


@pytest.fixture(scope="module")
def run_notebook_01_multi(multi_dataset_workspace):
    """
    Run notebook 01 for both entity-level and event-level datasets.
    This prepares the explorations folder for the multi-dataset notebook.
    """
    workspace, explorations_dir = multi_dataset_workspace
    notebook_path = NOTEBOOKS_DIR / "01_data_discovery.ipynb"

    if not notebook_path.exists():
        pytest.skip(f"Notebook not found: {notebook_path}")

    # Run for entity-level data
    if TEST_DATA_PATH.exists():
        pm.execute_notebook(
            str(notebook_path),
            str(workspace / "01_entity_output.ipynb"),
            parameters={
                "DATA_PATH": str(TEST_DATA_PATH),
                "OUTPUT_DIR": str(explorations_dir),
            },
            cwd=str(NOTEBOOKS_DIR),
            kernel_name="python3",
        )

    # Run for time series data
    if TEST_TS_DATA_PATH.exists():
        pm.execute_notebook(
            str(notebook_path),
            str(workspace / "01_ts_output.ipynb"),
            parameters={
                "DATA_PATH": str(TEST_TS_DATA_PATH),
                "OUTPUT_DIR": str(explorations_dir),
            },
            cwd=str(NOTEBOOKS_DIR),
            kernel_name="python3",
        )

    # Verify at least one findings file was created
    findings_files = list(explorations_dir.glob("*_findings.yaml"))
    assert len(findings_files) > 0, "No findings files created"

    return explorations_dir


def run_exploration_notebook(notebook_name: str, workspace: Path, explorations_dir: Path):
    """Helper to run a notebook and capture errors."""
    notebook_path = NOTEBOOKS_DIR / notebook_name
    output_path = workspace / f"{notebook_name.replace('.ipynb', '_output.ipynb')}"

    if not notebook_path.exists():
        pytest.skip(f"Notebook not found: {notebook_path}")

    try:
        pm.execute_notebook(
            str(notebook_path),
            str(output_path),
            cwd=str(NOTEBOOKS_DIR),
            kernel_name="python3",
        )
    except pm.PapermillExecutionError as e:
        # Extract useful error info
        pytest.fail(
            f"Notebook {notebook_name} failed at cell {e.cell_index}:\n"
            f"Cell source:\n{e.source}\n\n"
            f"Error:\n{e.ename}: {e.evalue}"
        )


class TestNotebook01DataDiscovery:
    """Test notebook 01: Data Discovery."""

    def test_notebook_executes(self, notebook_workspace):
        """Test that notebook 01 runs without errors."""
        workspace, explorations_dir = notebook_workspace
        notebook_path = NOTEBOOKS_DIR / "01_data_discovery.ipynb"
        output_path = workspace / "01_test_output.ipynb"

        if not notebook_path.exists():
            pytest.skip(f"Notebook not found: {notebook_path}")

        if not TEST_DATA_PATH.exists():
            pytest.skip(f"Test data not found: {TEST_DATA_PATH}")

        pm.execute_notebook(
            str(notebook_path),
            str(output_path),
            parameters={
                "DATA_PATH": str(TEST_DATA_PATH),
                "OUTPUT_DIR": str(explorations_dir),
            },
            cwd=str(NOTEBOOKS_DIR),
            kernel_name="python3",
        )

        # Verify output exists
        assert output_path.exists()

        # Verify findings were created
        findings_files = list(explorations_dir.glob("*_findings.yaml"))
        assert len(findings_files) > 0, "No findings file created"


class TestNotebook02ColumnDeepDive:
    """Test notebook 02: Column Deep Dive."""

    def test_notebook_executes(self, run_notebook_01, notebook_workspace):
        """Test that notebook 02 runs without errors."""
        workspace, _ = notebook_workspace
        run_exploration_notebook("02_column_deep_dive.ipynb", workspace, run_notebook_01)


class TestNotebook03QualityAssessment:
    """Test notebook 03: Quality Assessment."""

    def test_notebook_executes(self, run_notebook_01, notebook_workspace):
        """Test that notebook 03 runs without errors."""
        workspace, _ = notebook_workspace
        run_exploration_notebook("03_quality_assessment.ipynb", workspace, run_notebook_01)


class TestNotebook04RelationshipAnalysis:
    """Test notebook 04: Relationship Analysis."""

    def test_notebook_executes(self, run_notebook_01, notebook_workspace):
        """Test that notebook 04 runs without errors."""
        workspace, _ = notebook_workspace
        run_exploration_notebook("04_relationship_analysis.ipynb", workspace, run_notebook_01)


class TestNotebook06FeatureOpportunities:
    """Test notebook 06: Feature Opportunities."""

    def test_notebook_executes(self, run_notebook_01, notebook_workspace):
        """Test that notebook 06 runs without errors."""
        workspace, _ = notebook_workspace
        run_exploration_notebook("06_feature_opportunities.ipynb", workspace, run_notebook_01)


class TestNotebook07ModelingReadiness:
    """Test notebook 07: Modeling Readiness."""

    def test_notebook_executes(self, run_notebook_01, notebook_workspace):
        """Test that notebook 07 runs without errors."""
        workspace, _ = notebook_workspace
        run_exploration_notebook("07_modeling_readiness.ipynb", workspace, run_notebook_01)


class TestNotebook08BaselineExperiments:
    """Test notebook 08: Baseline Experiments."""

    def test_notebook_executes(self, run_notebook_01, notebook_workspace):
        """Test that notebook 08 runs without errors."""
        workspace, _ = notebook_workspace
        run_exploration_notebook("08_baseline_experiments.ipynb", workspace, run_notebook_01)


class TestNotebook09BusinessAlignment:
    """Test notebook 09: Business Alignment."""

    def test_notebook_executes(self, run_notebook_01, notebook_workspace):
        """Test that notebook 09 runs without errors."""
        workspace, _ = notebook_workspace
        run_exploration_notebook("09_business_alignment.ipynb", workspace, run_notebook_01)


class TestNotebook10SpecGeneration:
    """Test notebook 10: Spec Generation."""

    def test_notebook_executes(self, run_notebook_01, notebook_workspace):
        """Test that notebook 10 runs without errors."""
        workspace, _ = notebook_workspace
        run_exploration_notebook("10_spec_generation.ipynb", workspace, run_notebook_01)


# Convenience test to run all notebooks in sequence
class TestAllNotebooksSequential:
    """Run all notebooks in sequence (slower but thorough)."""

    @pytest.mark.slow
    def test_all_notebooks_execute(self, tmp_path):
        """Test that all notebooks run without errors in sequence."""
        workspace = tmp_path
        explorations_dir = workspace / "explorations"
        explorations_dir.mkdir()

        for notebook_name in EXPLORATION_NOTEBOOKS:
            notebook_path = NOTEBOOKS_DIR / notebook_name
            output_path = workspace / f"{notebook_name.replace('.ipynb', '_output.ipynb')}"

            if not notebook_path.exists():
                pytest.skip(f"Notebook not found: {notebook_path}")

            # For notebook 01, pass parameters
            params = {}
            if notebook_name == "01_data_discovery.ipynb":
                if not TEST_DATA_PATH.exists():
                    pytest.skip(f"Test data not found: {TEST_DATA_PATH}")
                params = {
                    "DATA_PATH": str(TEST_DATA_PATH),
                    "OUTPUT_DIR": str(explorations_dir),
                }

            try:
                pm.execute_notebook(
                    str(notebook_path),
                    str(output_path),
                    parameters=params,
                    cwd=str(NOTEBOOKS_DIR),
                    kernel_name="python3",
                )
            except pm.PapermillExecutionError as e:
                pytest.fail(
                    f"Notebook {notebook_name} failed at cell {e.cell_index}:\n"
                    f"Error: {e.ename}: {e.evalue}"
                )


# =============================================================================
# TIME SERIES TRACK NOTEBOOK TESTS
# =============================================================================


def run_ts_notebook(notebook_name: str, workspace: Path, explorations_dir: Path):
    """Helper to run a TS notebook and capture errors."""
    notebook_path = NOTEBOOKS_DIR / notebook_name
    output_path = workspace / f"{notebook_name.replace('.ipynb', '_output.ipynb')}"

    if not notebook_path.exists():
        pytest.skip(f"Notebook not found: {notebook_path}")

    try:
        pm.execute_notebook(
            str(notebook_path),
            str(output_path),
            cwd=str(NOTEBOOKS_DIR),
            kernel_name="python3",
        )
    except pm.PapermillExecutionError as e:
        pytest.fail(
            f"Notebook {notebook_name} failed at cell {e.cell_index}:\n"
            f"Cell source:\n{e.source}\n\n"
            f"Error:\n{e.ename}: {e.evalue}"
        )


class TestNotebook02TSTemporalDeepDive:
    """Test notebook 02TS: Temporal Deep Dive (Time Series Track)."""

    def test_notebook_executes(self, run_notebook_01_ts, ts_workspace):
        """Test that notebook 02TS runs without errors."""
        workspace, _ = ts_workspace
        run_ts_notebook("02TS_temporal_deep_dive.ipynb", workspace, run_notebook_01_ts)


class TestNotebook03TSTemporalQuality:
    """Test notebook 03TS: Temporal Quality Assessment (Time Series Track)."""

    def test_notebook_executes(self, run_notebook_01_ts, ts_workspace):
        """Test that notebook 03TS runs without errors."""
        workspace, _ = ts_workspace
        run_ts_notebook("03TS_temporal_quality.ipynb", workspace, run_notebook_01_ts)


class TestNotebook04TSTemporalPatterns:
    """Test notebook 04TS: Temporal Pattern Analysis (Time Series Track)."""

    def test_notebook_executes(self, run_notebook_01_ts, ts_workspace):
        """Test that notebook 04TS runs without errors."""
        workspace, _ = ts_workspace
        run_ts_notebook("04TS_temporal_patterns.ipynb", workspace, run_notebook_01_ts)


# =============================================================================
# MULTI-DATASET NOTEBOOK TEST
# =============================================================================


class TestNotebook05MultiDataset:
    """Test notebook 05: Multi-Dataset Relationships."""

    def test_notebook_executes(self, run_notebook_01_multi, multi_dataset_workspace):
        """Test that notebook 05 multi-dataset runs without errors."""
        workspace, _ = multi_dataset_workspace
        notebook_path = NOTEBOOKS_DIR / "05_multi_dataset.ipynb"
        output_path = workspace / "05_multi_dataset_output.ipynb"

        if not notebook_path.exists():
            pytest.skip(f"Notebook not found: {notebook_path}")

        try:
            pm.execute_notebook(
                str(notebook_path),
                str(output_path),
                cwd=str(NOTEBOOKS_DIR),
                kernel_name="python3",
            )
        except pm.PapermillExecutionError as e:
            pytest.fail(
                f"Notebook 05_multi_dataset failed at cell {e.cell_index}:\n"
                f"Cell source:\n{e.source}\n\n"
                f"Error:\n{e.ename}: {e.evalue}"
            )


# =============================================================================
# ALL TS NOTEBOOKS IN SEQUENCE
# =============================================================================


class TestAllTSNotebooksSequential:
    """Run all TS track notebooks in sequence (slower but thorough)."""

    @pytest.mark.slow
    def test_all_ts_notebooks_execute(self, tmp_path):
        """Test that all TS notebooks run without errors in sequence."""
        workspace = tmp_path
        explorations_dir = workspace / "explorations"
        explorations_dir.mkdir()

        # First run notebook 01 with time series data
        notebook_path = NOTEBOOKS_DIR / "01_data_discovery.ipynb"

        if not notebook_path.exists():
            pytest.skip(f"Notebook not found: {notebook_path}")

        if not TEST_TS_DATA_PATH.exists():
            pytest.skip(f"Time series test data not found: {TEST_TS_DATA_PATH}")

        pm.execute_notebook(
            str(notebook_path),
            str(workspace / "01_ts_output.ipynb"),
            parameters={
                "DATA_PATH": str(TEST_TS_DATA_PATH),
                "OUTPUT_DIR": str(explorations_dir),
            },
            cwd=str(NOTEBOOKS_DIR),
            kernel_name="python3",
        )

        # Then run all TS notebooks
        for notebook_name in TS_NOTEBOOKS:
            notebook_path = NOTEBOOKS_DIR / notebook_name
            output_path = workspace / f"{notebook_name.replace('.ipynb', '_output.ipynb')}"

            if not notebook_path.exists():
                pytest.skip(f"Notebook not found: {notebook_path}")

            try:
                pm.execute_notebook(
                    str(notebook_path),
                    str(output_path),
                    cwd=str(NOTEBOOKS_DIR),
                    kernel_name="python3",
                )
            except pm.PapermillExecutionError as e:
                pytest.fail(
                    f"Notebook {notebook_name} failed at cell {e.cell_index}:\n"
                    f"Error: {e.ename}: {e.evalue}"
                )
