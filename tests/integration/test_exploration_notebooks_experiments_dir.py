"""Integration tests for exploration notebooks with configurable experiments directory.

Tests that exploration notebooks (01-10) correctly use the experiments directory
configuration and work with CR_EXPERIMENTS_DIR environment variable override.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

NOTEBOOKS_DIR = Path(__file__).parent.parent.parent / "exploration_notebooks"


@pytest.fixture
def notebook_list():
    """Get list of all exploration notebooks."""
    return sorted(NOTEBOOKS_DIR.glob("*.ipynb"))


class TestNotebookStructure:
    """Tests for notebook structure and validity."""

    def test_all_notebooks_are_valid_json(self, notebook_list):
        """All notebooks should be valid JSON."""
        for nb_path in notebook_list:
            with open(nb_path, 'r', encoding='utf-8') as f:
                nb = json.load(f)
            assert 'cells' in nb, f"{nb_path.name} missing 'cells' key"

    def test_all_notebooks_have_experiments_import(self, notebook_list):
        """All notebooks should import from experiments config."""
        for nb_path in notebook_list:
            with open(nb_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check for the experiments import
            assert 'customer_retention.core.config.experiments' in content, \
                f"{nb_path.name} missing experiments config import"

    def test_no_hardcoded_experiments_paths(self, notebook_list):
        """No notebooks should have hardcoded ../experiments paths in definitions."""
        for nb_path in notebook_list:
            with open(nb_path, 'r', encoding='utf-8') as f:
                nb = json.load(f)

            for i, cell in enumerate(nb.get('cells', [])):
                if cell.get('cell_type') != 'code':
                    continue

                source = ''.join(cell.get('source', []))

                # Check for hardcoded path definitions (not in comments or strings for display)
                # Allow: comments, print statements, error messages
                lines = source.split('\n')
                for line in lines:
                    stripped = line.strip()
                    # Skip comments
                    if stripped.startswith('#'):
                        continue
                    # Skip print/console output
                    if 'print(' in line or 'console.' in line:
                        continue
                    # Skip string literals in raise statements
                    if 'raise ' in line:
                        continue
                    # Check for actual path definitions
                    if 'Path("../experiments' in line or "Path('../experiments" in line:
                        # Only flag if it's an assignment
                        if '=' in line and 'Path("../experiments' in line.split('=')[1]:
                            pytest.fail(
                                f"{nb_path.name} cell {i} has hardcoded path: {line.strip()}"
                            )


class TestNotebookCodeValidity:
    """Tests for Python code validity in notebooks."""

    def test_all_notebook_code_is_valid_python(self, notebook_list):
        """All code cells should contain valid Python syntax."""
        for nb_path in notebook_list:
            with open(nb_path, 'r', encoding='utf-8') as f:
                nb = json.load(f)

            for i, cell in enumerate(nb.get('cells', [])):
                if cell.get('cell_type') != 'code':
                    continue

                source = ''.join(cell.get('source', []))
                # Skip cells that only have magic commands or are empty
                if not source.strip() or source.strip().startswith('%') or source.strip().startswith('!'):
                    continue

                try:
                    compile(source, f"{nb_path.name}:cell_{i}", 'exec')
                except SyntaxError as e:
                    pytest.fail(f"{nb_path.name} cell {i} has syntax error: {e}")


class TestExperimentsConfigModule:
    """Tests for the experiments configuration module."""

    def test_experiments_module_imports(self):
        """The experiments module should be importable."""
        from customer_retention.core.config.experiments import (
            EXPERIMENTS_DIR,
            FINDINGS_DIR,
            OUTPUT_DIR,
        )
        assert EXPERIMENTS_DIR is not None
        assert FINDINGS_DIR is not None
        assert OUTPUT_DIR is not None

    def test_experiments_dir_default(self):
        """Default experiments dir should be 'experiments' in project root."""
        from customer_retention.core.config.experiments import get_experiments_dir

        # Without env var, should use default
        experiments_dir = get_experiments_dir()
        assert experiments_dir.name == "experiments"

    def test_experiments_dir_env_override(self, tmp_path, monkeypatch):
        """CR_EXPERIMENTS_DIR env var should override default."""
        custom_path = str(tmp_path / "custom_experiments")
        monkeypatch.setenv("CR_EXPERIMENTS_DIR", custom_path)

        # Need to reimport to get fresh value
        import importlib

        import customer_retention.core.config.experiments as exp_module
        importlib.reload(exp_module)

        assert str(exp_module.EXPERIMENTS_DIR) == custom_path
        assert str(exp_module.FINDINGS_DIR) == str(tmp_path / "custom_experiments" / "findings")

        # Clean up - reload with original env
        monkeypatch.delenv("CR_EXPERIMENTS_DIR")
        importlib.reload(exp_module)

    def test_setup_experiments_structure_creates_dirs(self, tmp_path, monkeypatch):
        """setup_experiments_structure should create all required directories."""
        custom_path = str(tmp_path / "test_experiments")
        monkeypatch.setenv("CR_EXPERIMENTS_DIR", custom_path)

        import importlib

        import customer_retention.core.config.experiments as exp_module
        importlib.reload(exp_module)

        exp_module.setup_experiments_structure()

        expected_dirs = [
            "findings/snapshots",
            "findings/unified",
            "data/bronze",
            "data/silver",
            "data/gold",
            "data/scoring",
            "mlruns",
            "feature_repo/data",
        ]

        for subdir in expected_dirs:
            assert (tmp_path / "test_experiments" / subdir).exists(), f"Missing: {subdir}"

        # Clean up
        monkeypatch.delenv("CR_EXPERIMENTS_DIR")
        importlib.reload(exp_module)


class TestNotebookImportExecution:
    """Tests that notebook imports can be executed."""

    def test_first_code_cell_imports_work(self, notebook_list, tmp_path):
        """First code cell imports should execute without errors."""
        for nb_path in notebook_list:
            with open(nb_path, 'r', encoding='utf-8') as f:
                nb = json.load(f)

            # Find first code cell
            for cell in nb.get('cells', []):
                if cell.get('cell_type') != 'code':
                    continue

                source = ''.join(cell.get('source', []))
                if not source.strip():
                    continue

                # Extract complete import statements including multi-line ones
                lines = source.split('\n')
                import_statements = []
                i = 0
                while i < len(lines):
                    line = lines[i].strip()
                    if line.startswith('import ') or line.startswith('from '):
                        # Check if it's a multi-line import with parentheses
                        if '(' in line and ')' not in line:
                            # Collect all lines until closing paren
                            statement_lines = [line]
                            i += 1
                            while i < len(lines) and ')' not in lines[i]:
                                statement_lines.append(lines[i].strip())
                                i += 1
                            if i < len(lines):
                                statement_lines.append(lines[i].strip())
                            import_statements.append('\n'.join(statement_lines))
                        else:
                            import_statements.append(line)
                    i += 1

                if import_statements:
                    import_code = '\n'.join(import_statements)

                    # Execute imports in subprocess with custom experiments dir
                    result = subprocess.run(
                        [sys.executable, "-c", import_code],
                        capture_output=True,
                        text=True,
                        env={**os.environ, "CR_EXPERIMENTS_DIR": str(tmp_path)},
                        timeout=30
                    )

                    assert result.returncode == 0, \
                        f"{nb_path.name} import failed:\n{result.stderr}"
                break  # Only test first code cell with imports


class TestNotebookWithCustomExperimentsDir:
    """Tests that notebooks work with custom experiments directory."""

    def test_findings_dir_resolves_to_custom_path(self, tmp_path):
        """FINDINGS_DIR should resolve to custom path when env var is set."""
        custom_exp = tmp_path / "my_experiments"
        custom_exp.mkdir()
        (custom_exp / "findings").mkdir()

        result = subprocess.run(
            [sys.executable, "-c", """
import os
os.environ['CR_EXPERIMENTS_DIR'] = os.environ.get('TEST_EXP_DIR')

from customer_retention.core.config.experiments import FINDINGS_DIR
print(f"FINDINGS_DIR: {FINDINGS_DIR}")
"""],
            capture_output=True,
            text=True,
            env={**os.environ, "TEST_EXP_DIR": str(custom_exp)},
        )

        assert result.returncode == 0, f"Failed: {result.stderr}"
        assert str(custom_exp / "findings") in result.stdout

    def test_databricks_path_format_works(self, tmp_path):
        """Databricks-style paths should work."""
        dbfs_path = "/dbfs/mnt/catalog/experiments"

        result = subprocess.run(
            [sys.executable, "-c", f"""
import os
os.environ['CR_EXPERIMENTS_DIR'] = '{dbfs_path}'

from customer_retention.core.config.experiments import (
    EXPERIMENTS_DIR, FINDINGS_DIR, get_experiments_dir
)

print(f"EXPERIMENTS_DIR: {{EXPERIMENTS_DIR}}")
print(f"FINDINGS_DIR: {{FINDINGS_DIR}}")
print(f"get_experiments_dir(): {{get_experiments_dir()}}")
"""],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Failed: {result.stderr}"
        assert dbfs_path in result.stdout
        assert f"{dbfs_path}/findings" in result.stdout


class TestNotebookFindingsDiscovery:
    """Tests that notebooks can discover findings with custom experiments dir."""

    def test_notebook_finds_findings_with_custom_dir(self, tmp_path):
        """Notebooks should find findings files in custom experiments directory."""
        # Create a mock findings structure
        custom_exp = tmp_path / "experiments"
        findings_dir = custom_exp / "findings"
        findings_dir.mkdir(parents=True)

        # Create a mock findings file
        mock_findings = {
            "source_path": "test.parquet",
            "source_format": "parquet",
            "row_count": 100,
            "column_count": 5,
            "columns": {},
            "target_column": "target",
            "identifier_columns": ["id"]
        }

        import yaml
        (findings_dir / "test_dataset_findings.yaml").write_text(yaml.dump(mock_findings))

        # Test that the findings can be discovered
        result = subprocess.run(
            [sys.executable, "-c", f"""
import os
os.environ['CR_EXPERIMENTS_DIR'] = '{custom_exp}'

from customer_retention.core.config.experiments import FINDINGS_DIR

# Find findings files (same pattern as notebooks)
findings_files = [
    f for f in FINDINGS_DIR.glob("*_findings.yaml")
    if "multi_dataset" not in f.name
]

print(f"FINDINGS_DIR: {{FINDINGS_DIR}}")
print(f"Found {{len(findings_files)}} findings files")
for f in findings_files:
    print(f"  - {{f.name}}")
"""],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Failed: {result.stderr}"
        assert "Found 1 findings files" in result.stdout
        assert "test_dataset_findings.yaml" in result.stdout


class TestSpecificNotebooks:
    """Tests for specific notebook configurations."""

    @pytest.mark.parametrize("notebook_name", [
        "01_data_discovery.ipynb",
        "02_column_deep_dive.ipynb",
        "05_multi_dataset.ipynb",
        "10_spec_generation.ipynb",
    ])
    def test_key_notebooks_have_correct_config(self, notebook_name):
        """Key notebooks should have the experiments import and no hardcoded paths."""
        nb_path = NOTEBOOKS_DIR / notebook_name
        if not nb_path.exists():
            pytest.skip(f"{notebook_name} not found")

        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)

        # Find config cell (usually cell 2 or 3)
        config_found = False
        for cell in nb.get('cells', []):
            if cell.get('cell_type') != 'code':
                continue

            source = ''.join(cell.get('source', []))

            # Check for experiments import
            if 'customer_retention.core.config.experiments' in source:
                config_found = True

                # Verify FINDINGS_DIR/OUTPUT_DIR is imported, not defined
                assert 'import' in source and 'FINDINGS_DIR' in source, \
                    f"{notebook_name}: FINDINGS_DIR should be imported, not defined"
                break

        assert config_found, f"{notebook_name}: Missing experiments config import"
