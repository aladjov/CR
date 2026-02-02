import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

NOTEBOOKS_DIR = Path(__file__).parent.parent.parent / "exploration_notebooks"


def _read_notebook_cells(path):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    return [
        (i, ''.join(cell.get('source', [])))
        for i, cell in enumerate(nb.get('cells', []))
        if cell.get('cell_type') == 'code'
    ]


def _is_hardcoded_path_assignment(line):
    stripped = line.strip()
    if stripped.startswith('#'):
        return False
    if 'print(' in line or 'console.' in line:
        return False
    if 'raise ' in line:
        return False
    if 'Path("../experiments' not in line and "Path('../experiments" not in line:
        return False
    if '=' in line and 'Path("../experiments' in line.split('=')[1]:
        return True
    return False


@pytest.fixture
def notebook_list():
    return sorted(NOTEBOOKS_DIR.glob("*.ipynb"))


class TestNotebookStructure:

    def test_all_notebooks_are_valid_json(self, notebook_list):
        for nb_path in notebook_list:
            with open(nb_path, 'r', encoding='utf-8') as f:
                nb = json.load(f)
            assert 'cells' in nb, f"{nb_path.name} missing 'cells' key"

    def test_all_notebooks_have_experiments_import(self, notebook_list):
        for nb_path in notebook_list:
            with open(nb_path, 'r', encoding='utf-8') as f:
                content = f.read()

            assert 'customer_retention.core.config.experiments' in content, \
                f"{nb_path.name} missing experiments config import"

    def test_no_hardcoded_experiments_paths(self, notebook_list):
        for nb_path in notebook_list:
            for i, source in _read_notebook_cells(nb_path):
                lines = source.split('\n')
                for line in lines:
                    if _is_hardcoded_path_assignment(line):
                        pytest.fail(
                            f"{nb_path.name} cell {i} has hardcoded path: {line.strip()}"
                        )


class TestNotebookCodeValidity:

    def test_all_notebook_code_is_valid_python(self, notebook_list):
        for nb_path in notebook_list:
            for i, source in _read_notebook_cells(nb_path):
                if not source.strip() or source.strip().startswith('%') or source.strip().startswith('!'):
                    continue

                try:
                    compile(source, f"{nb_path.name}:cell_{i}", 'exec')
                except SyntaxError as e:
                    pytest.fail(f"{nb_path.name} cell {i} has syntax error: {e}")


class TestExperimentsConfigModule:

    def test_experiments_module_imports(self):
        from customer_retention.core.config.experiments import (
            EXPERIMENTS_DIR,
            FINDINGS_DIR,
            OUTPUT_DIR,
        )
        assert EXPERIMENTS_DIR is not None
        assert FINDINGS_DIR is not None
        assert OUTPUT_DIR is not None

    def test_experiments_dir_default(self):
        from customer_retention.core.config.experiments import get_experiments_dir

        experiments_dir = get_experiments_dir()
        assert experiments_dir.name == "experiments"

    def test_experiments_dir_env_override(self, tmp_path, monkeypatch):
        custom_path = str(tmp_path / "custom_experiments")
        monkeypatch.setenv("CR_EXPERIMENTS_DIR", custom_path)

        import importlib

        import customer_retention.core.config.experiments as exp_module
        importlib.reload(exp_module)

        assert str(exp_module.EXPERIMENTS_DIR) == custom_path
        assert str(exp_module.FINDINGS_DIR) == str(tmp_path / "custom_experiments" / "findings")

        monkeypatch.delenv("CR_EXPERIMENTS_DIR")
        importlib.reload(exp_module)

    def test_setup_experiments_structure_creates_dirs(self, tmp_path, monkeypatch):
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

        monkeypatch.delenv("CR_EXPERIMENTS_DIR")
        importlib.reload(exp_module)


class TestNotebookImportExecution:

    def test_first_code_cell_imports_work(self, notebook_list, tmp_path):
        for nb_path in notebook_list:
            with open(nb_path, 'r', encoding='utf-8') as f:
                nb = json.load(f)

            for cell in nb.get('cells', []):
                if cell.get('cell_type') != 'code':
                    continue

                source = ''.join(cell.get('source', []))
                if not source.strip():
                    continue

                lines = source.split('\n')
                import_statements = []
                i = 0
                while i < len(lines):
                    line = lines[i].strip()
                    if line.startswith('import ') or line.startswith('from '):
                        if '(' in line and ')' not in line:
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

                    result = subprocess.run(
                        [sys.executable, "-c", import_code],
                        capture_output=True,
                        text=True,
                        env={**os.environ, "CR_EXPERIMENTS_DIR": str(tmp_path)},
                        timeout=30
                    )

                    assert result.returncode == 0, \
                        f"{nb_path.name} import failed:\n{result.stderr}"
                break


class TestNotebookWithCustomExperimentsDir:

    def test_findings_dir_resolves_to_custom_path(self, tmp_path):
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

    def test_notebook_finds_findings_with_custom_dir(self, tmp_path):
        custom_exp = tmp_path / "experiments"
        findings_dir = custom_exp / "findings"
        findings_dir.mkdir(parents=True)

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

        result = subprocess.run(
            [sys.executable, "-c", f"""
import os
os.environ['CR_EXPERIMENTS_DIR'] = '{custom_exp}'

from customer_retention.core.config.experiments import FINDINGS_DIR

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

    @pytest.mark.parametrize("notebook_name", [
        "01_data_discovery.ipynb",
        "02_column_deep_dive.ipynb",
        "05_multi_dataset.ipynb",
        "10_spec_generation.ipynb",
    ])
    def test_key_notebooks_have_correct_config(self, notebook_name):
        nb_path = NOTEBOOKS_DIR / notebook_name
        if not nb_path.exists():
            pytest.skip(f"{notebook_name} not found")

        for _, source in _read_notebook_cells(nb_path):
            if 'customer_retention.core.config.experiments' in source:
                assert 'import' in source and 'FINDINGS_DIR' in source, \
                    f"{notebook_name}: FINDINGS_DIR should be imported, not defined"
                return

        pytest.fail(f"{notebook_name}: Missing experiments config import")


class TestScoringValidationNotebook:

    NOTEBOOK_PATH = NOTEBOOKS_DIR / "11_scoring_validation.ipynb"

    def test_notebook_exists(self):
        assert self.NOTEBOOK_PATH.exists(), "11_scoring_validation.ipynb not found"

    def test_notebook_is_valid_json(self):
        with open(self.NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        assert 'cells' in nb
        assert 'metadata' in nb
        assert nb['nbformat'] >= 4

    def test_notebook_has_expected_sections(self):
        with open(self.NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
            content = f.read()

        expected_sections = [
            "11.1 Run Scoring",
            "11.2 Summary Metrics",
            "11.3 Model Comparison Grid",
            "11.4 Adversarial Pipeline Validation",
            "11.5 Transformation Validation",
            "11.6 Model Explanations (SHAP)",
            "11.7 Customer Browser",
            "11.8 Error Analysis",
            "11.9 Export Results",
        ]
        for section in expected_sections:
            assert section in content, f"Missing section: {section}"

    def test_notebook_imports_experiments_config(self):
        with open(self.NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
            content = f.read()
        assert 'customer_retention.core.config.experiments' in content

    def test_notebook_uses_transform_executor(self):
        with open(self.NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
            content = f.read()
        assert 'TransformExecutor' in content
        assert 'ArtifactStore' in content
        assert 'LabelEncoder().fit_transform' not in content

    def test_notebook_has_no_jinja2(self):
        with open(self.NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
            content = f.read()
        assert '{{ ' not in content
        assert '{% ' not in content

    def test_notebook_code_cells_are_valid_python(self):
        for i, source in _read_notebook_cells(self.NOTEBOOK_PATH):
            if not source.strip():
                continue
            try:
                compile(source, f"11_scoring_validation.ipynb:cell_{i}", 'exec')
            except SyntaxError as e:
                pytest.fail(f"Cell {i} has syntax error: {e}")

    def test_notebook_has_validation_import(self):
        with open(self.NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
            content = f.read()
        assert 'validate_feature_transformation' in content
