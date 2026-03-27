import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

NOTEBOOKS_DIR = Path(__file__).parent.parent.parent / "exploration_notebooks"


def _read_notebook_cells(path):
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    return [
        (i, "".join(cell.get("source", [])))
        for i, cell in enumerate(nb.get("cells", []))
        if cell.get("cell_type") == "code"
    ]


def _is_hardcoded_path_assignment(line):
    stripped = line.strip()
    if stripped.startswith("#"):
        return False
    if "print(" in line or "console." in line:
        return False
    if "raise " in line:
        return False
    if 'Path("../experiments' not in line and "Path('../experiments" not in line:
        return False
    if "=" in line and 'Path("../experiments' in line.split("=")[1]:
        return True
    return False


@pytest.fixture
def notebook_list():
    return sorted(NOTEBOOKS_DIR.glob("*.ipynb"))


class TestNotebookStructure:
    def test_all_notebooks_are_valid_json(self, notebook_list):
        for nb_path in notebook_list:
            with open(nb_path, "r", encoding="utf-8") as f:
                nb = json.load(f)
            assert "cells" in nb, f"{nb_path.name} missing 'cells' key"

    def test_all_notebooks_have_experiments_import(self, notebook_list):
        for nb_path in notebook_list:
            with open(nb_path, "r", encoding="utf-8") as f:
                content = f.read()

            assert "customer_retention.core.config.experiments" in content, (
                f"{nb_path.name} missing experiments config import"
            )

    def test_no_hardcoded_experiments_paths(self, notebook_list):
        for nb_path in notebook_list:
            for i, source in _read_notebook_cells(nb_path):
                lines = source.split("\n")
                for line in lines:
                    if _is_hardcoded_path_assignment(line):
                        pytest.fail(f"{nb_path.name} cell {i} has hardcoded path: {line.strip()}")


class TestNotebookCodeValidity:
    def test_all_notebook_code_is_valid_python(self, notebook_list):
        for nb_path in notebook_list:
            for i, source in _read_notebook_cells(nb_path):
                if not source.strip() or source.strip().startswith("%") or source.strip().startswith("!"):
                    continue

                try:
                    compile(source, f"{nb_path.name}:cell_{i}", "exec")
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
        imports_by_notebook: dict[str, str] = {}
        for nb_path in notebook_list:
            with open(nb_path, "r", encoding="utf-8") as f:
                nb = json.load(f)
            for cell in nb.get("cells", []):
                if cell.get("cell_type") != "code":
                    continue
                source = "".join(cell.get("source", []))
                if not source.strip():
                    continue
                lines = source.split("\n")
                import_statements = []
                i = 0
                while i < len(lines):
                    line = lines[i].strip()
                    if line.startswith("import ") or line.startswith("from "):
                        if "(" in line and ")" not in line:
                            statement_lines = [line]
                            i += 1
                            while i < len(lines) and ")" not in lines[i]:
                                statement_lines.append(lines[i].strip())
                                i += 1
                            if i < len(lines):
                                statement_lines.append(lines[i].strip())
                            import_statements.append("\n".join(statement_lines))
                        else:
                            import_statements.append(line)
                    i += 1
                if import_statements:
                    imports_by_notebook[nb_path.name] = "\n".join(import_statements)
                break

        script_lines = ["import sys", "failures = []"]
        for nb_name, code in imports_by_notebook.items():
            script_lines.append(
                f'try:\n    exec({code!r}, {{}})\nexcept Exception as e:\n    failures.append(f"{nb_name}: {{e}}")'
            )
        script_lines.append("if failures:\n    print('\\n'.join(failures), file=sys.stderr)\n    sys.exit(1)")
        result = subprocess.run(
            [sys.executable, "-c", "\n".join(script_lines)],
            capture_output=True,
            text=True,
            env={**os.environ, "CR_EXPERIMENTS_DIR": str(tmp_path)},
            timeout=60,
        )
        assert result.returncode == 0, f"Notebook imports failed:\n{result.stderr}"


class TestNotebookWithCustomExperimentsDir:
    def test_findings_dir_resolves_to_custom_path(self, tmp_path):
        custom_exp = tmp_path / "my_experiments"
        custom_exp.mkdir()
        (custom_exp / "findings").mkdir()

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import os
os.environ['CR_EXPERIMENTS_DIR'] = os.environ.get('TEST_EXP_DIR')

from customer_retention.core.config.experiments import FINDINGS_DIR
print(f"FINDINGS_DIR: {FINDINGS_DIR}")
""",
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "TEST_EXP_DIR": str(custom_exp)},
        )

        assert result.returncode == 0, f"Failed: {result.stderr}"
        assert str(custom_exp / "findings") in result.stdout

    def test_databricks_path_format_works(self, tmp_path):
        dbfs_path = "/dbfs/mnt/catalog/experiments"

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                f"""
import os
os.environ['CR_EXPERIMENTS_DIR'] = '{dbfs_path}'

from customer_retention.core.config.experiments import (
    EXPERIMENTS_DIR, FINDINGS_DIR, get_experiments_dir
)

print(f"EXPERIMENTS_DIR: {{EXPERIMENTS_DIR}}")
print(f"FINDINGS_DIR: {{FINDINGS_DIR}}")
print(f"get_experiments_dir(): {{get_experiments_dir()}}")
""",
            ],
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
            "identifier_columns": ["id"],
        }

        import yaml

        (findings_dir / "test_dataset_findings.yaml").write_text(yaml.dump(mock_findings))

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                f"""
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
""",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Failed: {result.stderr}"
        assert "Found 1 findings files" in result.stdout
        assert "test_dataset_findings.yaml" in result.stdout


class TestSpecificNotebooks:
    @pytest.mark.parametrize(
        "notebook_name",
        [
            "01_data_discovery.ipynb",
            "04_column_deep_dive.ipynb",
            "03_dataset_merge.ipynb",
            "10_spec_generation.ipynb",
        ],
    )
    def test_key_notebooks_have_correct_config(self, notebook_name):
        nb_path = NOTEBOOKS_DIR / notebook_name
        if not nb_path.exists():
            pytest.skip(f"{notebook_name} not found")

        for _, source in _read_notebook_cells(nb_path):
            if "customer_retention.core.config.experiments" in source:
                assert "import" in source and "FINDINGS_DIR" in source, (
                    f"{notebook_name}: FINDINGS_DIR should be imported, not defined"
                )
                return

        pytest.fail(f"{notebook_name}: Missing experiments config import")


class TestScoringValidationNotebook:
    NOTEBOOK_PATH = NOTEBOOKS_DIR / "11_scoring_validation.ipynb"

    def test_notebook_exists(self):
        assert self.NOTEBOOK_PATH.exists(), "11_scoring_validation.ipynb not found"

    def test_notebook_is_valid_json(self):
        with open(self.NOTEBOOK_PATH, "r", encoding="utf-8") as f:
            nb = json.load(f)
        assert "cells" in nb
        assert "metadata" in nb
        assert nb["nbformat"] >= 4

    def test_notebook_has_expected_sections(self):
        with open(self.NOTEBOOK_PATH, "r", encoding="utf-8") as f:
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
        with open(self.NOTEBOOK_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        assert "customer_retention.core.config.experiments" in content

    def test_notebook_uses_transform_executor(self):
        with open(self.NOTEBOOK_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        assert "TransformExecutor" in content
        assert "load_artifact_store" in content
        assert "LabelEncoder().fit_transform" not in content

    def test_notebook_has_no_jinja2(self):
        with open(self.NOTEBOOK_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        assert "{{ " not in content
        assert "{% " not in content

    def test_notebook_code_cells_are_valid_python(self):
        for i, source in _read_notebook_cells(self.NOTEBOOK_PATH):
            if not source.strip():
                continue
            try:
                compile(source, f"11_scoring_validation.ipynb:cell_{i}", "exec")
            except SyntaxError as e:
                pytest.fail(f"Cell {i} has syntax error: {e}")

    def test_notebook_has_validation_import(self):
        with open(self.NOTEBOOK_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        assert "validate_feature_transformation" in content


class TestNotebook03Architecture:
    NOTEBOOK_PATH = NOTEBOOKS_DIR / "03_dataset_merge.ipynb"

    def _all_code(self):
        return "\n".join(src for _, src in _read_notebook_cells(self.NOTEBOOK_PATH))

    def test_nb03_standardizes_entity_column(self):
        code = self._all_code()
        assert 'MergeConfig(entity_key="entity_id")' in code, "NB03 must always use MergeConfig(entity_key='entity_id')"

    def test_nb03_renames_entity_column(self):
        code = self._all_code()
        assert "entity_id" in code
        assert "rename" in code, "NB03 must rename original entity column to entity_id"


class TestNotebook04Architecture:
    NOTEBOOK_PATH = NOTEBOOKS_DIR / "04_column_deep_dive.ipynb"

    def _read_content(self):
        with open(self.NOTEBOOK_PATH, "r", encoding="utf-8") as f:
            return f.read()

    def test_nb04_uses_distributed_on_databricks(self):
        content = self._read_content()
        assert "require_silver_merged_distributed" in content, (
            "NB04 must use require_silver_merged_distributed for Databricks"
        )
        assert "is_databricks" in content, "NB04 must branch on is_databricks()"


class TestNotebook05Architecture:
    NOTEBOOK_PATH = NOTEBOOKS_DIR / "05_relationship_analysis.ipynb"

    def _read_content(self):
        with open(self.NOTEBOOK_PATH, "r", encoding="utf-8") as f:
            return f.read()

    def test_nb05_uses_distributed_on_databricks(self):
        content = self._read_content()
        assert "require_silver_merged_distributed" in content, (
            "NB05 must use require_silver_merged_distributed for Databricks"
        )
        assert "is_databricks" in content, "NB05 must branch on is_databricks()"


class TestNotebook06Architecture:
    NOTEBOOK_PATH = NOTEBOOKS_DIR / "06_feature_opportunities.ipynb"

    def _read_content(self):
        with open(self.NOTEBOOK_PATH, "r", encoding="utf-8") as f:
            return f.read()

    def test_nb06_uses_prefer_merged(self):
        content = self._read_content()
        assert "prefer_merged=True" in content, "NB06 must use prefer_merged=True for load_notebook_findings"

    def test_nb06_uses_require_silver_merged(self):
        content = self._read_content()
        assert "require_silver_merged" in content, "NB06 must use require_silver_merged"
        assert "load_active_dataset(" not in content, "NB06 must not fall back to load_active_dataset"

    def test_nb06_uses_distributed_on_databricks(self):
        content = self._read_content()
        assert "require_silver_merged_distributed" in content, (
            "NB06 must use require_silver_merged_distributed for Databricks"
        )
        assert "is_databricks" in content, "NB06 must branch on is_databricks()"

    def test_nb06_uses_merged_recommendations_path(self):
        content = self._read_content()
        assert "merged_recommendations_path" in content, "NB06 must use namespace merged_recommendations_path"


class TestNotebook07Architecture:
    NOTEBOOK_PATH = NOTEBOOKS_DIR / "07_modeling_readiness.ipynb"

    def _read_content(self):
        with open(self.NOTEBOOK_PATH, "r", encoding="utf-8") as f:
            return f.read()

    def test_nb07_uses_prefer_merged(self):
        content = self._read_content()
        assert "prefer_merged=True" in content, "NB07 must use prefer_merged=True for load_notebook_findings"

    def test_nb07_uses_require_silver_merged(self):
        content = self._read_content()
        assert "require_silver_merged" in content, "NB07 must use require_silver_merged"
        assert "load_active_dataset(" not in content, "NB07 must not fall back to load_active_dataset"

    def test_nb07_uses_distributed_on_databricks(self):
        content = self._read_content()
        assert "require_silver_merged_distributed" in content, (
            "NB07 must use require_silver_merged_distributed for Databricks"
        )
        assert "is_databricks" in content, "NB07 must branch on is_databricks()"


class TestNotebook10Architecture:
    NOTEBOOK_PATH = NOTEBOOKS_DIR / "10_spec_generation.ipynb"

    def _read_content(self):
        with open(self.NOTEBOOK_PATH, "r", encoding="utf-8") as f:
            return f.read()

    def test_nb10_uses_namespace_discovery(self):
        content = self._read_content()
        assert "discover_all_findings" in content or "merged_recommendations_path" in content, (
            "NB10 must use namespace discovery methods, not glob patterns"
        )

    def test_nb10_no_findings_dir_glob(self):
        content = self._read_content()
        assert 'findings_dir.glob("*_findings' not in content, (
            "NB10 must not glob findings_dir for findings files — use namespace discovery"
        )
        assert 'findings_dir.glob("*_recommendations' not in content, (
            "NB10 must not glob findings_dir for recommendations — use namespace.merged_recommendations_path"
        )


class TestNotebook08Architecture:
    NOTEBOOK_PATH = NOTEBOOKS_DIR / "08_baseline_experiments.ipynb"

    def _read_content(self):
        with open(self.NOTEBOOK_PATH, "r", encoding="utf-8") as f:
            return f.read()

    def test_nb08_uses_prefer_merged(self):
        content = self._read_content()
        assert "prefer_merged=True" in content, "NB08 must use prefer_merged=True for load_notebook_findings"
        assert "prefer_aggregated=True" not in content, "NB08 must not use prefer_aggregated=True"

    def test_nb08_uses_require_silver_merged(self):
        content = self._read_content()
        assert "require_silver_merged" in content, "NB08 must use require_silver_merged"
        assert "load_active_dataset(" not in content, "NB08 must not fall back to load_active_dataset"

    def test_nb08_uses_temporal_split_always(self):
        content = self._read_content()
        assert "SplitStrategy.TEMPORAL" in content or "TrainingPreparator" in content, (
            "NB08 must use SplitStrategy.TEMPORAL or TrainingPreparator (which enforces temporal)"
        )
        assert "RANDOM_STRATIFIED" not in content, "NB08 must not use RANDOM_STRATIFIED"

    def test_nb08_uses_temporal_cv_always(self):
        content = self._read_content()
        assert "CVStrategy.TEMPORAL_ENTITY" in content, "NB08 must use temporal entity CV"
        assert "cross_val_score" not in content, "NB08 must not use sklearn cross_val_score"

    def test_nb08_has_cv_folds_config_cell(self):
        import json

        with open(self.NOTEBOOK_PATH, "r", encoding="utf-8") as f:
            nb = json.load(f)
        config_cells = [
            c
            for c in nb["cells"]
            if c["cell_type"] == "code"
            and c["source"]
            and "@cr:config" in c["source"][0]
            and "CV_FOLDS" in "".join(c["source"])
        ]
        assert config_cells, "NB08 must have a @cr:config cell defining CV_FOLDS"

    def test_nb08_train_models_uses_cv_folds_variable(self):
        import json

        with open(self.NOTEBOOK_PATH, "r", encoding="utf-8") as f:
            nb = json.load(f)
        train_cell = next(
            c for c in nb["cells"] if c["cell_type"] == "code" and c["source"] and "train_models" in c["source"][0]
        )
        source = "".join(train_cell["source"])
        assert "n_splits=CV_FOLDS" in source, "train_models must use CV_FOLDS variable for n_splits"

    def test_nb08_validates_as_of_date_column(self):
        content = self._read_content()
        assert "as_of_date" in content and "missing" in content.lower(), "NB08 must validate as_of_date column"

    def test_nb08_validates_entity_id_column(self):
        content = self._read_content()
        assert "entity_id" in content and "missing" in content.lower(), "NB08 must validate entity_id column"


_CELL_ID_PATTERN = re.compile(r"^[a-f0-9]{8}$")


class TestCellIdStandardization:
    def test_all_cells_have_standardized_ids(self, notebook_list):
        for nb_path in notebook_list:
            with open(nb_path, "r", encoding="utf-8") as f:
                nb = json.load(f)
            for i, cell in enumerate(nb.get("cells", [])):
                cell_id = cell.get("id", "")
                assert _CELL_ID_PATTERN.match(cell_id), f"{nb_path.name} cell {i} has non-standard ID: {cell_id!r}"

    def test_no_duplicate_ids_within_notebook(self, notebook_list):
        for nb_path in notebook_list:
            with open(nb_path, "r", encoding="utf-8") as f:
                nb = json.load(f)
            ids = [cell.get("id") for cell in nb.get("cells", [])]
            assert len(ids) == len(set(ids)), f"{nb_path.name} has duplicate cell IDs"

    def test_cell_ids_are_valid_format(self, notebook_list):
        for nb_path in notebook_list:
            with open(nb_path, "r", encoding="utf-8") as f:
                nb = json.load(f)
            for cell in nb.get("cells", []):
                cell_id = cell.get("id", "")
                assert cell_id, f"{nb_path.name}: cell missing ID"
                assert len(cell_id) <= 20, f"{nb_path.name}: cell ID too long: {cell_id!r}"

    def test_all_code_cells_have_explicit_tags(self, notebook_list):
        tag_re = re.compile(
            r"^# @cr:(config|user_code|code)"
            r" name='[^']+'"
            r" id=[a-f0-9]{8}$"
        )
        for nb_path in notebook_list:
            with open(nb_path, "r", encoding="utf-8") as f:
                nb = json.load(f)
            for i, cell in enumerate(nb.get("cells", [])):
                if cell.get("cell_type") != "code":
                    continue
                source = "".join(cell.get("source", []))
                first_line = source.split("\n")[0].rstrip() if source else ""
                assert tag_re.match(first_line), (
                    f"{nb_path.name} cell {i} ({cell.get('id')}): code cell missing valid tag (got {first_line!r})"
                )

    def test_embedded_id_matches_cell_id(self, notebook_list):
        _code_id_re = re.compile(r"id=([a-f0-9]{8})")
        _doc_id_re = re.compile(r"\[//\]: # \(cr:doc\s.*?id=([a-f0-9]{8})\)")
        for nb_path in notebook_list:
            with open(nb_path, "r", encoding="utf-8") as f:
                nb = json.load(f)
            for i, cell in enumerate(nb.get("cells", [])):
                source = "".join(cell.get("source", []))
                first_line = source.split("\n")[0].rstrip() if source else ""
                if cell.get("cell_type") == "code":
                    m = _code_id_re.search(first_line)
                elif cell.get("cell_type") == "markdown":
                    m = _doc_id_re.match(first_line)
                else:
                    continue
                if m:
                    assert cell.get("id") == m.group(1), (
                        f"{nb_path.name} cell {i}: embedded id={m.group(1)} does not match cell.id={cell.get('id')!r}"
                    )

    def test_all_markdown_cells_have_doc_tags(self, notebook_list):
        doc_tag_re = re.compile(
            r"^\[//\]: # \(cr:doc"
            r" name='[^']+'"
            r" id=[a-f0-9]{8}"
            r"\)$"
        )
        for nb_path in notebook_list:
            with open(nb_path, "r", encoding="utf-8") as f:
                nb = json.load(f)
            for i, cell in enumerate(nb.get("cells", [])):
                if cell.get("cell_type") != "markdown":
                    continue
                source = "".join(cell.get("source", []))
                first_line = source.split("\n")[0].rstrip() if source else ""
                assert doc_tag_re.match(first_line), (
                    f"{nb_path.name} cell {i} ({cell.get('id')}): "
                    f"markdown cell missing valid doc tag (got {first_line!r})"
                )

    def test_config_cells_have_magic_comments(self, notebook_list):
        for nb_path in notebook_list:
            with open(nb_path, "r", encoding="utf-8") as f:
                nb = json.load(f)
            for i, cell in enumerate(nb.get("cells", [])):
                if cell.get("cell_type") != "code":
                    continue
                source = "".join(cell.get("source", []))
                lines = source.split("\n")
                all_caps_assignments = [
                    line for line in lines if re.match(r"^[A-Z][A-Z_0-9]+ *= *", line) and not line.startswith("#")
                ]
                has_non_config = any(
                    line.strip()
                    and not line.startswith("#")
                    and not re.match(r"^[A-Z][A-Z_0-9]+ *= *", line)
                    and not line.strip().startswith('"')
                    and not line.strip().startswith("'")
                    and not line.strip().startswith("{")
                    and not line.strip().startswith("[")
                    and not line.strip().startswith("(")
                    and not line.strip().startswith(")")
                    and not line.strip().startswith("]")
                    and not line.strip().startswith("}")
                    and not line.strip() == ""
                    for line in lines
                )
                if all_caps_assignments and not has_non_config:
                    first_line = lines[0].rstrip() if lines else ""
                    if not (first_line.startswith("# @cr:config") or first_line.startswith("# @cr:user_code")):
                        pytest.fail(
                            f"{nb_path.name} cell {i} ({cell.get('id')}): pure config cell missing magic comment"
                        )


class TestReleaseStageMemoryGuard:
    EXPECTED_NOTEBOOKS = [
        "00_start_here.ipynb",
        "01_data_discovery.ipynb",
        "01a_temporal_deep_dive.ipynb",
        "01a_a_temporal_text_deep_dive.ipynb",
        "01b_temporal_quality.ipynb",
        "01c_temporal_patterns.ipynb",
        "01d_event_aggregation.ipynb",
        "02_source_integrity.ipynb",
        "03_dataset_merge.ipynb",
        "04_column_deep_dive.ipynb",
        "04a_text_columns_deep_dive.ipynb",
        "05_relationship_analysis.ipynb",
        "06_feature_opportunities.ipynb",
        "07_modeling_readiness.ipynb",
        "08_baseline_experiments.ipynb",
        "09_business_alignment.ipynb",
        "10_spec_generation.ipynb",
        "11_scoring_validation.ipynb",
        "12_view_documentation.ipynb",
    ]

    def test_all_notebooks_have_release_stage_memory(self):
        for nb_name in self.EXPECTED_NOTEBOOKS:
            nb_path = NOTEBOOKS_DIR / nb_name
            assert nb_path.exists(), f"{nb_name} not found"
            code_cells = _read_notebook_cells(nb_path)
            has_release = any("release_stage_memory" in src for _, src in code_cells)
            assert has_release, f"{nb_name} missing release_stage_memory cleanup cell"

    def test_release_stage_memory_is_last_code_cell(self):
        for nb_name in self.EXPECTED_NOTEBOOKS:
            nb_path = NOTEBOOKS_DIR / nb_name
            code_cells = _read_notebook_cells(nb_path)
            assert code_cells, f"{nb_name} has no code cells"
            last_idx, last_src = code_cells[-1]
            assert "release_stage_memory" in last_src, (
                f"{nb_name}: release_stage_memory is not the last code cell "
                f"(last code cell {last_idx} contains: {last_src[:80]!r})"
            )


class TestNoBareExceptInCoreMetadata:
    CORE_FILES = [
        Path("src/customer_retention/analysis/auto_explorer/session.py"),
        Path("src/customer_retention/core/config/experiments.py"),
        Path("src/customer_retention/analysis/notebook_progress.py"),
        Path("scripts/notebooks/run_exploration.py"),
    ]

    _PATTERN = re.compile(
        r"except\s+Exception\s*:\s*\n\s*(?:pass|return\s+None)\s*$",
        re.MULTILINE,
    )

    def test_no_bare_except_in_core_metadata_files(self):
        project_root = Path(__file__).parent.parent.parent
        violations = []
        for rel_path in self.CORE_FILES:
            full_path = project_root / rel_path
            if not full_path.exists():
                continue
            content = full_path.read_text()
            for match in self._PATTERN.finditer(content):
                line_no = content[: match.start()].count("\n") + 1
                violations.append(f"{rel_path}:{line_no}")
        if violations:
            pytest.fail(
                "Found bare 'except Exception: pass/return None' in core files:\n"
                + "\n".join(f"  - {v}" for v in violations)
            )
