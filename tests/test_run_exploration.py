"""Tests for scripts/notebooks/run_exploration.py multi-dataset logic."""
import json

# Ensure the scripts directory is importable
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "notebooks"))

from run_exploration import (
    SETUP_NOTEBOOKS,
    _detect_skip_set_for_dataset,
    _execute_one,
    _load_dataset_context,
    _ordered_datasets,
    _run_multi_dataset_flow,
    _set_data_path,
    run_all,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def findings_dir(tmp_path):
    d = tmp_path / "findings"
    d.mkdir()
    return d


@pytest.fixture()
def multi_context_yaml(findings_dir):
    """Create a dataset_context.yaml with 3 datasets (target first)."""
    data = {
        "target_dataset": "profiles",
        "target_column": "churned",
        "entity_column": "customer_id",
        "datasets": {
            "profiles": {
                "path": "../fixtures/profiles.csv",
                "has_target": True,
                "target_column": "churned",
                "entity_column": "customer_id",
            },
            "transactions": {
                "path": "../fixtures/transactions.csv",
                "has_target": False,
                "join_key": "customer_id",
                "join_to": "profiles",
                "relationship": "many_to_one",
            },
            "tickets": {
                "path": "../fixtures/tickets.csv",
                "has_target": False,
                "join_key": "customer_id",
                "join_to": "profiles",
                "relationship": "many_to_one",
            },
        },
    }
    path = findings_dir / "dataset_context.yaml"
    path.write_text(yaml.dump(data, default_flow_style=False))
    return path


@pytest.fixture()
def single_context_yaml(findings_dir):
    """Create a dataset_context.yaml with 1 dataset."""
    data = {
        "target_dataset": "retail",
        "target_column": "churned",
        "entity_column": "customer_id",
        "datasets": {
            "retail": {
                "path": "../fixtures/retail.csv",
                "has_target": True,
                "target_column": "churned",
                "entity_column": "customer_id",
            },
        },
    }
    path = findings_dir / "dataset_context.yaml"
    path.write_text(yaml.dump(data, default_flow_style=False))
    return path


def _make_findings(columns_dict, **kwargs):
    """Helper to build ExplorationFindings with typed columns."""
    from customer_retention.analysis.auto_explorer.findings import (
        ColumnFinding,
        ExplorationFindings,
    )

    columns = {
        name: ColumnFinding(name=name, inferred_type=ctype, confidence=0.95, evidence=[])
        for name, ctype in columns_dict.items()
    }
    return ExplorationFindings(
        source_path="test.csv",
        source_format="csv",
        columns=columns,
        **kwargs,
    )


@pytest.fixture()
def entity_findings_yaml(findings_dir):
    """Create findings that represent an entity-level dataset (no events, no text)."""
    from customer_retention.core.config.column_config import ColumnType

    findings = _make_findings(
        {"age": ColumnType.NUMERIC_CONTINUOUS, "name": ColumnType.CATEGORICAL_NOMINAL},
        row_count=1000,
        column_count=10,
    )
    path = findings_dir / "profiles_findings.yaml"
    findings.save(str(path))
    return path


@pytest.fixture()
def event_findings_yaml(findings_dir):
    """Create findings that represent an event-level dataset."""
    from customer_retention.analysis.auto_explorer.findings import TimeSeriesMetadata
    from customer_retention.core.config.column_config import (
        ColumnType,
        DatasetGranularity,
    )

    findings = _make_findings(
        {"amount": ColumnType.NUMERIC_CONTINUOUS},
        row_count=50000,
        column_count=8,
    )
    findings.time_series_metadata = TimeSeriesMetadata(
        granularity=DatasetGranularity.EVENT_LEVEL,
        temporal_pattern="irregular",
        entity_column="customer_id",
        time_column="event_timestamp",
    )
    path = findings_dir / "transactions_findings.yaml"
    findings.save(str(path))
    return path


@pytest.fixture()
def text_findings_yaml(findings_dir):
    """Create findings with TEXT columns."""
    from customer_retention.core.config.column_config import ColumnType

    findings = _make_findings(
        {"description": ColumnType.TEXT, "status": ColumnType.CATEGORICAL_NOMINAL},
        row_count=5000,
        column_count=6,
    )
    path = findings_dir / "tickets_findings.yaml"
    findings.save(str(path))
    return path


@pytest.fixture()
def notebook_01(tmp_path):
    """Create a minimal notebook with DATA_PATH config cell."""
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": [
            {
                "cell_type": "code",
                "metadata": {},
                "source": ["import pandas as pd\n"],
                "outputs": [],
                "execution_count": None,
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    'DATA_PATH = "../fixtures/default.csv"\n',
                    "# DATA_PATH = '../fixtures/alt.csv'\n",
                    "TARGET_COLUMN = None\n",
                ],
                "outputs": [],
                "execution_count": None,
            },
        ],
    }
    path = tmp_path / "01_data_discovery.ipynb"
    path.write_text(json.dumps(nb, indent=1))
    return path


# ---------------------------------------------------------------------------
# _load_dataset_context
# ---------------------------------------------------------------------------


class TestLoadDatasetContext:
    def test_returns_context_for_multi_dataset(self, findings_dir, multi_context_yaml):
        ctx = _load_dataset_context(findings_dir)
        assert ctx is not None
        assert len(ctx.datasets) == 3
        assert ctx.target_dataset == "profiles"

    def test_returns_context_for_single_dataset(self, findings_dir, single_context_yaml):
        ctx = _load_dataset_context(findings_dir)
        assert ctx is not None
        assert len(ctx.datasets) == 1

    def test_returns_none_when_no_file(self, findings_dir):
        ctx = _load_dataset_context(findings_dir)
        assert ctx is None

    def test_returns_none_when_dir_missing(self, tmp_path):
        ctx = _load_dataset_context(tmp_path / "nonexistent")
        assert ctx is None


# ---------------------------------------------------------------------------
# _ordered_datasets
# ---------------------------------------------------------------------------


class TestOrderedDatasets:
    def test_target_first(self, findings_dir, multi_context_yaml):
        ctx = _load_dataset_context(findings_dir)
        ordered = _ordered_datasets(ctx)
        assert len(ordered) == 3
        assert ordered[0][0] == "profiles"
        assert ordered[0][1] == "../fixtures/profiles.csv"

    def test_non_targets_after(self, findings_dir, multi_context_yaml):
        ctx = _load_dataset_context(findings_dir)
        ordered = _ordered_datasets(ctx)
        non_target_names = [name for name, _ in ordered[1:]]
        assert "transactions" in non_target_names
        assert "tickets" in non_target_names


# ---------------------------------------------------------------------------
# _set_data_path
# ---------------------------------------------------------------------------


class TestSetDataPath:
    def test_replaces_active_data_path(self, notebook_01):
        _set_data_path(notebook_01, "../fixtures/new_data.csv")

        nb = json.loads(notebook_01.read_text())
        config_cell = nb["cells"][1]
        source = "".join(config_cell["source"])
        assert 'DATA_PATH = "../fixtures/new_data.csv"' in source

    def test_preserves_commented_lines(self, notebook_01):
        _set_data_path(notebook_01, "../fixtures/new_data.csv")

        nb = json.loads(notebook_01.read_text())
        config_cell = nb["cells"][1]
        source = "".join(config_cell["source"])
        assert "# DATA_PATH" in source

    def test_preserves_other_config(self, notebook_01):
        _set_data_path(notebook_01, "../fixtures/new_data.csv")

        nb = json.loads(notebook_01.read_text())
        config_cell = nb["cells"][1]
        source = "".join(config_cell["source"])
        assert "TARGET_COLUMN = None" in source

    def test_does_not_modify_non_data_path_cells(self, notebook_01):
        _set_data_path(notebook_01, "../fixtures/new_data.csv")

        nb = json.loads(notebook_01.read_text())
        import_cell = nb["cells"][0]
        source = "".join(import_cell["source"])
        assert "import pandas" in source

    def test_idempotent_double_set(self, notebook_01):
        _set_data_path(notebook_01, "../fixtures/first.csv")
        _set_data_path(notebook_01, "../fixtures/second.csv")

        nb = json.loads(notebook_01.read_text())
        config_cell = nb["cells"][1]
        source = "".join(config_cell["source"])
        assert 'DATA_PATH = "../fixtures/second.csv"' in source
        assert "first.csv" not in source


# ---------------------------------------------------------------------------
# _detect_skip_set_for_dataset
# ---------------------------------------------------------------------------


class TestDetectSkipSetForDataset:
    def test_entity_level_skips_temporal(self, findings_dir, entity_findings_yaml):
        skip, reasons = _detect_skip_set_for_dataset(findings_dir, "profiles")
        assert "01a_temporal_deep_dive" in skip
        assert "01b_temporal_quality" in skip
        assert "01c_temporal_patterns" in skip
        assert "01d_event_aggregation" in skip
        assert all("profiles" in r for r in reasons.values())

    def test_entity_level_skips_text(self, findings_dir, entity_findings_yaml):
        skip, reasons = _detect_skip_set_for_dataset(findings_dir, "profiles")
        assert "02a_text_columns_deep_dive" in skip

    def test_event_level_keeps_temporal(self, findings_dir, event_findings_yaml):
        skip, reasons = _detect_skip_set_for_dataset(findings_dir, "transactions")
        assert "01a_temporal_deep_dive" not in skip
        assert "01b_temporal_quality" not in skip

    def test_text_columns_keep_text_notebooks(self, findings_dir, text_findings_yaml):
        skip, reasons = _detect_skip_set_for_dataset(findings_dir, "tickets")
        assert "02a_text_columns_deep_dive" not in skip
        # 01a_a_temporal_text_deep_dive is in BOTH TEMPORAL and TEXT sets;
        # for a non-event dataset with text, temporal skip still applies
        assert "01a_a_temporal_text_deep_dive" in skip

    def test_missing_findings_skips_temporal_and_text(self, findings_dir):
        skip, reasons = _detect_skip_set_for_dataset(findings_dir, "nonexistent")
        assert "01a_temporal_deep_dive" in skip
        assert "02a_text_columns_deep_dive" in skip

    def test_context_many_to_one_enables_temporal(
        self, findings_dir, multi_context_yaml,
    ):
        """When findings have no time_series_metadata but context says many_to_one,
        temporal notebooks should NOT be skipped."""
        ctx = _load_dataset_context(findings_dir)
        skip, reasons = _detect_skip_set_for_dataset(
            findings_dir, "transactions", context=ctx,
        )
        assert "01a_temporal_deep_dive" not in skip
        assert "01b_temporal_quality" not in skip
        assert "01c_temporal_patterns" not in skip
        assert "01d_event_aggregation" not in skip

    def test_entity_findings_null_metadata_with_context_fallback(
        self, findings_dir, entity_findings_yaml, multi_context_yaml,
    ):
        """Entity-level findings for profiles (no time_series_metadata) +
        context confirms profiles has no relationship → skip temporal."""
        ctx = _load_dataset_context(findings_dir)
        skip, reasons = _detect_skip_set_for_dataset(
            findings_dir, "profiles", context=ctx,
        )
        assert "01a_temporal_deep_dive" in skip

    def test_null_metadata_findings_with_many_to_one_context(
        self, findings_dir, multi_context_yaml,
    ):
        """Findings exist but time_series_metadata is null; context has many_to_one.
        This reproduces the real-world 3set_edi_transactions issue."""
        from customer_retention.core.config.column_config import ColumnType

        findings = _make_findings(
            {"amount": ColumnType.NUMERIC_CONTINUOUS},
            row_count=50000, column_count=8,
        )
        # time_series_metadata deliberately left as None (the bug scenario)
        path = findings_dir / "transactions_findings.yaml"
        findings.save(str(path))

        ctx = _load_dataset_context(findings_dir)
        skip, reasons = _detect_skip_set_for_dataset(
            findings_dir, "transactions", context=ctx,
        )
        assert "01a_temporal_deep_dive" not in skip
        assert "01b_temporal_quality" not in skip
        assert "01c_temporal_patterns" not in skip
        assert "01d_event_aggregation" not in skip


# ---------------------------------------------------------------------------
# _execute_one
# ---------------------------------------------------------------------------


class TestExecuteOne:
    def test_skipped_notebook(self, tmp_path):
        nb_path = tmp_path / "01a_temporal_deep_dive.ipynb"
        nb_path.write_text("{}")
        results = {}
        timings = {}
        skip_set = {"01a_temporal_deep_dive"}
        skip_reasons = {"01a_temporal_deep_dive": "no event-level data"}

        ok = _execute_one(
            nb_path, "my_dataset", results, timings,
            skip_set, skip_reasons, False, 600, "python3",
        )
        assert ok is True
        assert "01a_temporal_deep_dive:my_dataset" in results
        assert results["01a_temporal_deep_dive:my_dataset"].startswith("SKIPPED")

    def test_dry_run(self, tmp_path):
        nb_path = tmp_path / "02_column_deep_dive.ipynb"
        nb_path.write_text("{}")
        results = {}
        timings = {}

        ok = _execute_one(
            nb_path, "profiles", results, timings,
            set(), {}, True, 600, "python3",
        )
        assert ok is True
        assert results["02_column_deep_dive:profiles"] == "DRY_RUN"

    def test_result_key_without_dataset(self, tmp_path):
        nb_path = tmp_path / "05_multi_dataset.ipynb"
        nb_path.write_text("{}")
        results = {}
        timings = {}

        ok = _execute_one(
            nb_path, None, results, timings,
            set(), {}, True, 600, "python3",
        )
        assert "05_multi_dataset" in results

    def test_result_key_with_dataset(self, tmp_path):
        nb_path = tmp_path / "01_data_discovery.ipynb"
        nb_path.write_text("{}")
        results = {}
        timings = {}

        ok = _execute_one(
            nb_path, "my_ds", results, timings,
            set(), {}, True, 600, "python3",
        )
        assert "01_data_discovery:my_ds" in results


# ---------------------------------------------------------------------------
# run_all multi-dataset dispatch
# ---------------------------------------------------------------------------


class TestRunAllDispatch:
    def _make_notebooks(self, notebooks_dir):
        """Create minimal dummy notebooks for all stems."""
        from run_exploration import NOTEBOOKS_ORDER

        nb_json = json.dumps({
            "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "cells": [],
        })
        for stem in NOTEBOOKS_ORDER:
            (notebooks_dir / f"{stem}.ipynb").write_text(nb_json)

    def test_detects_multi_dataset(self, tmp_path, multi_context_yaml):
        notebooks_dir = tmp_path / "notebooks"
        notebooks_dir.mkdir()
        self._make_notebooks(notebooks_dir)
        findings_dir = multi_context_yaml.parent

        results = run_all(
            notebooks_dir,
            findings_dir=findings_dir,
            dry_run=True,
        )

        per_ds_keys = [k for k in results if ":" in k]
        global_keys = [k for k in results if ":" not in k and k not in SETUP_NOTEBOOKS]

        assert len(per_ds_keys) > 0
        assert any("profiles" in k for k in per_ds_keys)
        assert any("transactions" in k for k in per_ds_keys)
        assert any("tickets" in k for k in per_ds_keys)
        assert "05_multi_dataset" in global_keys

    def test_single_dataset_no_colon_keys(self, tmp_path, single_context_yaml):
        notebooks_dir = tmp_path / "notebooks"
        notebooks_dir.mkdir()
        self._make_notebooks(notebooks_dir)
        findings_dir = single_context_yaml.parent

        results = run_all(
            notebooks_dir,
            findings_dir=findings_dir,
            dry_run=True,
        )

        per_ds_keys = [k for k in results if ":" in k]
        assert len(per_ds_keys) == 0

    def test_falls_back_to_single_when_no_context(self, tmp_path):
        notebooks_dir = tmp_path / "notebooks"
        notebooks_dir.mkdir()
        self._make_notebooks(notebooks_dir)
        findings_dir = tmp_path / "empty_findings"
        findings_dir.mkdir()

        results = run_all(
            notebooks_dir,
            findings_dir=findings_dir,
            dry_run=True,
        )

        per_ds_keys = [k for k in results if ":" in k]
        assert len(per_ds_keys) == 0
        assert "01_data_discovery" in results

    def test_per_dataset_stems_all_present(self, tmp_path, multi_context_yaml):
        notebooks_dir = tmp_path / "notebooks"
        notebooks_dir.mkdir()
        self._make_notebooks(notebooks_dir)
        findings_dir = multi_context_yaml.parent

        results = run_all(
            notebooks_dir,
            findings_dir=findings_dir,
            dry_run=True,
        )

        for ds_name in ["profiles", "transactions", "tickets"]:
            ds_keys = [k for k in results if f":{ds_name}" in k]
            ds_stems = {k.split(":")[0] for k in ds_keys}
            assert "01_data_discovery" in ds_stems


class TestMultiDatasetWithSkips:
    def _make_notebooks(self, notebooks_dir):
        from run_exploration import NOTEBOOKS_ORDER

        nb_json = json.dumps({
            "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "cells": [],
        })
        for stem in NOTEBOOKS_ORDER:
            (notebooks_dir / f"{stem}.ipynb").write_text(nb_json)

    def test_entity_dataset_skips_temporal_notebooks(
        self, tmp_path, multi_context_yaml, entity_findings_yaml,
    ):
        notebooks_dir = tmp_path / "notebooks"
        notebooks_dir.mkdir()
        self._make_notebooks(notebooks_dir)
        findings_dir = multi_context_yaml.parent

        results = run_all(
            notebooks_dir,
            findings_dir=findings_dir,
            dry_run=True,
        )

        temporal_keys = [
            k for k in results
            if ":profiles" in k and "temporal" in k.lower()
        ]
        for key in temporal_keys:
            assert results[key].startswith("SKIPPED")

    def test_event_dataset_runs_temporal_notebooks(
        self, tmp_path, multi_context_yaml, event_findings_yaml,
    ):
        notebooks_dir = tmp_path / "notebooks"
        notebooks_dir.mkdir()
        self._make_notebooks(notebooks_dir)
        findings_dir = multi_context_yaml.parent

        results = run_all(
            notebooks_dir,
            findings_dir=findings_dir,
            dry_run=True,
        )

        # Pure temporal notebooks (not requiring text) should run
        for stem in ("01a_temporal_deep_dive", "01b_temporal_quality",
                     "01c_temporal_patterns", "01d_event_aggregation"):
            key = f"{stem}:transactions"
            assert results[key] == "DRY_RUN", f"{key} should run for event dataset"

        # 01a_a_temporal_text_deep_dive requires TEXT columns too;
        # transactions has no text, so it may be skipped
        text_temporal_key = "01a_a_temporal_text_deep_dive:transactions"
        assert text_temporal_key in results

    def test_text_dataset_runs_text_notebooks(
        self, tmp_path, multi_context_yaml, text_findings_yaml,
    ):
        notebooks_dir = tmp_path / "notebooks"
        notebooks_dir.mkdir()
        self._make_notebooks(notebooks_dir)
        findings_dir = multi_context_yaml.parent

        results = run_all(
            notebooks_dir,
            findings_dir=findings_dir,
            dry_run=True,
        )

        text_key = "02a_text_columns_deep_dive:tickets"
        assert text_key in results
        assert results[text_key] == "DRY_RUN"


class TestMultiDatasetSetDataPath:
    def _make_notebook_with_data_path(self, path, default_path="../fixtures/default.csv"):
        nb = {
            "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "cells": [{
                "cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                "source": [
                    f'DATA_PATH = "{default_path}"\n',
                    "# DATA_PATH = '../fixtures/alt.csv'\n",
                ],
            }],
        }
        path.write_text(json.dumps(nb, indent=1))

    def _read_data_path(self, path):
        nb = json.loads(path.read_text())
        for cell in nb["cells"]:
            for line in cell.get("source", []):
                stripped = line.lstrip()
                if stripped.startswith("DATA_PATH") and "=" in stripped and not stripped.startswith("#"):
                    return stripped.split("=", 1)[1].strip().strip('"').strip("'")
        return None

    def test_set_data_path_applied_to_all_per_dataset_notebooks(self, tmp_path, multi_context_yaml, monkeypatch):
        from run_exploration import NOTEBOOKS_ORDER, PER_DATASET_STEMS

        notebooks_dir = tmp_path / "notebooks"
        notebooks_dir.mkdir()
        findings_dir = multi_context_yaml.parent

        nb_json = json.dumps({
            "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "cells": [],
        })
        for stem in NOTEBOOKS_ORDER:
            nb_path = notebooks_dir / f"{stem}.ipynb"
            if stem in PER_DATASET_STEMS:
                self._make_notebook_with_data_path(nb_path)
            else:
                nb_path.write_text(nb_json)

        monkeypatch.setattr("run_exploration._run_notebook", lambda *a, **kw: (True, None))
        context = _load_dataset_context(findings_dir)
        notebooks = [notebooks_dir / f"{s}.ipynb" for s in NOTEBOOKS_ORDER if (notebooks_dir / f"{s}.ipynb").exists()]
        results, timings = {}, {}

        _run_multi_dataset_flow(
            notebooks_dir, notebooks, findings_dir, context,
            results, timings, dry_run=False, timeout=600, kernel="python3",
        )

        last_dataset = list(context.datasets.values())[-1]
        for stem in PER_DATASET_STEMS:
            nb_path = notebooks_dir / f"{stem}.ipynb"
            assert self._read_data_path(nb_path) == last_dataset.path

    def test_set_data_path_noop_for_notebooks_without_data_path_cell(self, tmp_path, multi_context_yaml, monkeypatch):
        from run_exploration import NOTEBOOKS_ORDER, PER_DATASET_STEMS

        notebooks_dir = tmp_path / "notebooks"
        notebooks_dir.mkdir()
        findings_dir = multi_context_yaml.parent

        nb_json = json.dumps({
            "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "cells": [{"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": ["x = 1\n"]}],
        })
        for stem in NOTEBOOKS_ORDER:
            (notebooks_dir / f"{stem}.ipynb").write_text(nb_json)

        monkeypatch.setattr("run_exploration._run_notebook", lambda *a, **kw: (True, None))
        context = _load_dataset_context(findings_dir)
        notebooks = [notebooks_dir / f"{s}.ipynb" for s in NOTEBOOKS_ORDER if (notebooks_dir / f"{s}.ipynb").exists()]
        results, timings = {}, {}

        _run_multi_dataset_flow(
            notebooks_dir, notebooks, findings_dir, context,
            results, timings, dry_run=False, timeout=600, kernel="python3",
        )

        for stem in PER_DATASET_STEMS:
            nb = json.loads((notebooks_dir / f"{stem}.ipynb").read_text())
            source = "".join(nb["cells"][0]["source"])
            assert "DATA_PATH" not in source
