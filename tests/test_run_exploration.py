"""Tests for scripts/notebooks/run_exploration.py multi-dataset logic."""
import json
import os

# Ensure the scripts directory is importable
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "notebooks"))

from run_exploration import (
    _NOTEBOOK_TIMEOUT_MULTIPLIERS,
    PER_DATASET_STEMS,
    SETUP_NOTEBOOKS,
    ProgressiveErrorLog,
    _detect_global_skip_set,
    _detect_skip_set_for_dataset,
    _execute_one,
    _find_dataset_findings,
    _has_text_columns_for_dataset,
    _load_dataset_context,
    _ordered_datasets,
    _resolve_namespace,
    _run_multi_dataset_flow,
    _set_data_path,
    run_all,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_OBJECTIVE = {
    "objective": "immediate_risk",
    "priority": "primary",
    "anchor": "now",
}


def _project_ctx(*, datasets, target_dataset, target_column="churned", entity_column="customer_id"):
    return {
        "project_name": "test",
        "target_dataset": target_dataset,
        "target_column": target_column,
        "entity_column": entity_column,
        "primary_objective": "immediate_risk",
        "objectives": [_DEFAULT_OBJECTIVE],
        "datasets": datasets,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def findings_dir(tmp_path):
    d = tmp_path / "findings"
    d.mkdir()
    return d


_TEST_RUN_ID = "test-run"


@pytest.fixture()
def multi_context_yaml(tmp_path, findings_dir):
    """Create a project_context.yaml in a namespace run dir."""
    data = _project_ctx(
        target_dataset="profiles",
        datasets={
            "profiles": {
                "name": "profiles",
                "path": "../fixtures/profiles.csv",
                "role": "target",
                "entity_column": "customer_id",
                "target_candidates": ["churned"],
            },
            "transactions": {
                "name": "transactions",
                "path": "../fixtures/transactions.csv",
                "role": "feature",
                "join_keys": ["customer_id"],
                "join_to": "profiles",
                "relationship": "many_to_one",
            },
            "tickets": {
                "name": "tickets",
                "path": "../fixtures/tickets.csv",
                "role": "feature",
                "join_keys": ["customer_id"],
                "join_to": "profiles",
                "relationship": "many_to_one",
            },
        },
    )
    run_dir = tmp_path / "runs" / _TEST_RUN_ID
    run_dir.mkdir(parents=True)
    path = run_dir / "project_context.yaml"
    path.write_text(yaml.dump(data, default_flow_style=False))
    return path


@pytest.fixture()
def single_context_yaml(tmp_path, findings_dir):
    """Create a project_context.yaml in a namespace run dir."""
    data = _project_ctx(
        target_dataset="retail",
        datasets={
            "retail": {
                "name": "retail",
                "path": "../fixtures/retail.csv",
                "role": "target",
                "entity_column": "customer_id",
                "target_candidates": ["churned"],
            },
        },
    )
    run_dir = tmp_path / "runs" / _TEST_RUN_ID
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "project_context.yaml"
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
        assert "04a_text_columns_deep_dive" in skip

    def test_event_level_keeps_temporal(self, findings_dir, event_findings_yaml):
        skip, reasons = _detect_skip_set_for_dataset(findings_dir, "transactions")
        assert "01a_temporal_deep_dive" not in skip
        assert "01b_temporal_quality" not in skip

    def test_text_columns_keep_text_notebooks(self, findings_dir, text_findings_yaml):
        skip, reasons = _detect_skip_set_for_dataset(findings_dir, "tickets")
        assert "04a_text_columns_deep_dive" not in skip
        # 01a_a_temporal_text_deep_dive is in BOTH TEMPORAL and TEXT sets;
        # for a non-event dataset with text, temporal skip still applies
        assert "01a_a_temporal_text_deep_dive" in skip

    def test_missing_findings_skips_temporal_and_text(self, findings_dir):
        skip, reasons = _detect_skip_set_for_dataset(findings_dir, "nonexistent")
        assert "01a_temporal_deep_dive" in skip
        assert "04a_text_columns_deep_dive" in skip

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
        nb_path = tmp_path / "04_column_deep_dive.ipynb"
        nb_path.write_text("{}")
        results = {}
        timings = {}

        ok = _execute_one(
            nb_path, "profiles", results, timings,
            set(), {}, True, 600, "python3",
        )
        assert ok is True
        assert results["04_column_deep_dive:profiles"] == "DRY_RUN"

    def test_result_key_without_dataset(self, tmp_path):
        nb_path = tmp_path / "03_dataset_merge.ipynb"
        nb_path.write_text("{}")
        results = {}
        timings = {}

        ok = _execute_one(
            nb_path, None, results, timings,
            set(), {}, True, 600, "python3",
        )
        assert "03_dataset_merge" in results

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

    def test_timeout_multiplier_applied(self, tmp_path, monkeypatch):
        captured = {}

        def fake_run_notebook(nb_path, timeout=600, kernel="python3"):
            captured["timeout"] = timeout
            return True, None

        monkeypatch.setattr("run_exploration._run_notebook", fake_run_notebook)
        nb_path = tmp_path / "08_baseline_experiments.ipynb"
        nb_path.write_text("{}")

        _execute_one(nb_path, None, {}, {}, set(), {}, False, 600, "python3")
        assert captured["timeout"] == 600 * _NOTEBOOK_TIMEOUT_MULTIPLIERS["08_baseline_experiments"]

    def test_timeout_multiplier_default_is_1x(self, tmp_path, monkeypatch):
        captured = {}

        def fake_run_notebook(nb_path, timeout=600, kernel="python3"):
            captured["timeout"] = timeout
            return True, None

        monkeypatch.setattr("run_exploration._run_notebook", fake_run_notebook)
        nb_path = tmp_path / "03_dataset_merge.ipynb"
        nb_path.write_text("{}")

        _execute_one(nb_path, None, {}, {}, set(), {}, False, 600, "python3")
        assert captured["timeout"] == 600


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
        findings_dir = tmp_path / "findings"

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
        assert "03_dataset_merge" in global_keys

    def test_single_dataset_no_colon_keys(self, tmp_path, single_context_yaml):
        notebooks_dir = tmp_path / "notebooks"
        notebooks_dir.mkdir()
        self._make_notebooks(notebooks_dir)
        findings_dir = tmp_path / "findings"

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
        findings_dir = tmp_path / "findings"

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
        findings_dir = tmp_path / "findings"

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
        findings_dir = tmp_path / "findings"

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
        findings_dir = tmp_path / "findings"

        results = run_all(
            notebooks_dir,
            findings_dir=findings_dir,
            dry_run=True,
        )

        text_key = "04a_text_columns_deep_dive:tickets"
        assert text_key in results
        assert results[text_key] == "DRY_RUN"


class TestSkipDetectionWithoutFlatFindingsDir:
    def _make_notebooks(self, notebooks_dir):
        from run_exploration import NOTEBOOKS_ORDER

        nb_json = json.dumps({
            "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "cells": [],
        })
        for stem in NOTEBOOKS_ORDER:
            (notebooks_dir / f"{stem}.ipynb").write_text(nb_json)

    def test_entity_skips_temporal_without_entity_findings(self, tmp_path):
        """Entity dataset with context but no findings files still skips temporal.

        Reproduces: findings_dir has context (so multi-dataset triggers)
        but no individual findings files for the entity dataset.
        Skip detection must still work via context/raw-data fallback.
        """
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()

        ctx_data = _project_ctx(
            target_dataset="profiles",
            datasets={
                "profiles": {
                    "name": "profiles",
                    "path": "../fixtures/profiles.csv",
                    "role": "target",
                    "entity_column": "customer_id",
                    "target_candidates": ["churned"],
                },
                "transactions": {
                    "name": "transactions",
                    "path": "../fixtures/transactions.csv",
                    "role": "feature",
                    "join_keys": ["customer_id"],
                    "join_to": "profiles",
                    "relationship": "many_to_one",
                },
            },
        )
        run_dir = tmp_path / "runs" / _TEST_RUN_ID
        run_dir.mkdir(parents=True)
        (run_dir / "project_context.yaml").write_text(yaml.dump(ctx_data))

        notebooks_dir = tmp_path / "notebooks"
        notebooks_dir.mkdir()
        self._make_notebooks(notebooks_dir)

        results = run_all(
            notebooks_dir,
            findings_dir=findings_dir,
            dry_run=True,
        )

        for stem in ("01a_temporal_deep_dive", "01b_temporal_quality",
                      "01c_temporal_patterns", "01d_event_aggregation"):
            key = f"{stem}:profiles"
            assert key in results, f"{key} missing from results"
            assert results[key].startswith("SKIPPED"), (
                f"{key} should be SKIPPED for entity dataset but was {results[key]}"
            )

        key_01a_tx = "01a_temporal_deep_dive:transactions"
        assert results[key_01a_tx] == "DRY_RUN", "event dataset should run temporal"


class TestSingleDatasetSkipDetection:
    def _make_notebooks(self, notebooks_dir):
        from run_exploration import NOTEBOOKS_ORDER

        nb_json = json.dumps({
            "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "cells": [],
        })
        for stem in NOTEBOOKS_ORDER:
            (notebooks_dir / f"{stem}.ipynb").write_text(nb_json)

    def test_text_notebooks_skipped_when_no_text_columns(self, tmp_path, single_context_yaml):
        notebooks_dir = tmp_path / "notebooks"
        notebooks_dir.mkdir()
        self._make_notebooks(notebooks_dir)
        findings_dir = tmp_path / "findings"

        from customer_retention.core.config.column_config import ColumnType

        findings = _make_findings(
            {"age": ColumnType.NUMERIC_CONTINUOUS, "name": ColumnType.CATEGORICAL_NOMINAL},
            row_count=1000, column_count=2,
        )
        findings.save(str(findings_dir / "retail_findings.yaml"))

        results = run_all(
            notebooks_dir, findings_dir=findings_dir, dry_run=True,
        )

        assert results["01a_a_temporal_text_deep_dive"].startswith("SKIPPED")
        assert results["04a_text_columns_deep_dive"].startswith("SKIPPED")

    def test_text_notebooks_skipped_with_auto_drop(self, tmp_path, single_context_yaml):
        notebooks_dir = tmp_path / "notebooks"
        notebooks_dir.mkdir()
        self._make_notebooks(notebooks_dir)
        findings_dir = tmp_path / "findings"

        from customer_retention.core.config.column_config import ColumnType

        findings = _make_findings(
            {"description": ColumnType.TEXT, "status": ColumnType.CATEGORICAL_NOMINAL},
            row_count=5000, column_count=6,
        )
        findings.metadata["auto_drop_text_columns"] = True
        findings.save(str(findings_dir / "retail_findings.yaml"))

        results = run_all(
            notebooks_dir, findings_dir=findings_dir, dry_run=True,
        )

        assert results["01a_a_temporal_text_deep_dive"].startswith("SKIPPED")
        assert results["04a_text_columns_deep_dive"].startswith("SKIPPED")

    def test_text_notebooks_run_when_text_columns_present(self, tmp_path, single_context_yaml):
        notebooks_dir = tmp_path / "notebooks"
        notebooks_dir.mkdir()
        self._make_notebooks(notebooks_dir)
        findings_dir = tmp_path / "findings"

        from customer_retention.core.config.column_config import ColumnType

        findings = _make_findings(
            {"description": ColumnType.TEXT, "status": ColumnType.CATEGORICAL_NOMINAL},
            row_count=5000, column_count=6,
        )
        findings.save(str(findings_dir / "retail_findings.yaml"))

        results = run_all(
            notebooks_dir, findings_dir=findings_dir, dry_run=True,
        )

        assert results["04a_text_columns_deep_dive"] == "DRY_RUN"

    def test_skip_detection_uses_namespace(self, tmp_path):
        from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace

        experiments_dir = tmp_path / "experiments"
        findings_dir = experiments_dir / "findings"
        findings_dir.mkdir(parents=True)

        ns = RunNamespace(root=experiments_dir, run_id="run-001")
        ns_findings = ns.dataset_findings_dir("retail")
        ns_findings.mkdir(parents=True)

        from customer_retention.core.config.column_config import ColumnType

        findings = _make_findings(
            {"age": ColumnType.NUMERIC_CONTINUOUS},
            row_count=1000, column_count=2,
        )
        findings.save(str(ns_findings / "retail_findings.yaml"))

        ctx_data = _project_ctx(
            target_dataset="retail",
            datasets={
                "retail": {
                    "name": "retail",
                    "path": "../fixtures/retail.csv",
                    "role": "target",
                    "entity_column": "customer_id",
                    "target_candidates": ["churned"],
                },
            },
        )

        ns_ctx = ns.project_context_path
        ns_ctx.parent.mkdir(parents=True, exist_ok=True)
        ns_ctx.write_text(yaml.dump(ctx_data))

        (ns.root / "runs" / "run-001").mkdir(parents=True, exist_ok=True)

        notebooks_dir = tmp_path / "notebooks"
        notebooks_dir.mkdir()
        self._make_notebooks(notebooks_dir)

        results = run_all(
            notebooks_dir,
            findings_dir=findings_dir,
            dry_run=True,
            run_id="run-001",
        )

        assert results["04a_text_columns_deep_dive"].startswith("SKIPPED")


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
        findings_dir = tmp_path / "findings"

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
        findings_dir = tmp_path / "findings"

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


class TestSingleDatasetSetDataPath:
    """Single-dataset flow patches per-dataset notebooks from project context."""

    def _make_notebook_with_data_path(self, path, default_path="../fixtures/default.csv"):
        nb = {
            "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "cells": [{
                "cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                "source": [f'DATA_PATH = "{default_path}"\n'],
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

    def test_single_dataset_patches_per_dataset_notebooks(self, tmp_path, single_context_yaml, monkeypatch):
        from run_exploration import NOTEBOOKS_ORDER, PER_DATASET_STEMS

        notebooks_dir = tmp_path / "notebooks"
        notebooks_dir.mkdir()
        findings_dir = tmp_path / "findings"

        nb_json = json.dumps({
            "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "cells": [],
        })
        for stem in NOTEBOOKS_ORDER:
            nb_path = notebooks_dir / f"{stem}.ipynb"
            if stem in PER_DATASET_STEMS:
                self._make_notebook_with_data_path(nb_path, "../fixtures/WRONG.csv")
            else:
                nb_path.write_text(nb_json)

        monkeypatch.setattr("run_exploration._run_notebook", lambda *a, **kw: (True, None))

        run_all(notebooks_dir, findings_dir=findings_dir, dry_run=False, timeout=600)

        for stem in PER_DATASET_STEMS:
            nb_path = notebooks_dir / f"{stem}.ipynb"
            assert self._read_data_path(nb_path) == "../fixtures/retail.csv", (
                f"{stem} DATA_PATH not patched from project context"
            )

    def test_single_dataset_dry_run_does_not_patch(self, tmp_path, single_context_yaml):
        from run_exploration import NOTEBOOKS_ORDER, PER_DATASET_STEMS

        notebooks_dir = tmp_path / "notebooks"
        notebooks_dir.mkdir()
        findings_dir = tmp_path / "findings"

        nb_json = json.dumps({
            "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "cells": [],
        })
        for stem in NOTEBOOKS_ORDER:
            nb_path = notebooks_dir / f"{stem}.ipynb"
            if stem in PER_DATASET_STEMS:
                self._make_notebook_with_data_path(nb_path, "../fixtures/WRONG.csv")
            else:
                nb_path.write_text(nb_json)

        run_all(notebooks_dir, findings_dir=findings_dir, dry_run=True, timeout=600)

        for stem in PER_DATASET_STEMS:
            nb_path = notebooks_dir / f"{stem}.ipynb"
            assert self._read_data_path(nb_path) == "../fixtures/WRONG.csv"

    def test_no_context_does_not_patch(self, tmp_path):
        from run_exploration import NOTEBOOKS_ORDER, PER_DATASET_STEMS

        notebooks_dir = tmp_path / "notebooks"
        notebooks_dir.mkdir()
        findings_dir = tmp_path / "empty_findings"
        findings_dir.mkdir()

        nb_json = json.dumps({
            "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "cells": [],
        })
        for stem in NOTEBOOKS_ORDER:
            nb_path = notebooks_dir / f"{stem}.ipynb"
            if stem in PER_DATASET_STEMS:
                self._make_notebook_with_data_path(nb_path, "../fixtures/original.csv")
            else:
                nb_path.write_text(nb_json)

        run_all(notebooks_dir, findings_dir=findings_dir, dry_run=True, timeout=600)

        for stem in PER_DATASET_STEMS:
            nb_path = notebooks_dir / f"{stem}.ipynb"
            assert self._read_data_path(nb_path) == "../fixtures/original.csv"


# ---------------------------------------------------------------------------
# _resolve_namespace
# ---------------------------------------------------------------------------


class TestResolveNamespace:
    def test_returns_namespace_with_run_id(self, tmp_path):
        experiments_dir = tmp_path / "experiments"
        runs_dir = experiments_dir / "runs" / "my-run-123"
        runs_dir.mkdir(parents=True)
        findings_dir = experiments_dir / "findings"
        findings_dir.mkdir(parents=True)

        ns = _resolve_namespace(findings_dir, run_id="my-run-123")
        assert ns is not None
        assert ns.run_id == "my-run-123"
        assert ns.root == experiments_dir

    def test_returns_namespace_from_latest(self, tmp_path):
        experiments_dir = tmp_path / "experiments"
        run_dir = experiments_dir / "runs" / "latest-abc"
        run_dir.mkdir(parents=True)
        (run_dir / "project_context.yaml").write_text("run_id: latest-abc")
        findings_dir = experiments_dir / "findings"
        findings_dir.mkdir(parents=True)

        ns = _resolve_namespace(findings_dir, run_id=None)
        assert ns is not None
        assert ns.run_id == "latest-abc"

    def test_returns_none_when_no_runs(self, tmp_path):
        experiments_dir = tmp_path / "experiments"
        findings_dir = experiments_dir / "findings"
        findings_dir.mkdir(parents=True)

        ns = _resolve_namespace(findings_dir, run_id=None)
        assert ns is None

    def test_derives_experiments_dir_from_findings_parent(self, tmp_path):
        experiments_dir = tmp_path / "my_experiments"
        runs_dir = experiments_dir / "runs" / "run-xyz"
        runs_dir.mkdir(parents=True)
        findings_dir = experiments_dir / "findings"
        findings_dir.mkdir(parents=True)

        ns = _resolve_namespace(findings_dir, run_id="run-xyz")
        assert ns is not None
        assert ns.root == experiments_dir

    def test_handles_non_findings_dir_name(self, tmp_path):
        custom_dir = tmp_path / "custom_output"
        custom_dir.mkdir(parents=True)

        ns = _resolve_namespace(custom_dir, run_id=None)
        assert ns is None


# ---------------------------------------------------------------------------
# _find_dataset_findings with namespace
# ---------------------------------------------------------------------------


class TestFindDatasetFindingsNamespace:
    def test_finds_in_namespace_dir(self, tmp_path):
        from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace

        experiments_dir = tmp_path / "experiments"
        ns = RunNamespace(root=experiments_dir, run_id="run-001")
        ns_findings = ns.dataset_findings_dir("transactions")
        ns_findings.mkdir(parents=True)
        expected = ns_findings / "transactions_findings.yaml"
        expected.write_text("dummy: true")

        flat_dir = experiments_dir / "findings"
        flat_dir.mkdir(parents=True)

        result = _find_dataset_findings(flat_dir, "transactions", namespace=ns)
        assert result == expected

    def test_falls_back_to_flat_dir(self, tmp_path):
        from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace

        experiments_dir = tmp_path / "experiments"
        ns = RunNamespace(root=experiments_dir, run_id="run-001")

        flat_dir = experiments_dir / "findings"
        flat_dir.mkdir(parents=True)
        flat_file = flat_dir / "transactions_findings.yaml"
        flat_file.write_text("dummy: true")

        result = _find_dataset_findings(flat_dir, "transactions", namespace=ns)
        assert result == flat_file

    def test_prefers_namespace_over_flat(self, tmp_path):
        from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace

        experiments_dir = tmp_path / "experiments"
        ns = RunNamespace(root=experiments_dir, run_id="run-001")
        ns_findings = ns.dataset_findings_dir("transactions")
        ns_findings.mkdir(parents=True)
        ns_file = ns_findings / "transactions_findings.yaml"
        ns_file.write_text("namespace: true")

        flat_dir = experiments_dir / "findings"
        flat_dir.mkdir(parents=True)
        flat_file = flat_dir / "transactions_findings.yaml"
        flat_file.write_text("flat: true")

        result = _find_dataset_findings(flat_dir, "transactions", namespace=ns)
        assert result == ns_file

    def test_finds_aggregated_in_namespace(self, tmp_path):
        from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace

        experiments_dir = tmp_path / "experiments"
        ns = RunNamespace(root=experiments_dir, run_id="run-001")
        ns_findings = ns.dataset_findings_dir("transactions")
        ns_findings.mkdir(parents=True)
        agg = ns_findings / "transactions_aggregated_findings.yaml"
        agg.write_text("aggregated: true")

        flat_dir = experiments_dir / "findings"
        flat_dir.mkdir(parents=True)

        result = _find_dataset_findings(flat_dir, "transactions", namespace=ns)
        assert result == agg

    def test_no_namespace_uses_flat_dir(self, tmp_path):
        flat_dir = tmp_path / "findings"
        flat_dir.mkdir()
        flat_file = flat_dir / "sales_findings.yaml"
        flat_file.write_text("data: true")

        result = _find_dataset_findings(flat_dir, "sales", namespace=None)
        assert result == flat_file


# ---------------------------------------------------------------------------
# TestPerDatasetTextNotebook
# ---------------------------------------------------------------------------


class TestPerDatasetTextNotebook:
    def test_04a_in_per_dataset_stems(self):
        assert "04a_text_columns_deep_dive" in PER_DATASET_STEMS

    def test_04a_runs_per_dataset_with_text(self, tmp_path, multi_context_yaml):
        from run_exploration import NOTEBOOKS_ORDER

        notebooks_dir = tmp_path / "notebooks"
        notebooks_dir.mkdir()
        findings_dir = tmp_path / "findings"

        from customer_retention.core.config.column_config import ColumnType

        text_findings = _make_findings(
            {"description": ColumnType.TEXT, "status": ColumnType.CATEGORICAL_NOMINAL},
            row_count=5000, column_count=6,
        )
        text_findings.save(str(findings_dir / "tickets_findings.yaml"))

        nb_json = json.dumps({
            "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "cells": [],
        })
        for stem in NOTEBOOKS_ORDER:
            (notebooks_dir / f"{stem}.ipynb").write_text(nb_json)

        results = run_all(
            notebooks_dir, findings_dir=findings_dir, dry_run=True,
        )

        key = "04a_text_columns_deep_dive:tickets"
        assert key in results
        assert results[key] == "DRY_RUN"

    def test_04a_skipped_per_dataset_without_text(self, tmp_path, multi_context_yaml):
        from run_exploration import NOTEBOOKS_ORDER

        notebooks_dir = tmp_path / "notebooks"
        notebooks_dir.mkdir()
        findings_dir = tmp_path / "findings"

        from customer_retention.core.config.column_config import ColumnType

        no_text = _make_findings(
            {"age": ColumnType.NUMERIC_CONTINUOUS},
            row_count=1000, column_count=2,
        )
        no_text.save(str(findings_dir / "profiles_findings.yaml"))

        nb_json = json.dumps({
            "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "cells": [],
        })
        for stem in NOTEBOOKS_ORDER:
            (notebooks_dir / f"{stem}.ipynb").write_text(nb_json)

        results = run_all(
            notebooks_dir, findings_dir=findings_dir, dry_run=True,
        )

        key = "04a_text_columns_deep_dive:profiles"
        assert key in results
        assert results[key].startswith("SKIPPED")


# ---------------------------------------------------------------------------
# TestAutoDropTextSkip
# ---------------------------------------------------------------------------


class TestAutoDropTextSkip:
    def test_auto_drop_flag_skips_text_notebooks(self, findings_dir):
        from customer_retention.core.config.column_config import ColumnType

        findings = _make_findings(
            {"description": ColumnType.TEXT, "status": ColumnType.CATEGORICAL_NOMINAL},
            row_count=5000, column_count=6,
        )
        findings.metadata["auto_drop_text_columns"] = True
        findings.save(str(findings_dir / "tickets_findings.yaml"))

        assert _has_text_columns_for_dataset(findings_dir, "tickets") is False

    def test_auto_drop_flag_false_allows_text(self, findings_dir):
        from customer_retention.core.config.column_config import ColumnType

        findings = _make_findings(
            {"description": ColumnType.TEXT, "status": ColumnType.CATEGORICAL_NOMINAL},
            row_count=5000, column_count=6,
        )
        findings.metadata["auto_drop_text_columns"] = False
        findings.save(str(findings_dir / "tickets_findings.yaml"))

        assert _has_text_columns_for_dataset(findings_dir, "tickets") is True

    def test_no_auto_drop_flag_backward_compat(self, findings_dir):
        from customer_retention.core.config.column_config import ColumnType

        findings = _make_findings(
            {"description": ColumnType.TEXT, "status": ColumnType.CATEGORICAL_NOMINAL},
            row_count=5000, column_count=6,
        )
        findings.save(str(findings_dir / "tickets_findings.yaml"))

        assert _has_text_columns_for_dataset(findings_dir, "tickets") is True


# ---------------------------------------------------------------------------
# _detect_global_skip_set
# ---------------------------------------------------------------------------


class TestGlobalSkipDetection:
    def test_no_global_text_skip_when_04a_is_per_dataset(self, tmp_path):
        from customer_retention.core.config.column_config import ColumnType

        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()

        f1 = _make_findings(
            {"age": ColumnType.NUMERIC_CONTINUOUS}, row_count=100, column_count=2,
        )
        f1.save(str(findings_dir / "profiles_findings.yaml"))

        f2 = _make_findings(
            {"amount": ColumnType.NUMERIC_CONTINUOUS}, row_count=100, column_count=2,
        )
        f2.save(str(findings_dir / "transactions_findings.yaml"))

        context_data = _project_ctx(
            target_dataset="profiles",
            datasets={
                "profiles": {
                    "name": "profiles",
                    "path": "profiles.csv",
                    "role": "target",
                    "entity_column": "customer_id",
                    "target_candidates": ["churned"],
                },
                "transactions": {
                    "name": "transactions",
                    "path": "transactions.csv",
                    "role": "feature",
                    "join_keys": ["customer_id"],
                    "join_to": "profiles",
                },
            },
        )
        run_dir = tmp_path / "runs" / _TEST_RUN_ID
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "project_context.yaml").write_text(yaml.dump(context_data))
        context = _load_dataset_context(findings_dir)

        skip, reasons = _detect_global_skip_set(findings_dir, context)
        assert "04a_text_columns_deep_dive" not in skip

    def test_global_skip_empty_when_no_global_text_notebooks(self, tmp_path):
        from customer_retention.core.config.column_config import ColumnType

        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()

        f1 = _make_findings(
            {"age": ColumnType.NUMERIC_CONTINUOUS}, row_count=100, column_count=2,
        )
        f1.save(str(findings_dir / "profiles_findings.yaml"))

        f2 = _make_findings(
            {"description": ColumnType.TEXT}, row_count=100, column_count=2,
        )
        f2.save(str(findings_dir / "tickets_findings.yaml"))

        context_data = _project_ctx(
            target_dataset="profiles",
            datasets={
                "profiles": {
                    "name": "profiles",
                    "path": "profiles.csv",
                    "role": "target",
                    "entity_column": "customer_id",
                    "target_candidates": ["churned"],
                },
                "tickets": {
                    "name": "tickets",
                    "path": "tickets.csv",
                    "role": "feature",
                    "join_keys": ["customer_id"],
                    "join_to": "profiles",
                },
            },
        )
        run_dir = tmp_path / "runs" / _TEST_RUN_ID
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "project_context.yaml").write_text(yaml.dump(context_data))
        context = _load_dataset_context(findings_dir)

        skip, reasons = _detect_global_skip_set(findings_dir, context)
        assert "04a_text_columns_deep_dive" not in skip


# ---------------------------------------------------------------------------
# CR_DATASET_ID env var management
# ---------------------------------------------------------------------------


class TestDatasetIdEnvVar:
    def _make_notebooks(self, notebooks_dir):
        from run_exploration import NOTEBOOKS_ORDER

        nb_json = json.dumps({
            "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "cells": [],
        })
        for stem in NOTEBOOKS_ORDER:
            (notebooks_dir / f"{stem}.ipynb").write_text(nb_json)

    def test_cr_dataset_id_set_during_per_dataset_phase(
        self, tmp_path, multi_context_yaml, monkeypatch,
    ):
        notebooks_dir = tmp_path / "notebooks"
        notebooks_dir.mkdir()
        self._make_notebooks(notebooks_dir)
        findings_dir = tmp_path / "findings"

        captured_env: dict[str, str | None] = {}

        original_execute = _execute_one

        def spy_execute(nb_path, dataset_name, *args, **kwargs):
            if dataset_name:
                captured_env[f"{nb_path.stem}:{dataset_name}"] = os.environ.get("CR_DATASET_ID")
            return original_execute(nb_path, dataset_name, *args, **kwargs)

        monkeypatch.setattr("run_exploration._execute_one", spy_execute)

        context = _load_dataset_context(findings_dir)
        from run_exploration import NOTEBOOKS_ORDER

        notebooks = [
            notebooks_dir / f"{s}.ipynb"
            for s in NOTEBOOKS_ORDER
            if (notebooks_dir / f"{s}.ipynb").exists()
        ]
        results, timings = {}, {}

        _run_multi_dataset_flow(
            notebooks_dir, notebooks, findings_dir, context,
            results, timings, dry_run=True, timeout=600, kernel="python3",
        )

        for key, env_val in captured_env.items():
            ds_name = key.split(":")[1]
            assert env_val == ds_name, f"CR_DATASET_ID should be '{ds_name}' but was '{env_val}' for {key}"

    def test_cr_dataset_id_set_to_target_for_global_phase(
        self, tmp_path, multi_context_yaml, monkeypatch,
    ):
        notebooks_dir = tmp_path / "notebooks"
        notebooks_dir.mkdir()
        self._make_notebooks(notebooks_dir)
        findings_dir = tmp_path / "findings"

        global_env_vals: list[str | None] = []

        original_execute = _execute_one

        def spy_execute(nb_path, dataset_name, *args, **kwargs):
            if dataset_name is None and nb_path.stem != "00_start_here":
                global_env_vals.append(os.environ.get("CR_DATASET_ID"))
            return original_execute(nb_path, dataset_name, *args, **kwargs)

        monkeypatch.setattr("run_exploration._execute_one", spy_execute)

        context = _load_dataset_context(findings_dir)
        from run_exploration import NOTEBOOKS_ORDER

        notebooks = [
            notebooks_dir / f"{s}.ipynb"
            for s in NOTEBOOKS_ORDER
            if (notebooks_dir / f"{s}.ipynb").exists()
        ]
        results, timings = {}, {}

        _run_multi_dataset_flow(
            notebooks_dir, notebooks, findings_dir, context,
            results, timings, dry_run=True, timeout=600, kernel="python3",
        )

        assert len(global_env_vals) > 0
        for val in global_env_vals:
            assert val == "profiles", f"CR_DATASET_ID should be 'profiles' but was '{val}'"


# ---------------------------------------------------------------------------
# Critical global stop
# ---------------------------------------------------------------------------


class TestCriticalGlobalStop:
    def _make_notebooks(self, notebooks_dir):
        from run_exploration import NOTEBOOKS_ORDER

        nb_json = json.dumps({
            "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "cells": [],
        })
        for stem in NOTEBOOKS_ORDER:
            (notebooks_dir / f"{stem}.ipynb").write_text(nb_json)

    @staticmethod
    def _make_spy(fail_stem):
        def spy(nb_path, dataset_name, results, timings, *args, **kwargs):
            stem = nb_path.stem
            key = f"{stem}:{dataset_name}" if dataset_name else stem
            if stem == fail_stem and dataset_name is None:
                results[key] = "FAILED: mocked"
                return False
            results[key] = "OK (0s)"
            return True
        return spy

    def test_global_loop_stops_on_critical_failure(
        self, tmp_path, multi_context_yaml, monkeypatch,
    ):
        notebooks_dir = tmp_path / "notebooks"
        notebooks_dir.mkdir()
        self._make_notebooks(notebooks_dir)
        findings_dir = tmp_path / "findings"

        monkeypatch.setattr("run_exploration._execute_one", self._make_spy("03_dataset_merge"))

        context = _load_dataset_context(findings_dir)
        from run_exploration import NOTEBOOKS_ORDER

        notebooks = [
            notebooks_dir / f"{s}.ipynb"
            for s in NOTEBOOKS_ORDER
            if (notebooks_dir / f"{s}.ipynb").exists()
        ]
        results, timings = {}, {}

        _run_multi_dataset_flow(
            notebooks_dir, notebooks, findings_dir, context,
            results, timings, dry_run=False, timeout=600, kernel="python3",
        )

        global_keys = [k for k in results if ":" not in k and k not in SETUP_NOTEBOOKS]
        assert "03_dataset_merge" in global_keys
        assert "06_feature_opportunities" not in global_keys
        assert "07_modeling_readiness" not in global_keys
        assert "08_baseline_experiments" not in global_keys

    def test_global_loop_continues_on_non_critical_failure(
        self, tmp_path, multi_context_yaml, monkeypatch,
    ):
        notebooks_dir = tmp_path / "notebooks"
        notebooks_dir.mkdir()
        self._make_notebooks(notebooks_dir)
        findings_dir = tmp_path / "findings"

        monkeypatch.setattr("run_exploration._execute_one", self._make_spy("04_column_deep_dive"))

        context = _load_dataset_context(findings_dir)
        from run_exploration import NOTEBOOKS_ORDER

        notebooks = [
            notebooks_dir / f"{s}.ipynb"
            for s in NOTEBOOKS_ORDER
            if (notebooks_dir / f"{s}.ipynb").exists()
        ]
        results, timings = {}, {}

        _run_multi_dataset_flow(
            notebooks_dir, notebooks, findings_dir, context,
            results, timings, dry_run=False, timeout=600, kernel="python3",
        )

        global_keys = [k for k in results if ":" not in k and k not in SETUP_NOTEBOOKS]
        assert "06_feature_opportunities" in global_keys
        assert "07_modeling_readiness" in global_keys
        assert "08_baseline_experiments" in global_keys


# ---------------------------------------------------------------------------
# ProgressiveErrorLog header with run parameters
# ---------------------------------------------------------------------------


class TestProgressiveErrorLogHeader:
    def test_log_header_includes_run_params(self, tmp_path):
        params = {
            "notebooks_dir": "/path/to/notebooks",
            "findings_dir": "/path/to/findings",
            "timeout": 600,
            "kernel": "python3",
            "dry_run": False,
            "run_id": "run-123",
            "spark_remote": True,
            "sample_entities": 500,
            "max_grid_dates": 10,
        }
        log = ProgressiveErrorLog(tmp_path, run_params=params)
        content = log.log_path.read_text()
        assert "run_exploration" in content
        assert "timeout:" in content and "600" in content
        assert "kernel:" in content and "python3" in content
        assert "run_id:" in content and "run-123" in content
        assert "spark_remote:" in content and "yes" in content
        assert "sample_entities: 500" in content
        assert "max_grid_dates:" in content and "10" in content

    def test_log_header_omits_none_and_default_params(self, tmp_path):
        params = {
            "notebooks_dir": "/path/to/notebooks",
            "findings_dir": "/path/to/findings",
            "timeout": 600,
            "kernel": "python3",
            "dry_run": False,
            "run_id": None,
            "spark_remote": False,
            "sample_entities": None,
            "max_grid_dates": None,
        }
        log = ProgressiveErrorLog(tmp_path, run_params=params)
        content = log.log_path.read_text()
        assert "run_id" not in content
        assert "sample_entities" not in content
        assert "max_grid_dates" not in content
        assert "spark_remote" not in content

    def test_log_header_backward_compatible_without_params(self, tmp_path):
        log = ProgressiveErrorLog(tmp_path)
        content = log.log_path.read_text()
        assert "run_exploration" in content
        assert content.count("\n") >= 1

    def test_log_header_shows_dry_run(self, tmp_path):
        params = {
            "notebooks_dir": "/nb",
            "findings_dir": "/f",
            "timeout": 300,
            "kernel": "python3",
            "dry_run": True,
            "run_id": None,
            "spark_remote": False,
            "sample_entities": None,
            "max_grid_dates": None,
        }
        log = ProgressiveErrorLog(tmp_path, run_params=params)
        content = log.log_path.read_text()
        assert "dry_run:" in content and "yes" in content

    def test_run_all_writes_params_to_log(self, tmp_path, single_context_yaml):
        from run_exploration import NOTEBOOKS_ORDER

        notebooks_dir = tmp_path / "notebooks"
        notebooks_dir.mkdir()
        nb_json = json.dumps({
            "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "cells": [],
        })
        for stem in NOTEBOOKS_ORDER:
            (notebooks_dir / f"{stem}.ipynb").write_text(nb_json)

        run_all(
            notebooks_dir,
            findings_dir=tmp_path / "findings",
            dry_run=True,
            timeout=300,
            kernel="python3",
            run_id="test-run",
        )

        log_content = (notebooks_dir / "run_exploration.log").read_text()
        assert "timeout:" in log_content and "300" in log_content
        assert "run_id:" in log_content and "test-run" in log_content
        assert "dry_run:" in log_content and "yes" in log_content


class TestCrRunIdSetBeforeSetupNotebooks:
    """CR_RUN_ID must be in env before NB00 runs so the kernel inherits it."""

    def _make_notebooks(self, notebooks_dir):
        from run_exploration import NOTEBOOKS_ORDER

        nb_json = json.dumps({
            "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "cells": [],
        })
        for stem in NOTEBOOKS_ORDER:
            (notebooks_dir / f"{stem}.ipynb").write_text(nb_json)

    def test_cr_run_id_set_before_setup_notebooks(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CR_RUN_ID", raising=False)
        notebooks_dir = tmp_path / "notebooks"
        notebooks_dir.mkdir()
        self._make_notebooks(notebooks_dir)
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()

        captured_env = {}
        original_execute = _execute_one.__wrapped__ if hasattr(_execute_one, "__wrapped__") else _execute_one

        def spy_execute(nb_path, dataset, *args, **kwargs):
            if nb_path.stem in SETUP_NOTEBOOKS:
                captured_env["CR_RUN_ID"] = os.environ.get("CR_RUN_ID")
            return original_execute(nb_path, dataset, *args, **kwargs)

        monkeypatch.setattr("run_exploration._execute_one", spy_execute)
        run_all(
            notebooks_dir, findings_dir=findings_dir,
            dry_run=True, run_id="comparison-abc1234",
        )
        assert captured_env.get("CR_RUN_ID") == "comparison-abc1234"

    def test_cr_run_id_restored_after_run(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CR_RUN_ID", "previous-run")
        notebooks_dir = tmp_path / "notebooks"
        notebooks_dir.mkdir()
        self._make_notebooks(notebooks_dir)
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()

        run_all(
            notebooks_dir, findings_dir=findings_dir,
            dry_run=True, run_id="comparison-abc1234",
        )
        assert os.environ.get("CR_RUN_ID") == "previous-run"

    def test_cr_run_id_cleaned_when_no_prior_env(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CR_RUN_ID", raising=False)
        notebooks_dir = tmp_path / "notebooks"
        notebooks_dir.mkdir()
        self._make_notebooks(notebooks_dir)
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()

        run_all(
            notebooks_dir, findings_dir=findings_dir,
            dry_run=True, run_id="comparison-abc1234",
        )
        assert os.environ.get("CR_RUN_ID") is None

    def test_no_run_id_arg_does_not_set_env(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CR_RUN_ID", raising=False)
        notebooks_dir = tmp_path / "notebooks"
        notebooks_dir.mkdir()
        self._make_notebooks(notebooks_dir)
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()

        captured_env = {}

        def spy_execute(nb_path, dataset, *args, **kwargs):
            if nb_path.stem in SETUP_NOTEBOOKS:
                captured_env["CR_RUN_ID"] = os.environ.get("CR_RUN_ID")
            return _execute_one(nb_path, dataset, *args, **kwargs)

        monkeypatch.setattr("run_exploration._execute_one", spy_execute)
        run_all(
            notebooks_dir, findings_dir=findings_dir,
            dry_run=True, run_id=None,
        )
        assert captured_env.get("CR_RUN_ID") is None
