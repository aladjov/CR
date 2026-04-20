"""event_datasets surface invariant (Cycle 002).

Invariant under test: when `discover_findings(prefer_aggregated=True)` returns
a `<name>_aggregated_findings.yaml` whose `time_series_metadata` is null,
`ExplorationManager.list_datasets()` must still recognize the dataset as
event-level by consulting the sibling `<name>_findings.yaml`. Consequently
`MultiDatasetFindings.event_datasets` reflects the source granularity.

Regression target: run `spschurn-e4ad6e1b` produced
`multi_dataset_findings.yaml.event_datasets = []` even though
`contract_findings.yaml.time_series_metadata.granularity = event_level`. Root
cause: the aggregated findings (which `prefer_aggregated=True` returned) carry
`time_series_metadata = null`, so `is_time_series` returned False and the
granularity collapsed to ENTITY_LEVEL.

Fixtures are generic — no client identifiers.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from customer_retention.analysis.auto_explorer import ExplorationFindings
from customer_retention.analysis.auto_explorer.exploration_manager import (
    ExplorationManager,
)
from customer_retention.analysis.auto_explorer.findings import TimeSeriesMetadata
from customer_retention.core.config.column_config import DatasetGranularity


@pytest.fixture
def temp_explorations_dir():
    temp = tempfile.mkdtemp()
    yield Path(temp)
    shutil.rmtree(temp)


def _write_event_findings(dir_: Path, name: str) -> Path:
    findings = ExplorationFindings(
        source_path=f"/data/{name}.csv",
        source_format="csv",
        row_count=20000,
        column_count=10,
        columns={},
        time_series_metadata=TimeSeriesMetadata(
            granularity=DatasetGranularity.EVENT_LEVEL,
            entity_column="entity_id",
            time_column="event_timestamp",
            avg_events_per_entity=3.5,
            unique_entities=6000,
        ),
    )
    path = dir_ / f"{name}_findings.yaml"
    findings.save(str(path))
    return path


def _write_aggregated_findings(dir_: Path, name: str, ts_meta_null: bool) -> Path:
    ts_meta = None if ts_meta_null else TimeSeriesMetadata(
        granularity=DatasetGranularity.ENTITY_LEVEL,
    )
    findings = ExplorationFindings(
        source_path=f"/data/{name}_aggregated.parquet",
        source_format="parquet",
        row_count=6000,
        column_count=48,
        columns={},
        time_series_metadata=ts_meta,
    )
    path = dir_ / f"{name}_aggregated_findings.yaml"
    findings.save(str(path))
    return path


def _write_entity_findings(dir_: Path, name: str) -> Path:
    findings = ExplorationFindings(
        source_path=f"/data/{name}.csv",
        source_format="csv",
        row_count=3000,
        column_count=12,
        columns={},
        target_column="churned",
        target_type="binary",
    )
    path = dir_ / f"{name}_findings.yaml"
    findings.save(str(path))
    return path


def oracle_event_datasets(explorations_dir: Path) -> set[str]:
    """Expected event_datasets — computed from first principles.

    A dataset is event-level iff any findings file for it carries
    `time_series_metadata.granularity == EVENT_LEVEL`. The aggregated sibling
    carrying `time_series_metadata = null` does not erase that signal.
    """
    event_names: set[str] = set()
    for path in explorations_dir.glob("*_findings.yaml"):
        if "multi_dataset" in path.name:
            continue
        findings = ExplorationFindings.load(str(path))
        ts = findings.time_series_metadata
        if ts is not None and ts.granularity == DatasetGranularity.EVENT_LEVEL:
            stem = path.stem.replace("_findings", "")
            if stem.endswith("_aggregated"):
                stem = stem[: -len("_aggregated")]
            event_names.add(stem)
    return event_names


class TestEventDatasetsWithAggregatedSibling:
    def test_aggregated_with_null_ts_meta_still_event_level(self, temp_explorations_dir):
        _write_event_findings(temp_explorations_dir, "contract")
        _write_aggregated_findings(temp_explorations_dir, "contract", ts_meta_null=True)
        _write_entity_findings(temp_explorations_dir, "account")

        manager = ExplorationManager(explorations_dir=temp_explorations_dir)
        multi = manager.create_multi_dataset_findings()

        assert "contract" in multi.event_datasets, (
            f"expected 'contract' in event_datasets={multi.event_datasets}"
        )
        assert "account" not in multi.event_datasets

    def test_aggregated_with_entity_level_ts_meta_still_event_level(self, temp_explorations_dir):
        _write_event_findings(temp_explorations_dir, "subscription")
        _write_aggregated_findings(temp_explorations_dir, "subscription", ts_meta_null=False)
        _write_entity_findings(temp_explorations_dir, "account")

        manager = ExplorationManager(explorations_dir=temp_explorations_dir)
        multi = manager.create_multi_dataset_findings()

        assert "subscription" in multi.event_datasets

    def test_entity_only_dataset_not_in_event_datasets(self, temp_explorations_dir):
        _write_entity_findings(temp_explorations_dir, "account")

        manager = ExplorationManager(explorations_dir=temp_explorations_dir)
        multi = manager.create_multi_dataset_findings()

        assert multi.event_datasets == []

    def test_event_without_aggregated_sibling_still_event_level(self, temp_explorations_dir):
        _write_event_findings(temp_explorations_dir, "raw_events")
        _write_entity_findings(temp_explorations_dir, "account")

        manager = ExplorationManager(explorations_dir=temp_explorations_dir)
        multi = manager.create_multi_dataset_findings()

        assert "raw_events" in multi.event_datasets

    def test_datasets_dict_name_matches_base_name(self, temp_explorations_dir):
        _write_event_findings(temp_explorations_dir, "contract")
        _write_aggregated_findings(temp_explorations_dir, "contract", ts_meta_null=True)

        manager = ExplorationManager(explorations_dir=temp_explorations_dir)
        multi = manager.create_multi_dataset_findings()

        assert "contract" in multi.datasets
        assert multi.datasets["contract"].granularity == DatasetGranularity.EVENT_LEVEL


class TestOracleParity:
    def test_manager_matches_oracle_mixed_tree(self, temp_explorations_dir):
        _write_event_findings(temp_explorations_dir, "contract")
        _write_aggregated_findings(temp_explorations_dir, "contract", ts_meta_null=True)
        _write_event_findings(temp_explorations_dir, "subscription")
        _write_aggregated_findings(temp_explorations_dir, "subscription", ts_meta_null=False)
        _write_event_findings(temp_explorations_dir, "raw_events")
        _write_entity_findings(temp_explorations_dir, "account")
        _write_entity_findings(temp_explorations_dir, "profile")

        manager = ExplorationManager(explorations_dir=temp_explorations_dir)
        multi = manager.create_multi_dataset_findings()

        expected = oracle_event_datasets(temp_explorations_dir)
        assert set(multi.event_datasets) == expected


class TestRunSpschurnRegressionShape:
    """Reproduces the exact shape seen in debug/engagement_e4ad6e1b/run/.

    Pre-fix observed state:
    - `contract_findings.yaml.time_series_metadata.granularity = event_level`
    - `contract_aggregated_findings.yaml.time_series_metadata = null`
    - `multi_dataset_findings.yaml.event_datasets = []`  ← bug

    Post-fix: event_datasets contains both event-level datasets.
    """

    def test_contract_and_subscription_surface_as_event_datasets(self, temp_explorations_dir):
        for name in ("contract", "subscription"):
            _write_event_findings(temp_explorations_dir, name)
            _write_aggregated_findings(temp_explorations_dir, name, ts_meta_null=True)
        _write_entity_findings(temp_explorations_dir, "account")

        manager = ExplorationManager(explorations_dir=temp_explorations_dir)
        multi = manager.create_multi_dataset_findings()

        assert set(multi.event_datasets) >= {"contract", "subscription"}
        assert multi.event_datasets  # not empty — catches the regression directly
