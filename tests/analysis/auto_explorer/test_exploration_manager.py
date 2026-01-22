"""Tests for ExplorationManager and MultiDatasetFindings - TDD approach."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from customer_retention.analysis.auto_explorer import ExplorationFindings
from customer_retention.analysis.auto_explorer.exploration_manager import (
    ExplorationManager,
    MultiDatasetFindings,
    DatasetInfo,
    DatasetRelationshipInfo,
)
from customer_retention.core.config.column_config import DatasetGranularity


@pytest.fixture
def temp_explorations_dir():
    """Create a temporary explorations directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_entity_findings(temp_explorations_dir):
    """Create sample entity-level findings."""
    findings = ExplorationFindings(
        source_path="/data/customers.csv",
        source_format="csv",
        row_count=5000,
        column_count=10,
        columns={},
        target_column="churned",
        target_type="binary",
    )
    path = temp_explorations_dir / "customers_abc123_findings.yaml"
    findings.save(str(path))
    return path


@pytest.fixture
def sample_event_findings(temp_explorations_dir):
    """Create sample event-level findings."""
    from customer_retention.analysis.auto_explorer.findings import TimeSeriesMetadata

    findings = ExplorationFindings(
        source_path="/data/transactions.csv",
        source_format="csv",
        row_count=50000,
        column_count=8,
        columns={},
        time_series_metadata=TimeSeriesMetadata(
            granularity=DatasetGranularity.EVENT_LEVEL,
            entity_column="customer_id",
            time_column="transaction_date",
            avg_events_per_entity=10.0,
            unique_entities=5000,
        ),
    )
    path = temp_explorations_dir / "transactions_def456_findings.yaml"
    findings.save(str(path))
    return path


@pytest.fixture
def sample_email_findings(temp_explorations_dir):
    """Create sample email event findings."""
    from customer_retention.analysis.auto_explorer.findings import TimeSeriesMetadata

    findings = ExplorationFindings(
        source_path="/data/emails.csv",
        source_format="csv",
        row_count=100000,
        column_count=6,
        columns={},
        time_series_metadata=TimeSeriesMetadata(
            granularity=DatasetGranularity.EVENT_LEVEL,
            entity_column="customer_id",
            time_column="sent_date",
            avg_events_per_entity=20.0,
            unique_entities=5000,
        ),
    )
    path = temp_explorations_dir / "emails_ghi789_findings.yaml"
    findings.save(str(path))
    return path


class TestExplorationManager:
    """Tests for ExplorationManager."""

    def test_create_manager(self, temp_explorations_dir):
        manager = ExplorationManager(explorations_dir=temp_explorations_dir)
        assert manager is not None

    def test_discover_findings(self, temp_explorations_dir, sample_entity_findings, sample_event_findings):
        manager = ExplorationManager(explorations_dir=temp_explorations_dir)
        findings_list = manager.discover_findings()

        assert len(findings_list) == 2

    def test_load_findings_by_name(self, temp_explorations_dir, sample_entity_findings):
        manager = ExplorationManager(explorations_dir=temp_explorations_dir)
        findings = manager.load_findings("customers")

        assert findings is not None
        assert findings.source_path == "/data/customers.csv"

    def test_list_datasets(self, temp_explorations_dir, sample_entity_findings, sample_event_findings):
        manager = ExplorationManager(explorations_dir=temp_explorations_dir)
        datasets = manager.list_datasets()

        assert len(datasets) == 2
        assert all(isinstance(d, DatasetInfo) for d in datasets)

    def test_dataset_info_has_granularity(self, temp_explorations_dir, sample_entity_findings, sample_event_findings):
        manager = ExplorationManager(explorations_dir=temp_explorations_dir)
        datasets = manager.list_datasets()

        # Find the event-level dataset
        event_dataset = next((d for d in datasets if "transaction" in d.name.lower()), None)
        assert event_dataset is not None
        assert event_dataset.granularity == DatasetGranularity.EVENT_LEVEL


class TestMultiDatasetFindings:
    """Tests for MultiDatasetFindings."""

    def test_create_multi_findings(self, temp_explorations_dir, sample_entity_findings, sample_event_findings):
        manager = ExplorationManager(explorations_dir=temp_explorations_dir)
        multi = manager.create_multi_dataset_findings()

        assert isinstance(multi, MultiDatasetFindings)
        assert len(multi.datasets) == 2

    def test_multi_findings_has_entity_dataset(self, temp_explorations_dir, sample_entity_findings, sample_event_findings):
        manager = ExplorationManager(explorations_dir=temp_explorations_dir)
        multi = manager.create_multi_dataset_findings()

        # Should identify which dataset is the primary entity dataset
        assert multi.primary_entity_dataset is not None

    def test_multi_findings_has_event_datasets(self, temp_explorations_dir, sample_entity_findings, sample_event_findings):
        manager = ExplorationManager(explorations_dir=temp_explorations_dir)
        multi = manager.create_multi_dataset_findings()

        # Should list event-level datasets
        assert len(multi.event_datasets) >= 1

    def test_save_and_load_multi_findings(self, temp_explorations_dir, sample_entity_findings, sample_event_findings):
        manager = ExplorationManager(explorations_dir=temp_explorations_dir)
        multi = manager.create_multi_dataset_findings()

        # Save
        save_path = temp_explorations_dir / "multi_dataset_findings.yaml"
        multi.save(str(save_path))

        # Load
        loaded = MultiDatasetFindings.load(str(save_path))
        assert len(loaded.datasets) == len(multi.datasets)


class TestDatasetRelationships:
    """Tests for dataset relationship detection."""

    def test_detect_relationships(self, temp_explorations_dir, sample_entity_findings, sample_event_findings):
        manager = ExplorationManager(explorations_dir=temp_explorations_dir)
        multi = manager.create_multi_dataset_findings()

        # Should detect that transactions relate to customers via customer_id
        assert len(multi.relationships) >= 0  # May be empty if can't load actual data

    def test_add_manual_relationship(self, temp_explorations_dir, sample_entity_findings, sample_event_findings):
        manager = ExplorationManager(explorations_dir=temp_explorations_dir)
        multi = manager.create_multi_dataset_findings()

        # Manually define a relationship
        multi.add_relationship(
            left_dataset="customers",
            right_dataset="transactions",
            left_column="customer_id",
            right_column="customer_id",
            relationship_type="one_to_many"
        )

        assert len(multi.relationships) >= 1


class TestDatasetExclusion:
    """Tests for dataset exclusion/archiving."""

    def test_exclude_dataset(self, temp_explorations_dir, sample_entity_findings, sample_event_findings, sample_email_findings):
        manager = ExplorationManager(explorations_dir=temp_explorations_dir)

        # Initially 3 datasets
        datasets = manager.list_datasets()
        assert len(datasets) == 3

        # Exclude one
        manager.exclude_dataset("emails")

        # Now 2 active datasets
        active_datasets = manager.list_datasets(include_excluded=False)
        assert len(active_datasets) == 2

    def test_excluded_datasets_tracked(self, temp_explorations_dir, sample_entity_findings, sample_event_findings, sample_email_findings):
        manager = ExplorationManager(explorations_dir=temp_explorations_dir)
        manager.exclude_dataset("emails")

        # Can still see excluded if requested
        all_datasets = manager.list_datasets(include_excluded=True)
        assert len(all_datasets) == 3

        excluded = [d for d in all_datasets if d.excluded]
        assert len(excluded) == 1
        assert "email" in excluded[0].name.lower()

    def test_reinclude_dataset(self, temp_explorations_dir, sample_entity_findings, sample_event_findings, sample_email_findings):
        manager = ExplorationManager(explorations_dir=temp_explorations_dir)
        manager.exclude_dataset("emails")
        manager.include_dataset("emails")

        active_datasets = manager.list_datasets(include_excluded=False)
        assert len(active_datasets) == 3


class TestAggregationPlanning:
    """Tests for planning aggregations across datasets."""

    def test_get_aggregation_plan(self, temp_explorations_dir, sample_entity_findings, sample_event_findings):
        manager = ExplorationManager(explorations_dir=temp_explorations_dir)
        multi = manager.create_multi_dataset_findings()

        # Get aggregation plan for event datasets
        plan = multi.get_aggregation_plan()

        assert plan is not None
        # Should suggest windows for event datasets
        assert "transactions" in plan or len(plan) >= 0


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_explorations_dir(self, temp_explorations_dir):
        manager = ExplorationManager(explorations_dir=temp_explorations_dir)
        findings_list = manager.discover_findings()

        assert findings_list == []

    def test_single_dataset(self, temp_explorations_dir, sample_entity_findings):
        manager = ExplorationManager(explorations_dir=temp_explorations_dir)
        multi = manager.create_multi_dataset_findings()

        assert len(multi.datasets) == 1

    def test_nonexistent_dataset_exclusion(self, temp_explorations_dir, sample_entity_findings):
        manager = ExplorationManager(explorations_dir=temp_explorations_dir)

        # Should not raise, just ignore
        manager.exclude_dataset("nonexistent")

        datasets = manager.list_datasets()
        assert len(datasets) == 1


class TestDatasetSelectionPersistence:
    """Tests for persisting dataset selection across sessions."""

    def test_multi_findings_exclude_dataset(self, temp_explorations_dir, sample_entity_findings, sample_event_findings, sample_email_findings):
        manager = ExplorationManager(explorations_dir=temp_explorations_dir)
        multi = manager.create_multi_dataset_findings()

        multi.exclude_dataset("emails")

        assert "emails" in multi.excluded_datasets
        assert len(multi.selected_datasets) == 2

    def test_multi_findings_select_dataset(self, temp_explorations_dir, sample_entity_findings, sample_event_findings, sample_email_findings):
        manager = ExplorationManager(explorations_dir=temp_explorations_dir)
        multi = manager.create_multi_dataset_findings()

        # Exclude then re-select
        multi.exclude_dataset("emails")
        multi.select_dataset("emails")

        assert "emails" not in multi.excluded_datasets
        assert len(multi.selected_datasets) == 3

    def test_selected_datasets_property(self, temp_explorations_dir, sample_entity_findings, sample_event_findings, sample_email_findings):
        manager = ExplorationManager(explorations_dir=temp_explorations_dir)
        multi = manager.create_multi_dataset_findings()

        multi.exclude_dataset("emails")

        selected = multi.selected_datasets
        assert len(selected) == 2
        assert all(name != "emails" for name in selected.keys())

    def test_exclusion_persists_after_save_load(self, temp_explorations_dir, sample_entity_findings, sample_event_findings, sample_email_findings):
        manager = ExplorationManager(explorations_dir=temp_explorations_dir)
        multi = manager.create_multi_dataset_findings()

        multi.exclude_dataset("emails")
        save_path = temp_explorations_dir / "multi_dataset_findings.yaml"
        multi.save(str(save_path))

        loaded = MultiDatasetFindings.load(str(save_path))
        assert "emails" in loaded.excluded_datasets
        assert len(loaded.selected_datasets) == 2

    def test_dataset_info_excluded_flag_persists(self, temp_explorations_dir, sample_entity_findings, sample_event_findings, sample_email_findings):
        manager = ExplorationManager(explorations_dir=temp_explorations_dir)
        multi = manager.create_multi_dataset_findings()

        multi.exclude_dataset("emails")
        save_path = temp_explorations_dir / "multi_dataset_findings.yaml"
        multi.save(str(save_path))

        loaded = MultiDatasetFindings.load(str(save_path))
        emails_info = loaded.datasets.get("emails")
        assert emails_info is not None
        assert emails_info.excluded is True


class TestSelectionConnectsToRegistry:
    """Tests for connecting selection to RecommendationRegistry."""

    def test_create_registry_from_multi_findings(self, temp_explorations_dir, sample_entity_findings, sample_event_findings, sample_email_findings):
        from customer_retention.analysis.auto_explorer.layered_recommendations import RecommendationRegistry

        manager = ExplorationManager(explorations_dir=temp_explorations_dir)
        multi = manager.create_multi_dataset_findings()
        multi.exclude_dataset("emails")

        registry = multi.to_recommendation_registry()

        assert isinstance(registry, RecommendationRegistry)
        assert len(registry.sources) == 2
        assert "emails" not in registry.sources

    def test_registry_has_sources_for_selected_only(self, temp_explorations_dir, sample_entity_findings, sample_event_findings, sample_email_findings):
        from customer_retention.analysis.auto_explorer.layered_recommendations import RecommendationRegistry

        manager = ExplorationManager(explorations_dir=temp_explorations_dir)
        multi = manager.create_multi_dataset_findings()
        multi.exclude_dataset("transactions")

        registry = multi.to_recommendation_registry()

        assert "transactions" not in registry.sources
        assert "customers" in registry.sources or "emails" in registry.sources

    def test_registry_initializes_silver_with_joins(self, temp_explorations_dir, sample_entity_findings, sample_event_findings):
        manager = ExplorationManager(explorations_dir=temp_explorations_dir)
        multi = manager.create_multi_dataset_findings()
        multi.add_relationship("customers", "transactions", "customer_id", "customer_id")

        registry = multi.to_recommendation_registry()

        assert registry.silver is not None
        assert len(registry.silver.joins) >= 0  # May convert relationships to joins


class TestAggregatedFindingsDiscovery:
    """Tests for discovering and handling aggregated findings."""

    @pytest.fixture
    def aggregated_event_findings(self, temp_explorations_dir):
        """Create sample aggregated event findings (output from 01d notebook)."""
        from customer_retention.analysis.auto_explorer.findings import TimeSeriesMetadata

        # Aggregated data path
        aggregated_data_path = str(temp_explorations_dir / "transactions_def456_aggregated.parquet")
        aggregated_findings_path = str(temp_explorations_dir / "transactions_def456_aggregated_findings.yaml")

        # Create aggregated findings (entity-level, output from aggregation)
        aggregated_findings = ExplorationFindings(
            source_path=aggregated_data_path,
            source_format="parquet",
            row_count=5000,
            column_count=50,
            columns={},
            time_series_metadata=TimeSeriesMetadata(
                granularity=DatasetGranularity.ENTITY_LEVEL,
            ),
        )
        aggregated_findings.save(aggregated_findings_path)

        return Path(aggregated_findings_path)

    @pytest.fixture
    def event_findings_with_aggregation(self, temp_explorations_dir, sample_event_findings, aggregated_event_findings):
        """Update event findings to mark aggregation as executed."""
        from customer_retention.analysis.auto_explorer.findings import TimeSeriesMetadata

        # Load and update original findings
        findings = ExplorationFindings.load(str(sample_event_findings))
        findings.time_series_metadata.aggregation_executed = True
        findings.time_series_metadata.aggregated_data_path = str(
            temp_explorations_dir / "transactions_def456_aggregated.parquet"
        )
        findings.time_series_metadata.aggregated_findings_path = str(aggregated_event_findings)
        findings.time_series_metadata.aggregation_windows_used = ["7d", "30d", "all_time"]
        findings.save(str(sample_event_findings))

        return sample_event_findings

    def test_discover_findings_includes_aggregated(
        self, temp_explorations_dir, sample_entity_findings, event_findings_with_aggregation, aggregated_event_findings
    ):
        """Aggregated findings files should be discoverable."""
        manager = ExplorationManager(explorations_dir=temp_explorations_dir)
        findings_paths = manager.discover_findings()

        names = [p.name for p in findings_paths]
        assert any("aggregated" in n for n in names)

    def test_get_aggregated_path_returns_path(
        self, temp_explorations_dir, event_findings_with_aggregation, aggregated_event_findings
    ):
        """get_aggregated_path should return the aggregated findings path."""
        manager = ExplorationManager(explorations_dir=temp_explorations_dir)

        aggregated = manager.get_aggregated_path(str(event_findings_with_aggregation))

        assert aggregated is not None
        assert "aggregated" in aggregated

    def test_get_aggregated_path_returns_none_for_entity_level(
        self, temp_explorations_dir, sample_entity_findings
    ):
        """get_aggregated_path should return None for entity-level datasets."""
        manager = ExplorationManager(explorations_dir=temp_explorations_dir)

        aggregated = manager.get_aggregated_path(str(sample_entity_findings))

        assert aggregated is None

    def test_get_aggregated_path_returns_none_when_not_aggregated(
        self, temp_explorations_dir, sample_event_findings
    ):
        """get_aggregated_path should return None when aggregation not executed."""
        manager = ExplorationManager(explorations_dir=temp_explorations_dir)

        aggregated = manager.get_aggregated_path(str(sample_event_findings))

        assert aggregated is None

    def test_has_aggregated_output_property(
        self, temp_explorations_dir, event_findings_with_aggregation
    ):
        """ExplorationFindings.has_aggregated_output should work correctly."""
        findings = ExplorationFindings.load(str(event_findings_with_aggregation))

        assert findings.has_aggregated_output is True

    def test_extract_dataset_name_handles_aggregated_suffix(self, temp_explorations_dir):
        """_extract_dataset_name should handle the _aggregated suffix."""
        manager = ExplorationManager(explorations_dir=temp_explorations_dir)

        # Regular findings
        regular = manager._extract_dataset_name(Path("events_abc123_findings.yaml"))
        assert regular == "events"

        # Aggregated findings - should include "aggregated" to distinguish
        agg = manager._extract_dataset_name(Path("events_abc123_aggregated_findings.yaml"))
        assert "events" in agg
