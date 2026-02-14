"""Tests for ExplorationManager and MultiDatasetFindings - TDD approach."""
import shutil
import tempfile
from pathlib import Path

import pytest

from customer_retention.analysis.auto_explorer import ExplorationFindings
from customer_retention.analysis.auto_explorer.exploration_manager import (
    DatasetInfo,
    ExplorationManager,
    MultiDatasetFindings,
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
    path = temp_explorations_dir / "customers_findings.yaml"
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
    path = temp_explorations_dir / "transactions_findings.yaml"
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
    path = temp_explorations_dir / "emails_findings.yaml"
    findings.save(str(path))
    return path


class TestExplorationManager:
    """Tests for ExplorationManager."""

    def test_create_manager(self, temp_explorations_dir):
        manager = ExplorationManager(explorations_dir=temp_explorations_dir)
        assert isinstance(manager, ExplorationManager)
        assert manager.explorations_dir == temp_explorations_dir

    def test_discover_findings(self, temp_explorations_dir, sample_entity_findings, sample_event_findings):
        manager = ExplorationManager(explorations_dir=temp_explorations_dir)
        findings_list = manager.discover_findings()

        assert len(findings_list) == 2

    def test_load_findings_by_name(self, temp_explorations_dir, sample_entity_findings):
        manager = ExplorationManager(explorations_dir=temp_explorations_dir)
        findings = manager.load_findings("customers")

        assert isinstance(findings, ExplorationFindings)
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
        assert isinstance(multi.primary_entity_dataset, str)
        assert len(multi.primary_entity_dataset) > 0

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
        assert isinstance(multi.relationships, list)  # May be empty if can't load actual data

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

        assert isinstance(plan, dict)


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
        assert isinstance(emails_info, DatasetInfo)
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

        assert registry.silver is not None  # Silver layer should be initialized
        assert isinstance(registry.silver.joins, list)


class TestAggregatedFindingsDiscovery:
    """Tests for discovering and handling aggregated findings."""

    @pytest.fixture
    def aggregated_event_findings(self, temp_explorations_dir):
        """Create sample aggregated event findings (output from 01d notebook)."""
        from customer_retention.analysis.auto_explorer.findings import TimeSeriesMetadata

        aggregated_data_path = str(temp_explorations_dir / "transactions_aggregated.parquet")
        aggregated_findings_path = str(temp_explorations_dir / "transactions_aggregated_findings.yaml")

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

        # Load and update original findings
        findings = ExplorationFindings.load(str(sample_event_findings))
        findings.time_series_metadata.aggregation_executed = True
        findings.time_series_metadata.aggregated_data_path = str(
            temp_explorations_dir / "transactions_aggregated.parquet"
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

        assert isinstance(aggregated, str)
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

        regular = manager._extract_dataset_name(Path("events_findings.yaml"))
        assert regular == "events"

        agg = manager._extract_dataset_name(Path("events_aggregated_findings.yaml"))
        assert agg == "events"

    def test_discover_findings_prefer_aggregated_default(
        self, temp_explorations_dir, sample_entity_findings, event_findings_with_aggregation, aggregated_event_findings
    ):
        """By default, discover_findings should prefer aggregated over event-level."""
        manager = ExplorationManager(explorations_dir=temp_explorations_dir)
        findings_paths = manager.discover_findings()

        names = [p.name for p in findings_paths]
        assert any("aggregated" in n for n in names)
        assert not any(n == "transactions_findings.yaml" for n in names)

    def test_discover_findings_prefer_aggregated_false(
        self, temp_explorations_dir, sample_entity_findings, event_findings_with_aggregation, aggregated_event_findings
    ):
        """With prefer_aggregated=False, all findings should be returned."""
        manager = ExplorationManager(explorations_dir=temp_explorations_dir)
        findings_paths = manager.discover_findings(prefer_aggregated=False)

        names = [p.name for p in findings_paths]
        assert any("aggregated" in n for n in names)
        assert any(n == "transactions_findings.yaml" for n in names)

    def test_get_skipped_event_findings(
        self, temp_explorations_dir, sample_entity_findings, event_findings_with_aggregation, aggregated_event_findings
    ):
        """get_skipped_event_findings should return event files skipped in favor of aggregated."""
        manager = ExplorationManager(explorations_dir=temp_explorations_dir)
        skipped = manager.get_skipped_event_findings()

        assert len(skipped) == 1
        assert skipped[0].name == "transactions_findings.yaml"

    def test_get_skipped_event_findings_empty_when_no_aggregated(
        self, temp_explorations_dir, sample_entity_findings, sample_event_findings
    ):
        """get_skipped_event_findings should return empty when no aggregated files exist."""
        manager = ExplorationManager(explorations_dir=temp_explorations_dir)
        skipped = manager.get_skipped_event_findings()

        assert skipped == []

    def test_get_base_name_extracts_correctly(self, temp_explorations_dir):
        """_get_base_name should extract base dataset name without aggregated suffix."""
        manager = ExplorationManager(explorations_dir=temp_explorations_dir)

        assert manager._get_base_name(Path("events_findings.yaml")) == "events"
        assert manager._get_base_name(Path("events_aggregated_findings.yaml")) == "events"
        assert manager._get_base_name(Path("my_dataset_findings.yaml")) == "my_dataset"
        assert manager._get_base_name(Path("my_dataset_aggregated_findings.yaml")) == "my_dataset"


class TestDiscoverFindingsWithPaths:
    def test_findings_paths_used_directly(self, temp_explorations_dir, sample_entity_findings, sample_event_findings):
        custom_paths = [sample_entity_findings]
        manager = ExplorationManager(
            explorations_dir=temp_explorations_dir,
            findings_paths=custom_paths,
        )
        result = manager.discover_findings()
        assert result == custom_paths

    def test_findings_paths_skips_globbing(self, tmp_path):
        fake_dir = tmp_path / "nonexistent"
        paths = [tmp_path / "a_findings.yaml", tmp_path / "b_findings.yaml"]
        for p in paths:
            p.touch()
        manager = ExplorationManager(
            explorations_dir=fake_dir,
            findings_paths=paths,
        )
        result = manager.discover_findings()
        assert result == paths

    def test_findings_paths_none_falls_back_to_glob(self, temp_explorations_dir, sample_entity_findings, sample_event_findings):
        manager = ExplorationManager(
            explorations_dir=temp_explorations_dir,
            findings_paths=None,
        )
        result = manager.discover_findings()
        assert len(result) == 2


class TestCompositeDatasetNamePersistence:
    def test_composite_dataset_name_defaults_to_none(self):
        multi = MultiDatasetFindings()
        assert multi.composite_dataset_name is None

    def test_composite_dataset_name_persists_through_save_load(self, temp_explorations_dir, sample_entity_findings, sample_event_findings):
        manager = ExplorationManager(explorations_dir=temp_explorations_dir)
        multi = manager.create_multi_dataset_findings()
        multi.composite_dataset_name = "combined_abc123"

        save_path = temp_explorations_dir / "multi_dataset_findings.yaml"
        multi.save(str(save_path))

        loaded = MultiDatasetFindings.load(str(save_path))
        assert loaded.composite_dataset_name == "combined_abc123"

    def test_composite_dataset_name_none_persists_through_save_load(self, temp_explorations_dir, sample_entity_findings):
        manager = ExplorationManager(explorations_dir=temp_explorations_dir)
        multi = manager.create_multi_dataset_findings()

        save_path = temp_explorations_dir / "multi_dataset_findings.yaml"
        multi.save(str(save_path))

        loaded = MultiDatasetFindings.load(str(save_path))
        assert loaded.composite_dataset_name is None


class TestScaffoldToRelationships:

    def test_scaffold_to_relationships_converts_entries(self):
        from customer_retention.analysis.auto_explorer.project_context import MergeScaffoldEntry

        scaffold = [
            MergeScaffoldEntry(
                left_dataset="transactions",
                right_dataset="profiles",
                join_keys=["customer_id"],
                relationship="many_to_one",
            ),
        ]
        rels = ExplorationManager._scaffold_to_relationships(scaffold)
        assert len(rels) == 1
        rel = rels[0]
        assert rel.left_dataset == "profiles"
        assert rel.right_dataset == "transactions"
        assert rel.left_columns == ["customer_id"]
        assert rel.right_columns == ["customer_id"]
        assert rel.relationship_type == "many_to_one"

    def test_scaffold_to_relationships_composite_keys(self):
        from customer_retention.analysis.auto_explorer.project_context import MergeScaffoldEntry

        scaffold = [
            MergeScaffoldEntry(
                left_dataset="transactions",
                right_dataset="profiles",
                join_keys=["customer_id", "as_of_date"],
                relationship="many_to_one",
            ),
        ]
        rels = ExplorationManager._scaffold_to_relationships(scaffold)
        assert rels[0].left_columns == ["customer_id", "as_of_date"]
        assert rels[0].right_columns == ["customer_id", "as_of_date"]

    def test_scaffold_to_relationships_empty(self):
        rels = ExplorationManager._scaffold_to_relationships([])
        assert rels == []

    def test_create_multi_dataset_findings_with_scaffold(
        self, temp_explorations_dir, sample_entity_findings, sample_event_findings
    ):
        from customer_retention.analysis.auto_explorer.project_context import MergeScaffoldEntry

        scaffold = [
            MergeScaffoldEntry(
                left_dataset="transactions",
                right_dataset="customers",
                join_keys=["customer_id"],
                relationship="many_to_one",
            ),
        ]
        manager = ExplorationManager(explorations_dir=temp_explorations_dir)
        multi = manager.create_multi_dataset_findings(merge_scaffold=scaffold)
        assert len(multi.relationships) == 1
        assert multi.relationships[0].left_dataset == "customers"
        assert multi.relationships[0].right_dataset == "transactions"

    def test_create_multi_dataset_findings_without_scaffold(
        self, temp_explorations_dir, sample_entity_findings, sample_event_findings
    ):
        manager = ExplorationManager(explorations_dir=temp_explorations_dir)
        multi = manager.create_multi_dataset_findings()
        assert multi.relationships == []


class TestRelationshipInfoCompositeKeys:

    def test_relationship_info_left_right_columns(self):
        from customer_retention.analysis.auto_explorer.exploration_manager import DatasetRelationshipInfo

        rel = DatasetRelationshipInfo(
            left_dataset="a",
            right_dataset="b",
            left_columns=["id", "date"],
            right_columns=["id", "date"],
            relationship_type="one_to_many",
        )
        assert rel.left_columns == ["id", "date"]
        assert rel.left_column == "id"
        assert rel.right_column == "id"

    def test_save_load_with_composite_columns(self, temp_explorations_dir, sample_entity_findings, sample_event_findings):
        from customer_retention.analysis.auto_explorer.exploration_manager import DatasetRelationshipInfo

        manager = ExplorationManager(explorations_dir=temp_explorations_dir)
        multi = manager.create_multi_dataset_findings()
        multi.relationships = [
            DatasetRelationshipInfo(
                left_dataset="customers",
                right_dataset="transactions",
                left_columns=["customer_id", "as_of_date"],
                right_columns=["customer_id", "as_of_date"],
                relationship_type="one_to_many",
            )
        ]
        save_path = temp_explorations_dir / "multi_dataset_findings.yaml"
        multi.save(str(save_path))
        loaded = MultiDatasetFindings.load(str(save_path))
        assert loaded.relationships[0].left_columns == ["customer_id", "as_of_date"]
        assert loaded.relationships[0].right_columns == ["customer_id", "as_of_date"]

    def test_load_legacy_single_column_format(self, temp_explorations_dir):
        import yaml

        data = {
            "datasets": {},
            "relationships": [
                {
                    "left_dataset": "a",
                    "right_dataset": "b",
                    "left_column": "id",
                    "right_column": "id",
                    "relationship_type": "one_to_many",
                }
            ],
            "primary_entity_dataset": None,
            "event_datasets": [],
            "excluded_datasets": [],
        }
        path = temp_explorations_dir / "multi_dataset_findings.yaml"
        path.write_text(yaml.dump(data))
        loaded = MultiDatasetFindings.load(str(path))
        assert loaded.relationships[0].left_columns == ["id"]
        assert loaded.relationships[0].right_columns == ["id"]
