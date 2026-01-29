import tempfile
from pathlib import Path

from customer_retention.analysis.auto_explorer.findings import (
    ExplorationFindings,
    FeatureAvailabilityInfo,
    FeatureAvailabilityMetadata,
)


class TestFeatureAvailabilityStorage:
    def create_sample_availability_metadata(self) -> FeatureAvailabilityMetadata:
        return FeatureAvailabilityMetadata(
            data_start="2020-01-01",
            data_end="2022-12-31",
            time_span_days=1095,
            new_tracking=["loyalty_score", "app_version"],
            retired_tracking=["old_system_flag"],
            partial_window=["temp_campaign"],
            features={
                "loyalty_score": FeatureAvailabilityInfo(
                    first_valid_date="2021-06-01",
                    last_valid_date="2022-12-31",
                    coverage_pct=54.8,
                    availability_type="new_tracking",
                    days_from_start=517,
                    days_before_end=0,
                ),
                "old_system_flag": FeatureAvailabilityInfo(
                    first_valid_date="2020-01-01",
                    last_valid_date="2021-03-15",
                    coverage_pct=42.3,
                    availability_type="retired",
                    days_from_start=0,
                    days_before_end=656,
                ),
                "tenure_months": FeatureAvailabilityInfo(
                    first_valid_date="2020-01-01",
                    last_valid_date="2022-12-31",
                    coverage_pct=100.0,
                    availability_type="full",
                    days_from_start=0,
                    days_before_end=0,
                ),
            },
        )

    def test_store_and_load_availability_round_trip(self):
        findings = ExplorationFindings(
            source_path="events.csv",
            source_format="csv",
            feature_availability=self.create_sample_availability_metadata(),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "findings.yaml"
            findings.save(str(path))
            loaded = ExplorationFindings.load(str(path))
            assert loaded.feature_availability is not None
            assert loaded.feature_availability.data_start == "2020-01-01"
            assert loaded.feature_availability.data_end == "2022-12-31"
            assert loaded.feature_availability.time_span_days == 1095
            assert loaded.feature_availability.new_tracking == ["loyalty_score", "app_version"]
            assert loaded.feature_availability.retired_tracking == ["old_system_flag"]
            assert "loyalty_score" in loaded.feature_availability.features
            assert loaded.feature_availability.features["loyalty_score"].availability_type == "new_tracking"

    def test_empty_availability_serializes_correctly(self):
        findings = ExplorationFindings(
            source_path="clean_data.csv",
            source_format="csv",
            feature_availability=FeatureAvailabilityMetadata(
                data_start="2020-01-01",
                data_end="2022-12-31",
                time_span_days=1095,
                new_tracking=[],
                retired_tracking=[],
                partial_window=[],
                features={},
            ),
        )
        yaml_str = findings.to_yaml()
        restored = ExplorationFindings.from_yaml(yaml_str)
        assert restored.feature_availability is not None
        assert restored.feature_availability.new_tracking == []
        assert restored.feature_availability.retired_tracking == []
        assert restored.feature_availability.features == {}

    def test_new_tracking_columns_preserved(self):
        availability = FeatureAvailabilityMetadata(
            data_start="2020-01-01",
            data_end="2022-12-31",
            time_span_days=1095,
            new_tracking=["feature_a", "feature_b"],
            retired_tracking=[],
            partial_window=[],
            features={
                "feature_a": FeatureAvailabilityInfo(
                    first_valid_date="2021-07-01",
                    last_valid_date="2022-12-31",
                    coverage_pct=50.0,
                    availability_type="new_tracking",
                    days_from_start=547,
                    days_before_end=0,
                ),
            },
        )
        findings = ExplorationFindings(
            source_path="data.csv", source_format="csv", feature_availability=availability
        )
        data = findings.to_dict()
        restored = ExplorationFindings.from_dict(data)
        assert restored.feature_availability.new_tracking == ["feature_a", "feature_b"]
        assert restored.feature_availability.features["feature_a"].days_from_start == 547

    def test_retired_tracking_columns_preserved(self):
        availability = FeatureAvailabilityMetadata(
            data_start="2020-01-01",
            data_end="2022-12-31",
            time_span_days=1095,
            new_tracking=[],
            retired_tracking=["legacy_field"],
            partial_window=[],
            features={
                "legacy_field": FeatureAvailabilityInfo(
                    first_valid_date="2020-01-01",
                    last_valid_date="2021-06-30",
                    coverage_pct=50.0,
                    availability_type="retired",
                    days_from_start=0,
                    days_before_end=549,
                ),
            },
        )
        findings = ExplorationFindings(
            source_path="data.csv", source_format="csv", feature_availability=availability
        )
        json_str = findings.to_json()
        restored = ExplorationFindings.from_json(json_str)
        assert restored.feature_availability.retired_tracking == ["legacy_field"]
        assert restored.feature_availability.features["legacy_field"].days_before_end == 549

    def test_partial_window_columns_preserved(self):
        availability = FeatureAvailabilityMetadata(
            data_start="2020-01-01",
            data_end="2022-12-31",
            time_span_days=1095,
            new_tracking=[],
            retired_tracking=[],
            partial_window=["temp_feature"],
            features={
                "temp_feature": FeatureAvailabilityInfo(
                    first_valid_date="2021-01-01",
                    last_valid_date="2021-12-31",
                    coverage_pct=33.3,
                    availability_type="partial_window",
                    days_from_start=366,
                    days_before_end=365,
                ),
            },
        )
        findings = ExplorationFindings(
            source_path="data.csv", source_format="csv", feature_availability=availability
        )
        yaml_str = findings.to_yaml()
        restored = ExplorationFindings.from_yaml(yaml_str)
        assert restored.feature_availability.partial_window == ["temp_feature"]
        feat = restored.feature_availability.features["temp_feature"]
        assert feat.availability_type == "partial_window"
        assert feat.days_from_start == 366
        assert feat.days_before_end == 365

    def test_backward_compatible_with_existing_findings(self):
        old_findings_yaml = """
source_path: legacy_data.csv
source_format: csv
exploration_timestamp: '2024-01-01T00:00:00'
row_count: 1000
column_count: 5
memory_usage_mb: 1.5
columns: {}
target_column: null
target_type: null
identifier_columns: []
datetime_columns: []
overall_quality_score: 100.0
critical_issues: []
warnings: []
modeling_ready: false
blocking_issues: []
metadata: {}
time_series_metadata: null
text_processing: {}
iteration_id: null
parent_iteration_id: null
snapshot_id: null
snapshot_path: null
timestamp_scenario: null
timestamp_strategy: null
"""
        restored = ExplorationFindings.from_yaml(old_findings_yaml)
        assert restored.source_path == "legacy_data.csv"
        assert restored.feature_availability is None

    def test_availability_with_none_dates_serializes(self):
        availability = FeatureAvailabilityMetadata(
            data_start="2020-01-01",
            data_end="2022-12-31",
            time_span_days=1095,
            new_tracking=[],
            retired_tracking=[],
            partial_window=[],
            features={
                "empty_col": FeatureAvailabilityInfo(
                    first_valid_date=None,
                    last_valid_date=None,
                    coverage_pct=0.0,
                    availability_type="empty",
                    days_from_start=None,
                    days_before_end=None,
                ),
            },
        )
        findings = ExplorationFindings(
            source_path="data.csv", source_format="csv", feature_availability=availability
        )
        yaml_str = findings.to_yaml()
        restored = ExplorationFindings.from_yaml(yaml_str)
        feat = restored.feature_availability.features["empty_col"]
        assert feat.first_valid_date is None
        assert feat.last_valid_date is None
        assert feat.availability_type == "empty"

    def test_has_availability_issues_property(self):
        findings_with_issues = ExplorationFindings(
            source_path="data.csv",
            source_format="csv",
            feature_availability=self.create_sample_availability_metadata(),
        )
        assert findings_with_issues.has_availability_issues is True

        findings_clean = ExplorationFindings(
            source_path="data.csv",
            source_format="csv",
            feature_availability=FeatureAvailabilityMetadata(
                data_start="2020-01-01",
                data_end="2022-12-31",
                time_span_days=1095,
                new_tracking=[],
                retired_tracking=[],
                partial_window=[],
                features={},
            ),
        )
        assert findings_clean.has_availability_issues is False

        findings_none = ExplorationFindings(source_path="data.csv", source_format="csv")
        assert findings_none.has_availability_issues is False

    def test_problematic_columns_property(self):
        findings = ExplorationFindings(
            source_path="data.csv",
            source_format="csv",
            feature_availability=self.create_sample_availability_metadata(),
        )
        problematic = findings.problematic_availability_columns
        assert "loyalty_score" in problematic
        assert "app_version" in problematic
        assert "old_system_flag" in problematic
        assert "temp_campaign" in problematic
        assert len(problematic) == 4

    def test_get_feature_availability_info(self):
        findings = ExplorationFindings(
            source_path="data.csv",
            source_format="csv",
            feature_availability=self.create_sample_availability_metadata(),
        )
        info = findings.get_feature_availability("loyalty_score")
        assert info is not None
        assert info.availability_type == "new_tracking"
        assert info.coverage_pct == 54.8

        missing = findings.get_feature_availability("nonexistent")
        assert missing is None

        findings_no_avail = ExplorationFindings(source_path="data.csv", source_format="csv")
        assert findings_no_avail.get_feature_availability("any") is None
