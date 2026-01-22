import pytest
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from customer_retention.core.config.column_config import ColumnType
from customer_retention.analysis.auto_explorer.findings import ColumnFinding, ExplorationFindings, _convert_to_native


class TestConvertToNative:
    def test_converts_numpy_int(self):
        assert _convert_to_native(np.int64(42)) == 42
        assert isinstance(_convert_to_native(np.int64(42)), int)

    def test_converts_numpy_float(self):
        assert _convert_to_native(np.float64(3.14)) == 3.14
        assert isinstance(_convert_to_native(np.float64(3.14)), float)

    def test_converts_nested_dict(self):
        data = {"a": np.int64(1), "b": {"c": np.float64(2.5)}}
        result = _convert_to_native(data)
        assert result == {"a": 1, "b": {"c": 2.5}}
        assert isinstance(result["a"], int)
        assert isinstance(result["b"]["c"], float)

    def test_converts_list_with_numpy(self):
        data = [np.int64(1), np.float64(2.0), "text"]
        result = _convert_to_native(data)
        assert result == [1, 2.0, "text"]
        assert isinstance(result[0], int)

    def test_preserves_none(self):
        assert _convert_to_native(None) is None

    def test_preserves_native_types(self):
        assert _convert_to_native(42) == 42
        assert _convert_to_native("text") == "text"
        assert _convert_to_native(3.14) == 3.14


class TestColumnFinding:
    def test_creation_with_required_fields(self):
        finding = ColumnFinding(
            name="age",
            inferred_type=ColumnType.NUMERIC_CONTINUOUS,
            confidence=0.9,
            evidence=["Numeric with many unique values"]
        )
        assert finding.name == "age"
        assert finding.inferred_type == ColumnType.NUMERIC_CONTINUOUS
        assert finding.confidence == 0.9

    def test_default_values(self):
        finding = ColumnFinding(
            name="test",
            inferred_type=ColumnType.TEXT,
            confidence=0.5,
            evidence=[]
        )
        assert finding.alternatives == []
        assert finding.universal_metrics == {}
        assert finding.type_metrics == {}
        assert finding.quality_issues == []
        assert finding.quality_score == 100.0
        assert not finding.cleaning_needed

    def test_to_column_config_basic(self):
        finding = ColumnFinding(
            name="customer_id",
            inferred_type=ColumnType.IDENTIFIER,
            confidence=0.95,
            evidence=["All unique"],
            universal_metrics={"null_count": 0}
        )
        config = finding.to_column_config()
        assert config.name == "customer_id"
        assert config.column_type == ColumnType.IDENTIFIER
        assert not config.nullable

    def test_to_column_config_with_nulls(self):
        finding = ColumnFinding(
            name="age",
            inferred_type=ColumnType.NUMERIC_CONTINUOUS,
            confidence=0.8,
            evidence=[],
            universal_metrics={"null_count": 10}
        )
        config = finding.to_column_config()
        assert config.nullable


class TestExplorationFindings:
    def create_sample_findings(self) -> ExplorationFindings:
        columns = {
            "customer_id": ColumnFinding(
                name="customer_id",
                inferred_type=ColumnType.IDENTIFIER,
                confidence=0.95,
                evidence=["All unique"]
            ),
            "age": ColumnFinding(
                name="age",
                inferred_type=ColumnType.NUMERIC_CONTINUOUS,
                confidence=0.85,
                evidence=["Numeric with many values"]
            ),
            "churned": ColumnFinding(
                name="churned",
                inferred_type=ColumnType.TARGET,
                confidence=0.9,
                evidence=["Binary target"]
            )
        }
        return ExplorationFindings(
            source_path="test_data.csv",
            source_format="csv",
            row_count=1000,
            column_count=3,
            memory_usage_mb=1.5,
            columns=columns,
            target_column="churned",
            target_type="binary",
            identifier_columns=["customer_id"]
        )

    def test_creation(self):
        findings = self.create_sample_findings()
        assert findings.source_path == "test_data.csv"
        assert findings.row_count == 1000
        assert len(findings.columns) == 3

    def test_column_types_property(self):
        findings = self.create_sample_findings()
        types = findings.column_types
        assert types["customer_id"] == ColumnType.IDENTIFIER
        assert types["age"] == ColumnType.NUMERIC_CONTINUOUS
        assert types["churned"] == ColumnType.TARGET

    def test_column_configs_property(self):
        findings = self.create_sample_findings()
        configs = findings.column_configs
        assert len(configs) == 3
        assert configs["age"].column_type == ColumnType.NUMERIC_CONTINUOUS

    def test_to_dict(self):
        findings = self.create_sample_findings()
        data = findings.to_dict()
        assert data["source_path"] == "test_data.csv"
        assert data["row_count"] == 1000
        assert "columns" in data
        assert data["columns"]["age"]["inferred_type"] == "numeric_continuous"

    def test_to_json(self):
        findings = self.create_sample_findings()
        json_str = findings.to_json()
        parsed = json.loads(json_str)
        assert parsed["source_path"] == "test_data.csv"

    def test_to_yaml(self):
        findings = self.create_sample_findings()
        yaml_str = findings.to_yaml()
        assert "source_path: test_data.csv" in yaml_str

    def test_from_dict(self):
        original = self.create_sample_findings()
        data = original.to_dict()
        restored = ExplorationFindings.from_dict(data)
        assert restored.source_path == original.source_path
        assert restored.row_count == original.row_count
        assert restored.column_types["age"] == ColumnType.NUMERIC_CONTINUOUS

    def test_from_json(self):
        original = self.create_sample_findings()
        json_str = original.to_json()
        restored = ExplorationFindings.from_json(json_str)
        assert restored.target_column == "churned"

    def test_from_yaml(self):
        original = self.create_sample_findings()
        yaml_str = original.to_yaml()
        restored = ExplorationFindings.from_yaml(yaml_str)
        assert restored.target_type == "binary"

    def test_save_and_load_yaml(self):
        original = self.create_sample_findings()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "findings.yaml"
            original.save(str(path))
            loaded = ExplorationFindings.load(str(path))
            assert loaded.source_path == original.source_path
            assert loaded.column_types == original.column_types

    def test_save_and_load_json(self):
        original = self.create_sample_findings()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "findings.json"
            original.save(str(path))
            loaded = ExplorationFindings.load(str(path))
            assert loaded.row_count == original.row_count

    def test_empty_columns_property(self):
        findings = ExplorationFindings(
            source_path="empty.csv",
            source_format="csv"
        )
        assert findings.column_types == {}
        assert findings.column_configs == {}

    def test_overall_quality_score_default(self):
        findings = ExplorationFindings(
            source_path="test.csv",
            source_format="csv"
        )
        assert findings.overall_quality_score == 100.0

    def test_modeling_ready_default(self):
        findings = ExplorationFindings(
            source_path="test.csv",
            source_format="csv"
        )
        assert not findings.modeling_ready

    def test_metadata_default_empty(self):
        findings = ExplorationFindings(
            source_path="test.csv",
            source_format="csv"
        )
        assert findings.metadata == {}

    def test_metadata_can_be_set(self):
        findings = ExplorationFindings(
            source_path="test.csv",
            source_format="csv"
        )
        findings.metadata["business_context"] = {"objective": "reduce churn"}
        assert findings.metadata["business_context"]["objective"] == "reduce churn"

    def test_metadata_persists_in_serialization(self):
        findings = self.create_sample_findings()
        findings.metadata = {"project": "test", "version": 1}
        data = findings.to_dict()
        assert data["metadata"] == {"project": "test", "version": 1}

        restored = ExplorationFindings.from_dict(data)
        assert restored.metadata == {"project": "test", "version": 1}

    def test_numpy_types_converted_to_native(self):
        findings = ExplorationFindings(
            source_path="test.csv",
            source_format="csv",
            memory_usage_mb=np.float64(1.5),
            row_count=np.int64(1000),
            overall_quality_score=np.float32(95.5)
        )
        data = findings.to_dict()
        assert isinstance(data["memory_usage_mb"], float)
        assert isinstance(data["row_count"], int)
        assert isinstance(data["overall_quality_score"], float)
        yaml_str = findings.to_yaml()
        assert "!!python" not in yaml_str
        restored = ExplorationFindings.from_yaml(yaml_str)
        assert restored.row_count == 1000


class TestTimeSeriesMetadata:
    """Tests for TimeSeriesMetadata integration with ExplorationFindings."""

    def test_time_series_metadata_creation(self):
        from customer_retention.analysis.auto_explorer.findings import TimeSeriesMetadata
        from customer_retention.core.config import DatasetGranularity

        metadata = TimeSeriesMetadata(
            granularity=DatasetGranularity.EVENT_LEVEL,
            entity_column="customer_id",
            time_column="transaction_date",
            avg_events_per_entity=5.2,
            time_span_days=365
        )

        assert metadata.granularity == DatasetGranularity.EVENT_LEVEL
        assert metadata.entity_column == "customer_id"
        assert metadata.time_column == "transaction_date"

    def test_findings_with_time_series_metadata(self):
        from customer_retention.analysis.auto_explorer.findings import TimeSeriesMetadata
        from customer_retention.core.config import DatasetGranularity

        ts_metadata = TimeSeriesMetadata(
            granularity=DatasetGranularity.EVENT_LEVEL,
            entity_column="customer_id",
            time_column="event_date",
            avg_events_per_entity=10.5,
            time_span_days=180
        )

        findings = ExplorationFindings(
            source_path="events.csv",
            source_format="csv",
            row_count=10000,
            column_count=5,
            time_series_metadata=ts_metadata
        )

        assert findings.time_series_metadata is not None
        assert findings.time_series_metadata.granularity == DatasetGranularity.EVENT_LEVEL

    def test_findings_without_time_series_metadata(self):
        findings = ExplorationFindings(
            source_path="customers.csv",
            source_format="csv"
        )
        assert findings.time_series_metadata is None

    def test_time_series_metadata_serialization(self):
        from customer_retention.analysis.auto_explorer.findings import TimeSeriesMetadata
        from customer_retention.core.config import DatasetGranularity

        ts_metadata = TimeSeriesMetadata(
            granularity=DatasetGranularity.EVENT_LEVEL,
            entity_column="user_id",
            time_column="created_at",
            avg_events_per_entity=3.5,
            time_span_days=90
        )

        findings = ExplorationFindings(
            source_path="events.csv",
            source_format="csv",
            time_series_metadata=ts_metadata
        )

        # Test serialization
        data = findings.to_dict()
        assert "time_series_metadata" in data
        assert data["time_series_metadata"]["granularity"] == "event_level"
        assert data["time_series_metadata"]["entity_column"] == "user_id"

        # Test YAML serialization
        yaml_str = findings.to_yaml()
        assert "time_series_metadata:" in yaml_str
        assert "granularity: event_level" in yaml_str

    def test_time_series_metadata_deserialization(self):
        from customer_retention.analysis.auto_explorer.findings import TimeSeriesMetadata
        from customer_retention.core.config import DatasetGranularity

        ts_metadata = TimeSeriesMetadata(
            granularity=DatasetGranularity.EVENT_LEVEL,
            entity_column="cust_id",
            time_column="order_date",
            avg_events_per_entity=7.0,
            time_span_days=365
        )

        original = ExplorationFindings(
            source_path="orders.csv",
            source_format="csv",
            time_series_metadata=ts_metadata
        )

        # Round-trip through YAML
        yaml_str = original.to_yaml()
        restored = ExplorationFindings.from_yaml(yaml_str)

        assert restored.time_series_metadata is not None
        assert restored.time_series_metadata.granularity == DatasetGranularity.EVENT_LEVEL
        assert restored.time_series_metadata.entity_column == "cust_id"
        assert restored.time_series_metadata.avg_events_per_entity == 7.0

    def test_is_time_series_property(self):
        from customer_retention.analysis.auto_explorer.findings import TimeSeriesMetadata
        from customer_retention.core.config import DatasetGranularity

        # Entity-level data
        entity_findings = ExplorationFindings(
            source_path="customers.csv",
            source_format="csv",
            time_series_metadata=TimeSeriesMetadata(
                granularity=DatasetGranularity.ENTITY_LEVEL
            )
        )
        assert not entity_findings.is_time_series

        # Event-level data
        event_findings = ExplorationFindings(
            source_path="transactions.csv",
            source_format="csv",
            time_series_metadata=TimeSeriesMetadata(
                granularity=DatasetGranularity.EVENT_LEVEL,
                entity_column="customer_id",
                time_column="date"
            )
        )
        assert event_findings.is_time_series

        # No metadata
        no_metadata = ExplorationFindings(
            source_path="unknown.csv",
            source_format="csv"
        )
        assert not no_metadata.is_time_series

    def test_aggregation_tracking_fields_defaults(self):
        from customer_retention.analysis.auto_explorer.findings import TimeSeriesMetadata
        from customer_retention.core.config import DatasetGranularity

        metadata = TimeSeriesMetadata(granularity=DatasetGranularity.EVENT_LEVEL)
        assert metadata.aggregation_executed is False
        assert metadata.aggregated_data_path is None
        assert metadata.aggregated_findings_path is None
        assert metadata.aggregation_windows_used == []
        assert metadata.aggregation_timestamp is None

    def test_aggregation_tracking_fields_populated(self):
        from customer_retention.analysis.auto_explorer.findings import TimeSeriesMetadata
        from customer_retention.core.config import DatasetGranularity

        metadata = TimeSeriesMetadata(
            granularity=DatasetGranularity.EVENT_LEVEL,
            entity_column="customer_id",
            time_column="event_date",
            aggregation_executed=True,
            aggregated_data_path="/data/aggregated.parquet",
            aggregated_findings_path="/explorations/aggregated_findings.yaml",
            aggregation_windows_used=["7d", "30d", "all_time"],
            aggregation_timestamp="2024-01-15T10:30:00"
        )
        assert metadata.aggregation_executed is True
        assert metadata.aggregated_data_path == "/data/aggregated.parquet"
        assert metadata.aggregated_findings_path == "/explorations/aggregated_findings.yaml"
        assert metadata.aggregation_windows_used == ["7d", "30d", "all_time"]
        assert metadata.aggregation_timestamp == "2024-01-15T10:30:00"

    def test_aggregation_tracking_serialization(self):
        from customer_retention.analysis.auto_explorer.findings import TimeSeriesMetadata
        from customer_retention.core.config import DatasetGranularity

        ts_metadata = TimeSeriesMetadata(
            granularity=DatasetGranularity.EVENT_LEVEL,
            entity_column="user_id",
            time_column="created_at",
            aggregation_executed=True,
            aggregated_data_path="/data/users_aggregated.parquet",
            aggregated_findings_path="/explorations/users_aggregated_findings.yaml",
            aggregation_windows_used=["24h", "7d", "30d"],
            aggregation_timestamp="2024-02-01T14:00:00"
        )
        findings = ExplorationFindings(
            source_path="events.csv",
            source_format="csv",
            time_series_metadata=ts_metadata
        )

        yaml_str = findings.to_yaml()
        restored = ExplorationFindings.from_yaml(yaml_str)

        assert restored.time_series_metadata.aggregation_executed is True
        assert restored.time_series_metadata.aggregated_data_path == "/data/users_aggregated.parquet"
        assert restored.time_series_metadata.aggregation_windows_used == ["24h", "7d", "30d"]

    def test_has_aggregated_output_property(self):
        from customer_retention.analysis.auto_explorer.findings import TimeSeriesMetadata
        from customer_retention.core.config import DatasetGranularity

        # No metadata - no aggregation
        findings_no_meta = ExplorationFindings(source_path="test.csv", source_format="csv")
        assert findings_no_meta.has_aggregated_output is False

        # Metadata but not aggregated
        findings_not_agg = ExplorationFindings(
            source_path="test.csv",
            source_format="csv",
            time_series_metadata=TimeSeriesMetadata(
                granularity=DatasetGranularity.EVENT_LEVEL,
                aggregation_executed=False
            )
        )
        assert findings_not_agg.has_aggregated_output is False

        # Metadata with aggregation executed
        findings_agg = ExplorationFindings(
            source_path="test.csv",
            source_format="csv",
            time_series_metadata=TimeSeriesMetadata(
                granularity=DatasetGranularity.EVENT_LEVEL,
                aggregation_executed=True,
                aggregated_data_path="/data/agg.parquet"
            )
        )
        assert findings_agg.has_aggregated_output is True
