import decimal
import json
import tempfile
from pathlib import Path

import numpy as np

from customer_retention.analysis.auto_explorer.findings import ColumnFinding, ExplorationFindings, _convert_to_native
from customer_retention.core.config.column_config import ColumnType


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

    def test_converts_decimal(self):
        assert _convert_to_native(decimal.Decimal("3.14")) == 3.14
        assert isinstance(_convert_to_native(decimal.Decimal("3.14")), float)

    def test_converts_nested_decimal(self):
        data = {"a": decimal.Decimal("1.5"), "b": [decimal.Decimal("2.0")]}
        result = _convert_to_native(data)
        assert result == {"a": 1.5, "b": [2.0]}
        assert isinstance(result["a"], float)
        assert isinstance(result["b"][0], float)


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

    def test_datetime_derivation_sources_round_trip(self):
        original = self.create_sample_findings()
        original.datetime_derivation_sources = ["signup_date", "first_purchase_date"]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "findings.yaml"
            original.save(str(path))
            loaded = ExplorationFindings.load(str(path))
            assert loaded.datetime_derivation_sources == ["signup_date", "first_purchase_date"]

    def test_datetime_derivation_sources_default_empty(self):
        findings = ExplorationFindings(source_path="test.csv", source_format="csv")
        assert findings.datetime_derivation_sources == []

    def test_datetime_allow_future_columns_round_trip(self):
        original = self.create_sample_findings()
        original.datetime_derivation_sources = ["signup_date", "contract_end"]
        original.datetime_allow_future_columns = ["contract_end"]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "findings.yaml"
            original.save(str(path))
            loaded = ExplorationFindings.load(str(path))
            assert loaded.datetime_allow_future_columns == ["contract_end"]

    def test_datetime_allow_future_columns_default_empty(self):
        findings = ExplorationFindings(source_path="test.csv", source_format="csv")
        assert findings.datetime_allow_future_columns == []

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

    def test_decimal_types_converted_to_native(self):
        findings = ExplorationFindings(
            source_path="test.csv",
            source_format="csv",
            memory_usage_mb=decimal.Decimal("1.5"),
            row_count=1000,
            overall_quality_score=decimal.Decimal("95.5"),
        )
        data = findings.to_dict()
        assert isinstance(data["memory_usage_mb"], float)
        assert isinstance(data["overall_quality_score"], float)
        yaml_str = findings.to_yaml()
        assert "!!python" not in yaml_str
        assert "decimal" not in yaml_str.lower()
        restored = ExplorationFindings.from_yaml(yaml_str)
        assert restored.overall_quality_score == 95.5


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

    def test_drift_fields_have_none_defaults(self):
        from customer_retention.analysis.auto_explorer.findings import TimeSeriesMetadata

        metadata = TimeSeriesMetadata()
        assert metadata.drift_risk_level is None
        assert metadata.volume_drift_risk is None
        assert metadata.population_stability is None
        assert metadata.regime_count is None
        assert metadata.recommended_training_start is None

    def test_drift_fields_roundtrip_through_yaml(self):
        from customer_retention.analysis.auto_explorer.findings import TimeSeriesMetadata
        from customer_retention.core.config import DatasetGranularity

        ts_metadata = TimeSeriesMetadata(
            granularity=DatasetGranularity.EVENT_LEVEL,
            entity_column="user_id",
            time_column="event_date",
            drift_risk_level="moderate",
            volume_drift_risk="declining",
            population_stability=0.72,
            regime_count=2,
            recommended_training_start="2020-10-01T00:00:00",
        )
        findings = ExplorationFindings(
            source_path="events.csv",
            source_format="csv",
            time_series_metadata=ts_metadata,
        )

        yaml_str = findings.to_yaml()
        restored = ExplorationFindings.from_yaml(yaml_str)

        assert restored.time_series_metadata.drift_risk_level == "moderate"
        assert restored.time_series_metadata.volume_drift_risk == "declining"
        assert restored.time_series_metadata.population_stability == 0.72
        assert restored.time_series_metadata.regime_count == 2
        assert restored.time_series_metadata.recommended_training_start == "2020-10-01T00:00:00"


class TestMergeFromDatasets:

    @staticmethod
    def _make_findings(
        name: str,
        columns: dict[str, ColumnFinding],
        target_column: str | None = None,
        target_type: str | None = None,
        identifier_columns: list[str] | None = None,
        datetime_columns: list[str] | None = None,
    ) -> ExplorationFindings:
        return ExplorationFindings(
            source_path=f"{name}.csv",
            source_format="csv",
            row_count=100,
            column_count=len(columns),
            columns=columns,
            target_column=target_column,
            target_type=target_type,
            identifier_columns=identifier_columns or [],
            datetime_columns=datetime_columns or [],
        )

    @staticmethod
    def _col(name: str, col_type: ColumnType, quality_score: float = 100.0) -> ColumnFinding:
        return ColumnFinding(
            name=name,
            inferred_type=col_type,
            confidence=0.9,
            evidence=["test"],
            quality_score=quality_score,
        )

    def test_merge_combines_columns_from_multiple_findings(self):
        f1 = self._make_findings("ds1", {
            "age": self._col("age", ColumnType.NUMERIC_CONTINUOUS),
        })
        f2 = self._make_findings("ds2", {
            "plan_type": self._col("plan_type", ColumnType.CATEGORICAL_NOMINAL),
        })
        merged = ExplorationFindings.merge_from_datasets(
            [f1, f2], row_count=500, column_count=4, source_path="/silver"
        )
        assert "age" in merged.columns
        assert "plan_type" in merged.columns
        assert "entity_id" in merged.columns
        assert "as_of_date" in merged.columns
        assert len(merged.columns) == 4

    def test_merge_handles_renamed_columns(self):
        f1 = self._make_findings("ds1", {
            "status": self._col("status", ColumnType.CATEGORICAL_NOMINAL),
        })
        merged = ExplorationFindings.merge_from_datasets(
            [f1],
            row_count=100,
            column_count=3,
            source_path="/silver",
            renamed_columns={"status": "ds1_status"},
        )
        assert "ds1_status" in merged.columns
        assert "status" not in merged.columns
        assert merged.columns["ds1_status"].inferred_type == ColumnType.CATEGORICAL_NOMINAL

    def test_merge_adds_spine_columns(self):
        merged = ExplorationFindings.merge_from_datasets(
            [], row_count=0, column_count=2, source_path="/silver"
        )
        assert "entity_id" in merged.columns
        assert merged.columns["entity_id"].inferred_type == ColumnType.IDENTIFIER
        assert merged.columns["entity_id"].confidence == 1.0
        assert "as_of_date" in merged.columns
        assert merged.columns["as_of_date"].inferred_type == ColumnType.DATETIME
        assert merged.columns["as_of_date"].confidence == 1.0

    def test_merge_preserves_target_from_source(self):
        f1 = self._make_findings("ds1", {
            "age": self._col("age", ColumnType.NUMERIC_CONTINUOUS),
        })
        f2 = self._make_findings("ds2", {
            "churned": self._col("churned", ColumnType.TARGET),
        }, target_column="churned", target_type="binary")
        merged = ExplorationFindings.merge_from_datasets(
            [f1, f2], row_count=500, column_count=4, source_path="/silver"
        )
        assert merged.target_column == "churned"
        assert merged.target_type == "binary"

    def test_merge_sets_row_and_column_counts(self):
        merged = ExplorationFindings.merge_from_datasets(
            [], row_count=2000, column_count=50, source_path="/silver"
        )
        assert merged.row_count == 2000
        assert merged.column_count == 50

    def test_merge_computes_overall_quality_score(self):
        f1 = self._make_findings("ds1", {
            "a": self._col("a", ColumnType.NUMERIC_CONTINUOUS, quality_score=80.0),
        })
        f2 = self._make_findings("ds2", {
            "b": self._col("b", ColumnType.NUMERIC_CONTINUOUS, quality_score=60.0),
        })
        merged = ExplorationFindings.merge_from_datasets(
            [f1, f2], row_count=100, column_count=4, source_path="/silver"
        )
        scores = [c.quality_score for c in merged.columns.values()]
        expected = sum(scores) / len(scores)
        assert abs(merged.overall_quality_score - expected) < 0.01

    def test_merge_sets_identifier_and_datetime_columns(self):
        f1 = self._make_findings("ds1", {
            "cust_id": self._col("cust_id", ColumnType.IDENTIFIER),
        }, identifier_columns=["cust_id"])
        f2 = self._make_findings("ds2", {
            "signup_date": self._col("signup_date", ColumnType.DATETIME),
        }, datetime_columns=["signup_date"])
        merged = ExplorationFindings.merge_from_datasets(
            [f1, f2], row_count=100, column_count=5, source_path="/silver"
        )
        assert "cust_id" in merged.identifier_columns
        assert "entity_id" in merged.identifier_columns
        assert "signup_date" in merged.datetime_columns
        assert "as_of_date" in merged.datetime_columns

    def test_merge_skips_temporal_metadata_cols(self):
        f1 = self._make_findings("ds1", {
            "age": self._col("age", ColumnType.NUMERIC_CONTINUOUS),
            "feature_timestamp": self._col("feature_timestamp", ColumnType.FEATURE_TIMESTAMP),
            "label_timestamp": self._col("label_timestamp", ColumnType.LABEL_TIMESTAMP),
            "label_available_flag": self._col("label_available_flag", ColumnType.BINARY),
        })
        merged = ExplorationFindings.merge_from_datasets(
            [f1], row_count=100, column_count=3, source_path="/silver"
        )
        assert "age" in merged.columns
        assert "feature_timestamp" not in merged.columns
        assert "label_timestamp" not in merged.columns
        assert "label_available_flag" not in merged.columns

    def test_merge_serialization_roundtrip(self):
        f1 = self._make_findings("ds1", {
            "age": self._col("age", ColumnType.NUMERIC_CONTINUOUS, quality_score=90.0),
        })
        f2 = self._make_findings("ds2", {
            "churned": self._col("churned", ColumnType.TARGET),
        }, target_column="churned", target_type="binary")
        merged = ExplorationFindings.merge_from_datasets(
            [f1, f2], row_count=500, column_count=4, source_path="/silver"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "merged_findings.yaml"
            merged.save(str(path))
            loaded = ExplorationFindings.load(str(path))
            assert loaded.row_count == 500
            assert loaded.target_column == "churned"
            assert set(loaded.columns.keys()) == set(merged.columns.keys())
            for col_name in merged.columns:
                assert loaded.columns[col_name].inferred_type == merged.columns[col_name].inferred_type

    def test_merge_empty_findings_list(self):
        merged = ExplorationFindings.merge_from_datasets(
            [], row_count=0, column_count=2, source_path="/silver"
        )
        assert "entity_id" in merged.columns
        assert "as_of_date" in merged.columns
        assert len(merged.columns) == 2
        assert merged.target_column is None

    def test_merge_overlapping_columns(self):
        f1 = self._make_findings("ds1", {
            "score": self._col("score", ColumnType.NUMERIC_CONTINUOUS, quality_score=80.0),
        })
        f2 = self._make_findings("ds2", {
            "score": self._col("score", ColumnType.NUMERIC_DISCRETE, quality_score=90.0),
        })
        merged = ExplorationFindings.merge_from_datasets(
            [f1, f2], row_count=100, column_count=3, source_path="/silver"
        )
        assert merged.columns["score"].inferred_type == ColumnType.NUMERIC_DISCRETE

    def test_merge_custom_entity_key(self):
        merged = ExplorationFindings.merge_from_datasets(
            [], row_count=0, column_count=2, source_path="/silver",
            entity_key="customer_id",
        )
        assert "customer_id" in merged.columns
        assert "entity_id" not in merged.columns
        assert merged.columns["customer_id"].inferred_type == ColumnType.IDENTIFIER


class TestBuildDatetimeDiscoveryStats:
    def _make_findings(self, datetime_cols, columns, row_count=100):
        return ExplorationFindings(
            source_path="/test",
            source_format="csv",
            row_count=row_count,
            column_count=len(columns),
            columns=columns,
            datetime_columns=datetime_cols,
        )

    def test_basic_reconstruction(self):
        columns = {
            "created_at": ColumnFinding(
                name="created_at",
                inferred_type=ColumnType.DATETIME,
                confidence=1.0,
                evidence=["test"],
                universal_metrics={"null_count": 10},
                type_metrics={
                    "min_date": "2023-01-01T00:00:00",
                    "max_date": "2023-12-31T00:00:00",
                    "future_date_count": 5,
                },
            ),
        }
        findings = self._make_findings(["created_at"], columns, row_count=100)
        stats = findings.build_datetime_discovery_stats()
        assert "created_at" in stats
        s = stats["created_at"]
        assert s.coverage == 0.9
        assert s.future_fraction == 5 / 90
        assert s.min_date is not None
        assert s.max_date is not None

    def test_empty_datetime_columns(self):
        findings = self._make_findings([], {}, row_count=100)
        stats = findings.build_datetime_discovery_stats()
        assert stats == {}

    def test_missing_column_in_columns_dict(self):
        findings = self._make_findings(["nonexistent"], {}, row_count=100)
        stats = findings.build_datetime_discovery_stats()
        assert stats == {}

    def test_zero_row_count(self):
        columns = {
            "ts": ColumnFinding(
                name="ts",
                inferred_type=ColumnType.DATETIME,
                confidence=1.0,
                evidence=["test"],
                universal_metrics={"null_count": 0},
                type_metrics={"min_date": None, "max_date": None, "future_date_count": 0},
            ),
        }
        findings = self._make_findings(["ts"], columns, row_count=0)
        stats = findings.build_datetime_discovery_stats()
        assert stats["ts"].coverage == 0.0
        assert stats["ts"].future_fraction == 0.0

    def test_all_null_column(self):
        columns = {
            "ts": ColumnFinding(
                name="ts",
                inferred_type=ColumnType.DATETIME,
                confidence=1.0,
                evidence=["test"],
                universal_metrics={"null_count": 100},
                type_metrics={"min_date": None, "max_date": None, "future_date_count": 0},
            ),
        }
        findings = self._make_findings(["ts"], columns, row_count=100)
        stats = findings.build_datetime_discovery_stats()
        assert stats["ts"].coverage == 0.0
        assert stats["ts"].future_fraction == 0.0
        assert stats["ts"].min_date is None

    def test_multiple_datetime_columns(self):
        columns = {
            "created_at": ColumnFinding(
                name="created_at",
                inferred_type=ColumnType.DATETIME,
                confidence=1.0,
                evidence=["test"],
                universal_metrics={"null_count": 0},
                type_metrics={"min_date": "2023-01-01", "max_date": "2023-12-31", "future_date_count": 0},
            ),
            "updated_at": ColumnFinding(
                name="updated_at",
                inferred_type=ColumnType.DATETIME,
                confidence=1.0,
                evidence=["test"],
                universal_metrics={"null_count": 50},
                type_metrics={"min_date": "2023-06-01", "max_date": "2024-01-15", "future_date_count": 10},
            ),
        }
        findings = self._make_findings(["created_at", "updated_at"], columns, row_count=100)
        stats = findings.build_datetime_discovery_stats()
        assert len(stats) == 2
        assert stats["created_at"].coverage == 1.0
        assert stats["created_at"].future_fraction == 0.0
        assert stats["updated_at"].coverage == 0.5
        assert stats["updated_at"].future_fraction == 10 / 50
