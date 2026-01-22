import pytest
from customer_retention.core.config import (
    DedupStrategy, BronzeConfig, SilverConfig, GoldConfig,
    ModelingConfig, ValidationConfig, PathConfig, PipelineConfig,
    DataSourceConfig, SourceType, FileFormat, ColumnType, ColumnConfig
)


class TestDedupStrategy:
    def test_all_strategies(self):
        assert DedupStrategy.KEEP_FIRST == "keep_first"
        assert DedupStrategy.KEEP_LAST == "keep_last"
        assert DedupStrategy.KEEP_MOST_COMPLETE == "keep_most_complete"


class TestBronzeConfig:
    def test_defaults(self):
        config = BronzeConfig()
        assert config.deduplicate is True
        assert config.dedup_strategy == DedupStrategy.KEEP_LAST
        assert config.dedup_keys == ["custid"]
        assert config.max_missing_pct == 0.95
        assert config.min_distinct_values == 2

    def test_custom_values(self):
        config = BronzeConfig(
            deduplicate=False,
            dedup_strategy=DedupStrategy.KEEP_FIRST,
            dedup_keys=["id", "timestamp"],
            max_missing_pct=0.8,
            min_distinct_values=5
        )
        assert config.deduplicate is False
        assert config.dedup_strategy == DedupStrategy.KEEP_FIRST
        assert config.dedup_keys == ["id", "timestamp"]
        assert config.max_missing_pct == 0.8
        assert config.min_distinct_values == 5


class TestSilverConfig:
    def test_defaults(self):
        config = SilverConfig()
        assert config.entity_key == "custid"
        assert config.reference_date_column is None
        assert config.auto_detect_encoding is True
        assert config.auto_detect_scaling is True

    def test_custom_values(self):
        config = SilverConfig(
            entity_key="customer_id",
            reference_date_column="observation_date",
            auto_detect_encoding=False,
            auto_detect_scaling=False
        )
        assert config.entity_key == "customer_id"
        assert config.reference_date_column == "observation_date"
        assert config.auto_detect_encoding is False
        assert config.auto_detect_scaling is False


class TestGoldConfig:
    def test_defaults(self):
        config = GoldConfig()
        assert config.feature_store_catalog == "main"
        assert config.feature_store_schema == "feature_store"
        assert config.feature_table_name == "customer_features"
        assert config.version == "v1"

    def test_custom_values(self):
        config = GoldConfig(
            feature_store_catalog="prod",
            feature_store_schema="ml_features",
            feature_table_name="churn_features",
            version="v2"
        )
        assert config.feature_store_catalog == "prod"
        assert config.feature_store_schema == "ml_features"
        assert config.feature_table_name == "churn_features"
        assert config.version == "v2"

    def test_get_full_feature_table_name(self):
        config = GoldConfig(
            feature_store_catalog="prod",
            feature_store_schema="ml_features",
            feature_table_name="churn_features"
        )
        assert config.get_full_feature_table_name() == "prod.ml_features.churn_features"


class TestModelingConfig:
    def test_defaults(self):
        config = ModelingConfig()
        assert config.target_column == "retained"
        assert config.positive_class == 1
        assert config.test_size == 0.2
        assert config.stratify is True
        assert config.primary_metric == "average_precision"
        assert config.cost_false_negative == 100.0
        assert config.cost_false_positive == 10.0

    def test_custom_values(self):
        config = ModelingConfig(
            target_column="churned",
            positive_class=0,
            test_size=0.3,
            stratify=False,
            primary_metric="f1",
            cost_false_negative=200.0,
            cost_false_positive=5.0
        )
        assert config.target_column == "churned"
        assert config.positive_class == 0
        assert config.test_size == 0.3
        assert config.stratify is False
        assert config.primary_metric == "f1"

    def test_get_cost_ratio(self):
        config = ModelingConfig(cost_false_negative=100.0, cost_false_positive=10.0)
        assert config.get_cost_ratio() == 10.0

    def test_get_cost_ratio_custom(self):
        config = ModelingConfig(cost_false_negative=200.0, cost_false_positive=5.0)
        assert config.get_cost_ratio() == 40.0


class TestValidationConfig:
    def test_defaults(self):
        config = ValidationConfig()
        assert config.fail_on_critical is True
        assert config.fail_on_high is False
        assert config.leakage_correlation_threshold == 0.90
        assert config.max_overfit_gap == 0.15

    def test_custom_values(self):
        config = ValidationConfig(
            fail_on_critical=False,
            fail_on_high=True,
            leakage_correlation_threshold=0.95,
            max_overfit_gap=0.10
        )
        assert config.fail_on_critical is False
        assert config.fail_on_high is True
        assert config.leakage_correlation_threshold == 0.95
        assert config.max_overfit_gap == 0.10


class TestPathConfig:
    def test_defaults(self):
        config = PathConfig()
        assert config.landing_zone is None
        assert config.bronze is None
        assert config.silver is None
        assert config.gold is None

    def test_custom_values(self):
        config = PathConfig(
            landing_zone="/data/landing",
            bronze="/data/bronze",
            silver="/data/silver",
            gold="/data/gold"
        )
        assert config.landing_zone == "/data/landing"
        assert config.bronze == "/data/bronze"
        assert config.silver == "/data/silver"
        assert config.gold == "/data/gold"


class TestPipelineConfig:
    def test_minimal_config(self):
        config = PipelineConfig(project_name="test_project")
        assert config.project_name == "test_project"
        assert config.project_description is None
        assert config.version == "1.0.0"
        assert len(config.data_sources) == 0

    def test_with_description_and_version(self):
        config = PipelineConfig(
            project_name="test_project",
            project_description="Test description",
            version="2.0.0"
        )
        assert config.project_description == "Test description"
        assert config.version == "2.0.0"

    def test_with_data_sources(self):
        source = DataSourceConfig(
            name="test_source",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/data/test.csv",
            file_format=FileFormat.CSV
        )
        config = PipelineConfig(
            project_name="test_project",
            data_sources=[source]
        )
        assert len(config.data_sources) == 1
        assert config.data_sources[0].name == "test_source"

    def test_default_subconfigs_created(self):
        config = PipelineConfig(project_name="test_project")
        assert isinstance(config.bronze, BronzeConfig)
        assert isinstance(config.silver, SilverConfig)
        assert isinstance(config.gold, GoldConfig)
        assert isinstance(config.modeling, ModelingConfig)
        assert isinstance(config.validation, ValidationConfig)
        assert isinstance(config.paths, PathConfig)

    def test_custom_subconfigs(self):
        config = PipelineConfig(
            project_name="test_project",
            bronze=BronzeConfig(deduplicate=False),
            silver=SilverConfig(entity_key="customer_id"),
            modeling=ModelingConfig(target_column="churned")
        )
        assert config.bronze.deduplicate is False
        assert config.silver.entity_key == "customer_id"
        assert config.modeling.target_column == "churned"

    def test_get_source_by_name_exists(self):
        source1 = DataSourceConfig(
            name="source1",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/data/test1.csv",
            file_format=FileFormat.CSV
        )
        source2 = DataSourceConfig(
            name="source2",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/data/test2.csv",
            file_format=FileFormat.CSV
        )
        config = PipelineConfig(
            project_name="test_project",
            data_sources=[source1, source2]
        )
        result = config.get_source_by_name("source2")
        assert result is not None
        assert result.name == "source2"

    def test_get_source_by_name_not_exists(self):
        source = DataSourceConfig(
            name="source1",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/data/test1.csv",
            file_format=FileFormat.CSV
        )
        config = PipelineConfig(
            project_name="test_project",
            data_sources=[source]
        )
        result = config.get_source_by_name("nonexistent")
        assert result is None

    def test_get_target_source(self):
        columns_with_target = [
            ColumnConfig(name="id", column_type=ColumnType.IDENTIFIER),
            ColumnConfig(name="retained", column_type=ColumnType.TARGET)
        ]
        columns_without_target = [
            ColumnConfig(name="id", column_type=ColumnType.IDENTIFIER)
        ]
        source1 = DataSourceConfig(
            name="source1",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/data/test1.csv",
            file_format=FileFormat.CSV,
            columns=columns_without_target
        )
        source2 = DataSourceConfig(
            name="source2",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/data/test2.csv",
            file_format=FileFormat.CSV,
            columns=columns_with_target
        )
        config = PipelineConfig(
            project_name="test_project",
            data_sources=[source1, source2]
        )
        target_source = config.get_target_source()
        assert target_source is not None
        assert target_source.name == "source2"

    def test_get_target_source_none(self):
        source = DataSourceConfig(
            name="source1",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/data/test1.csv",
            file_format=FileFormat.CSV
        )
        config = PipelineConfig(
            project_name="test_project",
            data_sources=[source]
        )
        assert config.get_target_source() is None

    def test_get_all_feature_columns(self):
        columns1 = [
            ColumnConfig(name="id", column_type=ColumnType.IDENTIFIER),
            ColumnConfig(name="age", column_type=ColumnType.NUMERIC_CONTINUOUS)
        ]
        columns2 = [
            ColumnConfig(name="retained", column_type=ColumnType.TARGET),
            ColumnConfig(name="city", column_type=ColumnType.CATEGORICAL_NOMINAL)
        ]
        source1 = DataSourceConfig(
            name="source1",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/data/test1.csv",
            file_format=FileFormat.CSV,
            columns=columns1
        )
        source2 = DataSourceConfig(
            name="source2",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/data/test2.csv",
            file_format=FileFormat.CSV,
            columns=columns2
        )
        config = PipelineConfig(
            project_name="test_project",
            data_sources=[source1, source2]
        )
        feature_cols = config.get_all_feature_columns()
        assert len(feature_cols) == 2
        assert "age" in feature_cols
        assert "city" in feature_cols
        assert "id" not in feature_cols
        assert "retained" not in feature_cols

    def test_json_serialization(self):
        config = PipelineConfig(project_name="test_project")
        json_data = config.model_dump()
        assert json_data["project_name"] == "test_project"
        assert json_data["version"] == "1.0.0"

    def test_json_deserialization(self):
        data = {
            "project_name": "test_project",
            "version": "2.0.0"
        }
        config = PipelineConfig(**data)
        assert config.project_name == "test_project"
        assert config.version == "2.0.0"
