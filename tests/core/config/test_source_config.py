import pytest
from pydantic import ValidationError
from customer_retention.core.config import (
    SourceType, FileFormat, Grain, DataSourceConfig,
    ColumnType, ColumnConfig
)


class TestSourceType:
    def test_batch_file(self):
        assert SourceType.BATCH_FILE == "batch_file"

    def test_batch_table(self):
        assert SourceType.BATCH_TABLE == "batch_table"

    def test_stream(self):
        assert SourceType.STREAM == "stream"


class TestFileFormat:
    def test_all_formats(self):
        assert FileFormat.CSV == "csv"
        assert FileFormat.PARQUET == "parquet"
        assert FileFormat.DELTA == "delta"
        assert FileFormat.JSON == "json"
        assert FileFormat.ORC == "orc"
        assert FileFormat.AVRO == "avro"


class TestGrain:
    def test_all_grains(self):
        assert Grain.CUSTOMER == "customer"
        assert Grain.TRANSACTION == "transaction"
        assert Grain.EVENT == "event"


class TestDataSourceConfig:
    def test_create_minimal_batch_file_config(self):
        config = DataSourceConfig(
            name="test_source",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/data/test.csv",
            file_format=FileFormat.CSV
        )
        assert config.name == "test_source"
        assert config.source_type == SourceType.BATCH_FILE
        assert config.primary_key == "id"

    def test_batch_file_requires_path(self):
        with pytest.raises(ValidationError, match="path required"):
            DataSourceConfig(
                name="test_source",
                source_type=SourceType.BATCH_FILE,
                primary_key="id",
                file_format=FileFormat.CSV
            )

    def test_batch_file_requires_file_format(self):
        with pytest.raises(ValidationError, match="file_format required"):
            DataSourceConfig(
                name="test_source",
                source_type=SourceType.BATCH_FILE,
                primary_key="id",
                path="/data/test.csv"
            )

    def test_batch_table_requires_table(self):
        with pytest.raises(ValidationError, match="table required"):
            DataSourceConfig(
                name="test_source",
                source_type=SourceType.BATCH_TABLE,
                primary_key="id"
            )

    def test_batch_table_minimal(self):
        config = DataSourceConfig(
            name="test_source",
            source_type=SourceType.BATCH_TABLE,
            primary_key="id",
            table="customers"
        )
        assert config.table == "customers"

    def test_csv_defaults(self):
        config = DataSourceConfig(
            name="test_source",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/data/test.csv",
            file_format=FileFormat.CSV
        )
        assert config.delimiter == ","
        assert config.header is True
        assert config.quote_char == '"'
        assert config.encoding == "utf-8"

    def test_csv_custom_settings(self):
        config = DataSourceConfig(
            name="test_source",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/data/test.csv",
            file_format=FileFormat.CSV,
            delimiter="|",
            header=False,
            quote_char="'",
            encoding="latin1"
        )
        assert config.delimiter == "|"
        assert config.header is False
        assert config.quote_char == "'"
        assert config.encoding == "latin1"

    def test_get_full_table_name_simple(self):
        config = DataSourceConfig(
            name="test_source",
            source_type=SourceType.BATCH_TABLE,
            primary_key="id",
            table="customers"
        )
        assert config.get_full_table_name() == "customers"

    def test_get_full_table_name_with_schema(self):
        config = DataSourceConfig(
            name="test_source",
            source_type=SourceType.BATCH_TABLE,
            primary_key="id",
            schema_name="sales",
            table="customers"
        )
        assert config.get_full_table_name() == "sales.customers"

    def test_get_full_table_name_with_catalog(self):
        config = DataSourceConfig(
            name="test_source",
            source_type=SourceType.BATCH_TABLE,
            primary_key="id",
            catalog="main",
            schema_name="sales",
            table="customers"
        )
        assert config.get_full_table_name() == "main.sales.customers"

    def test_get_full_table_name_raises_for_batch_file(self):
        config = DataSourceConfig(
            name="test_source",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/data/test.csv",
            file_format=FileFormat.CSV
        )
        with pytest.raises(ValueError, match="only applicable for batch_table"):
            config.get_full_table_name()

    def test_columns_list(self):
        columns = [
            ColumnConfig(name="id", column_type=ColumnType.IDENTIFIER),
            ColumnConfig(name="age", column_type=ColumnType.NUMERIC_CONTINUOUS)
        ]
        config = DataSourceConfig(
            name="test_source",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/data/test.csv",
            file_format=FileFormat.CSV,
            columns=columns
        )
        assert len(config.columns) == 2
        assert config.columns[0].name == "id"

    def test_get_column_by_name_exists(self):
        columns = [
            ColumnConfig(name="id", column_type=ColumnType.IDENTIFIER),
            ColumnConfig(name="age", column_type=ColumnType.NUMERIC_CONTINUOUS)
        ]
        config = DataSourceConfig(
            name="test_source",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/data/test.csv",
            file_format=FileFormat.CSV,
            columns=columns
        )
        col = config.get_column_by_name("age")
        assert col is not None
        assert col.name == "age"

    def test_get_column_by_name_not_exists(self):
        columns = [
            ColumnConfig(name="id", column_type=ColumnType.IDENTIFIER)
        ]
        config = DataSourceConfig(
            name="test_source",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/data/test.csv",
            file_format=FileFormat.CSV,
            columns=columns
        )
        col = config.get_column_by_name("nonexistent")
        assert col is None

    def test_get_feature_columns(self):
        columns = [
            ColumnConfig(name="id", column_type=ColumnType.IDENTIFIER),
            ColumnConfig(name="retained", column_type=ColumnType.TARGET),
            ColumnConfig(name="age", column_type=ColumnType.NUMERIC_CONTINUOUS),
            ColumnConfig(name="city", column_type=ColumnType.CATEGORICAL_NOMINAL)
        ]
        config = DataSourceConfig(
            name="test_source",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/data/test.csv",
            file_format=FileFormat.CSV,
            columns=columns
        )
        features = config.get_feature_columns()
        assert len(features) == 2
        assert features[0].name == "age"
        assert features[1].name == "city"

    def test_is_cloud_path_s3(self):
        config = DataSourceConfig(
            name="test_source",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="s3://bucket/data.csv",
            file_format=FileFormat.CSV
        )
        assert config.is_cloud_path() is True

    def test_is_cloud_path_abfss(self):
        config = DataSourceConfig(
            name="test_source",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="abfss://container@account.dfs.core.windows.net/data.csv",
            file_format=FileFormat.CSV
        )
        assert config.is_cloud_path() is True

    def test_is_cloud_path_gs(self):
        config = DataSourceConfig(
            name="test_source",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="gs://bucket/data.csv",
            file_format=FileFormat.CSV
        )
        assert config.is_cloud_path() is True

    def test_is_cloud_path_local(self):
        config = DataSourceConfig(
            name="test_source",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/local/data.csv",
            file_format=FileFormat.CSV
        )
        assert config.is_cloud_path() is False

    def test_is_cloud_path_no_path(self):
        config = DataSourceConfig(
            name="test_source",
            source_type=SourceType.BATCH_TABLE,
            primary_key="id",
            table="customers"
        )
        assert config.is_cloud_path() is False

    def test_quality_expectations(self):
        config = DataSourceConfig(
            name="test_source",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/data/test.csv",
            file_format=FileFormat.CSV,
            expected_row_count_min=1000,
            expected_row_count_max=10000,
            expected_columns=["id", "age"],
            freshness_sla_hours=24
        )
        assert config.expected_row_count_min == 1000
        assert config.expected_row_count_max == 10000
        assert config.expected_columns == ["id", "age"]
        assert config.freshness_sla_hours == 24

    def test_timestamp_and_customer_key(self):
        config = DataSourceConfig(
            name="test_source",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/data/test.csv",
            file_format=FileFormat.CSV,
            timestamp_column="created_at",
            customer_key="customer_id",
            grain=Grain.TRANSACTION
        )
        assert config.timestamp_column == "created_at"
        assert config.customer_key == "customer_id"
        assert config.grain == Grain.TRANSACTION

    def test_json_serialization(self):
        config = DataSourceConfig(
            name="test_source",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/data/test.csv",
            file_format=FileFormat.CSV
        )
        json_data = config.model_dump()
        assert json_data["name"] == "test_source"
        assert json_data["source_type"] == "batch_file"

    def test_json_deserialization(self):
        data = {
            "name": "test_source",
            "source_type": "batch_file",
            "primary_key": "id",
            "path": "/data/test.csv",
            "file_format": "csv"
        }
        config = DataSourceConfig(**data)
        assert config.name == "test_source"
