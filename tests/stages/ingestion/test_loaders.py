from unittest.mock import MagicMock

import pandas as pd
import pytest

from customer_retention.core.config import DataSourceConfig, FileFormat, SourceType
from customer_retention.stages.ingestion import CSVLoader, DeltaLoader, LoaderFactory, ParquetLoader


class TestCSVLoader:
    def test_validate_source_valid(self):
        config = DataSourceConfig(
            name="test",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/data/test.csv",
            file_format=FileFormat.CSV
        )
        loader = CSVLoader()
        errors = loader.validate_source(config)
        assert len(errors) == 0

    def test_validate_source_missing_path(self):
        config = DataSourceConfig(
            name="test",
            source_type=SourceType.BATCH_TABLE,
            primary_key="id",
            table="test",
            file_format=FileFormat.CSV
        )
        loader = CSVLoader()
        errors = loader.validate_source(config)
        assert len(errors) > 0
        assert any("path" in err.lower() for err in errors)

    def test_validate_source_wrong_format(self):
        config = DataSourceConfig(
            name="test",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/data/test.parquet",
            file_format=FileFormat.PARQUET
        )
        loader = CSVLoader()
        errors = loader.validate_source(config)
        assert len(errors) > 0

    def test_load_csv_success(self, temp_csv_file):
        config = DataSourceConfig(
            name="test",
            source_type=SourceType.BATCH_FILE,
            primary_key="custid",
            path=temp_csv_file,
            file_format=FileFormat.CSV
        )
        loader = CSVLoader()
        df, result = loader.load(config)

        assert result.success is True
        assert result.row_count == 5
        assert result.column_count == 15
        assert len(df) == 5
        assert "custid" in df.columns

    def test_load_csv_with_sample(self, temp_csv_file):
        config = DataSourceConfig(
            name="test",
            source_type=SourceType.BATCH_FILE,
            primary_key="custid",
            path=temp_csv_file,
            file_format=FileFormat.CSV
        )
        loader = CSVLoader()
        df, result = loader.load(config, sample_size=3)

        assert result.success is True
        assert len(df) == 3
        assert result.row_count == 3

    def test_load_csv_invalid_path(self):
        config = DataSourceConfig(
            name="test",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/nonexistent/path.csv",
            file_format=FileFormat.CSV
        )
        loader = CSVLoader()
        df, result = loader.load(config)

        assert result.success is False
        assert result.has_errors() is True
        assert len(df) == 0

    def test_build_read_kwargs_defaults(self):
        config = DataSourceConfig(
            name="test",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/data/test.csv",
            file_format=FileFormat.CSV
        )
        loader = CSVLoader()
        kwargs = loader.build_read_kwargs(config, None)

        assert kwargs["delimiter"] == ","
        assert kwargs["header"] == 0
        assert kwargs["quotechar"] == '"'
        assert kwargs["encoding"] == "utf-8"
        assert "nrows" not in kwargs

    def test_build_read_kwargs_with_sample(self):
        config = DataSourceConfig(
            name="test",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/data/test.csv",
            file_format=FileFormat.CSV
        )
        loader = CSVLoader()
        kwargs = loader.build_read_kwargs(config, 100)

        assert kwargs["nrows"] == 100

    def test_build_read_kwargs_custom_delimiter(self):
        config = DataSourceConfig(
            name="test",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/data/test.csv",
            file_format=FileFormat.CSV,
            delimiter="|"
        )
        loader = CSVLoader()
        kwargs = loader.build_read_kwargs(config, None)

        assert kwargs["delimiter"] == "|"


class TestParquetLoader:
    def test_validate_source_valid(self):
        config = DataSourceConfig(
            name="test",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/data/test.parquet",
            file_format=FileFormat.PARQUET
        )
        loader = ParquetLoader()
        errors = loader.validate_source(config)
        assert len(errors) == 0

    def test_validate_source_missing_path(self):
        config = DataSourceConfig(
            name="test",
            source_type=SourceType.BATCH_TABLE,
            primary_key="id",
            table="test",
            file_format=FileFormat.PARQUET
        )
        loader = ParquetLoader()
        errors = loader.validate_source(config)
        assert len(errors) > 0

    def test_load_parquet_success(self, temp_parquet_file):
        config = DataSourceConfig(
            name="test",
            source_type=SourceType.BATCH_FILE,
            primary_key="custid",
            path=temp_parquet_file,
            file_format=FileFormat.PARQUET
        )
        loader = ParquetLoader()
        df, result = loader.load(config)

        assert result.success is True
        assert result.row_count == 5
        assert len(df) == 5
        assert "custid" in df.columns

    def test_load_parquet_with_sample(self, temp_parquet_file):
        config = DataSourceConfig(
            name="test",
            source_type=SourceType.BATCH_FILE,
            primary_key="custid",
            path=temp_parquet_file,
            file_format=FileFormat.PARQUET
        )
        loader = ParquetLoader()
        df, result = loader.load(config, sample_size=2)

        assert result.success is True
        assert len(df) == 2
        assert result.row_count == 2

    def test_load_parquet_invalid_path(self):
        config = DataSourceConfig(
            name="test",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/nonexistent/path.parquet",
            file_format=FileFormat.PARQUET
        )
        loader = ParquetLoader()
        df, result = loader.load(config)

        assert result.success is False
        assert result.has_errors() is True


class TestDeltaLoader:
    def test_validate_source_valid_file(self):
        config = DataSourceConfig(
            name="test",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/data/test",
            file_format=FileFormat.DELTA
        )
        loader = DeltaLoader()
        errors = loader.validate_source(config)
        assert len(errors) == 0

    def test_validate_source_valid_table(self):
        config = DataSourceConfig(
            name="test",
            source_type=SourceType.BATCH_TABLE,
            primary_key="id",
            table="test_table",
            file_format=FileFormat.DELTA
        )
        loader = DeltaLoader()
        errors = loader.validate_source(config)
        assert len(errors) == 0

    def test_validate_source_missing_path(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="path required"):
            DataSourceConfig(
                name="test",
                source_type=SourceType.BATCH_FILE,
                primary_key="id",
                file_format=FileFormat.DELTA
            )

    def test_validate_source_missing_table(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="table required"):
            DataSourceConfig(
                name="test",
                source_type=SourceType.BATCH_TABLE,
                primary_key="id",
                file_format=FileFormat.DELTA
            )

    def test_load_without_spark_session(self):
        config = DataSourceConfig(
            name="test",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/data/test",
            file_format=FileFormat.DELTA
        )
        loader = DeltaLoader()
        df, result = loader.load(config)

        assert result.success is False
        assert result.has_errors() is True
        assert any("Spark" in err for err in result.errors)


class TestLoaderFactory:
    def test_get_loader_csv(self):
        config = DataSourceConfig(
            name="test",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/data/test.csv",
            file_format=FileFormat.CSV
        )
        loader = LoaderFactory.get_loader(config)
        assert isinstance(loader, CSVLoader)

    def test_get_loader_parquet(self):
        config = DataSourceConfig(
            name="test",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/data/test.parquet",
            file_format=FileFormat.PARQUET
        )
        loader = LoaderFactory.get_loader(config)
        assert isinstance(loader, ParquetLoader)

    def test_get_loader_delta(self):
        config = DataSourceConfig(
            name="test",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/data/test",
            file_format=FileFormat.DELTA
        )
        loader = LoaderFactory.get_loader(config)
        assert isinstance(loader, DeltaLoader)

    def test_get_loader_no_format(self):
        config = DataSourceConfig(
            name="test",
            source_type=SourceType.BATCH_TABLE,
            primary_key="id",
            table="test"
        )
        with pytest.raises(ValueError, match="file_format required"):
            LoaderFactory.get_loader(config)

    def test_get_loader_unsupported_format(self):
        config = DataSourceConfig(
            name="test",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/data/test.json",
            file_format=FileFormat.JSON
        )
        with pytest.raises(ValueError, match="No loader available"):
            LoaderFactory.get_loader(config)

    def test_register_custom_loader(self):
        class CustomLoader(CSVLoader):
            pass

        LoaderFactory.register_loader(FileFormat.JSON, CustomLoader)

        config = DataSourceConfig(
            name="test",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/data/test.json",
            file_format=FileFormat.JSON
        )
        loader = LoaderFactory.get_loader(config)
        assert isinstance(loader, CustomLoader)


class TestDataLoaderBase:
    def test_create_load_result(self, sample_dataframe):
        config = DataSourceConfig(
            name="test",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/data/test.csv",
            file_format=FileFormat.CSV
        )
        loader = CSVLoader()
        result = loader.create_load_result(config, sample_dataframe, 1.5)

        assert result.success is True
        assert result.row_count == 5
        assert result.column_count == 15
        assert result.duration_seconds == 1.5
        assert result.source_name == "test"
        assert len(result.schema_info) == 15

    def test_apply_sample_with_limit(self, sample_dataframe):
        loader = CSVLoader()
        sampled_df = loader.apply_sample(sample_dataframe, 3)
        assert len(sampled_df) == 3

    def test_apply_sample_without_limit(self, sample_dataframe):
        loader = CSVLoader()
        sampled_df = loader.apply_sample(sample_dataframe, None)
        assert len(sampled_df) == 5


class TestCSVLoaderValidationErrors:
    def test_load_returns_early_on_validation_errors(self):
        config = DataSourceConfig(
            name="test",
            source_type=SourceType.BATCH_TABLE,
            primary_key="id",
            table="test",
            file_format=FileFormat.CSV
        )
        loader = CSVLoader()
        df, result = loader.load(config)

        assert result.success is False
        assert len(df) == 0
        assert any("path" in err.lower() for err in result.errors)
        assert result.duration_seconds >= 0

    def test_load_csv_sample_truncation(self, temp_csv_file):
        """Test the branch where len(df) > sample_size after read_csv with nrows."""
        config = DataSourceConfig(
            name="test",
            source_type=SourceType.BATCH_FILE,
            primary_key="custid",
            path=temp_csv_file,
            file_format=FileFormat.CSV
        )
        loader = CSVLoader()
        # Mock build_read_kwargs to not set nrows, so len(df) > sample_size
        original_build = loader.build_read_kwargs
        def patched_build(c, s):
            kwargs = original_build(c, None)  # Don't pass sample_size to avoid nrows
            return kwargs
        loader.build_read_kwargs = patched_build
        df, result = loader.load(config, sample_size=2)
        assert result.success is True
        assert len(df) == 2


class TestParquetLoaderEdgeCases:
    def test_validate_source_wrong_format(self):
        config = DataSourceConfig(
            name="test",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/data/test.csv",
            file_format=FileFormat.CSV
        )
        loader = ParquetLoader()
        errors = loader.validate_source(config)
        assert len(errors) > 0
        assert any("PARQUET" in err for err in errors)

    def test_load_returns_early_on_validation_errors(self):
        config = DataSourceConfig(
            name="test",
            source_type=SourceType.BATCH_TABLE,
            primary_key="id",
            table="test",
            file_format=FileFormat.PARQUET
        )
        loader = ParquetLoader()
        df, result = loader.load(config)

        assert result.success is False
        assert len(df) == 0
        assert any("path" in err.lower() for err in result.errors)
        assert result.duration_seconds >= 0


class TestDeltaLoaderEdgeCases:
    def test_validate_source_missing_table_for_batch_table(self):
        config = DataSourceConfig(
            name="test",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/data/test",
            file_format=FileFormat.DELTA
        )
        # Manually override to test the table branch
        config.source_type = SourceType.BATCH_TABLE
        config.table = None
        loader = DeltaLoader()
        errors = loader.validate_source(config)
        assert any("table" in err.lower() for err in errors)

    def test_validate_source_wrong_format(self):
        config = DataSourceConfig(
            name="test",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/data/test",
            file_format=FileFormat.CSV
        )
        loader = DeltaLoader()
        errors = loader.validate_source(config)
        assert any("DELTA" in err for err in errors)

    def test_load_returns_early_on_validation_errors(self):
        config = DataSourceConfig(
            name="test",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/data/test",
            file_format=FileFormat.CSV
        )
        loader = DeltaLoader()
        df, result = loader.load(config)

        assert result.success is False
        assert len(df) == 0
        assert result.duration_seconds >= 0

    def test_load_with_spark_session_file_source(self):
        mock_spark = MagicMock()
        mock_spark_df = MagicMock()
        mock_spark.read.format.return_value.load.return_value = mock_spark_df
        mock_spark_df.toPandas.return_value = pd.DataFrame({"id": [1, 2], "val": [10, 20]})

        config = DataSourceConfig(
            name="test",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/data/delta_table",
            file_format=FileFormat.DELTA
        )
        loader = DeltaLoader()
        df, result = loader.load(config, spark_session=mock_spark)

        assert result.success is True
        assert result.row_count == 2
        assert len(df) == 2
        mock_spark.read.format.assert_called_once_with("delta")

    def test_load_with_spark_session_table_source(self):
        mock_spark = MagicMock()
        mock_spark_df = MagicMock()
        mock_spark.read.table.return_value = mock_spark_df
        mock_spark_df.toPandas.return_value = pd.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})

        config = DataSourceConfig(
            name="test",
            source_type=SourceType.BATCH_TABLE,
            primary_key="id",
            table="my_table",
            file_format=FileFormat.DELTA
        )
        loader = DeltaLoader()
        df, result = loader.load(config, spark_session=mock_spark)

        assert result.success is True
        assert result.row_count == 3
        mock_spark.read.table.assert_called_once()

    def test_load_with_spark_session_and_sample_size(self):
        mock_spark = MagicMock()
        mock_spark_df = MagicMock()
        mock_limited_df = MagicMock()
        mock_spark.read.format.return_value.load.return_value = mock_spark_df
        mock_spark_df.limit.return_value = mock_limited_df
        mock_limited_df.toPandas.return_value = pd.DataFrame({"id": [1], "val": [10]})

        config = DataSourceConfig(
            name="test",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/data/delta_table",
            file_format=FileFormat.DELTA
        )
        loader = DeltaLoader()
        df, result = loader.load(config, spark_session=mock_spark, sample_size=1)

        assert result.success is True
        assert result.row_count == 1
        mock_spark_df.limit.assert_called_once_with(1)

    def test_load_spark_exception(self):
        mock_spark = MagicMock()
        mock_spark.read.format.return_value.load.side_effect = Exception("Connection failed")

        config = DataSourceConfig(
            name="test",
            source_type=SourceType.BATCH_FILE,
            primary_key="id",
            path="/data/delta_table",
            file_format=FileFormat.DELTA
        )
        loader = DeltaLoader()
        df, result = loader.load(config, spark_session=mock_spark)

        assert result.success is False
        assert result.has_errors()
        assert any("Failed to load Delta" in err for err in result.errors)
