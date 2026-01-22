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
