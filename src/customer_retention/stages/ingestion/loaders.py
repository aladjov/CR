from abc import ABC, abstractmethod
from typing import Optional, Any
import time

from customer_retention.core.compat import pd, DataFrame
from customer_retention.core.config.source_config import DataSourceConfig, FileFormat, SourceType
from .load_result import LoadResult


class DataLoader(ABC):
    @abstractmethod
    def load(self, config: DataSourceConfig, spark_session: Optional[Any] = None,
             sample_size: Optional[int] = None) -> tuple[DataFrame, LoadResult]:
        pass

    @abstractmethod
    def validate_source(self, config: DataSourceConfig) -> list[str]:
        pass

    def create_load_result(self, config: DataSourceConfig, df: DataFrame,
                          duration: float, success: bool = True) -> LoadResult:
        return LoadResult(
            success=success,
            row_count=len(df),
            column_count=len(df.columns),
            duration_seconds=duration,
            source_name=config.name,
            schema_info={col: str(dtype) for col, dtype in df.dtypes.items()}
        )

    def apply_sample(self, df: DataFrame, sample_size: Optional[int]) -> DataFrame:
        return df.head(sample_size) if sample_size else df


class CSVLoader(DataLoader):
    def validate_source(self, config: DataSourceConfig) -> list[str]:
        errors = []
        if not config.path:
            errors.append("CSV source requires path")
        if config.file_format != FileFormat.CSV:
            errors.append(f"CSVLoader requires CSV format, got {config.file_format}")
        return errors

    def load(self, config: DataSourceConfig, spark_session: Optional[Any] = None,
             sample_size: Optional[int] = None) -> tuple[DataFrame, LoadResult]:
        start_time = time.time()
        result = LoadResult(success=False, row_count=0, column_count=0,
                           duration_seconds=0, source_name=config.name)

        try:
            validation_errors = self.validate_source(config)
            if validation_errors:
                result.errors.extend(validation_errors)
                result.duration_seconds = time.time() - start_time
                return pd.DataFrame(), result

            read_kwargs = self.build_read_kwargs(config, sample_size)
            df = pd.read_csv(config.path, **read_kwargs)

            if sample_size and len(df) > sample_size:
                df = df.head(sample_size)

            duration = time.time() - start_time
            result = self.create_load_result(config, df, duration)
            return df, result

        except Exception as e:
            result.add_error(f"Failed to load CSV: {str(e)}")
            result.duration_seconds = time.time() - start_time
            return pd.DataFrame(), result

    def build_read_kwargs(self, config: DataSourceConfig, sample_size: Optional[int]) -> dict:
        kwargs = {
            "delimiter": config.delimiter,
            "header": 0 if config.header else None,
            "quotechar": config.quote_char,
            "encoding": config.encoding
        }
        if sample_size:
            kwargs["nrows"] = sample_size
        return kwargs


class ParquetLoader(DataLoader):
    def validate_source(self, config: DataSourceConfig) -> list[str]:
        errors = []
        if not config.path:
            errors.append("Parquet source requires path")
        if config.file_format != FileFormat.PARQUET:
            errors.append(f"ParquetLoader requires PARQUET format, got {config.file_format}")
        return errors

    def load(self, config: DataSourceConfig, spark_session: Optional[Any] = None,
             sample_size: Optional[int] = None) -> tuple[DataFrame, LoadResult]:
        start_time = time.time()
        result = LoadResult(success=False, row_count=0, column_count=0,
                           duration_seconds=0, source_name=config.name)

        try:
            validation_errors = self.validate_source(config)
            if validation_errors:
                result.errors.extend(validation_errors)
                result.duration_seconds = time.time() - start_time
                return pd.DataFrame(), result

            df = pd.read_parquet(config.path)
            df = self.apply_sample(df, sample_size)

            duration = time.time() - start_time
            result = self.create_load_result(config, df, duration)
            return df, result

        except Exception as e:
            result.add_error(f"Failed to load Parquet: {str(e)}")
            result.duration_seconds = time.time() - start_time
            return pd.DataFrame(), result


class DeltaLoader(DataLoader):
    def validate_source(self, config: DataSourceConfig) -> list[str]:
        errors = []
        if config.source_type == SourceType.BATCH_FILE and not config.path:
            errors.append("Delta file source requires path")
        if config.source_type == SourceType.BATCH_TABLE and not config.table:
            errors.append("Delta table source requires table name")
        if config.file_format != FileFormat.DELTA:
            errors.append(f"DeltaLoader requires DELTA format, got {config.file_format}")
        return errors

    def load(self, config: DataSourceConfig, spark_session: Optional[Any] = None,
             sample_size: Optional[int] = None) -> tuple[DataFrame, LoadResult]:
        start_time = time.time()
        result = LoadResult(success=False, row_count=0, column_count=0,
                           duration_seconds=0, source_name=config.name)

        try:
            validation_errors = self.validate_source(config)
            if validation_errors:
                result.errors.extend(validation_errors)
                result.duration_seconds = time.time() - start_time
                return pd.DataFrame(), result

            if not spark_session:
                result.add_error("Delta loader requires Spark session")
                result.duration_seconds = time.time() - start_time
                return pd.DataFrame(), result

            spark_df = self.load_spark_dataframe(config, spark_session, sample_size)
            df = spark_df.toPandas()

            duration = time.time() - start_time
            result = self.create_load_result(config, df, duration)
            return df, result

        except Exception as e:
            result.add_error(f"Failed to load Delta: {str(e)}")
            result.duration_seconds = time.time() - start_time
            return pd.DataFrame(), result

    def load_spark_dataframe(self, config: DataSourceConfig, spark_session: Any,
                            sample_size: Optional[int]) -> Any:
        if config.source_type == SourceType.BATCH_FILE:
            spark_df = spark_session.read.format("delta").load(config.path)
        else:
            table_name = config.get_full_table_name()
            spark_df = spark_session.read.table(table_name)

        if sample_size:
            spark_df = spark_df.limit(sample_size)

        return spark_df


class LoaderFactory:
    _loaders = {
        FileFormat.CSV: CSVLoader,
        FileFormat.PARQUET: ParquetLoader,
        FileFormat.DELTA: DeltaLoader
    }

    @classmethod
    def get_loader(cls, config: DataSourceConfig) -> DataLoader:
        if not config.file_format:
            raise ValueError(f"file_format required to determine loader for {config.name}")

        loader_class = cls._loaders.get(config.file_format)
        if not loader_class:
            raise ValueError(f"No loader available for format: {config.file_format}")

        return loader_class()

    @classmethod
    def register_loader(cls, file_format: FileFormat, loader_class: type[DataLoader]) -> None:
        cls._loaders[file_format] = loader_class
