from enum import Enum
from typing import Optional
from pydantic import BaseModel, model_validator
from .column_config import ColumnConfig


class SourceType(str, Enum):
    BATCH_FILE = "batch_file"
    BATCH_TABLE = "batch_table"
    STREAM = "stream"


class FileFormat(str, Enum):
    CSV = "csv"
    PARQUET = "parquet"
    DELTA = "delta"
    JSON = "json"
    ORC = "orc"
    AVRO = "avro"


class Grain(str, Enum):
    CUSTOMER = "customer"
    TRANSACTION = "transaction"
    EVENT = "event"


class DataSourceConfig(BaseModel):
    name: str
    source_type: SourceType
    primary_key: str

    path: Optional[str] = None
    file_format: Optional[FileFormat] = None

    catalog: Optional[str] = None
    schema_name: Optional[str] = None
    table: Optional[str] = None

    delimiter: str = ","
    header: bool = True
    quote_char: str = '"'
    encoding: str = "utf-8"

    columns: list[ColumnConfig] = []
    timestamp_column: Optional[str] = None
    customer_key: Optional[str] = None
    grain: Optional[Grain] = None

    expected_row_count_min: Optional[int] = None
    expected_row_count_max: Optional[int] = None
    expected_columns: Optional[list[str]] = None
    freshness_sla_hours: Optional[int] = None

    @model_validator(mode='after')
    def validate_source_requirements(self):
        if self.source_type == SourceType.BATCH_FILE:
            if not self.path:
                raise ValueError("path required for batch_file source_type")
            if not self.file_format:
                raise ValueError("file_format required for batch_file source_type")
        if self.source_type == SourceType.BATCH_TABLE and not self.table:
            raise ValueError("table required for batch_table source_type")
        return self

    def get_full_table_name(self) -> str:
        if self.source_type != SourceType.BATCH_TABLE:
            raise ValueError("full_table_name only applicable for batch_table")
        parts = [p for p in [self.catalog, self.schema_name, self.table] if p]
        return ".".join(parts)

    def get_column_by_name(self, name: str) -> Optional[ColumnConfig]:
        return next((c for c in self.columns if c.name == name), None)

    def get_feature_columns(self) -> list[ColumnConfig]:
        return [c for c in self.columns if c.should_be_used_as_feature()]

    def is_cloud_path(self) -> bool:
        if not self.path:
            return False
        return any(self.path.startswith(prefix) for prefix in ["s3://", "abfss://", "gs://", "wasb://", "adl://"])
