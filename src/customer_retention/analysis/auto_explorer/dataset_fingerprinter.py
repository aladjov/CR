from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from customer_retention.core.compat import is_dataframe, native_pd, pd, safe_to_datetime
from customer_retention.core.config.column_config import DatasetGranularity
from customer_retention.stages.profiling.type_detector import TypeDetector

_FILE_EXTENSIONS = frozenset((".csv", ".parquet", ".json", ".orc", ".avro", ".delta", ".txt"))


def is_table_name(source: str) -> bool:
    if "/" in source or "\\" in source:
        return False
    if "." not in source:
        return False
    return Path(source).suffix.lower() not in _FILE_EXTENSIONS


@dataclass
class DatasetFingerprint:
    name: str
    row_count: int
    column_count: int
    entity_columns: list[str]
    time_columns: list[str]
    target_candidates: list[str]
    granularity: DatasetGranularity
    entity_column: Optional[str] = None
    time_column: Optional[str] = None
    unique_entities: Optional[int] = None
    avg_rows_per_entity: Optional[float] = None
    temporal_span_days: Optional[int] = None
    data_start: Optional[str] = None
    data_end: Optional[str] = None
    sampled: bool = False


class DatasetFingerprinter:
    def __init__(self, nrows: int = 10000):
        self.nrows = nrows
        self._type_detector = TypeDetector()

    def fingerprint(
        self, name: str, data: pd.DataFrame | str | Path,
        known_entity_column: Optional[str] = None, known_time_column: Optional[str] = None,
        known_granularity: Optional[DatasetGranularity] = None,
    ) -> DatasetFingerprint:
        full_row_count = self._count_rows(data)
        df = self._load(data)
        sampled = len(df) < full_row_count

        self._validate_known_column(name, "entity_column", known_entity_column, df.columns)
        self._validate_known_column(name, "time_column", known_time_column, df.columns)

        all_known = known_entity_column and known_time_column and known_granularity
        if all_known:
            entity_columns = [known_entity_column]
            time_columns = [known_time_column]
            target_candidates: list[str] = []
            granularity = known_granularity
            best_entity = known_entity_column
            best_time = known_time_column
        else:
            entity_columns, time_columns, target_candidates = (
                self._type_detector.classify_structural_columns(df, total_count=full_row_count)
            )
            granularity_result = self._type_detector.detect_granularity(df)
            granularity = known_granularity or granularity_result.granularity
            best_entity = known_entity_column or granularity_result.entity_column or (entity_columns[0] if entity_columns else None)
            best_time = known_time_column or granularity_result.time_column or (time_columns[0] if time_columns else None)

        unique_entities = None
        avg_rows = None
        temporal_span = None
        data_start = None
        data_end = None

        if best_entity and best_entity in df.columns:
            unique_entities = df[best_entity].nunique()
            if unique_entities > 0:
                row_base = full_row_count if not sampled else len(df)
                avg_rows = round(row_base / unique_entities, 2)

        if best_time and best_time in df.columns:
            try:
                ts = safe_to_datetime(df[best_time], errors="coerce")
                ts_min = ts.min()
                ts_max = ts.max()
                span = ts_max - ts_min
                temporal_span = int(span.days) if not native_pd.isna(span) else None
                if not native_pd.isna(ts_min) and not native_pd.isna(ts_max):
                    data_start = str(ts_min.date())
                    data_end = str(ts_max.date())
            except Exception:
                pass

        return DatasetFingerprint(
            name=name,
            row_count=full_row_count,
            column_count=len(df.columns),
            entity_columns=entity_columns,
            time_columns=time_columns,
            target_candidates=target_candidates,
            granularity=granularity,
            entity_column=best_entity,
            time_column=best_time,
            unique_entities=unique_entities,
            avg_rows_per_entity=avg_rows,
            temporal_span_days=temporal_span,
            data_start=data_start,
            data_end=data_end,
            sampled=sampled,
        )

    @staticmethod
    def _validate_known_column(dataset_name: str, kind: str, column: Optional[str], available) -> None:
        if column is None or column in available:
            return
        raise ValueError(
            f"Configured {kind} '{column}' for dataset '{dataset_name}' is not present "
            f"in the loaded data. Available columns: {list(available)}. "
            f"This typically means the column was dropped or renamed by an upstream transform "
            f"(e.g., lifecycle enrichment replaces the raw start-date column with 'event_timestamp', "
            f"and DROP_COLUMNS in NB01 removes any column listed in it). "
            f"Update the semantics_overrides cell in NB00 to point at a column that survives "
            f"the transforms applied before landing."
        )

    def fingerprint_all(self, datasets: dict[str, pd.DataFrame | str | Path]) -> dict[str, DatasetFingerprint]:
        results: dict[str, DatasetFingerprint] = {}
        for name, data in datasets.items():
            try:
                results[name] = self.fingerprint(name, data)
            except Exception as exc:
                warnings.warn(f"Skipping dataset '{name}': {exc}", stacklevel=2)
        return results

    @staticmethod
    def to_summary_dataframe(fingerprints: dict[str, DatasetFingerprint]) -> native_pd.DataFrame:
        rows = []
        for fp in fingerprints.values():
            row = {
                "name": fp.name,
                "rows": fp.row_count,
                "columns": fp.column_count,
                "granularity": fp.granularity.value if fp.granularity else None,
                "entity_column": fp.entity_column,
                "time_column": fp.time_column,
                "target_candidates": ", ".join(fp.target_candidates) if fp.target_candidates else "",
                "sampled": fp.sampled,
            }
            rows.append(row)
        return native_pd.DataFrame(rows)

    def _count_rows(self, data) -> int:
        if is_dataframe(data):
            return len(data)
        path_str = str(data)
        if is_table_name(path_str):
            return self._count_rows_table(path_str)
        from customer_retention.core.compat.remote_path import RemotePath
        if isinstance(data, RemotePath):
            return self._count_rows_remote(data, path_str)
        if Path(path_str).exists():
            return self._count_rows_local(path_str)
        if self._is_remote():
            return self._count_rows_remote(data, path_str)
        return self._count_rows_local(path_str)

    @staticmethod
    def _count_rows_local(path_str: str) -> int:
        if path_str.endswith(".csv"):
            with Path(path_str).open("r") as f:
                return sum(1 for _ in f) - 1
        return len(native_pd.read_parquet(path_str, columns=[]))

    def _count_rows_remote(self, path, path_str: str) -> int:
        spark = self._ensure_spark()
        if spark and path_str.endswith(".csv"):
            return spark.read.option("header", "true").csv(path_str).count()
        if spark:
            return spark.read.parquet(path_str).count()
        return 0

    def _count_rows_table(self, table_name: str) -> int:
        spark = self._ensure_spark()
        if not spark:
            raise RuntimeError(f"No active Spark session to read table '{table_name}'")
        return spark.table(table_name).count()

    def _load(self, data) -> pd.DataFrame:
        if is_dataframe(data):
            return data.head(self.nrows) if len(data) > self.nrows else data
        path_str = str(data)
        if is_table_name(path_str):
            return self._load_table(path_str)
        from customer_retention.core.compat.remote_path import RemotePath
        if isinstance(data, RemotePath):
            return self._load_remote(path_str)
        if Path(path_str).exists():
            return self._load_local(path_str)
        if self._is_remote():
            return self._load_remote(path_str)
        return self._load_local(path_str)

    def _load_local(self, path_str: str) -> pd.DataFrame:
        if path_str.endswith(".csv"):
            return native_pd.read_csv(path_str, nrows=self.nrows)
        return native_pd.read_parquet(path_str).head(self.nrows)

    def _load_table(self, table_name: str) -> pd.DataFrame:
        spark = self._ensure_spark()
        if not spark:
            raise RuntimeError(f"No active Spark session to read table '{table_name}'")
        return spark.table(table_name).limit(self.nrows).toPandas()

    @staticmethod
    def _is_remote() -> bool:
        from customer_retention.core.compat.detection import is_remote_spark
        return is_remote_spark() or bool(os.environ.get("CR_SPARK_REMOTE"))

    @staticmethod
    def _ensure_spark():
        from customer_retention.core.compat.detection import get_spark_session
        spark = get_spark_session()
        if not spark and os.environ.get("CR_SPARK_REMOTE"):
            from customer_retention.core.compat.detection import connect_remote_spark
            spark = connect_remote_spark()
        return spark

    def _load_remote(self, path_str: str) -> pd.DataFrame:
        spark = self._ensure_spark()
        if not spark:
            raise RuntimeError("No active Spark session for remote data loading")
        if path_str.endswith(".csv"):
            sdf = spark.read.option("header", "true").option("inferSchema", "true").csv(path_str)
        else:
            sdf = spark.read.parquet(path_str)
        return sdf.limit(self.nrows).toPandas()
