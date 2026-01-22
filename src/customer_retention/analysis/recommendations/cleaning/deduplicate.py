from typing import Any, Dict, List, Optional
import pandas as pd

from ..base import CleaningRecommendation, RecommendationResult


class DeduplicateRecommendation(CleaningRecommendation):
    def __init__(
        self, key_columns: List[str], rationale: str = None, strategy: str = "keep_first",
        timestamp_column: Optional[str] = None, evidence: List[str] = None,
        priority: str = "medium", source_finding: Optional[Any] = None
    ):
        rationale = rationale or f"Remove duplicate rows using {strategy}"
        super().__init__(key_columns, rationale, evidence, priority, source_finding)
        self.key_columns = key_columns
        self.strategy = strategy
        self.timestamp_column = timestamp_column

    @property
    def recommendation_type(self) -> str:
        return f"deduplicate_{self.strategy}"

    def _fit_impl(self, df: pd.DataFrame) -> None:
        existing_keys = [k for k in self.key_columns if k in df.columns]
        if not existing_keys:
            self._fit_params["duplicate_count"] = 0
            self._fit_params["duplicate_keys"] = []
            return
        duplicated_mask = df.duplicated(subset=existing_keys, keep=False)
        duplicated_df = df[duplicated_mask]
        dup_count = len(duplicated_df) - duplicated_df.drop_duplicates(subset=existing_keys).shape[0]
        self._fit_params["duplicate_count"] = dup_count
        first_key = existing_keys[0]
        self._fit_params["duplicate_keys"] = duplicated_df[first_key].unique().tolist()

    def _transform_local(self, df: pd.DataFrame) -> RecommendationResult:
        df = df.copy()
        rows_before = len(df)
        existing_keys = [k for k in self.key_columns if k in df.columns]
        if not existing_keys:
            return RecommendationResult(
                data=df, columns_affected=self.key_columns, rows_before=rows_before,
                rows_after=rows_before, metadata={"duplicates_removed": 0}
            )
        if self.strategy == "keep_first":
            df = df.drop_duplicates(subset=existing_keys, keep="first")
        elif self.strategy == "keep_last":
            df = df.drop_duplicates(subset=existing_keys, keep="last")
        elif self.strategy == "keep_most_recent" and self.timestamp_column:
            df = df.sort_values(self.timestamp_column, ascending=False)
            df = df.drop_duplicates(subset=existing_keys, keep="first")
            df = df.sort_index()
        elif self.strategy == "drop_exact":
            df = df.drop_duplicates(subset=existing_keys, keep="first")
        rows_after = len(df)
        return RecommendationResult(
            data=df, columns_affected=self.key_columns, rows_before=rows_before,
            rows_after=rows_after, metadata={"duplicates_removed": rows_before - rows_after}
        )

    def _transform_databricks(self, df: pd.DataFrame) -> RecommendationResult:
        from customer_retention.core.compat import is_spark_available
        if not is_spark_available():
            return self._transform_local(df)
        return self._transform_local(df)

    def _generate_local_code(self) -> str:
        key_str = ", ".join(f"'{k}'" for k in self.key_columns)
        lines = [f"# Deduplicate: {self.rationale}"]
        if self.strategy == "keep_first":
            lines.append(f"df = df.drop_duplicates(subset=[{key_str}], keep='first')")
        elif self.strategy == "keep_last":
            lines.append(f"df = df.drop_duplicates(subset=[{key_str}], keep='last')")
        elif self.strategy == "keep_most_recent" and self.timestamp_column:
            lines.append(f"df = df.sort_values('{self.timestamp_column}', ascending=False)")
            lines.append(f"df = df.drop_duplicates(subset=[{key_str}], keep='first')")
            lines.append("df = df.sort_index()")
        elif self.strategy == "drop_exact":
            lines.append(f"df = df.drop_duplicates(subset=[{key_str}], keep='first')")
        return "\n".join(lines)

    def _generate_databricks_code(self) -> str:
        key_str = ", ".join(f"'{k}'" for k in self.key_columns)
        lines = [f"# Deduplicate: {self.rationale}"]
        if self.strategy == "keep_most_recent" and self.timestamp_column:
            lines.append("from pyspark.sql.window import Window")
            lines.append("from pyspark.sql.functions import row_number, desc")
            lines.append(f"window = Window.partitionBy([{key_str}]).orderBy(desc('{self.timestamp_column}'))")
            lines.append("df = df.withColumn('_row_num', row_number().over(window))")
            lines.append("df = df.filter(df._row_num == 1).drop('_row_num')")
        else:
            lines.append(f"df = df.dropDuplicates([{key_str}])")
        return "\n".join(lines)
