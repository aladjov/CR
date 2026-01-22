import re
from typing import Any, Dict, List, Optional
import pandas as pd

from ..base import CleaningRecommendation, RecommendationResult


class ConsistencyNormalizeRecommendation(CleaningRecommendation):
    def __init__(
        self, columns: List[str], rationale: str = None, normalization: str = "lowercase",
        evidence: List[str] = None, priority: str = "medium", source_finding: Optional[Any] = None
    ):
        rationale = rationale or f"Normalize values using {normalization}"
        super().__init__(columns, rationale, evidence, priority, source_finding)
        self.normalization = normalization
        self._unique_before: Dict[str, int] = {}

    @property
    def recommendation_type(self) -> str:
        return f"normalize_{self.normalization}"

    def _fit_impl(self, df: pd.DataFrame) -> None:
        variants = {}
        unique_before = {}
        for col in self.columns:
            if col not in df.columns:
                continue
            unique_before[col] = df[col].nunique()
            if df[col].dtype == object:
                variants[col] = df[col].dropna().unique().tolist()[:20]
        self._fit_params["variants"] = variants
        self._fit_params["unique_before"] = unique_before
        self._unique_before = unique_before

    def _normalize_series(self, series: pd.Series) -> pd.Series:
        if series.dtype != object:
            return series
        if self.normalization == "lowercase":
            return series.str.lower()
        if self.normalization == "uppercase":
            return series.str.upper()
        if self.normalization == "titlecase":
            return series.str.title()
        if self.normalization == "strip_whitespace":
            return series.str.strip()
        if self.normalization == "collapse_whitespace":
            return series.apply(lambda x: re.sub(r'\s+', ' ', x) if isinstance(x, str) else x)
        return series

    def _transform_local(self, df: pd.DataFrame) -> RecommendationResult:
        df = df.copy()
        rows_before = len(df)
        values_changed = {}
        unique_after = {}
        for col in self.columns:
            if col not in df.columns:
                continue
            original = df[col].copy()
            df[col] = self._normalize_series(df[col])
            changed = (original != df[col]) & original.notna()
            values_changed[col] = int(changed.sum())
            unique_after[col] = df[col].nunique()
        return RecommendationResult(
            data=df, columns_affected=self.columns, rows_before=rows_before,
            rows_after=len(df), metadata={
                "values_changed": values_changed, "unique_after": unique_after,
                "unique_before": self._unique_before
            }
        )

    def _transform_databricks(self, df: pd.DataFrame) -> RecommendationResult:
        from customer_retention.core.compat import is_spark_available
        if not is_spark_available():
            return self._transform_local(df)
        return self._transform_local(df)

    def _generate_local_code(self) -> str:
        lines = [f"# Normalize: {self.rationale}"]
        method_map = {
            "lowercase": "str.lower()",
            "uppercase": "str.upper()",
            "titlecase": "str.title()",
            "strip_whitespace": "str.strip()",
            "collapse_whitespace": "apply(lambda x: re.sub(r'\\s+', ' ', x) if isinstance(x, str) else x)",
        }
        method = method_map.get(self.normalization, "str.lower()")
        for col in self.columns:
            lines.append(f"df['{col}'] = df['{col}'].{method}")
        return "\n".join(lines)

    def _generate_databricks_code(self) -> str:
        func_map = {
            "lowercase": "lower",
            "uppercase": "upper",
            "strip_whitespace": "trim",
            "titlecase": "initcap",
            "collapse_whitespace": "regexp_replace",
        }
        func = func_map.get(self.normalization, "lower")
        lines = [f"# Normalize: {self.rationale}", f"from pyspark.sql.functions import {func}, col"]
        for col in self.columns:
            if self.normalization == "collapse_whitespace":
                lines.append(f"df = df.withColumn('{col}', regexp_replace(col('{col}'), r'\\s+', ' '))")
            else:
                lines.append(f"df = df.withColumn('{col}', {func}(col('{col}')))")
        return "\n".join(lines)
