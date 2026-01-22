from typing import Any, Dict, List, Optional

import pandas as pd

from ..base import CleaningRecommendation, RecommendationResult


class OutlierCapRecommendation(CleaningRecommendation):
    def __init__(
        self, columns: List[str], rationale: str = None, percentile: int = 99,
        evidence: List[str] = None, priority: str = "medium", source_finding: Optional[Any] = None
    ):
        rationale = rationale or f"Cap outliers at {percentile}th percentile"
        super().__init__(columns, rationale, evidence, priority, source_finding)
        self.percentile = percentile
        self._bounds: Dict[str, Dict[str, float]] = {}

    @property
    def recommendation_type(self) -> str:
        return f"cap_outliers_{self.percentile}"

    def _fit_impl(self, df: pd.DataFrame) -> None:
        lower_pct = (100 - self.percentile) / 100
        upper_pct = self.percentile / 100
        for col in self.columns:
            if col not in df.columns:
                continue
            series = df[col].dropna()
            self._bounds[col] = {
                "lower": float(series.quantile(lower_pct)),
                "upper": float(series.quantile(upper_pct)),
            }
        self._fit_params["bounds"] = self._bounds

    def _transform_local(self, df: pd.DataFrame) -> RecommendationResult:
        df = df.copy()
        rows_before = len(df)
        outliers_capped = {}
        for col in self.columns:
            if col in df.columns and col in self._bounds:
                bounds = self._bounds[col]
                outlier_mask = (df[col] < bounds["lower"]) | (df[col] > bounds["upper"])
                outliers_capped[col] = int(outlier_mask.sum())
                df[col] = df[col].clip(lower=bounds["lower"], upper=bounds["upper"])
        return RecommendationResult(
            data=df, columns_affected=self.columns, rows_before=rows_before,
            rows_after=len(df), metadata={"outliers_capped": outliers_capped, "bounds": self._bounds}
        )

    def _transform_databricks(self, df: pd.DataFrame) -> RecommendationResult:
        from customer_retention.core.compat import is_spark_available
        if not is_spark_available():
            return self._transform_local(df)
        return self._transform_local(df)

    def _generate_local_code(self) -> str:
        lines = [f"# Cap outliers: {self.rationale}"]
        for col, bounds in self._bounds.items():
            lines.append(f"df['{col}'] = df['{col}'].clip(lower={bounds['lower']}, upper={bounds['upper']})")
        return "\n".join(lines)

    def _generate_databricks_code(self) -> str:
        lines = [f"# Cap outliers: {self.rationale}", "from pyspark.sql.functions import when, col"]
        for col, bounds in self._bounds.items():
            lines.append(
                f"df = df.withColumn('{col}', "
                f"when(col('{col}') < {bounds['lower']}, {bounds['lower']})"
                f".when(col('{col}') > {bounds['upper']}, {bounds['upper']})"
                f".otherwise(col('{col}')))"
            )
        return "\n".join(lines)
