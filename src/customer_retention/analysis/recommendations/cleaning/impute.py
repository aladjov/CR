from typing import Any, Dict, List, Optional

import pandas as pd

from ..base import CleaningRecommendation, RecommendationResult


class ImputeRecommendation(CleaningRecommendation):
    def __init__(
        self, columns: List[str], rationale: str = None, strategy: str = "median",
        fill_value: Any = None, evidence: List[str] = None,
        priority: str = "medium", source_finding: Optional[Any] = None
    ):
        rationale = rationale or f"Impute missing values using {strategy}"
        super().__init__(columns, rationale, evidence, priority, source_finding)
        self.strategy = strategy
        self.fill_value = fill_value
        self._impute_values: Dict[str, Any] = {}

    @property
    def recommendation_type(self) -> str:
        return f"impute_{self.strategy}"

    def _fit_impl(self, df: pd.DataFrame) -> None:
        for col in self.columns:
            if col not in df.columns:
                continue
            series = df[col]
            if self.strategy == "median":
                self._impute_values[col] = series.median()
            elif self.strategy == "mean":
                self._impute_values[col] = series.mean()
            elif self.strategy == "mode":
                modes = series.mode()
                self._impute_values[col] = modes.iloc[0] if len(modes) > 0 else None
            elif self.strategy == "constant":
                self._impute_values[col] = self.fill_value
        self._fit_params["impute_values"] = self._impute_values

    def _transform_local(self, df: pd.DataFrame) -> RecommendationResult:
        df = df.copy()
        rows_before = len(df)
        nulls_imputed = {}
        for col in self.columns:
            if col in df.columns and col in self._impute_values:
                nulls = int(df[col].isna().sum())
                df[col] = df[col].fillna(self._impute_values[col])
                nulls_imputed[col] = nulls
        return RecommendationResult(
            data=df, columns_affected=self.columns, rows_before=rows_before,
            rows_after=len(df), metadata={"nulls_imputed": nulls_imputed}
        )

    def _transform_databricks(self, df: pd.DataFrame) -> RecommendationResult:
        from customer_retention.core.compat import is_spark_available
        if not is_spark_available():
            return self._transform_local(df)
        return self._transform_local(df)

    def _generate_local_code(self) -> str:
        lines = [f"# Impute: {self.rationale}"]
        for col, val in self._impute_values.items():
            lines.append(f"df['{col}'] = df['{col}'].fillna({repr(val)})")
        return "\n".join(lines)

    def _generate_databricks_code(self) -> str:
        return f"# Impute: {self.rationale}\ndf = df.fillna({self._impute_values})"
