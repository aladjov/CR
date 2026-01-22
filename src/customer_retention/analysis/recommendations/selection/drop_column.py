from typing import Any, List, Optional

import pandas as pd

from ..base import BaseRecommendation, RecommendationResult


class DropColumnRecommendation(BaseRecommendation):
    def __init__(
        self, columns: List[str], rationale: str = None, reason: str = "not_specified",
        evidence: List[str] = None, priority: str = "medium", source_finding: Optional[Any] = None
    ):
        rationale = rationale or f"Drop columns: {', '.join(columns)}"
        super().__init__(columns, rationale, evidence, priority, source_finding)
        self.reason = reason
        self._columns_to_drop: List[str] = []

    @property
    def category(self) -> str:
        return "feature_selection"

    @property
    def recommendation_type(self) -> str:
        return f"drop_{self.reason}"

    def _fit_impl(self, df: pd.DataFrame) -> None:
        self._columns_to_drop = [c for c in self.columns if c in df.columns]

    def _transform_local(self, df: pd.DataFrame) -> RecommendationResult:
        df = df.copy()
        rows_before = len(df)
        cols_to_drop = [c for c in self._columns_to_drop if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
        return RecommendationResult(
            data=df, columns_affected=self._columns_to_drop, rows_before=rows_before,
            rows_after=len(df), metadata={"dropped_columns": cols_to_drop}
        )

    def _transform_databricks(self, df: pd.DataFrame) -> RecommendationResult:
        from customer_retention.core.compat import is_spark_available
        if not is_spark_available():
            return self._transform_local(df)
        return self._transform_local(df)

    def _generate_local_code(self) -> str:
        cols_str = ", ".join(f"'{c}'" for c in self._columns_to_drop)
        lines = [f"# Drop columns: {self.rationale}"]
        lines.append(f"df = df.drop(columns=[{cols_str}])")
        return "\n".join(lines)

    def _generate_databricks_code(self) -> str:
        cols_str = ", ".join(f"'{c}'" for c in self._columns_to_drop)
        lines = [f"# Drop columns: {self.rationale}"]
        lines.append(f"df = df.drop([{cols_str}])")
        return "\n".join(lines)
