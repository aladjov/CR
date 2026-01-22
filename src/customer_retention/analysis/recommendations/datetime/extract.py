from datetime import datetime
from typing import Any, List, Optional

import pandas as pd

from ..base import DatetimeRecommendation, RecommendationResult


class ExtractMonthRecommendation(DatetimeRecommendation):
    def __init__(
        self, columns: List[str], rationale: str = None, evidence: List[str] = None,
        priority: str = "medium", source_finding: Optional[Any] = None
    ):
        rationale = rationale or "Extract month from datetime for seasonality analysis"
        super().__init__(columns, rationale, evidence, priority, source_finding)

    @property
    def recommendation_type(self) -> str:
        return "extract_month"

    def _fit_impl(self, df: pd.DataFrame) -> None:
        self._fit_params["columns"] = self.columns

    def _transform_local(self, df: pd.DataFrame) -> RecommendationResult:
        df = df.copy()
        new_cols = []
        for col in self.columns:
            if col in df.columns:
                new_col = f"{col}_month"
                df[new_col] = pd.to_datetime(df[col]).dt.month
                new_cols.append(new_col)
        return RecommendationResult(
            data=df, columns_affected=self.columns + new_cols,
            rows_before=len(df), rows_after=len(df), metadata={"new_columns": new_cols}
        )

    def _transform_databricks(self, df: pd.DataFrame) -> RecommendationResult:
        from customer_retention.core.compat import is_spark_available
        if not is_spark_available():
            return self._transform_local(df)
        return self._transform_local(df)

    def _generate_local_code(self) -> str:
        lines = [f"# Extract: {self.rationale}"]
        for col in self.columns:
            lines.append(f"df['{col}_month'] = pd.to_datetime(df['{col}']).dt.month")
        return "\n".join(lines)

    def _generate_databricks_code(self) -> str:
        lines = [f"# Extract: {self.rationale}", "from pyspark.sql.functions import month, col"]
        for col in self.columns:
            lines.append(f"df = df.withColumn('{col}_month', month(col('{col}')))")
        return "\n".join(lines)


class ExtractDayOfWeekRecommendation(DatetimeRecommendation):
    def __init__(
        self, columns: List[str], rationale: str = None, evidence: List[str] = None,
        priority: str = "medium", source_finding: Optional[Any] = None
    ):
        rationale = rationale or "Extract day of week from datetime for weekly patterns"
        super().__init__(columns, rationale, evidence, priority, source_finding)

    @property
    def recommendation_type(self) -> str:
        return "extract_dayofweek"

    def _fit_impl(self, df: pd.DataFrame) -> None:
        self._fit_params["columns"] = self.columns

    def _transform_local(self, df: pd.DataFrame) -> RecommendationResult:
        df = df.copy()
        new_cols = []
        for col in self.columns:
            if col in df.columns:
                new_col = f"{col}_dayofweek"
                df[new_col] = pd.to_datetime(df[col]).dt.dayofweek
                new_cols.append(new_col)
        return RecommendationResult(
            data=df, columns_affected=self.columns + new_cols,
            rows_before=len(df), rows_after=len(df), metadata={"new_columns": new_cols}
        )

    def _transform_databricks(self, df: pd.DataFrame) -> RecommendationResult:
        from customer_retention.core.compat import is_spark_available
        if not is_spark_available():
            return self._transform_local(df)
        return self._transform_local(df)

    def _generate_local_code(self) -> str:
        lines = [f"# Extract: {self.rationale}"]
        for col in self.columns:
            lines.append(f"df['{col}_dayofweek'] = pd.to_datetime(df['{col}']).dt.dayofweek")
        return "\n".join(lines)

    def _generate_databricks_code(self) -> str:
        lines = [f"# Extract: {self.rationale}", "from pyspark.sql.functions import dayofweek, col"]
        for col in self.columns:
            lines.append(f"df = df.withColumn('{col}_dayofweek', dayofweek(col('{col}')) - 1)")
        return "\n".join(lines)


class DaysSinceRecommendation(DatetimeRecommendation):
    def __init__(
        self, columns: List[str], rationale: str = None, reference_date: datetime = None,
        evidence: List[str] = None, priority: str = "medium", source_finding: Optional[Any] = None
    ):
        rationale = rationale or "Calculate days since datetime for recency features"
        super().__init__(columns, rationale, evidence, priority, source_finding)
        self.reference_date = reference_date or datetime.now()

    @property
    def recommendation_type(self) -> str:
        return "days_since"

    def _fit_impl(self, df: pd.DataFrame) -> None:
        self._fit_params["reference_date"] = str(self.reference_date)

    def _transform_local(self, df: pd.DataFrame) -> RecommendationResult:
        df = df.copy()
        new_cols = []
        for col in self.columns:
            if col in df.columns:
                new_col = f"{col}_days_since"
                df[new_col] = (pd.Timestamp(self.reference_date) - pd.to_datetime(df[col])).dt.days
                new_cols.append(new_col)
        return RecommendationResult(
            data=df, columns_affected=self.columns + new_cols,
            rows_before=len(df), rows_after=len(df), metadata={"reference_date": str(self.reference_date), "new_columns": new_cols}
        )

    def _transform_databricks(self, df: pd.DataFrame) -> RecommendationResult:
        from customer_retention.core.compat import is_spark_available
        if not is_spark_available():
            return self._transform_local(df)
        return self._transform_local(df)

    def _generate_local_code(self) -> str:
        lines = [f"# Extract: {self.rationale}", f"reference_date = pd.Timestamp('{self.reference_date}')"]
        for col in self.columns:
            lines.append(f"df['{col}_days_since'] = (reference_date - pd.to_datetime(df['{col}'])).dt.days")
        return "\n".join(lines)

    def _generate_databricks_code(self) -> str:
        lines = [f"# Extract: {self.rationale}", "from pyspark.sql.functions import datediff, lit, col, to_date"]
        lines.append(f"reference_date = '{self.reference_date.strftime('%Y-%m-%d')}'")
        for col in self.columns:
            lines.append(f"df = df.withColumn('{col}_days_since', datediff(lit(reference_date), to_date(col('{col}'))))")
        return "\n".join(lines)
