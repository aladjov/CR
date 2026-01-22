from typing import Any, List, Optional

import numpy as np
import pandas as pd

from ..base import RecommendationResult, TransformRecommendation


class LogTransformRecommendation(TransformRecommendation):
    def __init__(
        self, columns: List[str], rationale: str = None, evidence: List[str] = None,
        priority: str = "medium", source_finding: Optional[Any] = None
    ):
        rationale = rationale or "Apply log1p transform to reduce skewness"
        super().__init__(columns, rationale, evidence, priority, source_finding)

    @property
    def recommendation_type(self) -> str:
        return "log_transform"

    def _fit_impl(self, df: pd.DataFrame) -> None:
        self._fit_params["columns"] = self.columns

    def _transform_local(self, df: pd.DataFrame) -> RecommendationResult:
        df = df.copy()
        for col in self.columns:
            if col in df.columns:
                df[col] = np.log1p(df[col])
        return RecommendationResult(
            data=df, columns_affected=self.columns, rows_before=len(df),
            rows_after=len(df), metadata={"transform": "log1p"}
        )

    def _transform_databricks(self, df: pd.DataFrame) -> RecommendationResult:
        from customer_retention.core.compat import is_spark_available
        if not is_spark_available():
            return self._transform_local(df)
        return self._transform_local(df)

    def _generate_local_code(self) -> str:
        lines = [f"# Transform: {self.rationale}", "import numpy as np"]
        for col in self.columns:
            lines.append(f"df['{col}'] = np.log1p(df['{col}'])")
        return "\n".join(lines)

    def _generate_databricks_code(self) -> str:
        lines = [f"# Transform: {self.rationale}", "from pyspark.sql.functions import log1p, col"]
        for col in self.columns:
            lines.append(f"df = df.withColumn('{col}', log1p(col('{col}')))")
        return "\n".join(lines)


class SqrtTransformRecommendation(TransformRecommendation):
    def __init__(
        self, columns: List[str], rationale: str = None, evidence: List[str] = None,
        priority: str = "medium", source_finding: Optional[Any] = None
    ):
        rationale = rationale or "Apply sqrt transform to reduce moderate skewness"
        super().__init__(columns, rationale, evidence, priority, source_finding)

    @property
    def recommendation_type(self) -> str:
        return "sqrt_transform"

    def _fit_impl(self, df: pd.DataFrame) -> None:
        self._fit_params["columns"] = self.columns

    def _transform_local(self, df: pd.DataFrame) -> RecommendationResult:
        df = df.copy()
        for col in self.columns:
            if col in df.columns:
                df[col] = np.sqrt(df[col])
        return RecommendationResult(
            data=df, columns_affected=self.columns, rows_before=len(df),
            rows_after=len(df), metadata={"transform": "sqrt"}
        )

    def _transform_databricks(self, df: pd.DataFrame) -> RecommendationResult:
        from customer_retention.core.compat import is_spark_available
        if not is_spark_available():
            return self._transform_local(df)
        return self._transform_local(df)

    def _generate_local_code(self) -> str:
        lines = [f"# Transform: {self.rationale}", "import numpy as np"]
        for col in self.columns:
            lines.append(f"df['{col}'] = np.sqrt(df['{col}'])")
        return "\n".join(lines)

    def _generate_databricks_code(self) -> str:
        lines = [f"# Transform: {self.rationale}", "from pyspark.sql.functions import sqrt, col"]
        for col in self.columns:
            lines.append(f"df = df.withColumn('{col}', sqrt(col('{col}')))")
        return "\n".join(lines)
