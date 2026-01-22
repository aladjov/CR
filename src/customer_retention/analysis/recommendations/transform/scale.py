from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..base import RecommendationResult, TransformRecommendation


class StandardScaleRecommendation(TransformRecommendation):
    def __init__(
        self, columns: List[str], rationale: str = None, evidence: List[str] = None,
        priority: str = "medium", source_finding: Optional[Any] = None
    ):
        rationale = rationale or "Standardize features to zero mean and unit variance"
        super().__init__(columns, rationale, evidence, priority, source_finding)
        self._means: Dict[str, float] = {}
        self._stds: Dict[str, float] = {}

    @property
    def recommendation_type(self) -> str:
        return "standard_scale"

    def _fit_impl(self, df: pd.DataFrame) -> None:
        for col in self.columns:
            if col in df.columns:
                self._means[col] = float(df[col].mean())
                self._stds[col] = float(df[col].std(ddof=0))
        self._fit_params = {"means": self._means, "stds": self._stds}

    def _transform_local(self, df: pd.DataFrame) -> RecommendationResult:
        df = df.copy()
        for col in self.columns:
            if col in df.columns and col in self._means:
                std = self._stds[col] if self._stds[col] != 0 else 1.0
                df[col] = (df[col] - self._means[col]) / std
        return RecommendationResult(
            data=df, columns_affected=self.columns, rows_before=len(df),
            rows_after=len(df), metadata={"means": self._means, "stds": self._stds}
        )

    def _transform_databricks(self, df: pd.DataFrame) -> RecommendationResult:
        from customer_retention.core.compat import is_spark_available
        if not is_spark_available():
            return self._transform_local(df)
        return self._transform_local(df)

    def _generate_local_code(self) -> str:
        return f"""# Scale: {self.rationale}
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[{self.columns}] = scaler.fit_transform(df[{self.columns}])"""

    def _generate_databricks_code(self) -> str:
        return f"""# Scale: {self.rationale}
from pyspark.ml.feature import StandardScaler, VectorAssembler
assembler = VectorAssembler(inputCols={self.columns}, outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaled", withStd=True, withMean=True)
df = scaler.fit(assembler.transform(df)).transform(assembler.transform(df))"""


class MinMaxScaleRecommendation(TransformRecommendation):
    def __init__(
        self, columns: List[str], rationale: str = None, feature_range: Tuple[float, float] = (0, 1),
        evidence: List[str] = None, priority: str = "medium", source_finding: Optional[Any] = None
    ):
        rationale = rationale or f"Scale features to range {feature_range}"
        super().__init__(columns, rationale, evidence, priority, source_finding)
        self.feature_range = feature_range
        self._mins: Dict[str, float] = {}
        self._maxs: Dict[str, float] = {}

    @property
    def recommendation_type(self) -> str:
        return "minmax_scale"

    def _fit_impl(self, df: pd.DataFrame) -> None:
        for col in self.columns:
            if col in df.columns:
                self._mins[col] = float(df[col].min())
                self._maxs[col] = float(df[col].max())
        self._fit_params = {"mins": self._mins, "maxs": self._maxs, "feature_range": self.feature_range}

    def _transform_local(self, df: pd.DataFrame) -> RecommendationResult:
        df = df.copy()
        min_val, max_val = self.feature_range
        for col in self.columns:
            if col in df.columns and col in self._mins:
                col_min, col_max = self._mins[col], self._maxs[col]
                scale = (max_val - min_val) / (col_max - col_min) if col_max != col_min else 1.0
                df[col] = (df[col] - col_min) * scale + min_val
        return RecommendationResult(
            data=df, columns_affected=self.columns, rows_before=len(df),
            rows_after=len(df), metadata={"mins": self._mins, "maxs": self._maxs}
        )

    def _transform_databricks(self, df: pd.DataFrame) -> RecommendationResult:
        from customer_retention.core.compat import is_spark_available
        if not is_spark_available():
            return self._transform_local(df)
        return self._transform_local(df)

    def _generate_local_code(self) -> str:
        return f"""# Scale: {self.rationale}
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range={self.feature_range})
df[{self.columns}] = scaler.fit_transform(df[{self.columns}])"""

    def _generate_databricks_code(self) -> str:
        return f"""# Scale: {self.rationale}
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
assembler = VectorAssembler(inputCols={self.columns}, outputCol="features")
scaler = MinMaxScaler(inputCol="features", outputCol="scaled", min={self.feature_range[0]}, max={self.feature_range[1]})
df = scaler.fit(assembler.transform(df)).transform(assembler.transform(df))"""
