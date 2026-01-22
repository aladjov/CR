from typing import Any, Dict, List, Optional
import pandas as pd

from ..base import EncodingRecommendation, Platform, RecommendationResult


class OneHotEncodeRecommendation(EncodingRecommendation):
    def __init__(
        self, columns: List[str], rationale: str = None, drop_first: bool = False,
        evidence: List[str] = None, priority: str = "medium", source_finding: Optional[Any] = None
    ):
        rationale = rationale or "One-hot encode categorical features"
        super().__init__(columns, rationale, evidence, priority, source_finding)
        self.drop_first = drop_first
        self._categories: Dict[str, List[str]] = {}

    @property
    def recommendation_type(self) -> str:
        return "onehot_encode"

    def _fit_impl(self, df: pd.DataFrame) -> None:
        for col in self.columns:
            if col in df.columns:
                self._categories[col] = list(df[col].dropna().unique())
        self._fit_params["categories"] = self._categories

    def _transform_local(self, df: pd.DataFrame) -> RecommendationResult:
        df = df.copy()
        rows_before = len(df)
        new_cols = []
        for col in self.columns:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=self.drop_first)
                new_cols.extend(dummies.columns.tolist())
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(columns=[col])
        return RecommendationResult(
            data=df, columns_affected=self.columns + new_cols, rows_before=rows_before,
            rows_after=len(df), metadata={"categories": self._categories, "new_columns": new_cols}
        )

    def _transform_databricks(self, df: pd.DataFrame) -> RecommendationResult:
        from customer_retention.core.compat import is_spark_available
        if not is_spark_available():
            return self._transform_local(df)
        return self._transform_local(df)

    def _generate_local_code(self) -> str:
        lines = [f"# Encode: {self.rationale}"]
        for col in self.columns:
            lines.append(f"df = pd.concat([df, pd.get_dummies(df['{col}'], prefix='{col}')], axis=1).drop(columns=['{col}'])")
        return "\n".join(lines)

    def _generate_databricks_code(self) -> str:
        lines = [f"# Encode: {self.rationale}", "from pyspark.ml.feature import StringIndexer, OneHotEncoder"]
        for col in self.columns:
            lines.append(f"indexer = StringIndexer(inputCol='{col}', outputCol='{col}_idx')")
            lines.append(f"encoder = OneHotEncoder(inputCol='{col}_idx', outputCol='{col}_vec')")
            lines.append("df = encoder.fit(indexer.fit(df).transform(df)).transform(indexer.fit(df).transform(df))")
        return "\n".join(lines)


class LabelEncodeRecommendation(EncodingRecommendation):
    def __init__(
        self, columns: List[str], rationale: str = None, evidence: List[str] = None,
        priority: str = "medium", source_finding: Optional[Any] = None
    ):
        rationale = rationale or "Label encode categorical features to integers"
        super().__init__(columns, rationale, evidence, priority, source_finding)
        self._mappings: Dict[str, Dict[str, int]] = {}

    @property
    def recommendation_type(self) -> str:
        return "label_encode"

    def _fit_impl(self, df: pd.DataFrame) -> None:
        for col in self.columns:
            if col in df.columns:
                categories = sorted(df[col].dropna().unique())
                self._mappings[col] = {cat: idx for idx, cat in enumerate(categories)}
        self._fit_params["mappings"] = self._mappings

    def _transform_local(self, df: pd.DataFrame) -> RecommendationResult:
        df = df.copy()
        rows_before = len(df)
        for col in self.columns:
            if col in df.columns and col in self._mappings:
                df[col] = df[col].map(self._mappings[col])
        return RecommendationResult(
            data=df, columns_affected=self.columns, rows_before=rows_before,
            rows_after=len(df), metadata={"mappings": self._mappings}
        )

    def _transform_databricks(self, df: pd.DataFrame) -> RecommendationResult:
        from customer_retention.core.compat import is_spark_available
        if not is_spark_available():
            return self._transform_local(df)
        return self._transform_local(df)

    def _generate_local_code(self) -> str:
        lines = [f"# Encode: {self.rationale}", "from sklearn.preprocessing import LabelEncoder"]
        for col in self.columns:
            lines.append(f"le_{col} = LabelEncoder()")
            lines.append(f"df['{col}'] = le_{col}.fit_transform(df['{col}'].astype(str))")
        return "\n".join(lines)

    def _generate_databricks_code(self) -> str:
        lines = [f"# Encode: {self.rationale}", "from pyspark.ml.feature import StringIndexer"]
        for col in self.columns:
            lines.append(f"indexer = StringIndexer(inputCol='{col}', outputCol='{col}_idx')")
            lines.append("df = indexer.fit(df).transform(df)")
            lines.append(f"df = df.drop('{col}').withColumnRenamed('{col}_idx', '{col}')")
        return "\n".join(lines)
