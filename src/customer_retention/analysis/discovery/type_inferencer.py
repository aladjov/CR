from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union

import pandas as pd

from customer_retention.core.compat import ops
from customer_retention.core.config.column_config import ColumnType


class InferenceConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ColumnInference:
    column_name: str
    inferred_type: ColumnType
    confidence: InferenceConfidence
    evidence: List[str]
    alternatives: List[ColumnType] = field(default_factory=list)
    suggested_encoding: Optional[str] = None
    suggested_scaling: Optional[str] = None
    suggested_missing_strategy: Optional[str] = None


@dataclass
class InferenceResult:
    inferences: Dict[str, ColumnInference]
    target_column: Optional[str] = None
    identifier_columns: List[str] = field(default_factory=list)
    datetime_columns: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class TypeInferencer:
    TARGET_PATTERNS = ["target", "label", "churn", "retained", "outcome", "class", "y"]
    ID_PATTERNS = ["id", "key", "code", "identifier", "index"]

    def __init__(self):
        self.evidence: List[str] = []

    def infer(self, source: Union[str, pd.DataFrame]) -> InferenceResult:
        if isinstance(source, str):
            df = ops.read_csv(source)
        else:
            df = source
        inferences = {}
        target_column = None
        identifier_columns = []
        datetime_columns = []
        for col in df.columns:
            inference = self._infer_column(df[col], col)
            inferences[col] = inference
            if inference.inferred_type == ColumnType.TARGET:
                target_column = col
            elif inference.inferred_type == ColumnType.IDENTIFIER:
                identifier_columns.append(col)
            elif inference.inferred_type == ColumnType.DATETIME:
                datetime_columns.append(col)
        return InferenceResult(
            inferences=inferences,
            target_column=target_column,
            identifier_columns=identifier_columns,
            datetime_columns=datetime_columns
        )

    def _infer_column(self, series: pd.Series, column_name: str) -> ColumnInference:
        evidence = []
        col_lower = column_name.lower()
        if self._is_identifier(series, col_lower, evidence):
            return ColumnInference(column_name, ColumnType.IDENTIFIER, InferenceConfidence.HIGH, evidence)
        if self._is_target(series, col_lower, evidence):
            return ColumnInference(column_name, ColumnType.TARGET, InferenceConfidence.HIGH, evidence)
        if self._is_datetime(series, evidence):
            return ColumnInference(column_name, ColumnType.DATETIME, InferenceConfidence.HIGH, evidence)
        if self._is_binary(series, evidence):
            return ColumnInference(column_name, ColumnType.BINARY, InferenceConfidence.HIGH, evidence)
        if pd.api.types.is_numeric_dtype(series):
            return self._infer_numeric(series, column_name, evidence)
        return self._infer_categorical(series, column_name, evidence)

    def _is_identifier(self, series: pd.Series, col_lower: str, evidence: List[str]) -> bool:
        if any(p in col_lower for p in self.ID_PATTERNS):
            if series.nunique() == len(series):
                evidence.append("unique values, id pattern in name")
                return True
        if series.nunique() == len(series) and pd.api.types.is_integer_dtype(series):
            evidence.append("unique integer values")
            return True
        return False

    def _is_target(self, series: pd.Series, col_lower: str, evidence: List[str]) -> bool:
        if any(p in col_lower for p in self.TARGET_PATTERNS):
            if series.nunique() <= 10:
                evidence.append(f"target pattern in name, {series.nunique()} distinct values")
                return True
        return False

    def _is_datetime(self, series: pd.Series, evidence: List[str]) -> bool:
        if pd.api.types.is_datetime64_any_dtype(series):
            evidence.append("datetime dtype")
            return True
        if series.dtype == object:
            try:
                pd.to_datetime(series.dropna().head(100), format='mixed')
                evidence.append("parseable as datetime")
                return True
            except (ValueError, TypeError):
                pass
        return False

    def _is_binary(self, series: pd.Series, evidence: List[str]) -> bool:
        unique = series.dropna().unique()
        if len(unique) == 2:
            evidence.append("exactly 2 unique values")
            return True
        return False

    def _infer_numeric(self, series: pd.Series, column_name: str, evidence: List[str]) -> ColumnInference:
        nunique = series.nunique()
        if nunique <= 20:
            evidence.append(f"numeric with {nunique} unique values (discrete)")
            return ColumnInference(column_name, ColumnType.NUMERIC_DISCRETE, InferenceConfidence.HIGH, evidence,
                                   suggested_encoding="ordinal", suggested_missing_strategy="median")
        evidence.append(f"numeric with {nunique} unique values (continuous)")
        return ColumnInference(column_name, ColumnType.NUMERIC_CONTINUOUS, InferenceConfidence.HIGH, evidence,
                               suggested_scaling="standard", suggested_missing_strategy="median")

    def _infer_categorical(self, series: pd.Series, column_name: str, evidence: List[str]) -> ColumnInference:
        nunique = series.nunique()
        if nunique <= 10:
            evidence.append(f"categorical with {nunique} categories (low cardinality)")
            return ColumnInference(column_name, ColumnType.CATEGORICAL_NOMINAL, InferenceConfidence.HIGH, evidence,
                                   suggested_encoding="onehot", suggested_missing_strategy="mode")
        evidence.append(f"categorical with {nunique} categories (high cardinality)")
        return ColumnInference(column_name, ColumnType.CATEGORICAL_NOMINAL, InferenceConfidence.MEDIUM, evidence,
                               suggested_encoding="target", suggested_missing_strategy="mode")

    def show_report(self, result: InferenceResult) -> None:
        print(f"Target column: {result.target_column}")
        print(f"Identifier columns: {result.identifier_columns}")
        print(f"Datetime columns: {result.datetime_columns}")
        for col, inf in result.inferences.items():
            print(f"  {col}: {inf.inferred_type.value} ({inf.confidence.value}) - {', '.join(inf.evidence)}")
