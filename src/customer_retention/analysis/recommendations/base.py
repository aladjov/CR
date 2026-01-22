from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import pandas as pd

from customer_retention.core.components.enums import Platform

if TYPE_CHECKING:
    from customer_retention.analysis.auto_explorer.findings import ColumnFinding
    from customer_retention.stages.features.feature_definitions import FeatureDefinition


@dataclass
class RecommendationResult:
    data: pd.DataFrame
    columns_affected: List[str]
    rows_before: int
    rows_after: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class BaseRecommendation(ABC):
    def __init__(
        self, columns: List[str], rationale: str, evidence: List[str] = None,
        priority: str = "medium", source_finding: Optional["ColumnFinding"] = None
    ):
        self.columns = columns
        self.rationale = rationale
        self.evidence = evidence or []
        self.priority = priority
        self.source_finding = source_finding
        self._is_fitted = False
        self._fit_params: Dict[str, Any] = {}

    @property
    @abstractmethod
    def recommendation_type(self) -> str:
        pass

    @property
    @abstractmethod
    def category(self) -> str:
        pass

    def fit(self, df: pd.DataFrame) -> "BaseRecommendation":
        self._fit_impl(df)
        self._is_fitted = True
        return self

    @abstractmethod
    def _fit_impl(self, df: pd.DataFrame) -> None:
        pass

    def transform(
        self, df: pd.DataFrame, platform: Platform = Platform.LOCAL,
        mlflow_adapter: Optional[Any] = None
    ) -> RecommendationResult:
        if not self._is_fitted:
            raise ValueError(f"{self.__class__.__name__} not fitted. Call fit() first.")
        result = self._transform_databricks(df) if platform == Platform.DATABRICKS else self._transform_local(df)
        if mlflow_adapter:
            mlflow_adapter.log_params(self._fit_params)
            mlflow_adapter.log_metrics({k: v for k, v in result.metadata.items() if isinstance(v, (int, float))})
        return result

    @abstractmethod
    def _transform_local(self, df: pd.DataFrame) -> RecommendationResult:
        pass

    @abstractmethod
    def _transform_databricks(self, df: pd.DataFrame) -> RecommendationResult:
        pass

    def fit_transform(self, df: pd.DataFrame, platform: Platform = Platform.LOCAL) -> RecommendationResult:
        self.fit(df)
        return self.transform(df, platform)

    def generate_code(self, platform: Platform = Platform.LOCAL) -> str:
        return self._generate_databricks_code() if platform == Platform.DATABRICKS else self._generate_local_code()

    @abstractmethod
    def _generate_local_code(self) -> str:
        pass

    @abstractmethod
    def _generate_databricks_code(self) -> str:
        pass

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.recommendation_type,
            "category": self.category,
            "columns": self.columns,
            "rationale": self.rationale,
            "evidence": self.evidence,
            "priority": self.priority,
            "fit_params": self._fit_params,
            "is_fitted": self._is_fitted,
        }

    def describe(self) -> str:
        return f"{self.recommendation_type} on {self.columns}: {self.rationale}"

    def to_feature_definition(self) -> "FeatureDefinition":
        from customer_retention.stages.features.feature_definitions import (
            FeatureCategory,
            FeatureDefinition,
            LeakageRisk,
        )
        category_map = {
            "cleaning": FeatureCategory.AGGREGATE,
            "transform": FeatureCategory.AGGREGATE,
            "encoding": FeatureCategory.AGGREGATE,
            "datetime": FeatureCategory.TEMPORAL,
            "feature": FeatureCategory.AGGREGATE,
        }
        return FeatureDefinition(
            name=f"{self.columns[0]}_{self.recommendation_type}",
            description=self.rationale,
            category=category_map.get(self.category, FeatureCategory.AGGREGATE),
            derivation=self._generate_local_code(),
            source_columns=self.columns,
            data_type="float64",
            business_meaning=self.rationale,
            leakage_risk=LeakageRisk.LOW,
        )


class CleaningRecommendation(BaseRecommendation, ABC):
    @property
    def category(self) -> str:
        return "cleaning"


class TransformRecommendation(BaseRecommendation, ABC):
    @property
    def category(self) -> str:
        return "transform"


class EncodingRecommendation(BaseRecommendation, ABC):
    @property
    def category(self) -> str:
        return "encoding"


class DatetimeRecommendation(BaseRecommendation, ABC):
    @property
    def category(self) -> str:
        return "datetime"


class FeatureRecommendation(BaseRecommendation, ABC):
    @property
    def category(self) -> str:
        return "feature"
