from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional

import pandas as pd

from .base import BaseRecommendation, Platform

if TYPE_CHECKING:
    from customer_retention.analysis.auto_explorer.findings import ExplorationFindings
    from customer_retention.stages.features.feature_definitions import FeatureCatalog


class RecommendationPipeline:
    def __init__(self, recommendations: List[BaseRecommendation] = None):
        self.recommendations = recommendations or []
        self._is_fitted = False

    def add(self, recommendation: BaseRecommendation) -> "RecommendationPipeline":
        self.recommendations.append(recommendation)
        return self

    def fit(self, df: pd.DataFrame) -> "RecommendationPipeline":
        for rec in self.recommendations:
            rec.fit(df)
        self._is_fitted = True
        return self

    def transform(
        self, df: pd.DataFrame, platform: Platform = Platform.LOCAL,
        mlflow_adapter: Optional[Any] = None
    ) -> pd.DataFrame:
        for rec in self.recommendations:
            result = rec.transform(df, platform, mlflow_adapter=mlflow_adapter)
            df = result.data
        return df

    def fit_transform(
        self, df: pd.DataFrame, platform: Platform = Platform.LOCAL,
        mlflow_adapter: Optional[Any] = None
    ) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df, platform, mlflow_adapter=mlflow_adapter)

    def generate_code(self, platform: Platform = Platform.LOCAL) -> str:
        if not self.recommendations:
            return ""
        lines = []
        for rec in self.recommendations:
            lines.append(rec.generate_code(platform))
            lines.append("")
        return "\n".join(lines).strip()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommendations": [rec.to_dict() for rec in self.recommendations],
            "is_fitted": self._is_fitted,
        }

    def to_feature_catalog(self) -> "FeatureCatalog":
        from customer_retention.stages.features.feature_definitions import FeatureCatalog
        catalog = FeatureCatalog()
        for rec in self.recommendations:
            catalog.add(rec.to_feature_definition())
        return catalog

    @classmethod
    def from_findings(cls, findings: "ExplorationFindings") -> "RecommendationPipeline":
        from .registry import RecommendationRegistry
        return cls(RecommendationRegistry.from_findings(findings))

    def __len__(self) -> int:
        return len(self.recommendations)

    def __iter__(self) -> Iterator[BaseRecommendation]:
        return iter(self.recommendations)
