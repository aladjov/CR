from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
import numpy as np

from customer_retention.core.compat import pd, Series


class EncodingType(Enum):
    ONE_HOT = "one_hot"
    TARGET = "target"
    FREQUENCY = "frequency"
    ORDINAL = "ordinal"
    CYCLICAL = "cyclical"
    BINARY = "binary"


@dataclass
class CategoricalDistributionAnalysis:
    column_name: str
    category_count: int
    total_count: int
    imbalance_ratio: float
    entropy: float
    normalized_entropy: float
    top1_concentration: float
    top3_concentration: float
    rare_category_count: int
    rare_category_names: List[str]
    value_counts: Dict[str, int]

    IMBALANCE_THRESHOLD = 10.0
    LOW_DIVERSITY_THRESHOLD = 0.5
    HIGH_CONCENTRATION_THRESHOLD = 90.0

    @property
    def is_imbalanced(self) -> bool:
        return self.imbalance_ratio > self.IMBALANCE_THRESHOLD

    @property
    def has_low_diversity(self) -> bool:
        return self.normalized_entropy < self.LOW_DIVERSITY_THRESHOLD

    @property
    def has_rare_categories(self) -> bool:
        return self.rare_category_count > 0

    @property
    def is_highly_concentrated(self) -> bool:
        return self.top3_concentration > self.HIGH_CONCENTRATION_THRESHOLD

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column": self.column_name,
            "categories": self.category_count,
            "imbalance_ratio": round(self.imbalance_ratio, 2),
            "entropy": round(self.entropy, 3),
            "normalized_entropy": round(self.normalized_entropy, 3),
            "top1_concentration": round(self.top1_concentration, 1),
            "top3_concentration": round(self.top3_concentration, 1),
            "rare_categories": self.rare_category_count,
            "is_imbalanced": self.is_imbalanced,
            "has_low_diversity": self.has_low_diversity,
        }


@dataclass
class EncodingRecommendation:
    column_name: str
    encoding_type: EncodingType
    reason: str
    priority: str
    preprocessing_steps: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column": self.column_name,
            "encoding": self.encoding_type.value,
            "reason": self.reason,
            "priority": self.priority,
            "preprocessing_steps": self.preprocessing_steps,
            "warnings": self.warnings,
        }


class CategoricalDistributionAnalyzer:
    LOW_CARDINALITY_THRESHOLD = 5
    MEDIUM_CARDINALITY_THRESHOLD = 20
    RARE_CATEGORY_THRESHOLD = 0.01

    def analyze(self, series: Series, column_name: str) -> CategoricalDistributionAnalysis:
        clean_series = series.dropna()
        total_count = len(clean_series)

        if total_count == 0:
            return self._empty_analysis(column_name)

        value_counts = clean_series.value_counts()
        category_count = len(value_counts)

        if category_count == 0:
            return self._empty_analysis(column_name)

        imbalance_ratio = float(value_counts.iloc[0] / value_counts.iloc[-1]) if value_counts.iloc[-1] > 0 else float('inf')
        entropy, normalized_entropy = self._calculate_entropy(value_counts, total_count, category_count)
        top1_concentration = float(value_counts.iloc[0] / total_count * 100)
        top3_concentration = float(value_counts.head(3).sum() / total_count * 100)
        rare_threshold = total_count * self.RARE_CATEGORY_THRESHOLD
        rare_mask = value_counts < rare_threshold
        rare_category_count = int(rare_mask.sum())
        rare_category_names = value_counts[rare_mask].index.tolist()

        return CategoricalDistributionAnalysis(
            column_name=column_name,
            category_count=category_count,
            total_count=total_count,
            imbalance_ratio=imbalance_ratio,
            entropy=entropy,
            normalized_entropy=normalized_entropy,
            top1_concentration=top1_concentration,
            top3_concentration=top3_concentration,
            rare_category_count=rare_category_count,
            rare_category_names=rare_category_names[:10],
            value_counts=value_counts.head(20).to_dict(),
        )

    def _empty_analysis(self, column_name: str) -> CategoricalDistributionAnalysis:
        return CategoricalDistributionAnalysis(
            column_name=column_name, category_count=0, total_count=0,
            imbalance_ratio=0.0, entropy=0.0, normalized_entropy=0.0,
            top1_concentration=0.0, top3_concentration=0.0,
            rare_category_count=0, rare_category_names=[], value_counts={},
        )

    def _calculate_entropy(self, value_counts: Series, total: int, n_categories: int) -> tuple:
        probabilities = value_counts / total
        entropy = float(-np.sum(probabilities * np.log2(probabilities + 1e-10)))
        max_entropy = np.log2(n_categories) if n_categories > 1 else 1.0
        normalized = entropy / max_entropy if max_entropy > 0 else 0.0
        return entropy, normalized

    def recommend_encoding(
        self, analysis: CategoricalDistributionAnalysis, is_cyclical: bool = False, is_ordinal: bool = False
    ) -> EncodingRecommendation:
        preprocessing = []
        warnings = []

        if is_cyclical:
            return EncodingRecommendation(
                column_name=analysis.column_name,
                encoding_type=EncodingType.CYCLICAL,
                reason="Cyclical data benefits from sin/cos encoding to preserve circular relationships",
                priority="medium",
            )

        if is_ordinal:
            return EncodingRecommendation(
                column_name=analysis.column_name,
                encoding_type=EncodingType.ORDINAL,
                reason="Ordinal data should preserve category order",
                priority="medium",
            )

        if analysis.category_count == 2:
            return EncodingRecommendation(
                column_name=analysis.column_name,
                encoding_type=EncodingType.BINARY,
                reason="Binary categorical - simple 0/1 encoding",
                priority="low",
            )

        if analysis.has_rare_categories:
            preprocessing.append(f"Group {analysis.rare_category_count} rare categories into 'Other'")

        if analysis.is_imbalanced:
            warnings.append("Use stratified sampling to preserve rare category representation")

        if analysis.category_count <= self.LOW_CARDINALITY_THRESHOLD:
            return EncodingRecommendation(
                column_name=analysis.column_name,
                encoding_type=EncodingType.ONE_HOT,
                reason=f"Low cardinality ({analysis.category_count} categories) - safe feature expansion",
                priority="low",
                preprocessing_steps=preprocessing,
                warnings=warnings,
            )

        if analysis.category_count <= self.MEDIUM_CARDINALITY_THRESHOLD:
            if analysis.has_rare_categories:
                encoding = EncodingType.TARGET
                reason = f"Medium cardinality ({analysis.category_count}) with rare categories - target encoding preferred"
                priority = "medium"
            else:
                encoding = EncodingType.ONE_HOT
                reason = f"Medium cardinality ({analysis.category_count}) without rare categories"
                priority = "low"
            return EncodingRecommendation(
                column_name=analysis.column_name,
                encoding_type=encoding,
                reason=reason,
                priority=priority,
                preprocessing_steps=preprocessing,
                warnings=warnings,
            )

        warnings.append("High cardinality may require regularization with target encoding")
        return EncodingRecommendation(
            column_name=analysis.column_name,
            encoding_type=EncodingType.TARGET,
            reason=f"High cardinality ({analysis.category_count} categories) - target or frequency encoding",
            priority="high",
            preprocessing_steps=preprocessing,
            warnings=warnings,
        )

    def analyze_dataframe(
        self, df: pd.DataFrame, categorical_columns: Optional[List[str]] = None
    ) -> Dict[str, CategoricalDistributionAnalysis]:
        if categorical_columns is None:
            categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()

        return {col: self.analyze(df[col], col) for col in categorical_columns if col in df.columns}

    def get_all_recommendations(
        self, df: pd.DataFrame, categorical_columns: Optional[List[str]] = None,
        cyclical_columns: Optional[List[str]] = None, ordinal_columns: Optional[List[str]] = None
    ) -> List[EncodingRecommendation]:
        cyclical_columns = cyclical_columns or []
        ordinal_columns = ordinal_columns or []
        analyses = self.analyze_dataframe(df, categorical_columns)

        recommendations = []
        for col_name, analysis in analyses.items():
            is_cyclical = col_name in cyclical_columns
            is_ordinal = col_name in ordinal_columns
            rec = self.recommend_encoding(analysis, is_cyclical=is_cyclical, is_ordinal=is_ordinal)
            recommendations.append(rec)

        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda r: priority_order.get(r.priority, 3))
        return recommendations

    def generate_summary(
        self, df: pd.DataFrame, categorical_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        analyses = self.analyze_dataframe(df, categorical_columns)
        recommendations = self.get_all_recommendations(df, categorical_columns)

        imbalanced = [n for n, a in analyses.items() if a.is_imbalanced]
        low_diversity = [n for n, a in analyses.items() if a.has_low_diversity]
        with_rare = [n for n, a in analyses.items() if a.has_rare_categories]

        return {
            "summary": {
                "total_columns": len(analyses),
                "imbalanced_count": len(imbalanced),
                "low_diversity_count": len(low_diversity),
                "with_rare_categories_count": len(with_rare),
            },
            "categories": {
                "imbalanced": imbalanced,
                "low_diversity": low_diversity,
                "with_rare_categories": with_rare,
            },
            "analyses": {k: v.to_dict() for k, v in analyses.items()},
            "recommendations": [r.to_dict() for r in recommendations],
        }
