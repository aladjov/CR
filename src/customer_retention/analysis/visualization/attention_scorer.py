"""Attention scoring for column analysis — ranks columns by how much they need review."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from customer_retention.stages.profiling.categorical_distribution import (
        CategoricalDistributionAnalysis,
        EncodingRecommendation,
    )
    from customer_retention.stages.profiling.distribution_analysis import (
        DistributionAnalysis,
        TransformationRecommendation,
    )


@dataclass
class AttentionScore:
    """Score indicating how much attention a column deserves during review."""

    column_name: str
    column_type: str
    score: float
    reasons: List[str] = field(default_factory=list)
    priority: str = "low"

    def __post_init__(self):
        self.score = max(0.0, min(100.0, self.score))
        if self.score >= 70:
            self.priority = "high"
        elif self.score >= 40:
            self.priority = "medium"
        else:
            self.priority = "low"


class AttentionScorer:
    """Static methods to score columns by how interesting/problematic they are."""

    @staticmethod
    def score_numeric(
        analysis: DistributionAnalysis,
        recommendation: Optional[TransformationRecommendation] = None,
    ) -> AttentionScore:
        score = 0.0
        reasons: list[str] = []

        # Skewness contribution: up to 30 points
        skew_score = min(abs(analysis.skewness) * 10, 30)
        if skew_score > 0:
            score += skew_score
            reasons.append(f"skewness {analysis.skewness:.2f}")

        # Outlier contribution: up to 25 points
        outlier_score = min(analysis.outlier_percentage * 2, 25)
        if outlier_score > 0:
            score += outlier_score
            reasons.append(f"outliers {analysis.outlier_percentage:.1f}%")

        # Zero inflation: up to 15 points (only if > 20%)
        if analysis.zero_percentage > 20:
            zero_score = min(analysis.zero_percentage * 0.4, 15)
            score += zero_score
            reasons.append(f"zero-inflated {analysis.zero_percentage:.1f}%")

        # Kurtosis (excess): up to 15 points
        excess_kurt = max(analysis.kurtosis - 3, 0)
        kurt_score = min(excess_kurt * 2, 15)
        if kurt_score > 0:
            score += kurt_score
            reasons.append(f"kurtosis {analysis.kurtosis:.2f}")

        # Transform priority boost
        if recommendation is not None:
            if recommendation.priority == "high":
                score += 15
                reasons.append("high-priority transform")
            elif recommendation.priority == "medium":
                score += 8
                reasons.append("medium-priority transform")

        return AttentionScore(
            column_name=analysis.column_name,
            column_type="numeric",
            score=score,
            reasons=reasons,
        )

    @staticmethod
    def score_categorical(
        analysis: CategoricalDistributionAnalysis,
        recommendation: Optional[EncodingRecommendation] = None,
    ) -> AttentionScore:
        score = 0.0
        reasons: list[str] = []

        # Imbalance: up to 30 points (log10 scale, kicks in at ratio > 5)
        if analysis.imbalance_ratio > 5:
            imbalance_score = min(math.log10(analysis.imbalance_ratio) * 15, 30)
            score += imbalance_score
            reasons.append(f"imbalance {analysis.imbalance_ratio:.1f}x")

        # Rare categories: up to 20 points
        rare_score = min(analysis.rare_category_count * 3, 20)
        if rare_score > 0:
            score += rare_score
            reasons.append(f"{analysis.rare_category_count} rare categories")

        # Low diversity
        if analysis.normalized_entropy < 0.3:
            score += 20
            reasons.append(f"low diversity (entropy {analysis.normalized_entropy:.2f})")

        # High concentration
        if analysis.top3_concentration > 90:
            score += 15
            reasons.append(f"top-3 concentration {analysis.top3_concentration:.1f}%")

        # Encoding priority boost
        if recommendation is not None:
            if recommendation.priority == "high":
                score += 15
                reasons.append("high-priority encoding")
            elif recommendation.priority == "medium":
                score += 8
                reasons.append("medium-priority encoding")

        return AttentionScore(
            column_name=analysis.column_name,
            column_type="categorical",
            score=score,
            reasons=reasons,
        )

    @staticmethod
    def score_datetime(summary: dict) -> AttentionScore:
        """Score a datetime column from a pre-computed summary dict.

        Expected keys:
            column_name, quality_issue_count, has_seasonality,
            overall_growth_pct, feature_rec_count, modeling_rec_count
        """
        score = 0.0
        reasons: list[str] = []
        col_name = summary.get("column_name", "unknown")

        # Quality issues: 10 per issue, capped at 30
        quality_count = summary.get("quality_issue_count", 0)
        if quality_count > 0:
            quality_score = min(quality_count * 10, 30)
            score += quality_score
            reasons.append(f"{quality_count} quality issues")

        # Seasonality
        if summary.get("has_seasonality", False):
            score += 15
            reasons.append("seasonality detected")

        # Strong trend
        growth_pct = abs(summary.get("overall_growth_pct", 0))
        if growth_pct > 50:
            score += 15
            reasons.append(f"strong trend ({growth_pct:.0f}% growth)")

        # Feature recommendations: 5 per rec, capped at 25
        feature_count = summary.get("feature_rec_count", 0)
        if feature_count > 0:
            feat_score = min(feature_count * 5, 25)
            score += feat_score
            reasons.append(f"{feature_count} feature recommendations")

        # Modeling strategy recommendations
        if summary.get("modeling_rec_count", 0) > 0:
            score += 15
            reasons.append("modeling strategy recommendation")

        return AttentionScore(
            column_name=col_name,
            column_type="datetime",
            score=score,
            reasons=reasons,
        )
