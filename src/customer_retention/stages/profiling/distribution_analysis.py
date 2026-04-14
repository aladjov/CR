"""
Distribution analysis module for exploratory data analysis.

This module provides functions for analyzing distributions and recommending
appropriate transformations based on distribution characteristics.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from customer_retention.core.compat import Series, _is_spark_pandas, native_pd, pd, to_pandas
from customer_retention.core.compat.bulk_profiling import bulk_distribution_stats

logger = logging.getLogger(__name__)

ZERO_INFLATION_CONSIDER_THRESHOLD = 30.0


class DistributionTransformationType(Enum):
    """Types of transformations for skewed distributions."""
    NONE = "none"
    LOG_TRANSFORM = "log_transform"
    SQRT_TRANSFORM = "sqrt_transform"
    BOX_COX = "box_cox"
    YERO_JOHNSON = "yeo_johnson"
    CAP_OUTLIERS = "cap_outliers"
    CAP_THEN_LOG = "cap_then_log"
    ZERO_INFLATION_HANDLING = "zero_inflation_handling"


@dataclass
class DistributionAnalysis:
    """Result of distribution analysis for a numeric column."""
    column_name: str
    count: int
    mean: float
    std: float
    min_value: float
    max_value: float
    median: float
    q1: float
    q3: float
    iqr: float
    skewness: float
    kurtosis: float
    zero_count: int
    zero_percentage: float
    negative_count: int
    negative_percentage: float
    outlier_count_iqr: int
    outlier_percentage: float
    percentiles: Dict[str, float] = field(default_factory=dict)

    @property
    def is_highly_skewed(self) -> bool:
        """Check if distribution is highly skewed."""
        return abs(self.skewness) > 2.0

    @property
    def is_moderately_skewed(self) -> bool:
        """Check if distribution is moderately skewed."""
        return 1.0 < abs(self.skewness) <= 2.0

    @property
    def has_zero_inflation(self) -> bool:
        """Check if distribution has significant zero inflation.

        Threshold raised from 30% → 70% in 2026-04: on entity-level snapshot
        bronze (the framework default for non per-grid-date sources), most
        numeric columns trivially clear 30% zeros because the entity's full
        history is broadcast onto every grid date and the post-event tail is
        all zeros. The 30% trigger therefore promoted snapshot artifacts to
        zero-inflation transforms, which produced single-bit `_is_zero` flags
        that classified the target. 70% reserves the recommendation for
        genuinely zero-inflated columns where zero is a semantic state.
        """
        return self.zero_percentage > 70.0

    @property
    def has_heavy_tails(self) -> bool:
        """Check if distribution has heavy tails (high kurtosis)."""
        return self.kurtosis > 3.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for display."""
        return {
            "column": self.column_name,
            "count": self.count,
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "min": round(self.min_value, 4),
            "max": round(self.max_value, 4),
            "median": round(self.median, 4),
            "skewness": round(self.skewness, 4),
            "kurtosis": round(self.kurtosis, 4),
            "zero_pct": round(self.zero_percentage, 2),
            "outlier_pct": round(self.outlier_percentage, 2),
            "is_highly_skewed": self.is_highly_skewed,
            "has_zero_inflation": self.has_zero_inflation
        }


@dataclass
class TransformationRecommendation:
    """Recommendation for transforming a column."""
    column_name: str
    recommended_transform: DistributionTransformationType
    reason: str
    priority: str  # "high", "medium", "low"
    parameters: Dict[str, Any] = field(default_factory=dict)
    alternative_transforms: List[DistributionTransformationType] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for display."""
        return {
            "column": self.column_name,
            "transform": self.recommended_transform.value,
            "reason": self.reason,
            "priority": self.priority,
            "parameters": self.parameters,
            "alternatives": [t.value for t in self.alternative_transforms],
            "warnings": self.warnings
        }


class DistributionAnalyzer:
    """
    Analyzer for numeric distribution characteristics.

    Provides methods for comprehensive distribution analysis and
    transformation recommendations.
    """

    # Thresholds
    HIGH_SKEWNESS_THRESHOLD = 2.0
    MODERATE_SKEWNESS_THRESHOLD = 1.0
    ZERO_INFLATION_THRESHOLD = 70.0
    OUTLIER_THRESHOLD = 5.0
    HIGH_KURTOSIS_THRESHOLD = 7.0

    def analyze_distribution(
        self,
        series: Series,
        column_name: str
    ) -> DistributionAnalysis:
        """
        Comprehensive distribution analysis for a single column.

        Parameters
        ----------
        series : Series
            Numeric data to analyze
        column_name : str
            Name of the column

        Returns
        -------
        DistributionAnalysis
            Detailed distribution statistics
        """
        clean_series = series.dropna()

        if len(clean_series) == 0:
            return DistributionAnalysis(
                column_name=column_name,
                count=0,
                mean=0.0,
                std=0.0,
                min_value=0.0,
                max_value=0.0,
                median=0.0,
                q1=0.0,
                q3=0.0,
                iqr=0.0,
                skewness=0.0,
                kurtosis=0.0,
                zero_count=0,
                zero_percentage=0.0,
                negative_count=0,
                negative_percentage=0.0,
                outlier_count_iqr=0,
                outlier_percentage=0.0
            )

        count = len(clean_series)
        mean = float(clean_series.mean())
        std = float(clean_series.std())
        min_value = float(clean_series.min())
        max_value = float(clean_series.max())
        median = float(clean_series.median())

        q1 = float(clean_series.quantile(0.25))
        q3 = float(clean_series.quantile(0.75))
        iqr = q3 - q1

        try:
            skewness = float(clean_series.skew())
            kurtosis = float(clean_series.kurtosis())
        except Exception:
            skewness = 0.0
            kurtosis = 0.0

        # Zero analysis
        zero_count = int((clean_series == 0).sum())
        zero_percentage = (zero_count / count * 100) if count > 0 else 0.0

        # Negative analysis
        negative_count = int((clean_series < 0).sum())
        negative_percentage = (negative_count / count * 100) if count > 0 else 0.0

        # Outlier analysis (IQR method)
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_mask = (clean_series < lower_bound) | (clean_series > upper_bound)
        outlier_count_iqr = int(outlier_mask.sum())
        outlier_percentage = (outlier_count_iqr / count * 100) if count > 0 else 0.0

        # Percentiles
        percentiles = {
            "p1": float(clean_series.quantile(0.01)),
            "p5": float(clean_series.quantile(0.05)),
            "p10": float(clean_series.quantile(0.10)),
            "p25": float(q1),
            "p50": float(median),
            "p75": float(q3),
            "p90": float(clean_series.quantile(0.90)),
            "p95": float(clean_series.quantile(0.95)),
            "p99": float(clean_series.quantile(0.99))
        }

        return DistributionAnalysis(
            column_name=column_name,
            count=count,
            mean=mean,
            std=std,
            min_value=min_value,
            max_value=max_value,
            median=median,
            q1=q1,
            q3=q3,
            iqr=iqr,
            skewness=skewness,
            kurtosis=kurtosis,
            zero_count=zero_count,
            zero_percentage=zero_percentage,
            negative_count=negative_count,
            negative_percentage=negative_percentage,
            outlier_count_iqr=outlier_count_iqr,
            outlier_percentage=outlier_percentage,
            percentiles=percentiles
        )

    @staticmethod
    def count_zero_inflation_recommendations(
        recommendations: Iterable["TransformationRecommendation"],
    ) -> int:
        return sum(
            1
            for rec in recommendations
            if rec is not None
            and rec.recommended_transform == DistributionTransformationType.ZERO_INFLATION_HANDLING
        )

    def recommend_transformation(
        self,
        analysis: DistributionAnalysis
    ) -> TransformationRecommendation:
        """
        Recommend transformation strategy based on distribution analysis.

        Parameters
        ----------
        analysis : DistributionAnalysis
            Distribution analysis results

        Returns
        -------
        TransformationRecommendation
            Recommended transformation with rationale
        """
        warnings = []
        alternatives = []

        if (
            not analysis.has_zero_inflation
            and analysis.zero_percentage > ZERO_INFLATION_CONSIDER_THRESHOLD
        ):
            logger.info(
                "zero_inflation skipped %s: zero_pct=%.2f < threshold=%.2f",
                analysis.column_name,
                analysis.zero_percentage,
                self.ZERO_INFLATION_THRESHOLD,
            )

        # Decision tree for transformation recommendation
        if analysis.has_zero_inflation and analysis.is_highly_skewed:
            # Zero-inflated and highly skewed
            recommended = DistributionTransformationType.ZERO_INFLATION_HANDLING
            reason = f"Zero-inflation ({analysis.zero_percentage:.1f}%) combined with high skewness ({analysis.skewness:.2f})"
            priority = "high"
            parameters = {
                "strategy": "separate_indicator",
                "transform_non_zero": "log"
            }
            alternatives = [DistributionTransformationType.CAP_THEN_LOG]
            warnings.append("Consider creating a binary indicator for zeros plus log transform of non-zero values")

        elif analysis.has_zero_inflation:
            # Zero-inflated but not highly skewed
            recommended = DistributionTransformationType.ZERO_INFLATION_HANDLING
            reason = f"Significant zero-inflation ({analysis.zero_percentage:.1f}%)"
            priority = "medium"
            parameters = {"strategy": "binary_indicator"}
            alternatives = [DistributionTransformationType.SQRT_TRANSFORM]
            warnings.append("Many zero values may indicate a mixture distribution")

        elif analysis.negative_count > 0 and analysis.is_highly_skewed:
            # Has negatives and highly skewed - use Yeo-Johnson
            recommended = DistributionTransformationType.YERO_JOHNSON
            reason = f"High skewness ({analysis.skewness:.2f}) with negative values present"
            priority = "high"
            parameters = {}
            alternatives = [DistributionTransformationType.CAP_OUTLIERS]
            warnings.append("Yeo-Johnson handles negative values unlike log/sqrt")

        elif analysis.is_highly_skewed and analysis.outlier_percentage > self.OUTLIER_THRESHOLD:
            # Highly skewed with many outliers
            recommended = DistributionTransformationType.CAP_THEN_LOG
            reason = f"High skewness ({analysis.skewness:.2f}) with significant outliers ({analysis.outlier_percentage:.1f}%)"
            priority = "high"
            parameters = {
                "cap_method": "iqr",
                "cap_multiplier": 1.5
            }
            alternatives = [DistributionTransformationType.LOG_TRANSFORM, DistributionTransformationType.BOX_COX]

        elif analysis.is_highly_skewed:
            # Highly skewed without major outliers
            if analysis.min_value > 0:
                recommended = DistributionTransformationType.LOG_TRANSFORM
                reason = f"High positive skewness ({analysis.skewness:.2f}) with all positive values"
                priority = "high"
                parameters = {"base": "natural", "offset": 0}
                alternatives = [DistributionTransformationType.BOX_COX, DistributionTransformationType.SQRT_TRANSFORM]
            else:
                recommended = DistributionTransformationType.YERO_JOHNSON
                reason = f"High skewness ({analysis.skewness:.2f}) with non-positive values"
                priority = "high"
                parameters = {}
                alternatives = [DistributionTransformationType.BOX_COX]

        elif analysis.is_moderately_skewed:
            # Moderately skewed
            if analysis.min_value >= 0:
                recommended = DistributionTransformationType.SQRT_TRANSFORM
                reason = f"Moderate skewness ({analysis.skewness:.2f})"
                priority = "medium"
                parameters = {}
                alternatives = [DistributionTransformationType.LOG_TRANSFORM]
            else:
                recommended = DistributionTransformationType.YERO_JOHNSON
                reason = f"Moderate skewness ({analysis.skewness:.2f}) with negative values"
                priority = "medium"
                parameters = {}
                alternatives = []

        elif analysis.outlier_percentage > self.OUTLIER_THRESHOLD:
            # Not skewed but has outliers
            recommended = DistributionTransformationType.CAP_OUTLIERS
            reason = f"Significant outliers ({analysis.outlier_percentage:.1f}%) despite low skewness"
            priority = "medium"
            parameters = {
                "method": "iqr",
                "multiplier": 1.5
            }
            alternatives = []
            warnings.append("Consider investigating outlier causes before capping")

        else:
            # Distribution is relatively normal
            recommended = DistributionTransformationType.NONE
            reason = f"Distribution is approximately normal (skewness: {analysis.skewness:.2f})"
            priority = "low"
            parameters = {}
            alternatives = []

        return TransformationRecommendation(
            column_name=analysis.column_name,
            recommended_transform=recommended,
            reason=reason,
            priority=priority,
            parameters=parameters,
            alternative_transforms=alternatives,
            warnings=warnings
        )

    def analyze_dataframe(
        self, df: pd.DataFrame, numeric_columns: Optional[List[str]] = None
    ) -> Dict[str, DistributionAnalysis]:
        if numeric_columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        valid_cols = [c for c in numeric_columns if c in df.columns]
        if not valid_cols:
            return {}

        if _is_spark_pandas(df):
            return self._analyze_dataframe_bulk(df, valid_cols)

        numeric_df = df[valid_cols]
        desc = to_pandas(numeric_df.describe())
        extra_q = to_pandas(numeric_df.quantile([0.01, 0.05, 0.10, 0.90, 0.95, 0.99]))
        try:
            skew_vals = to_pandas(numeric_df.skew())
        except Exception:
            skew_vals = native_pd.Series(0.0, index=valid_cols)
        try:
            kurt_vals = to_pandas(numeric_df.kurtosis())
        except Exception:
            kurt_vals = native_pd.Series(0.0, index=valid_cols)
        zero_counts = to_pandas((numeric_df == 0).sum())
        neg_counts = to_pandas((numeric_df < 0).sum())

        # Vectorized outlier counts across all columns at once
        q1_all = desc.loc['25%']
        q3_all = desc.loc['75%']
        iqr_all = q3_all - q1_all
        lower_bounds = q1_all - 1.5 * iqr_all
        upper_bounds = q3_all + 1.5 * iqr_all
        outlier_counts = ((numeric_df < lower_bounds) | (numeric_df > upper_bounds)).sum()

        results = {}
        for col in valid_cols:
            count = int(desc.loc['count', col])
            if count == 0:
                results[col] = self._empty_distribution(col)
                continue

            q1 = float(desc.loc['25%', col])
            q3 = float(desc.loc['75%', col])
            iqr = q3 - q1
            zero_c = int(zero_counts[col])
            neg_c = int(neg_counts[col])
            outlier_c = int(outlier_counts[col])

            results[col] = DistributionAnalysis(
                column_name=col, count=count,
                mean=float(desc.loc['mean', col]), std=float(desc.loc['std', col]),
                min_value=float(desc.loc['min', col]), max_value=float(desc.loc['max', col]),
                median=float(desc.loc['50%', col]), q1=q1, q3=q3, iqr=iqr,
                skewness=float(skew_vals[col]), kurtosis=float(kurt_vals[col]),
                zero_count=zero_c, zero_percentage=zero_c / count * 100,
                negative_count=neg_c, negative_percentage=neg_c / count * 100,
                outlier_count_iqr=outlier_c, outlier_percentage=outlier_c / count * 100,
                percentiles={
                    "p1": float(extra_q.loc[0.01, col]), "p5": float(extra_q.loc[0.05, col]),
                    "p10": float(extra_q.loc[0.10, col]), "p25": q1, "p50": float(desc.loc['50%', col]),
                    "p75": q3, "p90": float(extra_q.loc[0.90, col]),
                    "p95": float(extra_q.loc[0.95, col]), "p99": float(extra_q.loc[0.99, col]),
                },
            )
        return results

    def _empty_distribution(self, column_name: str) -> DistributionAnalysis:
        return DistributionAnalysis(
            column_name=column_name, count=0, mean=0.0, std=0.0,
            min_value=0.0, max_value=0.0, median=0.0, q1=0.0, q3=0.0, iqr=0.0,
            skewness=0.0, kurtosis=0.0, zero_count=0, zero_percentage=0.0,
            negative_count=0, negative_percentage=0.0,
            outlier_count_iqr=0, outlier_percentage=0.0,
        )

    def _analyze_dataframe_bulk(
        self, df: pd.DataFrame, valid_cols: List[str],
    ) -> Dict[str, DistributionAnalysis]:
        """Analyze distributions using batched Spark aggregation."""
        bulk = bulk_distribution_stats(df, valid_cols)
        results: Dict[str, DistributionAnalysis] = {}
        for col in valid_cols:
            br = bulk.get(col)
            if br is None or br.non_null_count == 0:
                results[col] = self._empty_distribution(col)
                continue
            count = br.non_null_count
            q1 = br.q1 or 0.0
            q3 = br.q3 or 0.0
            iqr = q3 - q1
            zero_c = br.zero_count
            neg_c = br.negative_count
            outlier_c = br.outlier_count_iqr
            results[col] = DistributionAnalysis(
                column_name=col,
                count=count,
                mean=br.mean or 0.0,
                std=br.std or 0.0,
                min_value=br.min_val or 0.0,
                max_value=br.max_val or 0.0,
                median=br.median or 0.0,
                q1=q1,
                q3=q3,
                iqr=iqr,
                skewness=br.skewness or 0.0,
                kurtosis=br.kurtosis or 0.0,
                zero_count=zero_c,
                zero_percentage=(zero_c / count * 100) if count > 0 else 0.0,
                negative_count=neg_c,
                negative_percentage=(neg_c / count * 100) if count > 0 else 0.0,
                outlier_count_iqr=outlier_c,
                outlier_percentage=(outlier_c / count * 100) if count > 0 else 0.0,
                percentiles=br.percentiles,
            )
        return results

    def get_all_recommendations(
        self,
        df: pd.DataFrame,
        numeric_columns: Optional[List[str]] = None
    ) -> List[TransformationRecommendation]:
        """
        Get transformation recommendations for all numeric columns.

        Parameters
        ----------
        df : DataFrame
            Data to analyze
        numeric_columns : List[str], optional
            Columns to analyze. If None, analyzes all numeric columns.

        Returns
        -------
        List[TransformationRecommendation]
            Recommendations sorted by priority
        """
        analyses = self.analyze_dataframe(df, numeric_columns)
        recommendations = []

        for col_name, analysis in analyses.items():
            rec = self.recommend_transformation(analysis)
            if rec.recommended_transform != DistributionTransformationType.NONE:
                recommendations.append(rec)

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda r: priority_order.get(r.priority, 3))

        return recommendations

    def generate_report(
        self,
        df: pd.DataFrame,
        numeric_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive distribution analysis report.

        Parameters
        ----------
        df : DataFrame
            Data to analyze
        numeric_columns : List[str], optional
            Columns to analyze

        Returns
        -------
        Dict[str, Any]
            Comprehensive report with analyses and recommendations
        """
        analyses = self.analyze_dataframe(df, numeric_columns)
        recommendations = self.get_all_recommendations(df, numeric_columns)

        # Categorize columns by skewness
        highly_skewed = []
        moderately_skewed = []
        normal = []
        zero_inflated = []

        for col_name, analysis in analyses.items():
            if analysis.has_zero_inflation:
                zero_inflated.append(col_name)
            if analysis.is_highly_skewed:
                highly_skewed.append(col_name)
            elif analysis.is_moderately_skewed:
                moderately_skewed.append(col_name)
            else:
                normal.append(col_name)

        return {
            "summary": {
                "total_columns": len(analyses),
                "highly_skewed_count": len(highly_skewed),
                "moderately_skewed_count": len(moderately_skewed),
                "normal_count": len(normal),
                "zero_inflated_count": len(zero_inflated)
            },
            "categories": {
                "highly_skewed": highly_skewed,
                "moderately_skewed": moderately_skewed,
                "approximately_normal": normal,
                "zero_inflated": zero_inflated
            },
            "analyses": {k: v.to_dict() for k, v in analyses.items()},
            "recommendations": [r.to_dict() for r in recommendations]
        }
