"""Categorical feature analysis with respect to a binary target."""
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from customer_retention.core.compat import DataFrame, to_pandas


@dataclass
class CategoricalTargetResult:
    """Results from categorical-target association analysis."""
    categorical_col: str
    target_col: str
    n_categories: int
    cramers_v: float
    chi2_statistic: float
    p_value: float
    effect_strength: str  # 'weak', 'moderate', 'strong'
    category_stats: pd.DataFrame  # category, count, retention_rate, lift, pct_of_total
    high_risk_categories: List[str]  # categories with lift < threshold
    low_risk_categories: List[str]  # categories with lift > threshold
    overall_rate: float


class CategoricalTargetAnalyzer:
    """Analyzes association between categorical features and binary target.

    Computes:
    - Cramér's V: Measure of association strength (0-1)
    - Retention rate by category
    - Lift: Category rate relative to overall rate
    - High/low risk category identification
    """

    EFFECT_THRESHOLDS = {
        'weak': 0.1,
        'moderate': 0.3,
        'strong': 0.5
    }

    HIGH_RISK_LIFT_THRESHOLD = 0.9
    LOW_RISK_LIFT_THRESHOLD = 1.1

    def __init__(self, min_samples_per_category: int = 10):
        self.min_samples_per_category = min_samples_per_category

    def analyze(
        self,
        df: DataFrame,
        categorical_col: str,
        target_col: str
    ) -> CategoricalTargetResult:
        """Analyze relationship between categorical feature and binary target."""
        df = to_pandas(df)

        if len(df) == 0 or categorical_col not in df.columns or target_col not in df.columns:
            return self._empty_result(categorical_col, target_col)

        # Remove rows with missing values in either column
        clean_df = df[[categorical_col, target_col]].dropna()

        if len(clean_df) == 0:
            return self._empty_result(categorical_col, target_col)

        # Calculate overall retention rate
        overall_rate = clean_df[target_col].mean()

        # Calculate category statistics
        category_stats = self._calculate_category_stats(clean_df, categorical_col, target_col, overall_rate)

        # Calculate Cramér's V
        cramers_v, chi2_stat, p_value = self._calculate_cramers_v(clean_df, categorical_col, target_col)

        # Determine effect strength
        effect_strength = self._determine_effect_strength(cramers_v)

        # Identify high/low risk categories
        high_risk = category_stats[category_stats['lift'] < self.HIGH_RISK_LIFT_THRESHOLD]['category'].tolist()
        low_risk = category_stats[category_stats['lift'] > self.LOW_RISK_LIFT_THRESHOLD]['category'].tolist()

        return CategoricalTargetResult(
            categorical_col=categorical_col,
            target_col=target_col,
            n_categories=len(category_stats),
            cramers_v=cramers_v,
            chi2_statistic=chi2_stat,
            p_value=p_value,
            effect_strength=effect_strength,
            category_stats=category_stats,
            high_risk_categories=high_risk,
            low_risk_categories=low_risk,
            overall_rate=overall_rate
        )

    def _calculate_category_stats(
        self,
        df: pd.DataFrame,
        categorical_col: str,
        target_col: str,
        overall_rate: float
    ) -> pd.DataFrame:
        """Calculate retention rate, lift, and counts by category."""
        stats = df.groupby(categorical_col)[target_col].agg(['sum', 'count', 'mean']).reset_index()
        stats.columns = ['category', 'retained_count', 'total_count', 'retention_rate']
        stats['churned_count'] = stats['total_count'] - stats['retained_count']
        stats['lift'] = stats['retention_rate'] / overall_rate if overall_rate > 0 else 0
        stats['pct_of_total'] = stats['total_count'] / len(df)

        # Filter out small categories
        stats = stats[stats['total_count'] >= self.min_samples_per_category]

        return stats.sort_values('retention_rate', ascending=False).reset_index(drop=True)

    def _calculate_cramers_v(
        self,
        df: pd.DataFrame,
        categorical_col: str,
        target_col: str
    ) -> tuple:
        """Calculate Cramér's V statistic."""
        contingency = pd.crosstab(df[categorical_col], df[target_col])

        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            return 0.0, 0.0, 1.0

        try:
            chi2, p_value, dof, expected = chi2_contingency(contingency)
            n = contingency.sum().sum()
            min_dim = min(contingency.shape) - 1

            if min_dim == 0 or n == 0:
                return 0.0, chi2, p_value

            cramers_v = np.sqrt(chi2 / (n * min_dim))
            return float(cramers_v), float(chi2), float(p_value)
        except Exception:
            return 0.0, 0.0, 1.0

    def _determine_effect_strength(self, cramers_v: float) -> str:
        """Determine effect strength category based on Cramér's V."""
        if cramers_v >= self.EFFECT_THRESHOLDS['strong']:
            return 'strong'
        elif cramers_v >= self.EFFECT_THRESHOLDS['moderate']:
            return 'moderate'
        elif cramers_v >= self.EFFECT_THRESHOLDS['weak']:
            return 'weak'
        else:
            return 'negligible'

    def _empty_result(self, categorical_col: str, target_col: str) -> CategoricalTargetResult:
        """Return empty result for edge cases."""
        return CategoricalTargetResult(
            categorical_col=categorical_col,
            target_col=target_col,
            n_categories=0,
            cramers_v=0.0,
            chi2_statistic=0.0,
            p_value=1.0,
            effect_strength='negligible',
            category_stats=pd.DataFrame(columns=[
                'category', 'retained_count', 'total_count', 'retention_rate',
                'churned_count', 'lift', 'pct_of_total'
            ]),
            high_risk_categories=[],
            low_risk_categories=[],
            overall_rate=0.0
        )

    def analyze_multiple(
        self,
        df: DataFrame,
        categorical_cols: List[str],
        target_col: str
    ) -> pd.DataFrame:
        """Analyze multiple categorical columns and return summary."""
        results = []
        for col in categorical_cols:
            result = self.analyze(df, col, target_col)
            results.append({
                'feature': col,
                'n_categories': result.n_categories,
                'cramers_v': result.cramers_v,
                'p_value': result.p_value,
                'effect_strength': result.effect_strength,
                'high_risk_count': len(result.high_risk_categories),
                'low_risk_count': len(result.low_risk_categories)
            })

        return pd.DataFrame(results).sort_values('cramers_v', ascending=False).reset_index(drop=True)
