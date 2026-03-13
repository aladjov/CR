"""Tests for AttentionScorer — column attention scoring logic."""

from __future__ import annotations

from unittest.mock import MagicMock

from customer_retention.analysis.visualization.attention_scorer import (
    AttentionScore,
    AttentionScorer,
)

# ---------------------------------------------------------------------------
# Lightweight stubs so tests don't depend on profiling internals
# ---------------------------------------------------------------------------


def _make_numeric(
    column_name: str = "col",
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    outlier_percentage: float = 0.0,
    zero_percentage: float = 0.0,
) -> MagicMock:
    m = MagicMock()
    m.column_name = column_name
    m.skewness = skewness
    m.kurtosis = kurtosis
    m.outlier_percentage = outlier_percentage
    m.zero_percentage = zero_percentage
    return m


def _make_categorical(
    column_name: str = "cat_col",
    imbalance_ratio: float = 1.0,
    rare_category_count: int = 0,
    normalized_entropy: float = 0.8,
    top3_concentration: float = 50.0,
) -> MagicMock:
    m = MagicMock()
    m.column_name = column_name
    m.imbalance_ratio = imbalance_ratio
    m.rare_category_count = rare_category_count
    m.normalized_entropy = normalized_entropy
    m.top3_concentration = top3_concentration
    return m


def _make_rec(priority: str = "low") -> MagicMock:
    m = MagicMock()
    m.priority = priority
    return m


# ===================================================================
# AttentionScore dataclass
# ===================================================================


class TestAttentionScore:
    def test_score_clamped_to_0(self):
        s = AttentionScore(column_name="c", column_type="numeric", score=-10)
        assert s.score == 0.0

    def test_score_clamped_to_100(self):
        s = AttentionScore(column_name="c", column_type="numeric", score=200)
        assert s.score == 100.0

    def test_priority_high(self):
        s = AttentionScore(column_name="c", column_type="numeric", score=70)
        assert s.priority == "high"

    def test_priority_medium(self):
        s = AttentionScore(column_name="c", column_type="numeric", score=40)
        assert s.priority == "medium"

    def test_priority_low(self):
        s = AttentionScore(column_name="c", column_type="numeric", score=39)
        assert s.priority == "low"

    def test_priority_boundary_70(self):
        assert AttentionScore("c", "n", 70).priority == "high"
        assert AttentionScore("c", "n", 69.9).priority == "medium"

    def test_priority_boundary_40(self):
        assert AttentionScore("c", "n", 40).priority == "medium"
        assert AttentionScore("c", "n", 39.9).priority == "low"

    def test_reasons_default_empty(self):
        s = AttentionScore(column_name="c", column_type="numeric", score=50)
        assert s.reasons == []


# ===================================================================
# Numeric scoring
# ===================================================================


class TestScoreNumeric:
    def test_zero_contribution_gives_zero(self):
        analysis = _make_numeric(skewness=0.0, kurtosis=3.0, outlier_percentage=0.0, zero_percentage=0.0)
        result = AttentionScorer.score_numeric(analysis)
        assert result.score == 0.0
        assert result.priority == "low"

    def test_high_skewness(self):
        analysis = _make_numeric(skewness=5.0)
        result = AttentionScorer.score_numeric(analysis)
        # 5*10 = 50, capped at 30
        assert result.score == 30.0
        assert any("skewness" in r for r in result.reasons)

    def test_negative_skewness(self):
        analysis = _make_numeric(skewness=-3.0)
        result = AttentionScorer.score_numeric(analysis)
        assert result.score == min(3.0 * 10, 30)

    def test_high_outliers(self):
        analysis = _make_numeric(outlier_percentage=20.0)
        result = AttentionScorer.score_numeric(analysis)
        # 20*2 = 40, capped at 25
        assert result.score == 25.0
        assert any("outlier" in r for r in result.reasons)

    def test_zero_inflation_below_threshold(self):
        analysis = _make_numeric(zero_percentage=15.0)
        result = AttentionScorer.score_numeric(analysis)
        # Below 20% threshold — no zero inflation contribution
        assert result.score == 0.0

    def test_zero_inflation_above_threshold(self):
        analysis = _make_numeric(zero_percentage=50.0)
        result = AttentionScorer.score_numeric(analysis)
        # 50*0.4 = 20, capped at 15
        assert result.score == 15.0
        assert any("zero" in r for r in result.reasons)

    def test_high_kurtosis(self):
        analysis = _make_numeric(kurtosis=13.0)
        result = AttentionScorer.score_numeric(analysis)
        # excess = 10, 10*2 = 20, capped at 15
        assert result.score == 15.0
        assert any("kurtosis" in r for r in result.reasons)

    def test_kurtosis_at_baseline(self):
        """Kurtosis == 3 (normal) should contribute nothing."""
        analysis = _make_numeric(kurtosis=3.0)
        result = AttentionScorer.score_numeric(analysis)
        assert result.score == 0.0

    def test_high_priority_transform_boost(self):
        analysis = _make_numeric()
        rec = _make_rec(priority="high")
        result = AttentionScorer.score_numeric(analysis, rec)
        assert result.score == 15.0
        assert any("high-priority" in r for r in result.reasons)

    def test_medium_priority_transform_boost(self):
        analysis = _make_numeric()
        rec = _make_rec(priority="medium")
        result = AttentionScorer.score_numeric(analysis, rec)
        assert result.score == 8.0

    def test_low_priority_no_boost(self):
        analysis = _make_numeric()
        rec = _make_rec(priority="low")
        result = AttentionScorer.score_numeric(analysis, rec)
        assert result.score == 0.0

    def test_none_recommendation(self):
        analysis = _make_numeric(skewness=2.0)
        result = AttentionScorer.score_numeric(analysis, None)
        assert result.score == min(20, 30)

    def test_combined_high_score(self):
        analysis = _make_numeric(skewness=5, kurtosis=13, outlier_percentage=20, zero_percentage=50)
        rec = _make_rec(priority="high")
        result = AttentionScorer.score_numeric(analysis, rec)
        # 30 + 25 + 15 + 15 + 15 = 100
        assert result.score == 100.0
        assert result.priority == "high"

    def test_column_name_propagated(self):
        analysis = _make_numeric(column_name="my_col")
        result = AttentionScorer.score_numeric(analysis)
        assert result.column_name == "my_col"

    def test_column_type_is_numeric(self):
        analysis = _make_numeric()
        result = AttentionScorer.score_numeric(analysis)
        assert result.column_type == "numeric"


# ===================================================================
# Categorical scoring
# ===================================================================


class TestScoreCategorical:
    def test_zero_contribution_gives_zero(self):
        analysis = _make_categorical()
        result = AttentionScorer.score_categorical(analysis)
        assert result.score == 0.0

    def test_imbalance_below_threshold(self):
        analysis = _make_categorical(imbalance_ratio=4.0)
        result = AttentionScorer.score_categorical(analysis)
        # ratio <= 5 — no imbalance contribution
        assert result.score == 0.0

    def test_imbalance_above_threshold(self):
        analysis = _make_categorical(imbalance_ratio=100.0)
        result = AttentionScorer.score_categorical(analysis)
        # log10(100) = 2, 2*15 = 30
        assert result.score == 30.0
        assert any("imbalance" in r for r in result.reasons)

    def test_rare_categories(self):
        analysis = _make_categorical(rare_category_count=5)
        result = AttentionScorer.score_categorical(analysis)
        # 5*3 = 15
        assert result.score == 15.0
        assert any("rare" in r for r in result.reasons)

    def test_rare_categories_capped(self):
        analysis = _make_categorical(rare_category_count=100)
        result = AttentionScorer.score_categorical(analysis)
        # 100*3 = 300, capped at 20
        assert result.score == 20.0

    def test_low_diversity(self):
        analysis = _make_categorical(normalized_entropy=0.2)
        result = AttentionScorer.score_categorical(analysis)
        assert result.score == 20.0
        assert any("diversity" in r for r in result.reasons)

    def test_normal_diversity_no_contribution(self):
        analysis = _make_categorical(normalized_entropy=0.3)
        result = AttentionScorer.score_categorical(analysis)
        # exactly 0.3 is NOT < 0.3
        assert result.score == 0.0

    def test_high_concentration(self):
        analysis = _make_categorical(top3_concentration=95.0)
        result = AttentionScorer.score_categorical(analysis)
        assert result.score == 15.0
        assert any("concentration" in r for r in result.reasons)

    def test_concentration_at_boundary(self):
        analysis = _make_categorical(top3_concentration=90.0)
        result = AttentionScorer.score_categorical(analysis)
        # exactly 90 — NOT > 90
        assert result.score == 0.0

    def test_high_priority_encoding_boost(self):
        analysis = _make_categorical()
        rec = _make_rec(priority="high")
        result = AttentionScorer.score_categorical(analysis, rec)
        assert result.score == 15.0

    def test_medium_priority_encoding_boost(self):
        analysis = _make_categorical()
        rec = _make_rec(priority="medium")
        result = AttentionScorer.score_categorical(analysis, rec)
        assert result.score == 8.0

    def test_none_recommendation(self):
        analysis = _make_categorical()
        result = AttentionScorer.score_categorical(analysis, None)
        assert result.score == 0.0

    def test_column_type_is_categorical(self):
        analysis = _make_categorical()
        result = AttentionScorer.score_categorical(analysis)
        assert result.column_type == "categorical"

    def test_combined_high_score(self):
        analysis = _make_categorical(
            imbalance_ratio=100, rare_category_count=100,
            normalized_entropy=0.1, top3_concentration=95.0,
        )
        rec = _make_rec(priority="high")
        result = AttentionScorer.score_categorical(analysis, rec)
        # 30 + 20 + 20 + 15 + 15 = 100
        assert result.score == 100.0


# ===================================================================
# Datetime scoring
# ===================================================================


class TestScoreDatetime:
    def test_empty_summary(self):
        result = AttentionScorer.score_datetime({"column_name": "dt"})
        assert result.score == 0.0
        assert result.column_name == "dt"
        assert result.column_type == "datetime"

    def test_quality_issues(self):
        result = AttentionScorer.score_datetime({"column_name": "dt", "quality_issue_count": 2})
        assert result.score == 20.0
        assert any("quality" in r for r in result.reasons)

    def test_quality_issues_capped(self):
        result = AttentionScorer.score_datetime({"column_name": "dt", "quality_issue_count": 10})
        assert result.score == 30.0

    def test_seasonality(self):
        result = AttentionScorer.score_datetime({"column_name": "dt", "has_seasonality": True})
        assert result.score == 15.0

    def test_strong_trend(self):
        result = AttentionScorer.score_datetime({"column_name": "dt", "overall_growth_pct": 80})
        assert result.score == 15.0

    def test_negative_strong_trend(self):
        result = AttentionScorer.score_datetime({"column_name": "dt", "overall_growth_pct": -60})
        assert result.score == 15.0

    def test_weak_trend_no_contribution(self):
        result = AttentionScorer.score_datetime({"column_name": "dt", "overall_growth_pct": 30})
        assert result.score == 0.0

    def test_feature_recommendations(self):
        result = AttentionScorer.score_datetime({"column_name": "dt", "feature_rec_count": 3})
        assert result.score == 15.0

    def test_feature_recommendations_capped(self):
        result = AttentionScorer.score_datetime({"column_name": "dt", "feature_rec_count": 10})
        assert result.score == 25.0

    def test_modeling_recommendation(self):
        result = AttentionScorer.score_datetime({"column_name": "dt", "modeling_rec_count": 2})
        assert result.score == 15.0

    def test_missing_column_name(self):
        result = AttentionScorer.score_datetime({})
        assert result.column_name == "unknown"

    def test_combined_datetime(self):
        result = AttentionScorer.score_datetime({
            "column_name": "dt",
            "quality_issue_count": 5,
            "has_seasonality": True,
            "overall_growth_pct": 100,
            "feature_rec_count": 5,
            "modeling_rec_count": 1,
        })
        # 30 + 15 + 15 + 25 + 15 = 100
        assert result.score == 100.0
        assert result.priority == "high"
