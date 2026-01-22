"""Tests for CategoricalDistributionAnalyzer."""

import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.profiling.categorical_distribution import (
    CategoricalDistributionAnalysis,
    CategoricalDistributionAnalyzer,
    EncodingRecommendation,
    EncodingType,
)


class TestCategoricalDistributionAnalysis:
    """Tests for CategoricalDistributionAnalysis dataclass."""

    def test_is_imbalanced_with_high_ratio(self):
        analysis = CategoricalDistributionAnalysis(
            column_name="test",
            category_count=5,
            total_count=1000,
            imbalance_ratio=15.0,
            entropy=1.5,
            normalized_entropy=0.65,
            top1_concentration=50.0,
            top3_concentration=80.0,
            rare_category_count=2,
            rare_category_names=["cat_a", "cat_b"],
            value_counts={"A": 500, "B": 300, "C": 150, "D": 40, "E": 10},
        )
        assert analysis.is_imbalanced is True

    def test_is_imbalanced_with_low_ratio(self):
        analysis = CategoricalDistributionAnalysis(
            column_name="test",
            category_count=3,
            total_count=900,
            imbalance_ratio=1.5,
            entropy=1.58,
            normalized_entropy=0.99,
            top1_concentration=35.0,
            top3_concentration=100.0,
            rare_category_count=0,
            rare_category_names=[],
            value_counts={"A": 300, "B": 300, "C": 300},
        )
        assert analysis.is_imbalanced is False

    def test_has_low_diversity_with_low_entropy(self):
        analysis = CategoricalDistributionAnalysis(
            column_name="test",
            category_count=5,
            total_count=1000,
            imbalance_ratio=100.0,
            entropy=0.5,
            normalized_entropy=0.22,
            top1_concentration=95.0,
            top3_concentration=99.0,
            rare_category_count=4,
            rare_category_names=["B", "C", "D", "E"],
            value_counts={"A": 950, "B": 25, "C": 15, "D": 8, "E": 2},
        )
        assert analysis.has_low_diversity is True

    def test_has_rare_categories(self):
        analysis = CategoricalDistributionAnalysis(
            column_name="test",
            category_count=10,
            total_count=1000,
            imbalance_ratio=50.0,
            entropy=2.0,
            normalized_entropy=0.60,
            top1_concentration=40.0,
            top3_concentration=70.0,
            rare_category_count=3,
            rare_category_names=["H", "I", "J"],
            value_counts={},
        )
        assert analysis.has_rare_categories is True


class TestCategoricalDistributionAnalyzer:
    """Tests for CategoricalDistributionAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        return CategoricalDistributionAnalyzer()

    @pytest.fixture
    def balanced_series(self):
        return pd.Series(["A", "B", "C"] * 100)

    @pytest.fixture
    def imbalanced_series(self):
        return pd.Series(["A"] * 900 + ["B"] * 80 + ["C"] * 15 + ["D"] * 5)

    @pytest.fixture
    def high_cardinality_series(self):
        return pd.Series([f"cat_{i}" for i in range(100)] * 10)

    def test_analyze_balanced_distribution(self, analyzer, balanced_series):
        result = analyzer.analyze(balanced_series, "balanced_col")
        assert result.column_name == "balanced_col"
        assert result.category_count == 3
        assert result.total_count == 300
        assert result.imbalance_ratio == 1.0
        assert result.normalized_entropy > 0.95

    def test_analyze_imbalanced_distribution(self, analyzer, imbalanced_series):
        result = analyzer.analyze(imbalanced_series, "imbalanced_col")
        assert result.category_count == 4
        assert result.imbalance_ratio == 180.0  # 900 / 5
        assert result.top1_concentration == 90.0
        assert result.is_imbalanced is True

    def test_analyze_calculates_entropy_correctly(self, analyzer, balanced_series):
        result = analyzer.analyze(balanced_series, "test")
        expected_entropy = np.log2(3)  # Max entropy for 3 equal categories
        assert abs(result.entropy - expected_entropy) < 0.01

    def test_analyze_detects_rare_categories(self, analyzer, imbalanced_series):
        result = analyzer.analyze(imbalanced_series, "test")
        assert result.rare_category_count >= 1  # D is rare (<1% = <10 out of 1000)
        assert "D" in result.rare_category_names

    def test_analyze_handles_nulls(self, analyzer):
        series = pd.Series(["A", "B", "C", None, "A", None])
        result = analyzer.analyze(series, "with_nulls")
        assert result.total_count == 4  # Excludes nulls
        assert result.category_count == 3

    def test_analyze_empty_series(self, analyzer):
        series = pd.Series([], dtype=str)
        result = analyzer.analyze(series, "empty")
        assert result.category_count == 0
        assert result.entropy == 0.0

    def test_recommend_encoding_low_cardinality(self, analyzer, balanced_series):
        analysis = analyzer.analyze(balanced_series, "test")
        rec = analyzer.recommend_encoding(analysis)
        assert rec.encoding_type == EncodingType.ONE_HOT
        assert rec.priority == "low"

    def test_recommend_encoding_high_cardinality(self, analyzer, high_cardinality_series):
        analysis = analyzer.analyze(high_cardinality_series, "test")
        rec = analyzer.recommend_encoding(analysis)
        assert rec.encoding_type in [EncodingType.TARGET, EncodingType.FREQUENCY]
        assert "cardinality" in rec.reason.lower()

    def test_recommend_encoding_with_rare_categories(self, analyzer, imbalanced_series):
        analysis = analyzer.analyze(imbalanced_series, "test")
        rec = analyzer.recommend_encoding(analysis)
        assert len(rec.preprocessing_steps) > 0
        assert any("rare" in step.lower() or "group" in step.lower()
                   for step in rec.preprocessing_steps)

    def test_recommend_encoding_cyclical(self, analyzer):
        days = pd.Series(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"] * 100)
        analysis = analyzer.analyze(days, "day_of_week")
        rec = analyzer.recommend_encoding(analysis, is_cyclical=True)
        assert rec.encoding_type == EncodingType.CYCLICAL

    def test_analyze_dataframe(self, analyzer):
        df = pd.DataFrame({
            "cat1": ["A", "B", "C"] * 100,
            "cat2": ["X"] * 250 + ["Y"] * 50,
            "num": range(300),
        })
        results = analyzer.analyze_dataframe(df, ["cat1", "cat2"])
        assert len(results) == 2
        assert "cat1" in results
        assert "cat2" in results
        assert results["cat1"].category_count == 3
        assert results["cat2"].imbalance_ratio == 5.0

    def test_get_all_recommendations(self, analyzer):
        df = pd.DataFrame({
            "low_card": ["A", "B", "C"] * 100,
            "high_card": [f"cat_{i}" for i in range(50)] * 6,
        })
        recs = analyzer.get_all_recommendations(df, ["low_card", "high_card"])
        assert len(recs) == 2
        assert recs[0].column_name == "low_card" or recs[1].column_name == "low_card"


class TestEncodingRecommendation:
    """Tests for EncodingRecommendation dataclass."""

    def test_to_dict(self):
        rec = EncodingRecommendation(
            column_name="test_col",
            encoding_type=EncodingType.TARGET,
            reason="High cardinality (50 categories)",
            priority="high",
            preprocessing_steps=["Group rare categories into 'Other'"],
            warnings=["May need regularization"],
        )
        d = rec.to_dict()
        assert d["column"] == "test_col"
        assert d["encoding"] == "target"
        assert d["priority"] == "high"
        assert len(d["preprocessing_steps"]) == 1
