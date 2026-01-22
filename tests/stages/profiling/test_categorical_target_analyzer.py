"""Tests for CategoricalTargetAnalyzer."""
import pytest
import numpy as np
import pandas as pd

from customer_retention.stages.profiling import CategoricalTargetAnalyzer


class TestCategoricalTargetAnalyzerInit:
    def test_default_init(self):
        analyzer = CategoricalTargetAnalyzer()
        assert analyzer is not None

    def test_custom_min_samples(self):
        analyzer = CategoricalTargetAnalyzer(min_samples_per_category=50)
        assert analyzer.min_samples_per_category == 50


class TestCramersV:
    @pytest.fixture
    def strong_association_data(self):
        """Data with strong association between category and target."""
        np.random.seed(42)
        n = 1000
        # Category A has 90% retention, Category B has 10% retention
        categories = np.array(['A'] * 500 + ['B'] * 500)
        targets = np.concatenate([
            np.random.choice([0, 1], 500, p=[0.1, 0.9]),  # A: 90% retention
            np.random.choice([0, 1], 500, p=[0.9, 0.1])   # B: 10% retention
        ])
        return pd.DataFrame({'category': categories, 'target': targets})

    @pytest.fixture
    def weak_association_data(self):
        """Data with weak/no association between category and target."""
        np.random.seed(42)
        n = 1000
        categories = np.random.choice(['A', 'B', 'C'], n)
        targets = np.random.choice([0, 1], n, p=[0.2, 0.8])  # Same rate for all
        return pd.DataFrame({'category': categories, 'target': targets})

    def test_cramers_v_strong_association(self, strong_association_data):
        analyzer = CategoricalTargetAnalyzer()
        result = analyzer.analyze(
            strong_association_data,
            categorical_col='category',
            target_col='target'
        )
        assert result.cramers_v >= 0.7  # Strong association

    def test_cramers_v_weak_association(self, weak_association_data):
        analyzer = CategoricalTargetAnalyzer()
        result = analyzer.analyze(
            weak_association_data,
            categorical_col='category',
            target_col='target'
        )
        assert result.cramers_v < 0.1  # Weak association

    def test_cramers_v_range(self, strong_association_data):
        analyzer = CategoricalTargetAnalyzer()
        result = analyzer.analyze(
            strong_association_data,
            categorical_col='category',
            target_col='target'
        )
        assert 0 <= result.cramers_v <= 1


class TestRetentionByCategory:
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        return pd.DataFrame({
            'city': ['NYC'] * 100 + ['LA'] * 100 + ['CHI'] * 100,
            'retained': [1] * 80 + [0] * 20 +  # NYC: 80% retention
                       [1] * 60 + [0] * 40 +   # LA: 60% retention
                       [1] * 90 + [0] * 10     # CHI: 90% retention
        })

    def test_category_stats_calculated(self, sample_data):
        analyzer = CategoricalTargetAnalyzer()
        result = analyzer.analyze(sample_data, 'city', 'retained')

        assert len(result.category_stats) == 3
        assert 'NYC' in result.category_stats['category'].values

    def test_retention_rates_correct(self, sample_data):
        analyzer = CategoricalTargetAnalyzer()
        result = analyzer.analyze(sample_data, 'city', 'retained')

        nyc_stats = result.category_stats[result.category_stats['category'] == 'NYC'].iloc[0]
        assert abs(nyc_stats['retention_rate'] - 0.80) < 0.01

    def test_lift_calculated(self, sample_data):
        analyzer = CategoricalTargetAnalyzer()
        result = analyzer.analyze(sample_data, 'city', 'retained')

        # Overall retention is (80+60+90)/300 = 76.67%
        chi_stats = result.category_stats[result.category_stats['category'] == 'CHI'].iloc[0]
        assert chi_stats['lift'] > 1.0  # CHI has above-average retention


class TestHighRiskCategories:
    @pytest.fixture
    def risk_data(self):
        return pd.DataFrame({
            'segment': ['Premium'] * 100 + ['Basic'] * 100,
            'retained': [1] * 95 + [0] * 5 +   # Premium: 95% retention
                       [1] * 50 + [0] * 50     # Basic: 50% retention
        })

    def test_identifies_high_risk(self, risk_data):
        analyzer = CategoricalTargetAnalyzer()
        result = analyzer.analyze(risk_data, 'segment', 'retained')

        assert len(result.high_risk_categories) >= 1
        assert 'Basic' in result.high_risk_categories

    def test_identifies_low_risk(self, risk_data):
        analyzer = CategoricalTargetAnalyzer()
        result = analyzer.analyze(risk_data, 'segment', 'retained')

        assert len(result.low_risk_categories) >= 1
        assert 'Premium' in result.low_risk_categories


class TestEffectStrength:
    def test_strong_effect_detected(self):
        np.random.seed(42)
        df = pd.DataFrame({
            'cat': ['A'] * 500 + ['B'] * 500,
            'target': [1] * 450 + [0] * 50 + [1] * 50 + [0] * 450
        })
        analyzer = CategoricalTargetAnalyzer()
        result = analyzer.analyze(df, 'cat', 'target')

        assert result.effect_strength == 'strong'

    def test_moderate_effect_detected(self):
        np.random.seed(42)
        # Create stronger effect: A=85% retention, B=45% retention
        df = pd.DataFrame({
            'cat': ['A'] * 500 + ['B'] * 500,
            'target': [1] * 425 + [0] * 75 + [1] * 225 + [0] * 275
        })
        analyzer = CategoricalTargetAnalyzer()
        result = analyzer.analyze(df, 'cat', 'target')

        assert result.effect_strength in ['moderate', 'strong']


class TestEdgeCases:
    def test_empty_dataframe(self):
        analyzer = CategoricalTargetAnalyzer()
        df = pd.DataFrame({'cat': [], 'target': []})
        result = analyzer.analyze(df, 'cat', 'target')

        assert result.cramers_v == 0
        assert len(result.category_stats) == 0

    def test_single_category(self):
        analyzer = CategoricalTargetAnalyzer()
        df = pd.DataFrame({
            'cat': ['A'] * 100,
            'target': [1] * 80 + [0] * 20
        })
        result = analyzer.analyze(df, 'cat', 'target')

        assert result.cramers_v == 0  # No variation in category
        assert len(result.category_stats) == 1

    def test_missing_values_handled(self):
        # Use min_samples=1 to allow small categories
        analyzer = CategoricalTargetAnalyzer(min_samples_per_category=1)
        df = pd.DataFrame({
            'cat': ['A', 'B', None, 'A', 'B'] * 10,
            'target': [1, 0, 1, 1, 0] * 10
        })
        result = analyzer.analyze(df, 'cat', 'target')

        assert len(result.category_stats) == 2  # Only A and B, None excluded
