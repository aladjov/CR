import pytest

from customer_retention.analysis.diagnostics.feature_stability import (
    FeatureStabilityAnalyzer,
    FeatureStabilityResult,
)


@pytest.fixture
def analyzer():
    return FeatureStabilityAnalyzer()


def _make_identical_folds(n_folds=5):
    return [{"a": 0.5, "b": 0.3, "c": 0.2} for _ in range(n_folds)]


def _make_diverse_folds():
    return [
        {"a": 0.5, "b": 0.3, "c": 0.1, "d": 0.05, "e": 0.05},
        {"b": 0.5, "c": 0.3, "a": 0.1, "d": 0.05, "e": 0.05},
        {"c": 0.5, "a": 0.3, "b": 0.1, "d": 0.05, "e": 0.05},
        {"d": 0.5, "e": 0.3, "a": 0.1, "b": 0.05, "c": 0.05},
        {"e": 0.5, "d": 0.3, "c": 0.1, "a": 0.05, "b": 0.05},
    ]


class TestFeatureStabilityResult:
    def test_result_is_dataclass(self):
        result = FeatureStabilityResult(
            stable_features=["a"], volatile_features=["b"],
            importance_stats={"a": {"mean": 0.5, "std": 0.0, "stability_score": 1.0}},
            overall_stability=1.0,
        )
        assert result.stable_features == ["a"]


class TestPerfectStability:
    def test_identical_folds_all_stable(self, analyzer):
        folds = _make_identical_folds(5)
        result = analyzer.analyze(folds, top_k=3)

        assert set(result.stable_features) == {"a", "b", "c"}
        assert result.volatile_features == []

    def test_identical_folds_high_overall_stability(self, analyzer):
        folds = _make_identical_folds(5)
        result = analyzer.analyze(folds, top_k=3)

        assert result.overall_stability > 10.0

    def test_identical_folds_zero_std(self, analyzer):
        folds = _make_identical_folds(5)
        result = analyzer.analyze(folds, top_k=3)

        for feat in ["a", "b", "c"]:
            assert result.importance_stats[feat]["std"] == 0.0


class TestLowStability:
    def test_diverse_folds_detect_volatile(self, analyzer):
        folds = _make_diverse_folds()
        result = analyzer.analyze(folds, top_k=2)

        assert len(result.volatile_features) > 0

    def test_diverse_folds_lower_stability(self, analyzer):
        folds = _make_diverse_folds()
        result_diverse = analyzer.analyze(folds, top_k=2)

        folds_identical = _make_identical_folds(5)
        result_identical = analyzer.analyze(folds_identical, top_k=2)

        assert result_diverse.overall_stability < result_identical.overall_stability


class TestEdgeCases:
    def test_empty_folds(self, analyzer):
        result = analyzer.analyze([], top_k=5)

        assert result.stable_features == []
        assert result.volatile_features == []
        assert result.importance_stats == {}
        assert result.overall_stability == 0.0

    def test_single_fold(self, analyzer):
        folds = [{"a": 0.5, "b": 0.3, "c": 0.2}]
        result = analyzer.analyze(folds, top_k=3)

        assert set(result.stable_features) == {"a", "b", "c"}
        assert result.volatile_features == []


class TestStableVolatileDisjoint:
    def test_no_overlap(self, analyzer):
        folds = _make_diverse_folds()
        result = analyzer.analyze(folds, top_k=3)

        stable_set = set(result.stable_features)
        volatile_set = set(result.volatile_features)
        assert stable_set.isdisjoint(volatile_set)


class TestImportanceStats:
    def test_mean_computed_correctly(self, analyzer):
        folds = [{"a": 0.4, "b": 0.6}, {"a": 0.6, "b": 0.4}]
        result = analyzer.analyze(folds, top_k=2)

        assert abs(result.importance_stats["a"]["mean"] - 0.5) < 1e-6
        assert abs(result.importance_stats["b"]["mean"] - 0.5) < 1e-6

    def test_std_computed_correctly(self, analyzer):
        folds = [{"a": 0.4, "b": 0.6}, {"a": 0.6, "b": 0.4}]
        result = analyzer.analyze(folds, top_k=2)

        assert result.importance_stats["a"]["std"] > 0.0

    def test_stability_score_is_mean_over_std(self, analyzer):
        folds = [{"a": 0.4, "b": 0.6}, {"a": 0.6, "b": 0.4}]
        result = analyzer.analyze(folds, top_k=2)

        for feat in ["a", "b"]:
            stats = result.importance_stats[feat]
            expected = stats["mean"] / (stats["std"] + 1e-8)
            assert abs(stats["stability_score"] - expected) < 1e-6


class TestMajorityThreshold:
    def test_custom_threshold(self, analyzer):
        folds = [
            {"a": 0.5, "b": 0.3, "c": 0.2},
            {"a": 0.5, "b": 0.3, "c": 0.2},
            {"a": 0.5, "b": 0.3, "c": 0.2},
            {"d": 0.5, "e": 0.3, "c": 0.2},
            {"d": 0.5, "e": 0.3, "c": 0.2},
        ]
        result_strict = analyzer.analyze(folds, top_k=2, majority_threshold=0.8)
        result_lenient = analyzer.analyze(folds, top_k=2, majority_threshold=0.4)

        assert len(result_lenient.stable_features) >= len(result_strict.stable_features)
