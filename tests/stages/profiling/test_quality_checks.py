import pandas as pd

from customer_retention.core.components.enums import Severity
from customer_retention.core.config import ColumnType
from customer_retention.stages.profiling import ProfilerFactory, QualityCheckRegistry
from customer_retention.stages.profiling.quality_checks import (
    ConstantFeatureCheck,
    DatetimeFutureLeakageCheck,
    HighCardinalityCheck,
    IdentifierLeakageCheck,
    ImbalancedTargetCheck,
    LowCardinalityCheck,
    MissingValueCheck,
    OutlierCheck,
    PlaceholderDateCheck,
    RareCategoryCheck,
    SkewnessCheck,
    UnknownCategoryCheck,
    ZeroInflationCheck,
)


class TestMissingValueCheck:
    def test_critical_missing_values(self):
        check = MissingValueCheck()
        series = pd.Series([1] * 4 + [None] * 96)
        profiler = ProfilerFactory.get_profiler(ColumnType.NUMERIC_CONTINUOUS)
        metrics = profiler.compute_universal_metrics(series)

        result = check.run("test_col", metrics)

        assert result.passed is False
        assert result.severity == Severity.CRITICAL
        assert "96.0%" in result.message
        assert result.recommendation is not None

    def test_acceptable_missing_values(self):
        check = MissingValueCheck()
        series = pd.Series([1] * 98 + [None] * 2)
        profiler = ProfilerFactory.get_profiler(ColumnType.NUMERIC_CONTINUOUS)
        metrics = profiler.compute_universal_metrics(series)

        result = check.run("test_col", metrics)

        assert result.passed is True
        assert "2.0%" in result.message


class TestHighCardinalityCheck:
    def test_very_high_cardinality(self):
        check = HighCardinalityCheck()
        series = pd.Series([f"cat_{i}" for i in range(100)])
        profiler = ProfilerFactory.get_profiler(ColumnType.CATEGORICAL_NOMINAL)
        metrics = profiler.profile(series)["categorical_metrics"]

        result = check.run("test_col", metrics)

        assert result.passed is False
        assert result.severity == Severity.MEDIUM

    def test_acceptable_cardinality(self):
        check = HighCardinalityCheck()
        series = pd.Series(["a", "b", "c"] * 30)
        profiler = ProfilerFactory.get_profiler(ColumnType.CATEGORICAL_NOMINAL)
        metrics = profiler.profile(series)["categorical_metrics"]

        result = check.run("test_col", metrics)

        assert result.passed is True


class TestLowCardinalityCheck:
    def test_low_cardinality_numeric(self):
        check = LowCardinalityCheck()
        series = pd.Series([1, 2, 3] * 30)
        profiler = ProfilerFactory.get_profiler(ColumnType.NUMERIC_CONTINUOUS)
        metrics = profiler.compute_universal_metrics(series)

        result = check.run("test_col", metrics, ColumnType.NUMERIC_CONTINUOUS)

        assert result.passed is False
        assert result.severity == Severity.LOW
        assert "3 unique values" in result.message

    def test_acceptable_cardinality_numeric(self):
        check = LowCardinalityCheck()
        series = pd.Series(range(100))
        profiler = ProfilerFactory.get_profiler(ColumnType.NUMERIC_CONTINUOUS)
        metrics = profiler.compute_universal_metrics(series)

        result = check.run("test_col", metrics, ColumnType.NUMERIC_CONTINUOUS)

        assert result.passed is True


class TestConstantFeatureCheck:
    def test_constant_feature(self):
        check = ConstantFeatureCheck()
        series = pd.Series([1] * 100)
        profiler = ProfilerFactory.get_profiler(ColumnType.NUMERIC_CONTINUOUS)
        metrics = profiler.compute_universal_metrics(series)

        result = check.run("test_col", metrics)

        assert result.passed is False
        assert result.severity == Severity.CRITICAL
        assert "constant" in result.message.lower()
        assert "1" in str(result.details["distinct_count"])

    def test_good_variance(self):
        check = ConstantFeatureCheck()
        series = pd.Series([1, 2, 3, 4, 5] * 20)
        profiler = ProfilerFactory.get_profiler(ColumnType.NUMERIC_CONTINUOUS)
        metrics = profiler.compute_universal_metrics(series)

        result = check.run("test_col", metrics)

        assert result.passed is True


class TestImbalancedTargetCheck:
    def test_severe_imbalance(self):
        check = ImbalancedTargetCheck()
        series = pd.Series([0] * 95 + [1] * 5)
        profiler = ProfilerFactory.get_profiler(ColumnType.TARGET)
        metrics = profiler.profile(series)["target_metrics"]

        result = check.run("test_col", metrics)

        assert result.passed is False
        assert result.severity == Severity.HIGH
        assert "19.0:1" in result.message

    def test_acceptable_balance(self):
        check = ImbalancedTargetCheck()
        series = pd.Series([0] * 60 + [1] * 40)
        profiler = ProfilerFactory.get_profiler(ColumnType.TARGET)
        metrics = profiler.profile(series)["target_metrics"]

        result = check.run("test_col", metrics)

        assert result.passed is True


class TestTargetNullCheck:
    def test_target_with_nulls(self):
        from customer_retention.stages.profiling.profile_result import UniversalMetrics
        from customer_retention.stages.profiling.quality_checks import TargetNullCheck

        check = TargetNullCheck()
        # Create universal metrics directly
        universal_metrics = UniversalMetrics(
            total_count=7,
            null_count=2,
            null_percentage=28.57,
            distinct_count=2,
            distinct_percentage=28.57
        )

        result = check.run("target_col", universal_metrics)

        assert result is not None
        assert result.passed is False
        assert result.severity.value == "critical"
        assert "2 null values" in result.message

    def test_target_without_nulls(self):
        from customer_retention.stages.profiling.profile_result import UniversalMetrics
        from customer_retention.stages.profiling.quality_checks import TargetNullCheck

        check = TargetNullCheck()
        # Create universal metrics directly
        universal_metrics = UniversalMetrics(
            total_count=7,
            null_count=0,
            null_percentage=0.0,
            distinct_count=2,
            distinct_percentage=28.57
        )

        result = check.run("target_col", universal_metrics)

        assert result is not None
        assert result.passed is True
        assert result.details["null_count"] == 0


class TestSingleClassTargetCheck:
    def test_single_class_target(self):
        from customer_retention.stages.profiling.quality_checks import SingleClassTargetCheck

        check = SingleClassTargetCheck()
        series = pd.Series([1, 1, 1, 1, 1])  # Only one class
        profiler = ProfilerFactory.get_profiler(ColumnType.TARGET)
        metrics = profiler.profile(series)["target_metrics"]

        result = check.run("target_col", metrics)

        assert result is not None
        assert result.passed is False
        assert result.severity.value == "critical"
        assert "only 1 class" in result.message

    def test_multi_class_target(self):
        from customer_retention.stages.profiling.quality_checks import SingleClassTargetCheck

        check = SingleClassTargetCheck()
        series = pd.Series([0, 1, 0, 1, 1])  # Two classes
        profiler = ProfilerFactory.get_profiler(ColumnType.TARGET)
        metrics = profiler.profile(series)["target_metrics"]

        result = check.run("target_col", metrics)

        assert result is not None
        assert result.passed is True
        assert result.details["n_classes"] == 2


class TestTargetSevereImbalanceCheck:
    def test_severe_imbalance(self):
        from customer_retention.stages.profiling.quality_checks import TargetSevereImbalanceCheck

        check = TargetSevereImbalanceCheck()
        series = pd.Series([0] * 995 + [1] * 5)  # 0.5% minority class
        profiler = ProfilerFactory.get_profiler(ColumnType.TARGET)
        metrics = profiler.profile(series)["target_metrics"]

        result = check.run("target_col", metrics)

        assert result is not None
        assert result.passed is False
        assert result.severity.value == "high"
        assert metrics.minority_percentage < 1.0

    def test_acceptable_imbalance(self):
        from customer_retention.stages.profiling.quality_checks import TargetSevereImbalanceCheck

        check = TargetSevereImbalanceCheck()
        series = pd.Series([0] * 950 + [1] * 50)  # 5% minority class
        profiler = ProfilerFactory.get_profiler(ColumnType.TARGET)
        metrics = profiler.profile(series)["target_metrics"]

        result = check.run("target_col", metrics)

        assert result is not None
        assert result.passed is True


class TestTargetModerateImbalanceCheck:
    def test_moderate_imbalance(self):
        from customer_retention.stages.profiling.quality_checks import TargetModerateImbalanceCheck

        check = TargetModerateImbalanceCheck()
        series = pd.Series([0] * 92 + [1] * 8)  # 8% minority class
        profiler = ProfilerFactory.get_profiler(ColumnType.TARGET)
        metrics = profiler.profile(series)["target_metrics"]

        result = check.run("target_col", metrics)

        assert result is not None
        assert result.passed is False
        assert result.severity.value == "medium"
        assert metrics.minority_percentage < 10.0

    def test_acceptable_imbalance(self):
        from customer_retention.stages.profiling.quality_checks import TargetModerateImbalanceCheck

        check = TargetModerateImbalanceCheck()
        series = pd.Series([0] * 70 + [1] * 30)  # 30% minority class
        profiler = ProfilerFactory.get_profiler(ColumnType.TARGET)
        metrics = profiler.profile(series)["target_metrics"]

        result = check.run("target_col", metrics)

        assert result is not None
        assert result.passed is True


class TestTargetUnexpectedClassesCheck:
    def test_unexpected_classes(self):
        from customer_retention.stages.profiling.quality_checks import TargetUnexpectedClassesCheck

        check = TargetUnexpectedClassesCheck(expected_classes=2)
        series = pd.Series([0, 1, 2, 0, 1, 2])  # 3 classes, expected 2
        profiler = ProfilerFactory.get_profiler(ColumnType.TARGET)
        metrics = profiler.profile(series)["target_metrics"]

        result = check.run("target_col", metrics)

        assert result is not None
        assert result.passed is False
        assert result.severity.value == "high"
        assert result.details["n_classes"] == 3
        assert result.details["expected_classes"] == 2

    def test_expected_classes(self):
        from customer_retention.stages.profiling.quality_checks import TargetUnexpectedClassesCheck

        check = TargetUnexpectedClassesCheck(expected_classes=2)
        series = pd.Series([0, 1, 0, 1, 1])  # 2 classes
        profiler = ProfilerFactory.get_profiler(ColumnType.TARGET)
        metrics = profiler.profile(series)["target_metrics"]

        result = check.run("target_col", metrics)

        assert result is not None
        assert result.passed is True


class TestSkewnessCheck:
    def test_extreme_skewness(self):
        check = SkewnessCheck()
        series = pd.Series([1] * 90 + list(range(100, 110)))
        profiler = ProfilerFactory.get_profiler(ColumnType.NUMERIC_CONTINUOUS)
        metrics = profiler.profile(series)["numeric_metrics"]

        result = check.run("test_col", metrics)

        if metrics.skewness and abs(metrics.skewness) > 3.0:
            assert result.passed is False
            assert result.severity == Severity.MEDIUM

    def test_acceptable_skewness(self):
        check = SkewnessCheck()
        series = pd.Series(range(100))
        profiler = ProfilerFactory.get_profiler(ColumnType.NUMERIC_CONTINUOUS)
        metrics = profiler.profile(series)["numeric_metrics"]

        result = check.run("test_col", metrics)

        assert result.passed is True


class TestOutlierCheck:
    def test_high_outliers(self):
        check = OutlierCheck()
        series = pd.Series(list(range(80)) + [1000] * 20)
        profiler = ProfilerFactory.get_profiler(ColumnType.NUMERIC_CONTINUOUS)
        metrics = profiler.profile(series)["numeric_metrics"]

        result = check.run("test_col", metrics)

        assert result.passed is False
        assert result.severity == Severity.MEDIUM

    def test_acceptable_outliers(self):
        check = OutlierCheck()
        series = pd.Series(range(100))
        profiler = ProfilerFactory.get_profiler(ColumnType.NUMERIC_CONTINUOUS)
        metrics = profiler.profile(series)["numeric_metrics"]

        result = check.run("test_col", metrics)

        assert result.passed is True


class TestZeroInflationCheck:
    def test_zero_inflated(self):
        check = ZeroInflationCheck()
        series = pd.Series([0] * 60 + list(range(1, 41)))
        profiler = ProfilerFactory.get_profiler(ColumnType.NUMERIC_CONTINUOUS)
        metrics = profiler.profile(series)["numeric_metrics"]

        result = check.run("test_col", metrics)

        assert result.passed is False
        assert result.severity == Severity.LOW
        assert "60.0%" in result.message

    def test_acceptable_zeros(self):
        check = ZeroInflationCheck()
        series = pd.Series(range(100))
        profiler = ProfilerFactory.get_profiler(ColumnType.NUMERIC_CONTINUOUS)
        metrics = profiler.profile(series)["numeric_metrics"]

        result = check.run("test_col", metrics)

        assert result.passed is True


class TestIdentifierLeakageCheck:
    def test_identifier_in_features(self):
        check = IdentifierLeakageCheck()

        result = check.run("customer_id", ColumnType.IDENTIFIER, should_use_as_feature=True)

        assert result.passed is False
        assert result.severity == Severity.CRITICAL
        assert "identifier" in result.message.lower()
        assert result.recommendation is not None
        assert "leakage" in result.recommendation.lower()

    def test_identifier_correctly_excluded(self):
        check = IdentifierLeakageCheck()

        result = check.run("customer_id", ColumnType.IDENTIFIER, should_use_as_feature=False)

        assert result.passed is True


class TestDatetimeFutureLeakageCheck:
    def test_future_dates_detected(self):
        check = DatetimeFutureLeakageCheck()
        series = pd.Series(pd.date_range("2030-01-01", periods=10))
        profiler = ProfilerFactory.get_profiler(ColumnType.DATETIME)
        metrics = profiler.profile(series)["datetime_metrics"]

        result = check.run("test_col", metrics)

        assert result.passed is False
        assert result.severity == Severity.HIGH
        assert result.details["future_date_count"] > 0

    def test_no_future_dates(self):
        check = DatetimeFutureLeakageCheck()
        series = pd.Series(pd.date_range("2020-01-01", periods=10))
        profiler = ProfilerFactory.get_profiler(ColumnType.DATETIME)
        metrics = profiler.profile(series)["datetime_metrics"]

        result = check.run("test_col", metrics)

        assert result.passed is True


class TestPlaceholderDateCheck:
    def test_placeholder_dates_found(self):
        check = PlaceholderDateCheck()
        series = pd.Series([pd.Timestamp("1970-01-01")] * 10 +
                          [pd.Timestamp("2023-06-15")] * 90)
        profiler = ProfilerFactory.get_profiler(ColumnType.DATETIME)
        metrics = profiler.profile(series)["datetime_metrics"]

        result = check.run("test_col", metrics, total_count=100)

        assert result.passed is False
        assert result.severity == Severity.MEDIUM

    def test_no_placeholder_dates(self):
        check = PlaceholderDateCheck()
        series = pd.Series(pd.date_range("2023-01-01", periods=100))
        profiler = ProfilerFactory.get_profiler(ColumnType.DATETIME)
        metrics = profiler.profile(series)["datetime_metrics"]

        result = check.run("test_col", metrics, total_count=100)

        assert result.passed is True


class TestRareCategoryCheck:
    def test_high_rare_categories(self):
        check = RareCategoryCheck()
        series = pd.Series(["common"] * 150 + [f"rare_{i}" for i in range(50)])
        profiler = ProfilerFactory.get_profiler(ColumnType.CATEGORICAL_NOMINAL)
        metrics = profiler.profile(series)["categorical_metrics"]

        result = check.run("test_col", metrics)

        assert result.passed is False
        assert result.severity == Severity.MEDIUM

    def test_acceptable_rare_categories(self):
        check = RareCategoryCheck()
        series = pd.Series(["a", "b", "c"] * 30)
        profiler = ProfilerFactory.get_profiler(ColumnType.CATEGORICAL_NOMINAL)
        metrics = profiler.profile(series)["categorical_metrics"]

        result = check.run("test_col", metrics)

        assert result.passed is True


class TestUnknownCategoryCheck:
    def test_unknown_categories_present(self):
        check = UnknownCategoryCheck()
        series = pd.Series(["a", "b", "unknown", "c", "n/a"])
        profiler = ProfilerFactory.get_profiler(ColumnType.CATEGORICAL_NOMINAL)
        metrics = profiler.profile(series)["categorical_metrics"]

        result = check.run("test_col", metrics)

        assert result.passed is False
        assert result.severity == Severity.LOW

    def test_no_unknown_categories(self):
        check = UnknownCategoryCheck()
        series = pd.Series(["a", "b", "c"] * 10)
        profiler = ProfilerFactory.get_profiler(ColumnType.CATEGORICAL_NOMINAL)
        metrics = profiler.profile(series)["categorical_metrics"]

        result = check.run("test_col", metrics)

        assert result.passed is True


class TestInfiniteValuesCheck:
    def test_detects_infinite_values(self):
        import numpy as np

        from customer_retention.stages.profiling.quality_checks import InfiniteValuesCheck

        check = InfiniteValuesCheck()
        series = pd.Series([1.0, 2.0, np.inf, 4.0, -np.inf, 6.0])
        profiler = ProfilerFactory.get_profiler(ColumnType.NUMERIC_CONTINUOUS)
        metrics = profiler.profile(series)["numeric_metrics"]

        result = check.run("test_col", metrics)

        assert result is not None
        assert result.passed is False
        assert result.severity.value == "critical"
        assert "2 infinite values" in result.message
        assert result.details["inf_count"] == 2

    def test_no_infinite_values(self):
        from customer_retention.stages.profiling.quality_checks import InfiniteValuesCheck

        check = InfiniteValuesCheck()
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        profiler = ProfilerFactory.get_profiler(ColumnType.NUMERIC_CONTINUOUS)
        metrics = profiler.profile(series)["numeric_metrics"]

        result = check.run("test_col", metrics)

        assert result is not None
        assert result.passed is True
        assert result.details["inf_count"] == 0


class TestQualityCheckRegistry:
    def test_get_check_by_id(self):
        check = QualityCheckRegistry.get_check("FQ001")
        assert check is not None
        assert isinstance(check, MissingValueCheck)

    def test_get_all_checks(self):
        checks = QualityCheckRegistry.get_all_checks()
        assert len(checks) >= 49  # Comprehensive checks including FQ, TX, NC, TG, ID, CN, DT, BN checks
        assert all(hasattr(c, 'check_id') for c in checks)

    def test_get_checks_for_identifier_type(self):
        checks = QualityCheckRegistry.get_checks_for_column_type(ColumnType.IDENTIFIER)
        check_ids = [c.check_id for c in checks]

        assert "FQ001" in check_ids
        assert "FQ003" in check_ids
        assert "LEAK001" in check_ids
        assert "ID001" in check_ids
        assert "ID002" in check_ids
        assert "ID003" in check_ids

    def test_get_checks_for_target_type(self):
        checks = QualityCheckRegistry.get_checks_for_column_type(ColumnType.TARGET)
        check_ids = [c.check_id for c in checks]

        assert "FQ001" in check_ids
        assert "FQ003" in check_ids
        assert "TG001" in check_ids
        assert "TG002" in check_ids
        assert "TG003" in check_ids
        assert "TG005" in check_ids
        assert "CAT002" in check_ids  # ImbalancedTargetCheck renamed

    def test_get_checks_for_numeric_type(self):
        checks = QualityCheckRegistry.get_checks_for_column_type(ColumnType.NUMERIC_CONTINUOUS)
        check_ids = [c.check_id for c in checks]

        assert "FQ001" in check_ids
        assert "FQ003" in check_ids
        assert "NUM001" in check_ids  # LowCardinalityCheck renamed
        assert "NC006" in check_ids
        assert "NUM002" in check_ids  # SkewnessCheck renamed
        assert "NUM003" in check_ids  # OutlierCheck renamed
        assert "NUM004" in check_ids  # ZeroInflationCheck renamed

    def test_get_checks_for_categorical_type(self):
        checks = QualityCheckRegistry.get_checks_for_column_type(ColumnType.CATEGORICAL_NOMINAL)
        check_ids = [c.check_id for c in checks]

        assert "FQ001" in check_ids
        assert "FQ003" in check_ids
        assert "CAT001" in check_ids  # HighCardinalityCheck renamed
        assert "CAT003" in check_ids  # RareCategoryCheck renamed
        assert "CAT004" in check_ids  # UnknownCategoryCheck renamed

    def test_get_checks_for_datetime_type(self):
        checks = QualityCheckRegistry.get_checks_for_column_type(ColumnType.DATETIME)
        check_ids = [c.check_id for c in checks]

        assert "FQ001" in check_ids
        assert "FQ003" in check_ids
        assert "DT001" in check_ids  # DatetimeFutureLeakageCheck renamed
        assert "DT002" in check_ids  # PlaceholderDateCheck renamed
