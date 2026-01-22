import pytest
import pandas as pd
import numpy as np
from customer_retention.stages.monitoring import (
    PerformanceMonitor, PerformanceResult, PerformanceStatus,
    ProxyMetrics, MonitoringConfig
)


@pytest.fixture
def predictions_with_labels():
    np.random.seed(42)
    n = 500
    y_true = np.random.choice([0, 1], n, p=[0.75, 0.25])
    y_prob = np.clip(y_true * 0.7 + np.random.normal(0.3, 0.15, n), 0, 1)
    return pd.DataFrame({
        "customer_id": [f"CUST{i:04d}" for i in range(n)],
        "y_true": y_true,
        "y_prob": y_prob,
        "y_pred": (y_prob >= 0.5).astype(int)
    })


@pytest.fixture
def baseline_metrics():
    return {
        "pr_auc": 0.55,
        "roc_auc": 0.75,
        "precision": 0.60,
        "recall": 0.65,
        "brier_score": 0.15
    }


class TestPerformanceStatus:
    def test_status_enum_values(self):
        assert PerformanceStatus.OK.value == "OK"
        assert PerformanceStatus.WARNING.value == "WARNING"
        assert PerformanceStatus.CRITICAL.value == "CRITICAL"


class TestMonitoringConfig:
    def test_config_has_default_thresholds(self):
        config = MonitoringConfig()
        assert config.pr_auc_warning_drop == 0.10
        assert config.pr_auc_critical_drop == 0.15
        assert config.roc_auc_warning_drop == 0.08
        assert config.roc_auc_critical_drop == 0.10

    def test_config_accepts_custom_thresholds(self):
        config = MonitoringConfig(pr_auc_warning_drop=0.12, brier_warning_increase=0.03)
        assert config.pr_auc_warning_drop == 0.12
        assert config.brier_warning_increase == 0.03


class TestPerformanceMetricsCalculation:
    def test_calculates_pr_auc(self, predictions_with_labels, baseline_metrics):
        monitor = PerformanceMonitor(baseline_metrics=baseline_metrics)
        result = monitor.evaluate(
            y_true=predictions_with_labels["y_true"],
            y_prob=predictions_with_labels["y_prob"]
        )
        assert result.current_metrics["pr_auc"] is not None
        assert 0 <= result.current_metrics["pr_auc"] <= 1

    def test_calculates_roc_auc(self, predictions_with_labels, baseline_metrics):
        monitor = PerformanceMonitor(baseline_metrics=baseline_metrics)
        result = monitor.evaluate(
            y_true=predictions_with_labels["y_true"],
            y_prob=predictions_with_labels["y_prob"]
        )
        assert result.current_metrics["roc_auc"] is not None
        assert 0 <= result.current_metrics["roc_auc"] <= 1

    def test_calculates_precision_recall(self, predictions_with_labels, baseline_metrics):
        monitor = PerformanceMonitor(baseline_metrics=baseline_metrics)
        result = monitor.evaluate(
            y_true=predictions_with_labels["y_true"],
            y_prob=predictions_with_labels["y_prob"],
            y_pred=predictions_with_labels["y_pred"]
        )
        assert result.current_metrics["precision"] is not None
        assert result.current_metrics["recall"] is not None

    def test_calculates_brier_score(self, predictions_with_labels, baseline_metrics):
        monitor = PerformanceMonitor(baseline_metrics=baseline_metrics)
        result = monitor.evaluate(
            y_true=predictions_with_labels["y_true"],
            y_prob=predictions_with_labels["y_prob"]
        )
        assert result.current_metrics["brier_score"] is not None
        assert result.current_metrics["brier_score"] >= 0


class TestPerformanceComparison:
    def test_compares_to_baseline(self, predictions_with_labels, baseline_metrics):
        monitor = PerformanceMonitor(baseline_metrics=baseline_metrics)
        result = monitor.evaluate(
            y_true=predictions_with_labels["y_true"],
            y_prob=predictions_with_labels["y_prob"]
        )
        assert result.comparison is not None
        assert "pr_auc_change_pct" in result.comparison

    def test_detects_performance_degradation(self, baseline_metrics):
        monitor = PerformanceMonitor(baseline_metrics=baseline_metrics)
        degraded_metrics = {"pr_auc": 0.40, "roc_auc": 0.60}
        result = monitor.compare_metrics(degraded_metrics)
        assert result.status in [PerformanceStatus.WARNING, PerformanceStatus.CRITICAL]

    def test_status_ok_when_performance_maintained(self, baseline_metrics):
        monitor = PerformanceMonitor(baseline_metrics=baseline_metrics)
        good_metrics = {"pr_auc": 0.54, "roc_auc": 0.74}
        result = monitor.compare_metrics(good_metrics)
        assert result.status == PerformanceStatus.OK


class TestPerformanceAlertThresholds:
    def test_pr_auc_drop_triggers_warning(self, baseline_metrics):
        monitor = PerformanceMonitor(
            baseline_metrics=baseline_metrics,
            config=MonitoringConfig(pr_auc_warning_drop=0.10, pr_auc_critical_drop=0.15)
        )
        metrics = {"pr_auc": 0.44}
        result = monitor.compare_metrics(metrics)
        assert result.status == PerformanceStatus.WARNING

    def test_pr_auc_drop_triggers_critical(self, baseline_metrics):
        monitor = PerformanceMonitor(
            baseline_metrics=baseline_metrics,
            config=MonitoringConfig(pr_auc_critical_drop=0.15)
        )
        metrics = {"pr_auc": 0.40}
        result = monitor.compare_metrics(metrics)
        assert result.status == PerformanceStatus.CRITICAL

    def test_brier_score_increase_triggers_warning(self, baseline_metrics):
        monitor = PerformanceMonitor(
            baseline_metrics=baseline_metrics,
            config=MonitoringConfig(brier_warning_increase=0.05)
        )
        metrics = {"brier_score": 0.22}
        result = monitor.compare_metrics(metrics)
        assert result.status in [PerformanceStatus.WARNING, PerformanceStatus.CRITICAL]


class TestProxyMetrics:
    def test_calculates_prediction_distribution(self):
        np.random.seed(42)
        y_prob = pd.Series(np.random.uniform(0, 1, 500))
        proxy = ProxyMetrics()
        result = proxy.analyze_prediction_distribution(y_prob)
        assert result.mean is not None
        assert result.std is not None
        assert result.percentiles is not None

    def test_calculates_segment_proportions(self):
        np.random.seed(42)
        segments = pd.Series(np.random.choice(["Critical", "High", "Medium", "Low"], 500))
        proxy = ProxyMetrics()
        result = proxy.analyze_segment_proportions(segments)
        assert len(result.proportions) == 4

    def test_compares_distributions(self):
        np.random.seed(42)
        reference_probs = pd.Series(np.random.uniform(0, 1, 500))
        current_probs = pd.Series(np.random.uniform(0, 1, 500))
        proxy = ProxyMetrics()
        result = proxy.compare_distributions(reference_probs, current_probs)
        assert result.distribution_shift_detected is not None


class TestLabelDelay:
    def test_works_with_delayed_labels(self, predictions_with_labels, baseline_metrics):
        partial_labels = predictions_with_labels.copy()
        partial_labels.loc[:300, "y_true"] = np.nan
        monitor = PerformanceMonitor(baseline_metrics=baseline_metrics)
        result = monitor.evaluate(
            y_true=partial_labels["y_true"].dropna(),
            y_prob=partial_labels.loc[partial_labels["y_true"].notna(), "y_prob"]
        )
        assert result.labels_available == 199

    def test_reports_proxy_metrics_without_labels(self, baseline_metrics):
        np.random.seed(42)
        y_prob = pd.Series(np.random.uniform(0, 1, 500))
        monitor = PerformanceMonitor(baseline_metrics=baseline_metrics)
        result = monitor.evaluate_without_labels(y_prob=y_prob)
        assert result.proxy_metrics is not None


class TestPerformanceResult:
    def test_result_contains_required_fields(self, predictions_with_labels, baseline_metrics):
        monitor = PerformanceMonitor(baseline_metrics=baseline_metrics)
        result = monitor.evaluate(
            y_true=predictions_with_labels["y_true"],
            y_prob=predictions_with_labels["y_prob"]
        )
        assert hasattr(result, "current_metrics")
        assert hasattr(result, "baseline_metrics")
        assert hasattr(result, "status")
        assert hasattr(result, "monitoring_date")

    def test_result_includes_labels_count(self, predictions_with_labels, baseline_metrics):
        monitor = PerformanceMonitor(baseline_metrics=baseline_metrics)
        result = monitor.evaluate(
            y_true=predictions_with_labels["y_true"],
            y_prob=predictions_with_labels["y_prob"]
        )
        assert result.labels_available == 500


class TestPerformanceHistory:
    def test_tracks_performance_over_time(self, predictions_with_labels, baseline_metrics):
        monitor = PerformanceMonitor(baseline_metrics=baseline_metrics)
        for i in range(3):
            monitor.evaluate(
                y_true=predictions_with_labels["y_true"],
                y_prob=predictions_with_labels["y_prob"]
            )
        history = monitor.get_history()
        assert len(history) == 3

    def test_generates_trend_report(self, predictions_with_labels, baseline_metrics):
        monitor = PerformanceMonitor(baseline_metrics=baseline_metrics)
        for i in range(5):
            monitor.evaluate(
                y_true=predictions_with_labels["y_true"],
                y_prob=predictions_with_labels["y_prob"]
            )
        trend = monitor.get_trend_report()
        assert trend is not None
        assert "pr_auc_trend" in trend


class TestCalibrationMonitoring:
    def test_monitors_calibration(self, predictions_with_labels, baseline_metrics):
        monitor = PerformanceMonitor(baseline_metrics=baseline_metrics)
        result = monitor.evaluate(
            y_true=predictions_with_labels["y_true"],
            y_prob=predictions_with_labels["y_prob"]
        )
        assert "calibration_error" in result.current_metrics or "brier_score" in result.current_metrics

    def test_generates_calibration_curve_data(self, predictions_with_labels, baseline_metrics):
        monitor = PerformanceMonitor(baseline_metrics=baseline_metrics)
        curve_data = monitor.get_calibration_curve(
            y_true=predictions_with_labels["y_true"],
            y_prob=predictions_with_labels["y_prob"],
            n_bins=10
        )
        assert len(curve_data.bin_means) == 10
        assert len(curve_data.actual_rates) == 10
