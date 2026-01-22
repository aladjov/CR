import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from sklearn.ensemble import RandomForestClassifier

from customer_retention.stages.deployment import (
    ModelRegistry, ModelStage, BatchScorer, ScoringConfig,
    RetrainingTrigger, RetrainingTriggerType, ChampionChallenger, RollbackManager
)
from customer_retention.stages.monitoring import (
    DriftDetector, PerformanceMonitor, PerformanceStatus,
    AlertManager, AlertLevel
)
from customer_retention.core.components.enums import Severity


@pytest.fixture
def retail_data():
    retail_path = Path(__file__).parent.parent / "fixtures" / "customer_retention_retail.csv"
    return pd.read_csv(retail_path)


@pytest.fixture
def feature_columns():
    return ["avgorder", "ordfreq", "eopenrate", "eclickrate", "paperless", "refill", "doorstep"]


@pytest.fixture
def trained_model(retail_data, feature_columns):
    X = retail_data[feature_columns].fillna(0)
    y = retail_data["retained"]
    model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    model.fit(X, y)
    return model


class TestModelRegistry:
    def test_ac8_1_model_registered_successfully(self, trained_model):
        with patch("customer_retention.stages.deployment.model_registry.mlflow") as mock_mlflow:
            mock_mlflow.register_model.return_value = MagicMock(version="1")
            registry = ModelRegistry()
            result = registry.register_model(
                model=trained_model,
                model_name="churn_prediction_model",
                run_id="test_run"
            )
            assert result.success is True

    def test_ac8_2_all_artifacts_logged(self, trained_model):
        with patch("customer_retention.stages.deployment.model_registry.mlflow") as mock_mlflow:
            registry = ModelRegistry()
            scaler = MagicMock()
            feature_manifest = {"features": ["f1", "f2"], "version": "1.0"}
            registry.register_model(
                model=trained_model,
                model_name="test_model",
                run_id="test_run",
                scaler=scaler,
                feature_manifest=feature_manifest,
                threshold=0.5,
                metrics={"pr_auc": 0.75}
            )
            assert mock_mlflow.log_artifact.called or mock_mlflow.log_dict.called or mock_mlflow.sklearn.log_model.called

    def test_ac8_3_stage_transitions_work(self):
        with patch("customer_retention.stages.deployment.model_registry.mlflow") as mock_mlflow:
            mock_client = MagicMock()
            mock_mlflow.tracking.MlflowClient.return_value = mock_client
            registry = ModelRegistry()
            registry.transition_stage("churn_model", version="1", stage=ModelStage.STAGING)
            registry.transition_stage("churn_model", version="1", stage=ModelStage.PRODUCTION)
            assert mock_client.transition_model_version_stage.call_count == 2

    def test_ac8_4_metadata_complete(self):
        with patch("customer_retention.stages.deployment.model_registry.mlflow") as mock_mlflow:
            mock_client = MagicMock()
            mock_version = MagicMock()
            mock_version.run_id = "run_123"
            mock_version.version = "1"
            mock_version.current_stage = "Production"
            mock_version.tags = {"model_type": "rf"}
            mock_client.get_model_version.return_value = mock_version
            mock_mlflow.tracking.MlflowClient.return_value = mock_client
            registry = ModelRegistry()
            metadata = registry.get_metadata("churn_model", version="1")
            assert metadata.run_id is not None
            assert metadata.version is not None


class TestBatchScoring:
    def test_ac8_5_pipeline_runs_without_error(self, retail_data, trained_model, feature_columns):
        scorer = BatchScorer(model=trained_model)
        result = scorer.score(
            retail_data,
            feature_columns=feature_columns,
            id_column="custid"
        )
        assert result is not None

    def test_ac8_6_output_schema_correct(self, retail_data, trained_model, feature_columns):
        scorer = BatchScorer(model=trained_model)
        result = scorer.score(
            retail_data,
            feature_columns=feature_columns,
            id_column="custid"
        )
        required_columns = ["customer_id", "churn_probability", "risk_segment", "predicted_churn", "score_timestamp"]
        for col in required_columns:
            assert col in result.predictions.columns

    def test_ac8_7_all_customers_scored(self, retail_data, trained_model, feature_columns):
        scorer = BatchScorer(model=trained_model)
        result = scorer.score(
            retail_data,
            feature_columns=feature_columns,
            id_column="custid"
        )
        assert len(result.predictions) == len(retail_data)

    def test_ac8_8_latency_acceptable(self, retail_data, trained_model, feature_columns):
        scorer = BatchScorer(model=trained_model)
        result = scorer.score(
            retail_data,
            feature_columns=feature_columns,
            id_column="custid"
        )
        assert result.scoring_duration_seconds < 60


class TestDriftMonitoring:
    def test_ac8_9_drift_metrics_calculated(self, retail_data, feature_columns):
        reference_data = retail_data[feature_columns].fillna(0).head(15000)
        current_data = retail_data[feature_columns].fillna(0).tail(15000)
        detector = DriftDetector(reference_data=reference_data)
        result = detector.detect_drift(current_data, method="psi")
        assert result is not None
        assert len(result.feature_results) > 0

    def test_ac8_10_psi_computed_correctly(self, retail_data, feature_columns):
        reference_data = retail_data[feature_columns].fillna(0).head(15000)
        current_data = retail_data[feature_columns].fillna(0).tail(15000)
        detector = DriftDetector(reference_data=reference_data)
        result = detector.detect_drift(current_data, method="psi")
        for feature_result in result.feature_results:
            assert feature_result.metric_value >= 0

    def test_ac8_11_alerts_triggered_on_drift(self, retail_data, feature_columns):
        reference_data = retail_data[feature_columns].fillna(0).head(15000)
        shifted_data = retail_data[feature_columns].fillna(0).tail(15000).copy()
        shifted_data["avgorder"] = shifted_data["avgorder"] * 2
        detector = DriftDetector(reference_data=reference_data)
        result = detector.detect_drift(shifted_data, method="psi")
        drifted = [r for r in result.feature_results if r.drift_detected]
        assert len(drifted) > 0

    def test_ac8_12_dashboard_updated(self, retail_data, feature_columns):
        reference_data = retail_data[feature_columns].fillna(0).head(15000)
        current_data = retail_data[feature_columns].fillna(0).tail(15000)
        detector = DriftDetector(reference_data=reference_data)
        result = detector.detect_drift(current_data, method="psi")
        assert result.monitoring_timestamp is not None


class TestPerformanceMonitoring:
    def test_ac8_13_metrics_calculated_with_labels(self, retail_data, trained_model, feature_columns):
        X = retail_data[feature_columns].fillna(0)
        y = retail_data["retained"]
        y_prob = trained_model.predict_proba(X)[:, 1]
        baseline = {"pr_auc": 0.55, "roc_auc": 0.75}
        monitor = PerformanceMonitor(baseline_metrics=baseline)
        result = monitor.evaluate(y_true=y, y_prob=pd.Series(y_prob))
        assert result.current_metrics["pr_auc"] is not None
        assert result.current_metrics["roc_auc"] is not None

    def test_ac8_14_proxy_metrics_without_labels(self, retail_data, trained_model, feature_columns):
        X = retail_data[feature_columns].fillna(0)
        y_prob = trained_model.predict_proba(X)[:, 1]
        baseline = {"pr_auc": 0.55}
        monitor = PerformanceMonitor(baseline_metrics=baseline)
        result = monitor.evaluate_without_labels(y_prob=pd.Series(y_prob))
        assert result.proxy_metrics is not None

    def test_ac8_15_comparison_to_baseline(self, retail_data, trained_model, feature_columns):
        X = retail_data[feature_columns].fillna(0)
        y = retail_data["retained"]
        y_prob = trained_model.predict_proba(X)[:, 1]
        baseline = {"pr_auc": 0.55, "roc_auc": 0.75}
        monitor = PerformanceMonitor(baseline_metrics=baseline)
        result = monitor.evaluate(y_true=y, y_prob=pd.Series(y_prob))
        assert result.comparison is not None


class TestAlerting:
    def test_ac8_16_alerts_sent_to_correct_channel(self):
        with patch("customer_retention.stages.monitoring.alert_manager.EmailSender") as mock_email:
            from customer_retention.stages.monitoring import AlertConfig, AlertChannel, Alert
            config = AlertConfig(channels=[AlertChannel.EMAIL])
            manager = AlertManager(config=config)
            alert = Alert(
                alert_id="test_alert",
                condition_id="AL001",
                level=AlertLevel.WARNING,
                message="Test alert",
                timestamp=datetime.now()
            )
            manager.send_alert(alert)
            mock_email.return_value.send.assert_called()

    def test_ac8_17_alert_levels_correct(self):
        from customer_retention.stages.monitoring import AlertCondition
        manager = AlertManager()
        manager.load_predefined_conditions()
        al001 = next((c for c in manager.conditions if c.alert_id == "AL001"), None)
        al004 = next((c for c in manager.conditions if c.alert_id == "AL004"), None)
        assert al001.level == AlertLevel.CRITICAL
        assert al004.level == AlertLevel.WARNING

    def test_ac8_18_aggregation_works(self):
        from customer_retention.stages.monitoring import AlertConfig, Alert
        config = AlertConfig(aggregation_window_minutes=60)
        manager = AlertManager(config=config)
        alerts = [
            Alert(alert_id=f"alert_{i}", condition_id="AL009", level=AlertLevel.INFO,
                  message="Drift", timestamp=datetime.now())
            for i in range(5)
        ]
        aggregated = manager.aggregate_alerts(alerts)
        assert len(aggregated) <= len(alerts)


class TestRetraining:
    def test_ac8_19_trigger_detection_works(self):
        trigger = RetrainingTrigger()
        metrics = {"pr_auc": {"current": 0.35, "baseline": 0.55}}
        result = trigger.evaluate_performance(metrics)
        assert result.should_retrain is True
        assert result.trigger_type == RetrainingTriggerType.PERFORMANCE_DEGRADATION

    def test_ac8_20_retraining_pipeline_conceptually_runs(self, retail_data, feature_columns):
        trigger = RetrainingTrigger()
        performance_metrics = {"pr_auc": {"current": 0.40, "baseline": 0.55}}
        drift_metrics = {"avgorder": {"psi": 0.25}}
        result = trigger.evaluate_all(
            performance_metrics=performance_metrics,
            drift_metrics=drift_metrics,
            last_training_date=datetime.now() - timedelta(days=100)
        )
        assert result.final_decision is not None
        assert result.final_decision.should_retrain is True

    def test_ac8_21_champion_challenger_comparison(self, retail_data, trained_model, feature_columns):
        X = retail_data[feature_columns].fillna(0)
        y = retail_data["retained"]
        challenger_model = RandomForestClassifier(n_estimators=15, max_depth=6, random_state=44)
        challenger_model.fit(X, y)
        cc = ChampionChallenger()
        cc.set_champion(trained_model, model_name="champion", version="1")
        cc.set_challenger(challenger_model, model_name="challenger", version="2")
        result = cc.compare(X, y)
        assert result.champion_metrics is not None
        assert result.challenger_metrics is not None
        assert result.recommendation in ["promote_challenger", "keep_champion"]


class TestEndToEndDeploymentWorkflow:
    def test_full_scoring_and_monitoring_pipeline(self, retail_data, trained_model, feature_columns):
        scorer = BatchScorer(model=trained_model)
        scoring_result = scorer.score(
            retail_data,
            feature_columns=feature_columns,
            id_column="custid"
        )
        assert len(scoring_result.predictions) == len(retail_data)
        reference = retail_data[feature_columns].fillna(0).head(15000)
        current = retail_data[feature_columns].fillna(0).tail(15000)
        drift_detector = DriftDetector(reference_data=reference)
        drift_result = drift_detector.detect_drift(current, method="psi")
        assert drift_result is not None
        X = retail_data[feature_columns].fillna(0)
        y = retail_data["retained"]
        y_prob = trained_model.predict_proba(X)[:, 1]
        baseline = {"pr_auc": 0.55, "roc_auc": 0.75}
        perf_monitor = PerformanceMonitor(baseline_metrics=baseline)
        perf_result = perf_monitor.evaluate(y_true=y, y_prob=pd.Series(y_prob))
        assert perf_result.status in [PerformanceStatus.OK, PerformanceStatus.WARNING, PerformanceStatus.CRITICAL]

    def test_retraining_decision_workflow(self, retail_data, trained_model, feature_columns):
        X = retail_data[feature_columns].fillna(0)
        y = retail_data["retained"]
        y_prob = trained_model.predict_proba(X)[:, 1]
        baseline = {"pr_auc": 0.55}
        perf_monitor = PerformanceMonitor(baseline_metrics=baseline)
        perf_result = perf_monitor.evaluate(y_true=y, y_prob=pd.Series(y_prob))
        reference = retail_data[feature_columns].fillna(0).head(15000)
        current = retail_data[feature_columns].fillna(0).tail(15000)
        drift_detector = DriftDetector(reference_data=reference)
        drift_result = drift_detector.detect_drift(current, method="psi")
        trigger = RetrainingTrigger()
        performance_degraded = perf_result.status in [PerformanceStatus.WARNING, PerformanceStatus.CRITICAL]
        drift_detected = drift_result.overall_drift_detected
        decision = trigger.make_decision(
            performance_degraded=performance_degraded,
            drift_detected=drift_detected
        )
        assert decision.action in ["immediate_retrain", "investigate_and_prepare",
                                   "investigate_possible_retrain", "continue_monitoring"]


class TestRetailDatasetExpectedValues:
    def test_expected_scoring_configuration(self, retail_data, trained_model, feature_columns):
        scorer = BatchScorer(model=trained_model, batch_size=10000)
        result = scorer.score(
            retail_data,
            feature_columns=feature_columns,
            id_column="custid"
        )
        assert result.total_scored == len(retail_data)
        assert result.scoring_duration_seconds < 300

    def test_expected_monitoring_thresholds(self, retail_data, trained_model, feature_columns):
        X = retail_data[feature_columns].fillna(0)
        y = retail_data["retained"]
        y_prob = trained_model.predict_proba(X)[:, 1]
        from customer_retention.stages.monitoring import MonitoringConfig
        config = MonitoringConfig(pr_auc_warning_drop=0.10, pr_auc_critical_drop=0.15)
        baseline = {"pr_auc": 0.55}
        monitor = PerformanceMonitor(baseline_metrics=baseline, config=config)
        result = monitor.evaluate(y_true=y, y_prob=pd.Series(y_prob))
        assert result.status is not None
