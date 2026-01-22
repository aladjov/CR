
import numpy as np
import pandas as pd
import pytest


class TestFullIterationLoop:
    @pytest.fixture
    def setup_environment(self, tmp_path):
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        (findings_dir / "iterations").mkdir()
        (findings_dir / "recommendations").mkdir()
        (findings_dir / "feedback").mkdir()

        data_dir = tmp_path / "data"
        data_dir.mkdir()

        np.random.seed(42)
        customers_df = pd.DataFrame({
            "customer_id": range(1, 101),
            "age": np.random.normal(35, 10, 100),
            "income": np.random.normal(60000, 15000, 100),
            "tenure": np.random.normal(24, 12, 100),
            "churn": np.random.choice([0, 1], 100, p=[0.8, 0.2])
        })
        customers_df.to_csv(data_dir / "customers.csv", index=False)

        return {
            "findings_dir": findings_dir,
            "data_dir": data_dir,
            "tmp_path": tmp_path
        }

    def test_complete_iteration_workflow(self, setup_environment):
        from customer_retention.integrations.iteration import (
            IterationOrchestrator,
            IterationStatus,
            IterationTrigger,
            ModelFeedback,
            RecommendationType,
            TrackedRecommendation,
        )

        findings_dir = setup_environment["findings_dir"]
        orchestrator = IterationOrchestrator(str(findings_dir))

        # 1. Start initial iteration
        ctx1 = orchestrator.start_new_iteration(IterationTrigger.INITIAL)
        assert ctx1.iteration_number == 1
        assert ctx1.trigger == IterationTrigger.INITIAL

        # 2. Track recommendations
        rec1 = TrackedRecommendation(
            recommendation_id="rec_impute_age",
            recommendation_type=RecommendationType.CLEANING,
            source_column="age",
            action="impute_median",
            description="Impute missing age values"
        )
        rec2 = TrackedRecommendation(
            recommendation_id="rec_scale_income",
            recommendation_type=RecommendationType.TRANSFORM,
            source_column="income",
            action="standard_scale",
            description="Standardize income"
        )
        orchestrator.track_recommendation(rec1)
        orchestrator.track_recommendation(rec2)

        # 3. Apply some recommendations
        orchestrator.apply_recommendation("rec_impute_age")
        orchestrator.skip_recommendation("rec_scale_income", "Using RobustScaler instead")

        # 4. Update status to training
        orchestrator.update_status(IterationStatus.TRAINING)

        # 5. Record model metrics
        orchestrator.record_model_metrics(
            {"roc_auc": 0.82, "pr_auc": 0.68},
            artifact_path="/models/v1"
        )

        # 6. Collect feedback
        feedback = ModelFeedback(
            iteration_id=ctx1.iteration_id,
            model_type="RandomForestClassifier",
            metrics={"roc_auc": 0.82, "pr_auc": 0.68},
            feature_importances={
                "age": 0.25,
                "income": 0.35,
                "tenure": 0.39,
                "unused_feature": 0.005  # Below drop threshold of 0.01
            }
        )
        orchestrator.collect_feedback(feedback)

        # 7. Complete iteration
        orchestrator.update_status(IterationStatus.COMPLETED)

        # 8. Start child iteration for improvement
        ctx2 = orchestrator.start_child_iteration(IterationTrigger.MANUAL)
        assert ctx2.iteration_number == 2
        assert ctx2.parent_iteration_id == ctx1.iteration_id

        # 9. Get refined recommendations based on feedback
        from customer_retention.analysis.auto_explorer.findings import ColumnFinding, ExplorationFindings
        from customer_retention.core.config.column_config import ColumnType

        findings = ExplorationFindings(
            source_path="/data/test.csv",
            source_format="csv",
            row_count=100,
            column_count=4,
            columns={
                "age": ColumnFinding(name="age", inferred_type=ColumnType.NUMERIC_CONTINUOUS, confidence=0.9, evidence=[]),
                "income": ColumnFinding(name="income", inferred_type=ColumnType.NUMERIC_CONTINUOUS, confidence=0.9, evidence=[]),
                "tenure": ColumnFinding(name="tenure", inferred_type=ColumnType.NUMERIC_CONTINUOUS, confidence=0.9, evidence=[]),
                "unused_feature": ColumnFinding(name="unused_feature", inferred_type=ColumnType.NUMERIC_CONTINUOUS, confidence=0.9, evidence=[])
            }
        )

        refined = orchestrator.get_refined_recommendations(findings, feedback)
        assert "unused_feature" in refined["features_to_drop"]
        assert "tenure" in refined["top_features"] or "income" in refined["top_features"]

        # 10. Save and reload state
        orchestrator.save_state()

        new_orchestrator = IterationOrchestrator(str(findings_dir))
        new_orchestrator.load_state()

        current = new_orchestrator.get_current_iteration()
        assert current.iteration_id == ctx2.iteration_id

        history = new_orchestrator.get_iteration_history()
        assert len(history) == 2

    def test_drift_triggered_iteration(self, setup_environment):
        import numpy as np
        import pandas as pd

        from customer_retention.integrations.iteration import (
            IterationOrchestrator,
            IterationStatus,
            IterationTrigger,
            SignalAggregator,
        )
        from customer_retention.stages.monitoring.drift_detector import DriftDetector
        from customer_retention.stages.monitoring.performance_monitor import PerformanceMonitor

        findings_dir = setup_environment["findings_dir"]

        np.random.seed(42)
        reference_data = pd.DataFrame({
            "age": np.random.normal(35, 10, 100),
            "income": np.random.normal(60000, 15000, 100)
        })

        drift_detector = DriftDetector(reference_data=reference_data)
        perf_monitor = PerformanceMonitor(baseline_metrics={"roc_auc": 0.85})

        signal_aggregator = SignalAggregator(
            drift_detector=drift_detector,
            performance_monitor=perf_monitor
        )

        orchestrator = IterationOrchestrator(
            str(findings_dir),
            signal_aggregator=signal_aggregator
        )

        ctx1 = orchestrator.start_new_iteration(IterationTrigger.INITIAL)
        orchestrator.update_status(IterationStatus.COMPLETED)

        drifted_data = pd.DataFrame({
            "age": np.random.normal(50, 15, 100),
            "income": np.random.normal(90000, 25000, 100)
        })
        signal_aggregator.check_drift_signals(drifted_data)

        should_trigger, trigger = orchestrator.check_for_iteration_triggers()
        assert should_trigger is True
        assert trigger == IterationTrigger.DRIFT_DETECTED

        ctx2 = orchestrator.start_child_iteration(trigger)
        assert ctx2.trigger == IterationTrigger.DRIFT_DETECTED

    def test_performance_drop_triggered_iteration(self, setup_environment):
        from customer_retention.integrations.iteration import (
            IterationOrchestrator,
            IterationStatus,
            IterationTrigger,
            SignalAggregator,
        )
        from customer_retention.stages.monitoring.performance_monitor import PerformanceMonitor

        findings_dir = setup_environment["findings_dir"]

        perf_monitor = PerformanceMonitor(baseline_metrics={"roc_auc": 0.85})
        signal_aggregator = SignalAggregator(performance_monitor=perf_monitor)

        orchestrator = IterationOrchestrator(
            str(findings_dir),
            signal_aggregator=signal_aggregator
        )

        ctx1 = orchestrator.start_new_iteration(IterationTrigger.INITIAL)
        orchestrator.update_status(IterationStatus.COMPLETED)

        signal_aggregator.check_performance_signals({"roc_auc": 0.70})

        should_trigger, trigger = orchestrator.check_for_iteration_triggers()
        assert should_trigger is True
        assert trigger == IterationTrigger.PERFORMANCE_DROP

    def test_iteration_with_findings_integration(self, setup_environment):
        from customer_retention.analysis.auto_explorer.findings import ColumnFinding, ExplorationFindings
        from customer_retention.core.config.column_config import ColumnType
        from customer_retention.integrations.iteration import IterationOrchestrator, IterationTrigger

        findings_dir = setup_environment["findings_dir"]
        orchestrator = IterationOrchestrator(str(findings_dir))

        ctx = orchestrator.start_new_iteration(IterationTrigger.INITIAL)

        findings = ExplorationFindings(
            source_path="/data/test.csv",
            source_format="csv",
            row_count=100,
            column_count=3,
            columns={
                "id": ColumnFinding(name="id", inferred_type=ColumnType.IDENTIFIER, confidence=0.95, evidence=[]),
                "value": ColumnFinding(name="value", inferred_type=ColumnType.NUMERIC_CONTINUOUS, confidence=0.9, evidence=[])
            },
            iteration_id=ctx.iteration_id
        )

        findings_path = findings_dir / "test_findings.yaml"
        findings.save(str(findings_path))

        loaded = ExplorationFindings.load(str(findings_path))
        assert loaded.iteration_id == ctx.iteration_id

    def test_recommendation_tracking_across_iterations(self, setup_environment):
        from customer_retention.integrations.iteration import (
            IterationOrchestrator,
            IterationStatus,
            IterationTrigger,
            RecommendationType,
            TrackedRecommendation,
        )

        findings_dir = setup_environment["findings_dir"]
        orchestrator = IterationOrchestrator(str(findings_dir))

        ctx1 = orchestrator.start_new_iteration(IterationTrigger.INITIAL)

        rec1 = TrackedRecommendation(
            recommendation_id="rec_001",
            recommendation_type=RecommendationType.CLEANING,
            source_column="age",
            action="impute",
            description="Impute age"
        )
        rec2 = TrackedRecommendation(
            recommendation_id="rec_002",
            recommendation_type=RecommendationType.TRANSFORM,
            source_column="income",
            action="log",
            description="Log income"
        )
        orchestrator.track_recommendation(rec1)
        orchestrator.track_recommendation(rec2)

        orchestrator.apply_recommendation("rec_001")
        orchestrator.update_status(IterationStatus.COMPLETED)

        ctx2 = orchestrator.start_child_iteration(IterationTrigger.MANUAL)

        tracker = orchestrator.get_recommendation_tracker()
        pending = tracker.get_pending()
        assert len(pending) == 1
        assert pending[0].recommendation_id == "rec_002"

        orchestrator.apply_recommendation("rec_002")

        applied = tracker.get_applied()
        assert len(applied) == 2
        assert applied[0].applied_in_iteration == ctx1.iteration_id
        assert applied[1].applied_in_iteration == ctx2.iteration_id

    def test_compare_iterations_metrics(self, setup_environment):
        from customer_retention.integrations.iteration import IterationOrchestrator, IterationStatus, IterationTrigger

        findings_dir = setup_environment["findings_dir"]
        orchestrator = IterationOrchestrator(str(findings_dir))

        ctx1 = orchestrator.start_new_iteration(IterationTrigger.INITIAL)
        orchestrator.record_model_metrics({"roc_auc": 0.80, "pr_auc": 0.65})
        orchestrator.update_status(IterationStatus.COMPLETED)

        ctx2 = orchestrator.start_child_iteration(IterationTrigger.MANUAL)
        orchestrator.record_model_metrics({"roc_auc": 0.85, "pr_auc": 0.72})

        comparison = orchestrator.compare_iterations(ctx1.iteration_id, ctx2.iteration_id)
        assert comparison["metric_changes"]["roc_auc"] == pytest.approx(0.05)
        assert comparison["metric_changes"]["pr_auc"] == pytest.approx(0.07)
        assert comparison["iteration_diff"] == 1
