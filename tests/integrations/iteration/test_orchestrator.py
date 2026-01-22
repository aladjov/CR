
import pytest


class TestIterationOrchestrator:
    @pytest.fixture
    def setup_findings_dir(self, tmp_path):
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        (findings_dir / "iterations").mkdir()
        (findings_dir / "recommendations").mkdir()
        (findings_dir / "feedback").mkdir()
        return findings_dir

    def test_create_orchestrator(self, setup_findings_dir):
        from customer_retention.integrations.iteration.orchestrator import IterationOrchestrator
        orchestrator = IterationOrchestrator(str(setup_findings_dir))
        assert orchestrator is not None

    def test_start_new_iteration(self, setup_findings_dir):
        from customer_retention.integrations.iteration.context import IterationStatus, IterationTrigger
        from customer_retention.integrations.iteration.orchestrator import IterationOrchestrator

        orchestrator = IterationOrchestrator(str(setup_findings_dir))
        ctx = orchestrator.start_new_iteration(IterationTrigger.INITIAL)

        assert ctx is not None
        assert ctx.iteration_number == 1
        assert ctx.trigger == IterationTrigger.INITIAL
        assert ctx.status == IterationStatus.EXPLORING

    def test_get_current_iteration(self, setup_findings_dir):
        from customer_retention.integrations.iteration.context import IterationTrigger
        from customer_retention.integrations.iteration.orchestrator import IterationOrchestrator

        orchestrator = IterationOrchestrator(str(setup_findings_dir))
        ctx = orchestrator.start_new_iteration(IterationTrigger.INITIAL)

        current = orchestrator.get_current_iteration()
        assert current is not None
        assert current.iteration_id == ctx.iteration_id

    def test_start_child_iteration(self, setup_findings_dir):
        from customer_retention.integrations.iteration.context import IterationTrigger
        from customer_retention.integrations.iteration.orchestrator import IterationOrchestrator

        orchestrator = IterationOrchestrator(str(setup_findings_dir))
        parent = orchestrator.start_new_iteration(IterationTrigger.INITIAL)

        child = orchestrator.start_child_iteration(IterationTrigger.DRIFT_DETECTED)
        assert child.iteration_number == 2
        assert child.parent_iteration_id == parent.iteration_id

    def test_update_iteration_status(self, setup_findings_dir):
        from customer_retention.integrations.iteration.context import IterationStatus, IterationTrigger
        from customer_retention.integrations.iteration.orchestrator import IterationOrchestrator

        orchestrator = IterationOrchestrator(str(setup_findings_dir))
        orchestrator.start_new_iteration(IterationTrigger.INITIAL)

        orchestrator.update_status(IterationStatus.TRAINING)
        current = orchestrator.get_current_iteration()
        assert current.status == IterationStatus.TRAINING

    def test_record_model_metrics(self, setup_findings_dir):
        from customer_retention.integrations.iteration.context import IterationTrigger
        from customer_retention.integrations.iteration.orchestrator import IterationOrchestrator

        orchestrator = IterationOrchestrator(str(setup_findings_dir))
        orchestrator.start_new_iteration(IterationTrigger.INITIAL)

        metrics = {"roc_auc": 0.85, "pr_auc": 0.72}
        orchestrator.record_model_metrics(metrics, artifact_path="/models/v1")

        current = orchestrator.get_current_iteration()
        assert current.model_metrics == metrics
        assert current.model_artifact_path == "/models/v1"

    def test_track_recommendation(self, setup_findings_dir):
        from customer_retention.integrations.iteration.context import IterationTrigger
        from customer_retention.integrations.iteration.orchestrator import IterationOrchestrator
        from customer_retention.integrations.iteration.recommendation_tracker import (
            RecommendationType,
            TrackedRecommendation,
        )

        orchestrator = IterationOrchestrator(str(setup_findings_dir))
        orchestrator.start_new_iteration(IterationTrigger.INITIAL)

        rec = TrackedRecommendation(
            recommendation_id="rec_001",
            recommendation_type=RecommendationType.CLEANING,
            source_column="age",
            action="impute_median",
            description="Impute missing values"
        )
        orchestrator.track_recommendation(rec)

        tracker = orchestrator.get_recommendation_tracker()
        assert tracker.get("rec_001") is not None

    def test_apply_recommendation(self, setup_findings_dir):
        from customer_retention.integrations.iteration.context import IterationTrigger
        from customer_retention.integrations.iteration.orchestrator import IterationOrchestrator
        from customer_retention.integrations.iteration.recommendation_tracker import (
            RecommendationStatus,
            RecommendationType,
            TrackedRecommendation,
        )

        orchestrator = IterationOrchestrator(str(setup_findings_dir))
        ctx = orchestrator.start_new_iteration(IterationTrigger.INITIAL)

        rec = TrackedRecommendation(
            recommendation_id="rec_001",
            recommendation_type=RecommendationType.CLEANING,
            source_column="age",
            action="impute_median",
            description="Impute missing values"
        )
        orchestrator.track_recommendation(rec)
        orchestrator.apply_recommendation("rec_001")

        tracker = orchestrator.get_recommendation_tracker()
        updated = tracker.get("rec_001")
        assert updated.status == RecommendationStatus.APPLIED
        assert updated.applied_in_iteration == ctx.iteration_id

        # Also check iteration context
        current = orchestrator.get_current_iteration()
        assert "rec_001" in current.applied_recommendations

    def test_skip_recommendation(self, setup_findings_dir):
        from customer_retention.integrations.iteration.context import IterationTrigger
        from customer_retention.integrations.iteration.orchestrator import IterationOrchestrator
        from customer_retention.integrations.iteration.recommendation_tracker import (
            RecommendationStatus,
            RecommendationType,
            TrackedRecommendation,
        )

        orchestrator = IterationOrchestrator(str(setup_findings_dir))
        orchestrator.start_new_iteration(IterationTrigger.INITIAL)

        rec = TrackedRecommendation(
            recommendation_id="rec_002",
            recommendation_type=RecommendationType.TRANSFORM,
            source_column="income",
            action="log_transform",
            description="Log transform"
        )
        orchestrator.track_recommendation(rec)
        orchestrator.skip_recommendation("rec_002", "Not needed")

        tracker = orchestrator.get_recommendation_tracker()
        updated = tracker.get("rec_002")
        assert updated.status == RecommendationStatus.SKIPPED

        current = orchestrator.get_current_iteration()
        assert "rec_002" in current.skipped_recommendations

    def test_collect_feedback(self, setup_findings_dir):
        from customer_retention.integrations.iteration.context import IterationTrigger
        from customer_retention.integrations.iteration.feedback_collector import ModelFeedback
        from customer_retention.integrations.iteration.orchestrator import IterationOrchestrator

        orchestrator = IterationOrchestrator(str(setup_findings_dir))
        orchestrator.start_new_iteration(IterationTrigger.INITIAL)

        feedback = ModelFeedback(
            iteration_id="iter_001",
            model_type="RandomForest",
            metrics={"roc_auc": 0.85},
            feature_importances={"age": 0.3, "income": 0.7}
        )
        orchestrator.collect_feedback(feedback)

        saved_feedback = orchestrator.get_latest_feedback()
        assert saved_feedback is not None
        assert saved_feedback.model_type == "RandomForest"

    def test_check_for_iteration_triggers(self, setup_findings_dir):
        from customer_retention.integrations.iteration.context import IterationTrigger
        from customer_retention.integrations.iteration.orchestrator import IterationOrchestrator

        orchestrator = IterationOrchestrator(str(setup_findings_dir))
        orchestrator.start_new_iteration(IterationTrigger.INITIAL)

        # No signals, should not trigger
        should_trigger, trigger = orchestrator.check_for_iteration_triggers()
        assert should_trigger is False

    def test_trigger_manual_iteration(self, setup_findings_dir):
        from customer_retention.integrations.iteration.context import IterationTrigger
        from customer_retention.integrations.iteration.orchestrator import IterationOrchestrator

        orchestrator = IterationOrchestrator(str(setup_findings_dir))
        orchestrator.start_new_iteration(IterationTrigger.INITIAL)

        orchestrator.trigger_manual_iteration("Testing new features")
        should_trigger, trigger = orchestrator.check_for_iteration_triggers()

        assert should_trigger is True
        assert trigger == IterationTrigger.MANUAL

    def test_prepare_iteration_from_feedback(self, setup_findings_dir):
        from customer_retention.integrations.iteration.context import IterationStatus, IterationTrigger
        from customer_retention.integrations.iteration.feedback_collector import ModelFeedback
        from customer_retention.integrations.iteration.orchestrator import IterationOrchestrator

        orchestrator = IterationOrchestrator(str(setup_findings_dir))
        orchestrator.start_new_iteration(IterationTrigger.INITIAL)
        orchestrator.update_status(IterationStatus.COMPLETED)

        feedback = ModelFeedback(
            iteration_id="iter_001",
            model_type="RandomForest",
            metrics={"roc_auc": 0.85},
            feature_importances={"age": 0.3, "income": 0.7}
        )
        orchestrator.collect_feedback(feedback)

        new_ctx = orchestrator.prepare_iteration_from_feedback(
            feedback, IterationTrigger.MANUAL
        )
        assert new_ctx.iteration_number == 2
        assert new_ctx.trigger == IterationTrigger.MANUAL

    def test_get_refined_recommendations(self, setup_findings_dir):
        from customer_retention.analysis.auto_explorer.findings import ColumnFinding, ExplorationFindings
        from customer_retention.core.config.column_config import ColumnType
        from customer_retention.integrations.iteration.context import IterationTrigger
        from customer_retention.integrations.iteration.feedback_collector import ModelFeedback
        from customer_retention.integrations.iteration.orchestrator import IterationOrchestrator

        orchestrator = IterationOrchestrator(str(setup_findings_dir))
        orchestrator.start_new_iteration(IterationTrigger.INITIAL)

        findings = ExplorationFindings(
            source_path="/data/test.csv",
            source_format="csv",
            row_count=100,
            column_count=3,
            columns={
                "age": ColumnFinding(
                    name="age",
                    inferred_type=ColumnType.NUMERIC_CONTINUOUS,
                    confidence=0.9,
                    evidence=[]
                ),
                "income": ColumnFinding(
                    name="income",
                    inferred_type=ColumnType.NUMERIC_CONTINUOUS,
                    confidence=0.9,
                    evidence=[]
                ),
                "unused": ColumnFinding(
                    name="unused",
                    inferred_type=ColumnType.NUMERIC_CONTINUOUS,
                    confidence=0.9,
                    evidence=[]
                )
            }
        )

        feedback = ModelFeedback(
            iteration_id="iter_001",
            model_type="RandomForest",
            metrics={"roc_auc": 0.85},
            feature_importances={
                "age": 0.4,
                "income": 0.59,
                "unused": 0.001  # Very low importance
            }
        )

        refined = orchestrator.get_refined_recommendations(findings, feedback)
        assert "features_to_drop" in refined
        assert "unused" in refined["features_to_drop"]
        assert "top_features" in refined

    def test_get_iteration_history(self, setup_findings_dir):
        from customer_retention.integrations.iteration.context import IterationStatus, IterationTrigger
        from customer_retention.integrations.iteration.orchestrator import IterationOrchestrator

        orchestrator = IterationOrchestrator(str(setup_findings_dir))

        # Create 3 iterations
        ctx1 = orchestrator.start_new_iteration(IterationTrigger.INITIAL)
        orchestrator.update_status(IterationStatus.COMPLETED)

        ctx2 = orchestrator.start_child_iteration(IterationTrigger.MANUAL)
        orchestrator.update_status(IterationStatus.COMPLETED)

        ctx3 = orchestrator.start_child_iteration(IterationTrigger.DRIFT_DETECTED)

        history = orchestrator.get_iteration_history()
        assert len(history) == 3
        assert history[0].iteration_number == 1
        assert history[2].iteration_number == 3

    def test_compare_iterations(self, setup_findings_dir):
        from customer_retention.integrations.iteration.context import IterationStatus, IterationTrigger
        from customer_retention.integrations.iteration.orchestrator import IterationOrchestrator

        orchestrator = IterationOrchestrator(str(setup_findings_dir))

        ctx1 = orchestrator.start_new_iteration(IterationTrigger.INITIAL)
        orchestrator.record_model_metrics({"roc_auc": 0.80}, "/models/v1")
        orchestrator.update_status(IterationStatus.COMPLETED)

        ctx2 = orchestrator.start_child_iteration(IterationTrigger.MANUAL)
        orchestrator.record_model_metrics({"roc_auc": 0.85}, "/models/v2")

        comparison = orchestrator.compare_iterations(ctx1.iteration_id, ctx2.iteration_id)
        assert comparison["metric_changes"]["roc_auc"] == pytest.approx(0.05)

    def test_save_state(self, setup_findings_dir):
        from customer_retention.integrations.iteration.context import IterationTrigger
        from customer_retention.integrations.iteration.orchestrator import IterationOrchestrator
        from customer_retention.integrations.iteration.recommendation_tracker import (
            RecommendationType,
            TrackedRecommendation,
        )

        orchestrator = IterationOrchestrator(str(setup_findings_dir))
        orchestrator.start_new_iteration(IterationTrigger.INITIAL)

        rec = TrackedRecommendation(
            recommendation_id="rec_001",
            recommendation_type=RecommendationType.CLEANING,
            source_column="age",
            action="impute",
            description="Impute"
        )
        orchestrator.track_recommendation(rec)
        orchestrator.apply_recommendation("rec_001")

        orchestrator.save_state()

        # Verify files exist
        iterations_dir = setup_findings_dir / "iterations"
        recommendations_path = setup_findings_dir / "recommendations" / "tracked.yaml"
        assert any(iterations_dir.glob("iteration_*.yaml"))
        assert recommendations_path.exists()

    def test_load_state(self, setup_findings_dir):
        from customer_retention.integrations.iteration.context import IterationTrigger
        from customer_retention.integrations.iteration.orchestrator import IterationOrchestrator
        from customer_retention.integrations.iteration.recommendation_tracker import (
            RecommendationStatus,
            RecommendationType,
            TrackedRecommendation,
        )

        # Create and save state
        orchestrator1 = IterationOrchestrator(str(setup_findings_dir))
        ctx = orchestrator1.start_new_iteration(IterationTrigger.INITIAL)
        rec = TrackedRecommendation(
            recommendation_id="rec_001",
            recommendation_type=RecommendationType.CLEANING,
            source_column="age",
            action="impute",
            description="Impute"
        )
        orchestrator1.track_recommendation(rec)
        orchestrator1.apply_recommendation("rec_001")
        orchestrator1.save_state()

        # Load in new orchestrator
        orchestrator2 = IterationOrchestrator(str(setup_findings_dir))
        orchestrator2.load_state()

        current = orchestrator2.get_current_iteration()
        assert current is not None
        assert current.iteration_id == ctx.iteration_id

        tracker = orchestrator2.get_recommendation_tracker()
        loaded_rec = tracker.get("rec_001")
        assert loaded_rec is not None
        assert loaded_rec.status == RecommendationStatus.APPLIED
