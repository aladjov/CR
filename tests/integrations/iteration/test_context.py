
import pytest


class TestIterationStatus:
    def test_iteration_status_values(self):
        from customer_retention.integrations.iteration.context import IterationStatus
        assert IterationStatus.EXPLORING.value == "exploring"
        assert IterationStatus.TRAINING.value == "training"
        assert IterationStatus.EVALUATING.value == "evaluating"
        assert IterationStatus.COMPLETED.value == "completed"
        assert IterationStatus.FAILED.value == "failed"


class TestIterationTrigger:
    def test_iteration_trigger_values(self):
        from customer_retention.integrations.iteration.context import IterationTrigger
        assert IterationTrigger.INITIAL.value == "initial"
        assert IterationTrigger.MANUAL.value == "manual"
        assert IterationTrigger.DRIFT_DETECTED.value == "drift_detected"
        assert IterationTrigger.PERFORMANCE_DROP.value == "performance_drop"
        assert IterationTrigger.SCHEDULED.value == "scheduled"


class TestIterationContext:
    def test_create_new_iteration(self):
        from customer_retention.integrations.iteration.context import IterationContext, IterationTrigger
        ctx = IterationContext.create_new(
            findings_dir="/path/to/findings",
            trigger=IterationTrigger.INITIAL
        )
        assert ctx.iteration_id is not None
        assert len(ctx.iteration_id) == 36  # UUID format
        assert ctx.iteration_number == 1
        assert ctx.parent_iteration_id is None
        assert ctx.trigger == IterationTrigger.INITIAL
        assert ctx.findings_path.startswith("/path/to/findings")
        assert ctx.recommendations_path.startswith("/path/to/findings")

    def test_create_child_iteration(self):
        from customer_retention.integrations.iteration.context import IterationContext, IterationTrigger
        parent = IterationContext.create_new(
            findings_dir="/path/to/findings",
            trigger=IterationTrigger.INITIAL
        )
        child = parent.create_child(trigger=IterationTrigger.DRIFT_DETECTED)
        assert child.parent_iteration_id == parent.iteration_id
        assert child.iteration_number == parent.iteration_number + 1
        assert child.trigger == IterationTrigger.DRIFT_DETECTED
        assert child.iteration_id != parent.iteration_id

    def test_iteration_context_default_values(self):
        from customer_retention.integrations.iteration.context import (
            IterationContext,
            IterationStatus,
            IterationTrigger,
        )
        ctx = IterationContext.create_new("/findings", IterationTrigger.INITIAL)
        assert ctx.status == IterationStatus.EXPLORING
        assert ctx.model_artifact_path is None
        assert ctx.model_metrics is None
        assert ctx.feature_count == 0
        assert ctx.applied_recommendations == []
        assert ctx.skipped_recommendations == []

    def test_update_status(self):
        from customer_retention.integrations.iteration.context import (
            IterationContext,
            IterationStatus,
            IterationTrigger,
        )
        ctx = IterationContext.create_new("/findings", IterationTrigger.INITIAL)
        ctx.update_status(IterationStatus.TRAINING)
        assert ctx.status == IterationStatus.TRAINING

    def test_set_model_metrics(self):
        from customer_retention.integrations.iteration.context import IterationContext, IterationTrigger
        ctx = IterationContext.create_new("/findings", IterationTrigger.INITIAL)
        metrics = {"roc_auc": 0.85, "pr_auc": 0.72}
        ctx.set_model_metrics(metrics, artifact_path="/models/model_v1")
        assert ctx.model_metrics == metrics
        assert ctx.model_artifact_path == "/models/model_v1"

    def test_add_applied_recommendation(self):
        from customer_retention.integrations.iteration.context import IterationContext, IterationTrigger
        ctx = IterationContext.create_new("/findings", IterationTrigger.INITIAL)
        ctx.add_applied_recommendation("rec_001")
        ctx.add_applied_recommendation("rec_002")
        assert "rec_001" in ctx.applied_recommendations
        assert "rec_002" in ctx.applied_recommendations
        assert len(ctx.applied_recommendations) == 2

    def test_add_skipped_recommendation(self):
        from customer_retention.integrations.iteration.context import IterationContext, IterationTrigger
        ctx = IterationContext.create_new("/findings", IterationTrigger.INITIAL)
        ctx.add_skipped_recommendation("rec_003")
        assert "rec_003" in ctx.skipped_recommendations

    def test_save_and_load(self, tmp_path):
        from customer_retention.integrations.iteration.context import (
            IterationContext,
            IterationStatus,
            IterationTrigger,
        )
        ctx = IterationContext.create_new(str(tmp_path), IterationTrigger.INITIAL)
        ctx.update_status(IterationStatus.TRAINING)
        ctx.set_model_metrics({"roc_auc": 0.85}, "/models/v1")
        ctx.add_applied_recommendation("rec_001")
        ctx.feature_count = 15

        save_path = tmp_path / "iteration.yaml"
        ctx.save(str(save_path))
        assert save_path.exists()

        loaded = IterationContext.load(str(save_path))
        assert loaded.iteration_id == ctx.iteration_id
        assert loaded.iteration_number == ctx.iteration_number
        assert loaded.status == IterationStatus.TRAINING
        assert loaded.trigger == IterationTrigger.INITIAL
        assert loaded.model_metrics == {"roc_auc": 0.85}
        assert loaded.model_artifact_path == "/models/v1"
        assert loaded.applied_recommendations == ["rec_001"]
        assert loaded.feature_count == 15

    def test_to_dict(self):
        from customer_retention.integrations.iteration.context import IterationContext, IterationTrigger
        ctx = IterationContext.create_new("/findings", IterationTrigger.INITIAL)
        data = ctx.to_dict()
        assert "iteration_id" in data
        assert "iteration_number" in data
        assert "status" in data
        assert "trigger" in data
        assert data["status"] == "exploring"
        assert data["trigger"] == "initial"

    def test_compare_iterations(self):
        from customer_retention.integrations.iteration.context import IterationContext, IterationTrigger
        ctx1 = IterationContext.create_new("/findings", IterationTrigger.INITIAL)
        ctx1.set_model_metrics({"roc_auc": 0.80, "pr_auc": 0.65}, "/models/v1")
        ctx1.feature_count = 10
        ctx1.add_applied_recommendation("rec_001")

        ctx2 = ctx1.create_child(IterationTrigger.MANUAL)
        ctx2.set_model_metrics({"roc_auc": 0.85, "pr_auc": 0.70}, "/models/v2")
        ctx2.feature_count = 12
        ctx2.add_applied_recommendation("rec_002")

        comparison = ctx2.compare(ctx1)
        assert comparison["iteration_diff"] == 1
        assert comparison["metric_changes"]["roc_auc"] == pytest.approx(0.05)
        assert comparison["metric_changes"]["pr_auc"] == pytest.approx(0.05)
        assert comparison["feature_count_change"] == 2
        assert comparison["new_recommendations"] == ["rec_002"]

    def test_compare_iterations_no_metrics(self):
        from customer_retention.integrations.iteration.context import IterationContext, IterationTrigger
        ctx1 = IterationContext.create_new("/findings", IterationTrigger.INITIAL)
        ctx2 = ctx1.create_child(IterationTrigger.MANUAL)
        comparison = ctx2.compare(ctx1)
        assert comparison["metric_changes"] == {}

    def test_get_iteration_filename(self):
        from customer_retention.integrations.iteration.context import IterationContext, IterationTrigger
        ctx = IterationContext.create_new("/findings", IterationTrigger.INITIAL)
        filename = ctx.get_iteration_filename()
        assert filename.startswith("iteration_001_")
        assert filename.endswith(".yaml")


class TestIterationContextManager:
    def test_list_iterations(self, tmp_path):
        from customer_retention.integrations.iteration.context import (
            IterationContext,
            IterationContextManager,
            IterationTrigger,
        )
        iterations_dir = tmp_path / "iterations"
        iterations_dir.mkdir()

        ctx1 = IterationContext.create_new(str(tmp_path), IterationTrigger.INITIAL)
        ctx1.save(str(iterations_dir / ctx1.get_iteration_filename()))

        ctx2 = ctx1.create_child(IterationTrigger.MANUAL)
        ctx2.save(str(iterations_dir / ctx2.get_iteration_filename()))

        manager = IterationContextManager(str(iterations_dir))
        iterations = manager.list_iterations()
        assert len(iterations) == 2

    def test_get_current_iteration(self, tmp_path):
        from customer_retention.integrations.iteration.context import (
            IterationContext,
            IterationContextManager,
            IterationTrigger,
        )
        iterations_dir = tmp_path / "iterations"
        iterations_dir.mkdir()

        ctx = IterationContext.create_new(str(tmp_path), IterationTrigger.INITIAL)
        ctx.save(str(iterations_dir / ctx.get_iteration_filename()))

        manager = IterationContextManager(str(iterations_dir))
        manager.set_current(ctx.iteration_id)

        current = manager.get_current()
        assert current is not None
        assert current.iteration_id == ctx.iteration_id

    def test_get_iteration_by_id(self, tmp_path):
        from customer_retention.integrations.iteration.context import (
            IterationContext,
            IterationContextManager,
            IterationTrigger,
        )
        iterations_dir = tmp_path / "iterations"
        iterations_dir.mkdir()

        ctx = IterationContext.create_new(str(tmp_path), IterationTrigger.INITIAL)
        ctx.save(str(iterations_dir / ctx.get_iteration_filename()))

        manager = IterationContextManager(str(iterations_dir))
        loaded = manager.get_by_id(ctx.iteration_id)
        assert loaded is not None
        assert loaded.iteration_id == ctx.iteration_id

    def test_get_iteration_history(self, tmp_path):
        from customer_retention.integrations.iteration.context import (
            IterationContext,
            IterationContextManager,
            IterationTrigger,
        )
        iterations_dir = tmp_path / "iterations"
        iterations_dir.mkdir()

        ctx1 = IterationContext.create_new(str(tmp_path), IterationTrigger.INITIAL)
        ctx1.save(str(iterations_dir / ctx1.get_iteration_filename()))

        ctx2 = ctx1.create_child(IterationTrigger.DRIFT_DETECTED)
        ctx2.save(str(iterations_dir / ctx2.get_iteration_filename()))

        ctx3 = ctx2.create_child(IterationTrigger.MANUAL)
        ctx3.save(str(iterations_dir / ctx3.get_iteration_filename()))

        manager = IterationContextManager(str(iterations_dir))
        history = manager.get_iteration_history(ctx3.iteration_id)
        assert len(history) == 3
        assert history[0].iteration_id == ctx1.iteration_id
        assert history[1].iteration_id == ctx2.iteration_id
        assert history[2].iteration_id == ctx3.iteration_id
