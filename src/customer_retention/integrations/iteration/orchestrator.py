from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .context import IterationContext, IterationContextManager, IterationStatus, IterationTrigger
from .feedback_collector import ModelFeedback, ModelFeedbackCollector
from .recommendation_tracker import RecommendationTracker, TrackedRecommendation
from .signals import SignalAggregator


class IterationOrchestrator:
    def __init__(self, findings_dir: str,
                 signal_aggregator: Optional[SignalAggregator] = None):
        self.findings_dir = Path(findings_dir)
        self.iterations_dir = self.findings_dir / "iterations"
        self.recommendations_dir = self.findings_dir / "recommendations"
        self.feedback_dir = self.findings_dir / "feedback"

        self.iterations_dir.mkdir(parents=True, exist_ok=True)
        self.recommendations_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)

        self._context_manager = IterationContextManager(str(self.iterations_dir))
        self._recommendation_tracker = RecommendationTracker(
            str(self.recommendations_dir / "tracked.yaml")
        )
        self._feedback_collector = ModelFeedbackCollector()
        self._signal_aggregator = signal_aggregator or SignalAggregator()

        self._current_context: Optional[IterationContext] = None
        self._latest_feedback: Optional[ModelFeedback] = None

    def start_new_iteration(self, trigger: IterationTrigger) -> IterationContext:
        ctx = IterationContext.create_new(str(self.findings_dir), trigger)
        self._current_context = ctx
        self._save_current_context()
        self._context_manager.set_current(ctx.iteration_id)
        return ctx

    def start_child_iteration(self, trigger: IterationTrigger) -> IterationContext:
        if self._current_context is None:
            return self.start_new_iteration(trigger)

        child = self._current_context.create_child(trigger)
        self._current_context = child
        self._save_current_context()
        self._context_manager.set_current(child.iteration_id)
        return child

    def get_current_iteration(self) -> Optional[IterationContext]:
        if self._current_context is not None:
            return self._current_context
        return self._context_manager.get_current()

    def update_status(self, status: IterationStatus) -> None:
        if self._current_context:
            self._current_context.update_status(status)
            self._save_current_context()

    def record_model_metrics(self, metrics: Dict[str, float],
                            artifact_path: Optional[str] = None) -> None:
        if self._current_context:
            self._current_context.set_model_metrics(metrics, artifact_path)
            self._save_current_context()

    def get_recommendation_tracker(self) -> RecommendationTracker:
        return self._recommendation_tracker

    def track_recommendation(self, recommendation: TrackedRecommendation) -> None:
        self._recommendation_tracker.add(recommendation)

    def apply_recommendation(self, recommendation_id: str) -> None:
        if self._current_context:
            self._recommendation_tracker.mark_applied(
                recommendation_id, self._current_context.iteration_id
            )
            self._current_context.add_applied_recommendation(recommendation_id)
            self._save_current_context()

    def skip_recommendation(self, recommendation_id: str, reason: str) -> None:
        self._recommendation_tracker.mark_skipped(recommendation_id, reason)
        if self._current_context:
            self._current_context.add_skipped_recommendation(recommendation_id)
            self._save_current_context()

    def collect_feedback(self, feedback: ModelFeedback) -> None:
        self._latest_feedback = feedback
        feedback_path = self.feedback_dir / f"feedback_{feedback.iteration_id}.yaml"
        self._feedback_collector.save_feedback(feedback, str(feedback_path))

    def get_latest_feedback(self) -> Optional[ModelFeedback]:
        return self._latest_feedback

    def check_for_iteration_triggers(self) -> Tuple[bool, Optional[IterationTrigger]]:
        return self._signal_aggregator.should_trigger_iteration()

    def trigger_manual_iteration(self, reason: str) -> None:
        self._signal_aggregator.add_manual_signal(reason, {})

    def prepare_iteration_from_feedback(self, feedback: ModelFeedback,
                                        trigger: IterationTrigger) -> IterationContext:
        new_ctx = self.start_child_iteration(trigger)

        insights = self._feedback_collector.analyze_feature_importance(feedback)
        [i.feature_name for i in insights if i.recommendation_to_drop]

        return new_ctx

    def get_refined_recommendations(self, findings, feedback: ModelFeedback) -> Dict[str, Any]:
        insights = self._feedback_collector.analyze_feature_importance(feedback)

        features_to_drop = [i.feature_name for i in insights if i.recommendation_to_drop]
        top_features = self._feedback_collector.get_top_features(feedback, n=5)

        refined = {
            "features_to_drop": features_to_drop,
            "top_features": [f[0] for f in top_features],
            "feature_insights": [
                {
                    "name": i.feature_name,
                    "rank": i.importance_rank,
                    "score": i.importance_score,
                    "drop": i.recommendation_to_drop
                }
                for i in insights
            ],
            "next_actions": self._feedback_collector.suggest_next_actions(feedback)
        }

        return refined

    def get_iteration_history(self) -> List[IterationContext]:
        if self._current_context:
            return self._context_manager.get_iteration_history(
                self._current_context.iteration_id
            )
        return self._context_manager.list_iterations()

    def compare_iterations(self, iteration_id_1: str,
                          iteration_id_2: str) -> Dict[str, Any]:
        ctx1 = self._context_manager.get_by_id(iteration_id_1)
        ctx2 = self._context_manager.get_by_id(iteration_id_2)

        if ctx1 is None or ctx2 is None:
            return {"error": "Iteration not found"}

        return ctx2.compare(ctx1)

    def save_state(self) -> None:
        if self._current_context:
            self._save_current_context()
        self._recommendation_tracker.save()

    def load_state(self) -> None:
        self._recommendation_tracker.load()

        current = self._context_manager.get_current()
        if current:
            self._current_context = current

        feedback_files = sorted(self.feedback_dir.glob("feedback_*.yaml"))
        if feedback_files:
            self._latest_feedback = self._feedback_collector.load_feedback(
                str(feedback_files[-1])
            )

    def _save_current_context(self) -> None:
        if self._current_context:
            self._context_manager.save_iteration(self._current_context)
