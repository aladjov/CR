import pytest
from datetime import datetime
from pathlib import Path


class TestModelFeedback:
    def test_create_model_feedback(self):
        from customer_retention.integrations.iteration.feedback_collector import ModelFeedback
        feedback = ModelFeedback(
            iteration_id="iter_001",
            model_type="RandomForestClassifier",
            metrics={"roc_auc": 0.85, "pr_auc": 0.72},
            feature_importances={"age": 0.25, "income": 0.35, "tenure": 0.40}
        )
        assert feedback.iteration_id == "iter_001"
        assert feedback.model_type == "RandomForestClassifier"
        assert feedback.metrics["roc_auc"] == 0.85
        assert len(feedback.feature_importances) == 3

    def test_model_feedback_with_confusion_matrix(self):
        from customer_retention.integrations.iteration.feedback_collector import ModelFeedback
        feedback = ModelFeedback(
            iteration_id="iter_001",
            model_type="XGBClassifier",
            metrics={"roc_auc": 0.88},
            feature_importances={"a": 0.5, "b": 0.5},
            confusion_matrix=[[100, 20], [15, 85]]
        )
        assert feedback.confusion_matrix == [[100, 20], [15, 85]]

    def test_to_dict(self):
        from customer_retention.integrations.iteration.feedback_collector import ModelFeedback
        feedback = ModelFeedback(
            iteration_id="iter_001",
            model_type="LogisticRegression",
            metrics={"roc_auc": 0.80},
            feature_importances={"x": 0.5, "y": 0.5}
        )
        data = feedback.to_dict()
        assert data["iteration_id"] == "iter_001"
        assert data["model_type"] == "LogisticRegression"
        assert "collected_at" in data

    def test_from_dict(self):
        from customer_retention.integrations.iteration.feedback_collector import ModelFeedback
        data = {
            "iteration_id": "iter_002",
            "model_type": "GradientBoosting",
            "metrics": {"roc_auc": 0.90},
            "feature_importances": {"a": 0.3, "b": 0.7},
            "confusion_matrix": [[80, 10], [5, 95]],
            "collected_at": "2024-01-15T10:00:00"
        }
        feedback = ModelFeedback.from_dict(data)
        assert feedback.iteration_id == "iter_002"
        assert feedback.metrics["roc_auc"] == 0.90


class TestFeatureInsight:
    def test_create_feature_insight(self):
        from customer_retention.integrations.iteration.feedback_collector import FeatureInsight
        insight = FeatureInsight(
            feature_name="age",
            importance_rank=1,
            importance_score=0.35,
            recommendation_to_drop=False,
            recommendation_to_engineer=None
        )
        assert insight.feature_name == "age"
        assert insight.importance_rank == 1
        assert insight.recommendation_to_drop is False

    def test_feature_insight_with_recommendation(self):
        from customer_retention.integrations.iteration.feedback_collector import FeatureInsight
        insight = FeatureInsight(
            feature_name="income",
            importance_rank=2,
            importance_score=0.25,
            recommendation_to_drop=False,
            recommendation_to_engineer="Consider binning into income brackets"
        )
        assert insight.recommendation_to_engineer is not None

    def test_feature_insight_to_drop(self):
        from customer_retention.integrations.iteration.feedback_collector import FeatureInsight
        insight = FeatureInsight(
            feature_name="unused_col",
            importance_rank=15,
            importance_score=0.001,
            recommendation_to_drop=True,
            recommendation_to_engineer=None
        )
        assert insight.recommendation_to_drop is True


class TestModelFeedbackCollector:
    def test_create_collector(self):
        from customer_retention.integrations.iteration.feedback_collector import ModelFeedbackCollector
        collector = ModelFeedbackCollector()
        assert collector is not None

    def test_create_from_sklearn_model(self):
        from customer_retention.integrations.iteration.feedback_collector import ModelFeedbackCollector
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])
        model.fit(X, y)

        collector = ModelFeedbackCollector()
        feedback = collector.create_from_sklearn(
            model=model,
            iteration_id="iter_001",
            feature_names=["feature_a", "feature_b"],
            metrics={"roc_auc": 0.85}
        )
        assert feedback.iteration_id == "iter_001"
        assert feedback.model_type == "RandomForestClassifier"
        assert len(feedback.feature_importances) == 2
        assert "feature_a" in feedback.feature_importances

    def test_analyze_feature_importance(self):
        from customer_retention.integrations.iteration.feedback_collector import (
            ModelFeedbackCollector, ModelFeedback
        )
        feedback = ModelFeedback(
            iteration_id="iter_001",
            model_type="RandomForest",
            metrics={"roc_auc": 0.85},
            feature_importances={
                "high_imp": 0.40,
                "med_imp": 0.25,
                "low_imp": 0.10,
                "very_low": 0.001
            }
        )
        collector = ModelFeedbackCollector()
        insights = collector.analyze_feature_importance(feedback)

        assert len(insights) == 4
        assert insights[0].feature_name == "high_imp"
        assert insights[0].importance_rank == 1
        assert insights[-1].recommendation_to_drop is True

    def test_analyze_feature_importance_drop_threshold(self):
        from customer_retention.integrations.iteration.feedback_collector import (
            ModelFeedbackCollector, ModelFeedback
        )
        feedback = ModelFeedback(
            iteration_id="iter_001",
            model_type="RandomForest",
            metrics={},
            feature_importances={
                "important": 0.50,
                "borderline": 0.009,
                "useless": 0.001
            }
        )
        collector = ModelFeedbackCollector(drop_threshold=0.01)
        insights = collector.analyze_feature_importance(feedback)

        to_drop = [i for i in insights if i.recommendation_to_drop]
        assert len(to_drop) == 2  # borderline and useless

    def test_suggest_next_actions(self):
        from customer_retention.integrations.iteration.feedback_collector import (
            ModelFeedbackCollector, ModelFeedback
        )
        feedback = ModelFeedback(
            iteration_id="iter_001",
            model_type="RandomForest",
            metrics={"roc_auc": 0.85, "pr_auc": 0.70},
            feature_importances={
                "age": 0.30,
                "income": 0.25,
                "unused1": 0.001,
                "unused2": 0.002
            }
        )
        collector = ModelFeedbackCollector()
        actions = collector.suggest_next_actions(feedback)

        assert len(actions) > 0
        # Should suggest dropping low importance features
        drop_action = any("drop" in a.lower() for a in actions)
        assert drop_action

    def test_suggest_actions_good_performance(self):
        from customer_retention.integrations.iteration.feedback_collector import (
            ModelFeedbackCollector, ModelFeedback
        )
        feedback = ModelFeedback(
            iteration_id="iter_001",
            model_type="RandomForest",
            metrics={"roc_auc": 0.95, "pr_auc": 0.92},
            feature_importances={"a": 0.5, "b": 0.5}
        )
        collector = ModelFeedbackCollector()
        actions = collector.suggest_next_actions(feedback)

        # Should note good performance
        good_action = any("good" in a.lower() or "excellent" in a.lower() for a in actions)
        assert good_action

    def test_compare_feedback(self):
        from customer_retention.integrations.iteration.feedback_collector import (
            ModelFeedbackCollector, ModelFeedback
        )
        feedback1 = ModelFeedback(
            iteration_id="iter_001",
            model_type="RandomForest",
            metrics={"roc_auc": 0.80, "pr_auc": 0.65},
            feature_importances={"a": 0.3, "b": 0.7}
        )
        feedback2 = ModelFeedback(
            iteration_id="iter_002",
            model_type="RandomForest",
            metrics={"roc_auc": 0.85, "pr_auc": 0.72},
            feature_importances={"a": 0.4, "b": 0.6}
        )
        collector = ModelFeedbackCollector()
        comparison = collector.compare_feedback(feedback1, feedback2)

        assert comparison["metric_improvements"]["roc_auc"] == pytest.approx(0.05)
        assert comparison["metric_improvements"]["pr_auc"] == pytest.approx(0.07)
        assert "improved" in comparison["overall_trend"]

    def test_compare_feedback_degraded(self):
        from customer_retention.integrations.iteration.feedback_collector import (
            ModelFeedbackCollector, ModelFeedback
        )
        feedback1 = ModelFeedback(
            iteration_id="iter_001",
            model_type="RF",
            metrics={"roc_auc": 0.90},
            feature_importances={}
        )
        feedback2 = ModelFeedback(
            iteration_id="iter_002",
            model_type="RF",
            metrics={"roc_auc": 0.82},
            feature_importances={}
        )
        collector = ModelFeedbackCollector()
        comparison = collector.compare_feedback(feedback1, feedback2)

        assert comparison["metric_improvements"]["roc_auc"] == pytest.approx(-0.08)
        assert "degraded" in comparison["overall_trend"]

    def test_save_and_load_feedback(self, tmp_path):
        from customer_retention.integrations.iteration.feedback_collector import (
            ModelFeedbackCollector, ModelFeedback
        )
        feedback = ModelFeedback(
            iteration_id="iter_001",
            model_type="XGBoost",
            metrics={"roc_auc": 0.88},
            feature_importances={"x": 0.4, "y": 0.6}
        )
        collector = ModelFeedbackCollector()
        path = tmp_path / "feedback.yaml"
        collector.save_feedback(feedback, str(path))

        loaded = collector.load_feedback(str(path))
        assert loaded.iteration_id == "iter_001"
        assert loaded.metrics["roc_auc"] == 0.88

    def test_get_top_features(self):
        from customer_retention.integrations.iteration.feedback_collector import (
            ModelFeedbackCollector, ModelFeedback
        )
        feedback = ModelFeedback(
            iteration_id="iter_001",
            model_type="RF",
            metrics={},
            feature_importances={
                "f1": 0.10, "f2": 0.05, "f3": 0.30,
                "f4": 0.20, "f5": 0.35
            }
        )
        collector = ModelFeedbackCollector()
        top = collector.get_top_features(feedback, n=3)

        assert len(top) == 3
        assert top[0][0] == "f5"  # Highest importance
        assert top[1][0] == "f3"
        assert top[2][0] == "f4"

    def test_get_low_importance_features(self):
        from customer_retention.integrations.iteration.feedback_collector import (
            ModelFeedbackCollector, ModelFeedback
        )
        feedback = ModelFeedback(
            iteration_id="iter_001",
            model_type="RF",
            metrics={},
            feature_importances={
                "good1": 0.30, "good2": 0.25,
                "bad1": 0.005, "bad2": 0.003, "bad3": 0.001
            }
        )
        collector = ModelFeedbackCollector(drop_threshold=0.01)
        low = collector.get_low_importance_features(feedback)

        assert len(low) == 3
        assert all(name in low for name in ["bad1", "bad2", "bad3"])
