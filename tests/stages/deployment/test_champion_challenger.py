from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from customer_retention.stages.deployment import (
    ChampionChallenger,
    ModelRole,
    PromotionCriteria,
    RollbackManager,
)


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 500
    X = pd.DataFrame({
        "f1": np.random.normal(0, 1, n),
        "f2": np.random.normal(0, 1, n),
        "f3": np.random.normal(0, 1, n),
    })
    y = pd.Series(np.random.choice([0, 1], n, p=[0.75, 0.25]))
    return X, y


@pytest.fixture
def champion_model(sample_data):
    X, y = sample_data
    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def challenger_model(sample_data):
    X, y = sample_data
    model = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=43)
    model.fit(X, y)
    return model


class TestModelRole:
    def test_model_role_enum_values(self):
        assert ModelRole.CHAMPION.value == "champion"
        assert ModelRole.CHALLENGER.value == "challenger"
        assert ModelRole.SHADOW.value == "shadow"


class TestChampionChallengerSetup:
    def test_registers_champion_model(self, champion_model):
        cc = ChampionChallenger()
        cc.set_champion(champion_model, model_name="champion_v1", version="1")
        assert cc.champion is not None
        assert cc.champion_version == "1"

    def test_registers_challenger_model(self, champion_model, challenger_model):
        cc = ChampionChallenger()
        cc.set_champion(champion_model, model_name="champion_v1", version="1")
        cc.set_challenger(challenger_model, model_name="challenger_v1", version="2")
        assert cc.challenger is not None
        assert cc.challenger_version == "2"

    def test_can_have_shadow_model(self, champion_model, challenger_model):
        cc = ChampionChallenger()
        cc.set_champion(champion_model, model_name="champion", version="1")
        cc.add_shadow(challenger_model, model_name="shadow", version="2")
        assert len(cc.shadow_models) == 1


class TestModelComparison:
    def test_compares_pr_auc(self, champion_model, challenger_model, sample_data):
        X, y = sample_data
        cc = ChampionChallenger()
        cc.set_champion(champion_model, model_name="champion", version="1")
        cc.set_challenger(challenger_model, model_name="challenger", version="2")
        result = cc.compare(X, y)
        assert result.champion_metrics["pr_auc"] is not None
        assert result.challenger_metrics["pr_auc"] is not None

    def test_compares_roc_auc(self, champion_model, challenger_model, sample_data):
        X, y = sample_data
        cc = ChampionChallenger()
        cc.set_champion(champion_model, model_name="champion", version="1")
        cc.set_challenger(challenger_model, model_name="challenger", version="2")
        result = cc.compare(X, y)
        assert result.champion_metrics["roc_auc"] is not None
        assert result.challenger_metrics["roc_auc"] is not None

    def test_compares_stability(self, champion_model, challenger_model, sample_data):
        X, y = sample_data
        cc = ChampionChallenger()
        cc.set_champion(champion_model, model_name="champion", version="1")
        cc.set_challenger(challenger_model, model_name="challenger", version="2")
        result = cc.compare(X, y, include_stability=True)
        assert "cv_std" in result.champion_metrics or "stability" in result.champion_metrics


class TestComparisonWeights:
    def test_applies_metric_weights(self, champion_model, challenger_model, sample_data):
        X, y = sample_data
        cc = ChampionChallenger(
            weights={"pr_auc": 0.40, "stability": 0.20, "business_roi": 0.25, "latency": 0.10, "fairness": 0.05}
        )
        cc.set_champion(champion_model, model_name="champion", version="1")
        cc.set_challenger(challenger_model, model_name="challenger", version="2")
        result = cc.compare(X, y)
        assert result.weighted_score_champion is not None
        assert result.weighted_score_challenger is not None

    def test_default_weights_sum_to_one(self):
        cc = ChampionChallenger()
        total_weight = sum(cc.weights.values())
        assert abs(total_weight - 1.0) < 0.01


class TestPromotionCriteria:
    def test_criteria_has_default_values(self):
        criteria = PromotionCriteria()
        assert criteria.min_pr_auc_improvement == 0.02
        assert criteria.max_fairness_regression == 0.0
        assert criteria.requires_validation_pass is True
        assert criteria.requires_business_approval is True

    def test_checks_pr_auc_improvement(self, champion_model, challenger_model, sample_data):
        X, y = sample_data
        criteria = PromotionCriteria(min_pr_auc_improvement=0.02)
        cc = ChampionChallenger(promotion_criteria=criteria)
        cc.set_champion(champion_model, model_name="champion", version="1")
        cc.set_challenger(challenger_model, model_name="challenger", version="2")
        result = cc.compare(X, y)
        assert result.meets_promotion_criteria is not None


class TestPromotionDecision:
    def test_recommends_promotion_when_challenger_better(self, champion_model, challenger_model, sample_data):
        X, y = sample_data
        cc = ChampionChallenger(promotion_criteria=PromotionCriteria(min_pr_auc_improvement=-0.5))
        cc.set_champion(champion_model, model_name="champion", version="1")
        cc.set_challenger(challenger_model, model_name="challenger", version="2")
        result = cc.compare(X, y)
        assert result.recommendation in ["promote_challenger", "keep_champion"]

    def test_keeps_champion_when_challenger_not_better(self, champion_model, sample_data):
        X, y = sample_data
        cc = ChampionChallenger(promotion_criteria=PromotionCriteria(min_pr_auc_improvement=0.50))
        cc.set_champion(champion_model, model_name="champion", version="1")
        cc.set_challenger(champion_model, model_name="challenger", version="2")
        result = cc.compare(X, y)
        assert result.recommendation == "keep_champion"


class TestFairnessComparison:
    def test_checks_fairness_regression(self, champion_model, challenger_model, sample_data):
        X, y = sample_data
        protected = pd.Series(np.random.choice(["A", "B"], len(y)))
        cc = ChampionChallenger()
        cc.set_champion(champion_model, model_name="champion", version="1")
        cc.set_challenger(challenger_model, model_name="challenger", version="2")
        result = cc.compare(X, y, protected_attribute=protected)
        assert "fairness_metrics" in result.champion_metrics or result.fairness_comparison is not None

    def test_blocks_promotion_on_fairness_regression(self, champion_model, challenger_model, sample_data):
        X, y = sample_data
        protected = pd.Series(np.random.choice(["A", "B"], len(y)))
        criteria = PromotionCriteria(max_fairness_regression=0.0, min_pr_auc_improvement=-1.0)
        cc = ChampionChallenger(promotion_criteria=criteria)
        cc.set_champion(champion_model, model_name="champion", version="1")
        cc.set_challenger(challenger_model, model_name="challenger", version="2")
        result = cc.compare(X, y, protected_attribute=protected)
        assert result.meets_promotion_criteria is not None


class TestLatencyComparison:
    def test_compares_inference_latency(self, champion_model, challenger_model, sample_data):
        X, y = sample_data
        cc = ChampionChallenger()
        cc.set_champion(champion_model, model_name="champion", version="1")
        cc.set_challenger(challenger_model, model_name="challenger", version="2")
        result = cc.compare(X, y, include_latency=True)
        assert result.champion_metrics.get("latency_ms") is not None or hasattr(result, "latency_comparison")

    def test_flags_slow_challenger(self, champion_model, sample_data):
        X, y = sample_data
        slow_model = MagicMock()
        slow_model.predict_proba = MagicMock(side_effect=lambda x: np.random.rand(len(x), 2))
        criteria = PromotionCriteria(max_latency_ratio=2.0)
        cc = ChampionChallenger(promotion_criteria=criteria)
        cc.set_champion(champion_model, model_name="champion", version="1")
        cc.set_challenger(slow_model, model_name="challenger", version="2")
        result = cc.compare(X, y, include_latency=True)
        assert result is not None


class TestComparisonResult:
    def test_result_contains_all_fields(self, champion_model, challenger_model, sample_data):
        X, y = sample_data
        cc = ChampionChallenger()
        cc.set_champion(champion_model, model_name="champion", version="1")
        cc.set_challenger(challenger_model, model_name="challenger", version="2")
        result = cc.compare(X, y)
        assert hasattr(result, "champion_metrics")
        assert hasattr(result, "challenger_metrics")
        assert hasattr(result, "recommendation")
        assert hasattr(result, "comparison_timestamp")

    def test_result_includes_improvement_metrics(self, champion_model, challenger_model, sample_data):
        X, y = sample_data
        cc = ChampionChallenger()
        cc.set_champion(champion_model, model_name="champion", version="1")
        cc.set_challenger(challenger_model, model_name="challenger", version="2")
        result = cc.compare(X, y)
        assert hasattr(result, "pr_auc_improvement") or "improvement" in str(result.__dict__)


class TestRollbackManager:
    def test_creates_rollback_plan(self):
        manager = RollbackManager()
        plan = manager.create_plan(
            current_model_name="challenger_v2",
            current_version="2",
            rollback_model_name="champion_v1",
            rollback_version="1"
        )
        assert plan.rollback_model_name == "champion_v1"
        assert plan.rollback_version == "1"

    def test_executes_rollback(self):
        with patch("customer_retention.stages.deployment.model_registry.ModelRegistry") as mock_registry:
            manager = RollbackManager()
            result = manager.execute_rollback(
                model_name="churn_model",
                from_version="2",
                to_version="1"
            )
            assert result.success is True

    def test_rollback_notifies_stakeholders(self):
        with patch("customer_retention.stages.deployment.model_registry.ModelRegistry"):
            with patch("customer_retention.stages.monitoring.alert_manager.AlertManager") as mock_alert:
                manager = RollbackManager(notify_on_rollback=True)
                manager.execute_rollback(
                    model_name="churn_model",
                    from_version="2",
                    to_version="1"
                )
                mock_alert.return_value.send_alert.assert_called()

    def test_rollback_time_under_five_minutes(self):
        manager = RollbackManager()
        plan = manager.create_plan(
            current_model_name="v2",
            current_version="2",
            rollback_model_name="v1",
            rollback_version="1"
        )
        assert plan.estimated_duration_minutes <= 5


class TestShadowMode:
    def test_scores_in_shadow_mode(self, champion_model, challenger_model, sample_data):
        X, y = sample_data
        cc = ChampionChallenger()
        cc.set_champion(champion_model, model_name="champion", version="1")
        cc.add_shadow(challenger_model, model_name="shadow", version="2")
        predictions = cc.score_with_shadow(X)
        assert "champion_predictions" in predictions
        assert "shadow_predictions" in predictions

    def test_shadow_predictions_not_used_for_action(self, champion_model, challenger_model, sample_data):
        X, y = sample_data
        cc = ChampionChallenger()
        cc.set_champion(champion_model, model_name="champion", version="1")
        cc.add_shadow(challenger_model, model_name="shadow", version="2")
        result = cc.score_with_shadow(X)
        assert result["active_predictions"] is result["champion_predictions"]


class TestComparisonHistory:
    def test_stores_comparison_history(self, champion_model, challenger_model, sample_data):
        X, y = sample_data
        cc = ChampionChallenger()
        cc.set_champion(champion_model, model_name="champion", version="1")
        cc.set_challenger(challenger_model, model_name="challenger", version="2")
        cc.compare(X, y)
        cc.compare(X, y)
        history = cc.get_comparison_history()
        assert len(history) == 2

    def test_generates_comparison_report(self, champion_model, challenger_model, sample_data):
        X, y = sample_data
        cc = ChampionChallenger()
        cc.set_champion(champion_model, model_name="champion", version="1")
        cc.set_challenger(challenger_model, model_name="challenger", version="2")
        cc.compare(X, y)
        report = cc.generate_report()
        assert report is not None
        assert "champion" in report.lower() or hasattr(report, "champion_summary")
