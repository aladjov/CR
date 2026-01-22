from customer_retention.core.components.enums import Severity
from customer_retention.stages.validation import BusinessSenseGate


class TestBusinessSenseChecks:
    def test_bs001_top_features_explainable(self):
        feature_importance = {
            "days_since_last_order": 0.35,
            "email_engagement": 0.25,
            "tenure_days": 0.20,
        }
        gate = BusinessSenseGate()
        result = gate.check_feature_explainability(feature_importance)
        assert hasattr(result, "checks")

    def test_bs005_roi_positive(self):
        cost_benefit = {
            "intervention_cost": 50,
            "customer_value": 500,
            "expected_lift": 0.10,
            "target_population": 1000,
        }
        gate = BusinessSenseGate()
        result = gate.check_roi(cost_benefit)
        assert hasattr(result, "roi")
        assert hasattr(result, "checks")

    def test_bs005_roi_negative_flagged(self):
        cost_benefit = {
            "intervention_cost": 500,
            "customer_value": 100,
            "expected_lift": 0.05,
            "target_population": 100,
        }
        gate = BusinessSenseGate()
        result = gate.check_roi(cost_benefit)
        critical_issues = [c for c in result.checks if c.severity == Severity.CRITICAL]
        assert len(critical_issues) > 0


class TestBusinessSenseGateRun:
    def test_gate_runs_all_checks(self):
        metrics = {
            "pr_auc_test": 0.65,
            "roc_auc_test": 0.75,
            "recall": 0.60,
        }
        feature_importance = {
            "days_since_last_order": 0.30,
            "email_engagement": 0.25,
        }
        gate = BusinessSenseGate()
        result = gate.run(metrics, feature_importance)
        assert hasattr(result, "passed")
        assert hasattr(result, "checks")

    def test_gate_blocks_on_critical(self):
        metrics = {"pr_auc_test": 0.98}
        feature_importance = {"unknown_feature_xyz": 0.90}
        gate = BusinessSenseGate()
        result = gate.run(metrics, feature_importance)
        assert result.passed is False


class TestBusinessSenseResult:
    def test_result_contains_required_fields(self):
        metrics = {"pr_auc_test": 0.65}
        feature_importance = {"feature1": 0.30}
        gate = BusinessSenseGate()
        result = gate.run(metrics, feature_importance)
        assert hasattr(result, "passed")
        assert hasattr(result, "checks")
        assert hasattr(result, "critical_issues")
        assert hasattr(result, "review_notes")
        assert hasattr(result, "recommendation")


class TestSignOffs:
    def test_tracks_sign_offs(self):
        gate = BusinessSenseGate()
        gate.add_sign_off("Data Engineer", "Features available at prediction time")
        gate.add_sign_off("Domain Expert", "Top features make business sense")
        result = gate.get_sign_offs()
        assert len(result) == 2

    def test_sign_off_required_for_production(self):
        gate = BusinessSenseGate(required_sign_offs=["Data Engineer", "Domain Expert"])
        gate.add_sign_off("Data Engineer", "Approved")
        result = gate.check_sign_offs()
        assert result.passed is False

        gate.add_sign_off("Domain Expert", "Approved")
        result = gate.check_sign_offs()
        assert result.passed is True
