from customer_retention.stages.validation import ModelValidityGate


class TestSuspiciousPRAUC:
    def test_mv001_flags_suspicious_pr_auc(self):
        metrics = {
            "pr_auc_test": 0.95,
            "roc_auc_test": 0.85,
            "train_test_gap": 0.05,
            "cv_std": 0.02,
            "recall": 0.50,
        }

        gate = ModelValidityGate()
        result = gate.run(metrics)

        assert not result.passed
        assert any("MV001" in str(issue) for issue in result.critical_issues)

    def test_mv001_passes_normal_pr_auc(self):
        metrics = {
            "pr_auc_test": 0.60,
            "roc_auc_test": 0.75,
            "train_test_gap": 0.05,
            "cv_std": 0.02,
            "recall": 0.50,
        }

        gate = ModelValidityGate()
        result = gate.run(metrics)

        mv001_issues = [i for i in result.critical_issues if "MV001" in str(i)]
        assert len(mv001_issues) == 0


class TestSuspiciousROCAUC:
    def test_mv002_flags_suspicious_roc_auc(self):
        metrics = {
            "pr_auc_test": 0.70,
            "roc_auc_test": 0.98,
            "train_test_gap": 0.05,
            "cv_std": 0.02,
            "recall": 0.50,
        }

        gate = ModelValidityGate()
        result = gate.run(metrics)

        assert any("MV002" in str(issue) for issue in result.high_issues)


class TestSevereOverfitting:
    def test_mv003_flags_severe_overfitting(self):
        metrics = {
            "pr_auc_test": 0.50,
            "pr_auc_train": 0.70,
            "roc_auc_test": 0.70,
            "train_test_gap": 0.20,
            "cv_std": 0.02,
            "recall": 0.50,
        }

        gate = ModelValidityGate()
        result = gate.run(metrics)

        assert not result.passed
        assert any("MV003" in str(issue) for issue in result.critical_issues)

    def test_mv004_flags_moderate_overfitting(self):
        metrics = {
            "pr_auc_test": 0.55,
            "roc_auc_test": 0.75,
            "train_test_gap": 0.12,
            "cv_std": 0.02,
            "recall": 0.50,
        }

        gate = ModelValidityGate()
        result = gate.run(metrics)

        assert any("MV004" in str(issue) for issue in result.high_issues)


class TestHighCVVariance:
    def test_mv005_flags_high_cv_variance(self):
        metrics = {
            "pr_auc_test": 0.55,
            "roc_auc_test": 0.75,
            "train_test_gap": 0.05,
            "cv_std": 0.15,
            "recall": 0.50,
        }

        gate = ModelValidityGate()
        result = gate.run(metrics)

        assert any("MV005" in str(issue) for issue in result.high_issues)


class TestPoorMinorityRecall:
    def test_mv006_flags_poor_minority_recall(self):
        metrics = {
            "pr_auc_test": 0.55,
            "roc_auc_test": 0.75,
            "train_test_gap": 0.05,
            "cv_std": 0.02,
            "recall": 0.20,
        }

        gate = ModelValidityGate()
        result = gate.run(metrics)

        assert any("MV006" in str(issue) for issue in result.high_issues)


class TestWorstThanBaseline:
    def test_mv007_flags_worse_than_baseline(self):
        metrics = {
            "pr_auc_test": 0.15,
            "roc_auc_test": 0.55,
            "train_test_gap": 0.05,
            "cv_std": 0.02,
            "recall": 0.50,
            "class_proportion": 0.25,
        }

        gate = ModelValidityGate()
        result = gate.run(metrics)

        assert not result.passed
        assert any("MV007" in str(issue) for issue in result.critical_issues)


class TestPerfectPredictions:
    def test_mv008_flags_perfect_predictions(self):
        metrics = {
            "pr_auc_test": 1.0,
            "roc_auc_test": 1.0,
            "train_test_gap": 0.0,
            "cv_std": 0.0,
            "recall": 1.0,
        }

        gate = ModelValidityGate()
        result = gate.run(metrics)

        assert not result.passed
        assert any("MV008" in str(issue) for issue in result.critical_issues)


class TestModelValidityResult:
    def test_result_contains_required_fields(self):
        metrics = {
            "pr_auc_test": 0.60,
            "roc_auc_test": 0.75,
            "train_test_gap": 0.05,
            "cv_std": 0.02,
            "recall": 0.50,
        }

        gate = ModelValidityGate()
        result = gate.run(metrics)

        assert hasattr(result, "passed")
        assert hasattr(result, "critical_issues")
        assert hasattr(result, "high_issues")
        assert hasattr(result, "warnings")
        assert hasattr(result, "recommendation")
        assert hasattr(result, "diagnostic_hints")


class TestGateRecommendation:
    def test_provides_proceed_recommendation_when_passed(self):
        metrics = {
            "pr_auc_test": 0.55,
            "roc_auc_test": 0.75,
            "train_test_gap": 0.05,
            "cv_std": 0.02,
            "recall": 0.50,
        }

        gate = ModelValidityGate()
        result = gate.run(metrics)

        assert result.passed
        assert "proceed" in result.recommendation.lower()

    def test_provides_investigate_recommendation_when_failed(self):
        metrics = {
            "pr_auc_test": 0.95,
            "roc_auc_test": 0.98,
            "train_test_gap": 0.05,
            "cv_std": 0.02,
            "recall": 0.50,
        }

        gate = ModelValidityGate()
        result = gate.run(metrics)

        assert not result.passed
        assert "investigate" in result.recommendation.lower()


class TestDiagnosticHints:
    def test_provides_diagnostic_hints_on_failure(self):
        metrics = {
            "pr_auc_test": 0.95,
            "roc_auc_test": 0.85,
            "train_test_gap": 0.05,
            "cv_std": 0.02,
            "recall": 0.50,
        }

        gate = ModelValidityGate()
        result = gate.run(metrics)

        assert len(result.diagnostic_hints) > 0


class TestCustomThresholds:
    def test_custom_pr_auc_threshold(self):
        metrics = {
            "pr_auc_test": 0.85,
            "roc_auc_test": 0.80,
            "train_test_gap": 0.05,
            "cv_std": 0.02,
            "recall": 0.50,
        }

        gate = ModelValidityGate(pr_auc_threshold_suspicious=0.80)
        result = gate.run(metrics)

        assert any("MV001" in str(issue) for issue in result.critical_issues)


class TestCleanModelPasses:
    def test_clean_model_passes_all_checks(self):
        metrics = {
            "pr_auc_test": 0.55,
            "roc_auc_test": 0.75,
            "train_test_gap": 0.04,
            "cv_std": 0.03,
            "recall": 0.45,
        }

        gate = ModelValidityGate()
        result = gate.run(metrics)

        assert result.passed
        assert len(result.critical_issues) == 0
