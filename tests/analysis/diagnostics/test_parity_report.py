import json

import pytest

from customer_retention.analysis.diagnostics.parity_report import (
    ParityFinding,
    ParitySeverity,
    compare_runs,
    compare_runs_from_paths,
)
from customer_retention.stages.modeling.feature_spec import (
    FeatureSpec,
    FittedTransform,
    Verdict,
)


def _spec(features, *, verdict_status="solid", cv_mean=0.8, cv_std=0.02, run_id="r1"):
    return FeatureSpec(
        exploration_run_id=run_id,
        target_column="churn",
        entity_column="entity_id",
        timestamp_column="as_of_date",
        horizon_days=180,
        selected_features=list(features),
        fitted_transforms=[
            FittedTransform(column=c, action="impute", method="median") for c in features
        ],
        verdict=Verdict(status=verdict_status, cv_mean=cv_mean, cv_std=cv_std),
    )


class TestFeatureSetIdentical:
    def test_empty_findings_when_sets_match(self):
        spec = _spec(["a", "b"])
        report = compare_runs(
            spec,
            exploration_diagnostics={"feature_names": ["a", "b"]},
            production_diagnostics={"feature_names": ["a", "b"]},
        )
        assert [f for f in report.findings if f.check_id.startswith("feature_set_identical")] == []
        assert report.passed

    def test_production_missing_feature_is_critical(self):
        spec = _spec(["a", "b", "c"])
        report = compare_runs(
            spec,
            production_diagnostics={"feature_names": ["a", "b"]},
        )
        prod_fail = [f for f in report.findings if f.check_id == "feature_set_identical:production"]
        assert len(prod_fail) == 1
        assert prod_fail[0].severity == ParitySeverity.CRITICAL
        assert not report.passed

    def test_production_extra_feature_is_critical(self):
        spec = _spec(["a", "b"])
        report = compare_runs(
            spec, production_diagnostics={"feature_names": ["a", "b", "rogue"]},
        )
        assert not report.passed
        prod_fail = next(f for f in report.findings if f.check_id == "feature_set_identical:production")
        assert "rogue" in prod_fail.details["extra_in_actual"]

    def test_exploration_mismatch_is_critical(self):
        spec = _spec(["a", "b"])
        report = compare_runs(spec, exploration_diagnostics={"feature_names": ["a"]})
        exp_fail = next(f for f in report.findings if f.check_id == "feature_set_identical:exploration")
        assert exp_fail.severity == ParitySeverity.CRITICAL


class TestClassProportionDelta:
    def test_within_2x_no_finding(self):
        spec = _spec(["a"])
        report = compare_runs(
            spec,
            exploration_diagnostics={"feature_names": ["a"], "class_proportion": 0.50},
            production_diagnostics={"feature_names": ["a"], "label_rate_test": 0.45},
        )
        assert [f for f in report.findings if f.check_id == "class_proportion_delta"] == []

    def test_2_to_5x_medium(self):
        spec = _spec(["a"])
        report = compare_runs(
            spec,
            exploration_diagnostics={"feature_names": ["a"], "class_proportion": 0.50},
            production_diagnostics={"feature_names": ["a"], "label_rate_test": 0.20},
        )
        cp = next(f for f in report.findings if f.check_id == "class_proportion_delta")
        assert cp.severity == ParitySeverity.MEDIUM

    def test_above_5x_high(self):
        spec = _spec(["a"])
        report = compare_runs(
            spec,
            exploration_diagnostics={"feature_names": ["a"], "class_proportion": 0.50},
            production_diagnostics={"feature_names": ["a"], "label_rate_test": 0.02},
        )
        cp = next(f for f in report.findings if f.check_id == "class_proportion_delta")
        assert cp.severity == ParitySeverity.HIGH


class TestCVMeanDelta:
    def test_info_when_present(self):
        spec = _spec(["a"], cv_mean=0.80)
        report = compare_runs(
            spec,
            production_diagnostics={
                "feature_names": ["a"],
                "best_model_name": "rf",
                "cv_results": {"rf": {"cv_mean": 0.78}},
            },
        )
        cv = next(f for f in report.findings if f.check_id == "cv_mean_delta")
        assert cv.severity == ParitySeverity.INFO
        assert cv.details["delta"] == pytest.approx(0.02)

    def test_absent_when_no_prod_cv(self):
        spec = _spec(["a"], cv_mean=0.80)
        report = compare_runs(spec, production_diagnostics={"feature_names": ["a"]})
        assert [f for f in report.findings if f.check_id == "cv_mean_delta"] == []


class TestVerdictConsistency:
    def test_match_no_finding(self):
        spec = _spec(["a"], verdict_status="solid")
        report = compare_runs(
            spec,
            production_diagnostics={"feature_names": ["a"], "verdict": {"status": "solid"}},
        )
        assert [f for f in report.findings if f.check_id == "verdict_consistency"] == []

    def test_mismatch_medium(self):
        spec = _spec(["a"], verdict_status="solid")
        report = compare_runs(
            spec,
            production_diagnostics={"feature_names": ["a"], "verdict": {"status": "unstable"}},
        )
        v = next(f for f in report.findings if f.check_id == "verdict_consistency")
        assert v.severity == ParitySeverity.MEDIUM


class TestParityReportSerialization:
    def test_to_dict_includes_passed_flag(self):
        spec = _spec(["a"])
        report = compare_runs(spec, production_diagnostics={"feature_names": ["a"]})
        d = report.to_dict()
        assert d["passed"] is True
        assert d["exploration_run_id"] == "r1"
        assert isinstance(d["findings"], list)

    def test_save_writes_valid_json(self, tmp_path):
        spec = _spec(["a"])
        report = compare_runs(spec, production_diagnostics={"feature_names": ["a"]})
        path = tmp_path / "parity.json"
        report.save(path)
        loaded = json.loads(path.read_text())
        assert loaded["passed"] is True


class TestCompareRunsFromPaths:
    def test_loads_and_runs(self, tmp_path):
        spec = _spec(["a", "b"])
        spec_path = tmp_path / "feature_spec.yaml"
        spec.save(spec_path)

        exp_path = tmp_path / "exp.json"
        exp_path.write_text(json.dumps({"feature_names": ["a", "b"], "class_proportion": 0.4}))

        prod_path = tmp_path / "prod.json"
        prod_path.write_text(json.dumps({
            "feature_names": ["a", "b"],
            "label_rate_test": 0.35,
            "best_model_name": "rf",
            "cv_results": {"rf": {"cv_mean": 0.79}},
            "verdict": {"status": "solid"},
            "run_type": "production",
        }))

        report = compare_runs_from_paths(spec_path, exp_path, prod_path)
        assert report.passed
        cv = next(f for f in report.findings if f.check_id == "cv_mean_delta")
        assert cv.severity == ParitySeverity.INFO

    def test_missing_production_diagnostics_fine(self, tmp_path):
        spec = _spec(["a"])
        spec_path = tmp_path / "spec.yaml"
        spec.save(spec_path)
        report = compare_runs_from_paths(spec_path, None, tmp_path / "missing.json")
        assert report.passed

    def test_critical_failure_flips_passed_false(self):
        spec = _spec(["a", "b"])
        report = compare_runs(spec, production_diagnostics={"feature_names": ["a"]})
        assert report.passed is False
        assert len(report.critical) == 1


class TestParityFindingEquality:
    def test_distinct_fingerprints(self):
        a = ParityFinding(check_id="x", severity=ParitySeverity.HIGH, message="m")
        b = ParityFinding(check_id="x", severity=ParitySeverity.HIGH, message="m")
        assert a == b
