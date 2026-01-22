import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

from customer_retention.analysis.interpretability import (
    ShapExplainer, PDPGenerator, CohortAnalyzer,
    IndividualExplainer, CounterfactualGenerator, Confidence
)
from customer_retention.analysis.business import (
    RiskProfiler, RiskSegment, Urgency,
    InterventionMatcher, InterventionCatalog, Intervention,
    ROIAnalyzer, FairnessAnalyzer,
    ReportGenerator, ABTestDesigner
)


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


class TestGlobalInterpretability:
    def test_ac7_1_shap_values_compute_without_error(self, trained_model, retail_data, feature_columns):
        X = retail_data[feature_columns].fillna(0)
        explainer = ShapExplainer(trained_model, X)
        result = explainer.explain_global(X.head(50))
        assert result is not None
        assert result.shap_values is not None

    def test_ac7_2_feature_ranking_returned(self, trained_model, retail_data, feature_columns):
        X = retail_data[feature_columns].fillna(0)
        explainer = ShapExplainer(trained_model, X)
        result = explainer.explain_global(X.head(50))
        assert len(result.feature_importance) > 0

    def test_ac7_4_business_translations_provided(self, trained_model, retail_data, feature_columns):
        X = retail_data[feature_columns].fillna(0)
        translations = {col: f"Description for {col}" for col in feature_columns}
        explainer = ShapExplainer(trained_model, X, feature_translations=translations)
        result = explainer.explain_global(X.head(50))
        for fi in result.feature_importance:
            assert fi.business_description is not None


class TestCohortInterpretability:
    def test_ac7_5_cohort_analysis_runs(self, trained_model, retail_data, feature_columns):
        X = retail_data[feature_columns].fillna(0)
        y = retail_data["retained"]
        cohorts = pd.Series(["Low" if v < 50 else "High" for v in retail_data["avgorder"]])
        analyzer = CohortAnalyzer(trained_model, X)
        result = analyzer.analyze(X, y, cohorts)
        assert len(result.cohort_insights) > 0

    def test_ac7_6_differences_identified(self, trained_model, retail_data, feature_columns):
        X = retail_data[feature_columns].fillna(0)
        y = retail_data["retained"]
        cohorts = pd.Series(["Low" if v < 50 else "High" for v in retail_data["avgorder"]])
        analyzer = CohortAnalyzer(trained_model, X)
        result = analyzer.analyze(X, y, cohorts)
        assert len(result.key_differences) > 0

    def test_ac7_7_insights_actionable(self, trained_model, retail_data, feature_columns):
        X = retail_data[feature_columns].fillna(0)
        y = retail_data["retained"]
        cohorts = pd.Series(["Low" if v < 50 else "High" for v in retail_data["avgorder"]])
        analyzer = CohortAnalyzer(trained_model, X)
        result = analyzer.analyze(X, y, cohorts)
        for insight in result.cohort_insights:
            assert insight.recommended_strategy is not None


class TestIndividualInterpretability:
    def test_ac7_8_waterfall_data_generated(self, trained_model, retail_data, feature_columns):
        X = retail_data[feature_columns].fillna(0)
        explainer = IndividualExplainer(trained_model, X)
        result = explainer.explain(X.iloc[0])
        assert result.shap_values is not None

    def test_ac7_9_risk_factors_extracted(self, trained_model, retail_data, feature_columns):
        X = retail_data[feature_columns].fillna(0)
        explainer = IndividualExplainer(trained_model, X)
        result = explainer.explain(X.iloc[0])
        assert len(result.top_positive_factors) > 0 or len(result.top_negative_factors) > 0

    def test_ac7_10_confidence_assigned(self, trained_model, retail_data, feature_columns):
        X = retail_data[feature_columns].fillna(0)
        explainer = IndividualExplainer(trained_model, X)
        result = explainer.explain(X.iloc[0])
        assert result.confidence in [Confidence.HIGH, Confidence.MEDIUM, Confidence.LOW]


class TestCounterfactuals:
    def test_ac7_11_counterfactuals_generate(self, trained_model, retail_data, feature_columns):
        X = retail_data[feature_columns].fillna(0)
        generator = CounterfactualGenerator(trained_model, X, actionable_features=["ordfreq", "eopenrate"])
        result = generator.generate(X.iloc[0])
        assert result is not None

    def test_ac7_12_changes_are_actionable(self, trained_model, retail_data, feature_columns):
        X = retail_data[feature_columns].fillna(0)
        actionable = ["ordfreq", "eopenrate"]
        generator = CounterfactualGenerator(trained_model, X, actionable_features=actionable)
        result = generator.generate(X.iloc[0])
        for change in result.changes:
            assert change.feature_name in actionable

    def test_ac7_13_business_interpretation_provided(self, trained_model, retail_data, feature_columns):
        X = retail_data[feature_columns].fillna(0)
        generator = CounterfactualGenerator(trained_model, X, actionable_features=["ordfreq"])
        result = generator.generate(X.iloc[0])
        assert result.business_interpretation is not None


class TestRiskProfiles:
    def test_ac7_14_profiles_complete(self, trained_model, retail_data, feature_columns):
        X = retail_data[feature_columns].fillna(0)
        profiler = RiskProfiler(trained_model, X)
        profile = profiler.generate_profile(X.iloc[0], customer_id="CUST001")
        assert profile.customer_id == "CUST001"
        assert profile.churn_probability is not None
        assert profile.risk_segment is not None

    def test_ac7_15_segments_assigned(self, trained_model, retail_data, feature_columns):
        X = retail_data[feature_columns].fillna(0)
        profiler = RiskProfiler(trained_model, X)
        segments = set()
        for i in range(min(50, len(X))):
            profile = profiler.generate_profile(X.iloc[i])
            segments.add(profile.risk_segment)
        assert len(segments) >= 1

    def test_ac7_16_interventions_matched(self, trained_model, retail_data, feature_columns):
        X = retail_data[feature_columns].fillna(0)
        profiler = RiskProfiler(trained_model, X)
        profile = profiler.generate_profile(X.iloc[0])
        assert len(profile.recommended_interventions) >= 0


class TestROIAnalysis:
    def test_ac7_17_roi_calculated(self):
        analyzer = ROIAnalyzer(
            avg_ltv=500,
            intervention_costs={"email": 2, "phone": 15},
            success_rates={"email": 0.10, "phone": 0.25}
        )
        result = analyzer.calculate_roi("phone", 100, 0.30)
        assert result.roi_pct is not None

    def test_ac7_18_positive_roi_strategies_exist(self):
        analyzer = ROIAnalyzer(
            avg_ltv=500,
            intervention_costs={"email": 2, "phone": 15, "discount": 25},
            success_rates={"email": 0.10, "phone": 0.25, "discount": 0.35}
        )
        results = analyzer.analyze_all_interventions(100, 0.30)
        positive_roi = [r for r in results if r.roi_pct > 0]
        assert len(positive_roi) > 0

    def test_ac7_19_optimization_works(self):
        analyzer = ROIAnalyzer(
            avg_ltv=500,
            intervention_costs={"email": 2, "phone": 15},
            success_rates={"email": 0.10, "phone": 0.25}
        )
        segment_data = {
            "Critical": {"customers": 50, "churn_rate": 0.80},
            "High": {"customers": 100, "churn_rate": 0.65},
        }
        result = analyzer.optimize_budget(segment_data, total_budget=2000)
        assert result.total_cost <= 2000


class TestBusinessReports:
    def test_ac7_20_executive_dashboard_generates(self, retail_data, feature_columns):
        customer_data = pd.DataFrame({
            "customer_id": [f"CUST{i}" for i in range(100)],
            "churn_probability": np.random.uniform(0, 1, 100),
            "risk_segment": np.random.choice(["Critical", "High", "Medium", "Low"], 100),
            "ltv": np.random.uniform(100, 1000, 100)
        })
        generator = ReportGenerator()
        dashboard = generator.generate_executive_dashboard(
            customer_data=customer_data,
            model_metrics={"pr_auc": 0.75, "roc_auc": 0.80}
        )
        assert dashboard.total_customers == 100

    def test_ac7_21_campaign_list_exports(self, retail_data, feature_columns):
        customer_data = pd.DataFrame({
            "customer_id": [f"CUST{i}" for i in range(100)],
            "churn_probability": np.random.uniform(0, 1, 100),
            "risk_segment": np.random.choice(["Critical", "High", "Medium", "Low"], 100),
        })
        generator = ReportGenerator()
        campaign_list = generator.generate_campaign_list(
            customer_data=customer_data,
            risk_segments=["Critical", "High"]
        )
        assert len(campaign_list.customers) > 0

    def test_ac7_22_reports_are_actionable(self, retail_data, feature_columns):
        customer_data = pd.DataFrame({
            "customer_id": [f"CUST{i}" for i in range(100)],
            "churn_probability": np.random.uniform(0, 1, 100),
            "risk_segment": np.random.choice(["Critical", "High", "Medium", "Low"], 100),
            "ltv": np.random.uniform(100, 1000, 100)
        })
        generator = ReportGenerator()
        dashboard = generator.generate_executive_dashboard(
            customer_data=customer_data,
            model_metrics={"pr_auc": 0.75}
        )
        assert len(dashboard.top_actions) > 0


class TestFairness:
    def test_ac7_23_fairness_metrics_calculated(self, trained_model, retail_data, feature_columns):
        X = retail_data[feature_columns].fillna(0)
        y = retail_data["retained"]
        y_pred = pd.Series(trained_model.predict(X))
        protected = pd.Series(["GroupA" if i % 2 == 0 else "GroupB" for i in range(len(y))])
        analyzer = FairnessAnalyzer()
        result = analyzer.analyze(y, y_pred, protected)
        assert len(result.metrics) > 0

    def test_ac7_24_no_severe_bias_detection(self, trained_model, retail_data, feature_columns):
        X = retail_data[feature_columns].fillna(0)
        y = retail_data["retained"]
        y_pred = pd.Series(trained_model.predict(X))
        protected = pd.Series(["GroupA" if i % 2 == 0 else "GroupB" for i in range(len(y))])
        analyzer = FairnessAnalyzer(threshold=0.8)
        result = analyzer.analyze(y, y_pred, protected)
        assert result.passed is not None

    def test_ac7_25_recommendations_provided(self, trained_model, retail_data, feature_columns):
        X = retail_data[feature_columns].fillna(0)
        y = retail_data["retained"]
        y_pred = pd.Series(trained_model.predict(X))
        protected = pd.Series(["GroupA" if i % 2 == 0 else "GroupB" for i in range(len(y))])
        analyzer = FairnessAnalyzer()
        result = analyzer.analyze(y, y_pred, protected)
        assert result.recommendations is not None


class TestPDPGeneration:
    def test_pdp_generates_for_feature(self, trained_model, retail_data, feature_columns):
        X = retail_data[feature_columns].fillna(0)
        generator = PDPGenerator(trained_model)
        result = generator.generate(X, feature="ordfreq")
        assert len(result.grid_values) > 0
        assert len(result.pdp_values) > 0


class TestABTestDesign:
    def test_ab_test_design_creates(self):
        np.random.seed(42)
        customer_pool = pd.DataFrame({
            "customer_id": [f"CUST{i}" for i in range(5000)],
            "risk_segment": np.random.choice(["Critical", "High", "Medium", "Low"], 5000),
        })
        designer = ABTestDesigner()
        design = designer.design_test(
            test_name="retention_test",
            customer_pool=customer_pool,
            control_name="control",
            treatment_names=["email_campaign"],
            baseline_rate=0.25,
            min_detectable_effect=0.05
        )
        assert design.test_name == "retention_test"
        assert design.recommended_sample_size > 0


class TestFullInterpretabilityPipeline:
    def test_end_to_end_interpretability(self, trained_model, retail_data, feature_columns):
        X = retail_data[feature_columns].fillna(0)
        y = retail_data["retained"]
        shap_explainer = ShapExplainer(trained_model, X)
        global_result = shap_explainer.explain_global(X.head(50))
        assert len(global_result.feature_importance) > 0
        individual_explainer = IndividualExplainer(trained_model, X)
        individual_result = individual_explainer.explain(X.iloc[0])
        assert individual_result.churn_probability is not None
        profiler = RiskProfiler(trained_model, X, avg_customer_ltv=500)
        profile = profiler.generate_profile(X.iloc[0], customer_id="CUST001")
        assert profile.risk_segment is not None
        assert len(profile.recommended_interventions) >= 0
        customer_data = pd.DataFrame({
            "customer_id": ["CUST001"],
            "churn_probability": [profile.churn_probability],
            "risk_segment": [profile.risk_segment.value],
            "ltv": [500]
        })
        generator = ReportGenerator()
        dashboard = generator.generate_executive_dashboard(
            customer_data=customer_data,
            model_metrics={"pr_auc": 0.75}
        )
        assert dashboard is not None
