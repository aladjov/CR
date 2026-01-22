import numpy as np
import pandas as pd
import pytest

from customer_retention.analysis.business import (
    CampaignList,
    CustomerServiceReport,
    ExecutiveDashboard,
    GovernanceReport,
    ProductInsights,
    ReportGenerator,
)


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "customer_id": [f"CUST{i:04d}" for i in range(n)],
        "churn_probability": np.random.uniform(0, 1, n),
        "risk_segment": np.random.choice(["Critical", "High", "Medium", "Low", "Very Low"], n),
        "ltv": np.random.uniform(100, 1000, n),
        "recency": np.random.randint(1, 365, n),
        "frequency": np.random.randint(1, 50, n),
        "engagement": np.random.uniform(0, 1, n),
    })


@pytest.fixture
def feature_importance():
    return {
        "recency": 0.30,
        "frequency": 0.25,
        "engagement": 0.20,
        "ltv": 0.15,
        "tenure": 0.10,
    }


@pytest.fixture
def model_metrics():
    return {
        "pr_auc": 0.75,
        "roc_auc": 0.82,
        "accuracy": 0.80,
        "precision": 0.78,
        "recall": 0.72,
    }


class TestExecutiveDashboard:
    def test_ac7_20_executive_dashboard_generates(self, sample_data, model_metrics):
        generator = ReportGenerator()
        dashboard = generator.generate_executive_dashboard(
            customer_data=sample_data,
            model_metrics=model_metrics
        )
        assert isinstance(dashboard, ExecutiveDashboard)

    def test_dashboard_contains_overview(self, sample_data, model_metrics):
        generator = ReportGenerator()
        dashboard = generator.generate_executive_dashboard(
            customer_data=sample_data,
            model_metrics=model_metrics
        )
        assert dashboard.total_customers == len(sample_data)
        assert dashboard.revenue_at_risk is not None

    def test_dashboard_contains_risk_distribution(self, sample_data, model_metrics):
        generator = ReportGenerator()
        dashboard = generator.generate_executive_dashboard(
            customer_data=sample_data,
            model_metrics=model_metrics
        )
        assert len(dashboard.risk_distribution) > 0

    def test_dashboard_contains_intervention_impact(self, sample_data, model_metrics):
        generator = ReportGenerator()
        dashboard = generator.generate_executive_dashboard(
            customer_data=sample_data,
            model_metrics=model_metrics,
            intervention_data={"expected_saves": 50, "expected_roi": 150}
        )
        assert dashboard.expected_saves is not None
        assert dashboard.expected_roi is not None


class TestCampaignList:
    def test_ac7_21_campaign_list_exports(self, sample_data):
        generator = ReportGenerator()
        campaign_list = generator.generate_campaign_list(
            customer_data=sample_data,
            risk_segments=["Critical", "High"]
        )
        assert isinstance(campaign_list, CampaignList)
        assert len(campaign_list.customers) > 0

    def test_campaign_list_contains_required_fields(self, sample_data):
        generator = ReportGenerator()
        campaign_list = generator.generate_campaign_list(
            customer_data=sample_data,
            risk_segments=["Critical", "High"]
        )
        for customer in campaign_list.customers:
            assert "customer_id" in customer
            assert "risk_segment" in customer
            assert "churn_probability" in customer

    def test_campaign_list_filtered_by_segment(self, sample_data):
        generator = ReportGenerator()
        campaign_list = generator.generate_campaign_list(
            customer_data=sample_data,
            risk_segments=["Critical"]
        )
        for customer in campaign_list.customers:
            assert customer["risk_segment"] == "Critical"


class TestCampaignListExport:
    def test_exports_to_dict_list(self, sample_data):
        generator = ReportGenerator()
        campaign_list = generator.generate_campaign_list(
            customer_data=sample_data,
            risk_segments=["Critical", "High"]
        )
        exported = campaign_list.to_dict_list()
        assert isinstance(exported, list)
        assert all(isinstance(item, dict) for item in exported)

    def test_exports_to_dataframe(self, sample_data):
        generator = ReportGenerator()
        campaign_list = generator.generate_campaign_list(
            customer_data=sample_data,
            risk_segments=["Critical", "High"]
        )
        df = campaign_list.to_dataframe()
        assert isinstance(df, pd.DataFrame)


class TestCustomerServiceReport:
    def test_generates_cs_report(self, sample_data):
        generator = ReportGenerator()
        report = generator.generate_customer_service_report(
            customer_id="CUST0001",
            customer_data=sample_data[sample_data["customer_id"] == "CUST0001"].iloc[0],
            risk_factors=[{"name": "Low engagement", "impact": 0.25}]
        )
        assert isinstance(report, CustomerServiceReport)

    def test_cs_report_contains_talking_points(self, sample_data):
        generator = ReportGenerator()
        report = generator.generate_customer_service_report(
            customer_id="CUST0001",
            customer_data=sample_data[sample_data["customer_id"] == "CUST0001"].iloc[0],
            risk_factors=[{"name": "Low engagement", "impact": 0.25}]
        )
        assert len(report.talking_points) > 0


class TestProductInsights:
    def test_generates_product_insights(self, sample_data, feature_importance):
        generator = ReportGenerator()
        insights = generator.generate_product_insights(
            customer_data=sample_data,
            feature_importance=feature_importance
        )
        assert isinstance(insights, ProductInsights)

    def test_product_insights_contains_churn_drivers(self, sample_data, feature_importance):
        generator = ReportGenerator()
        insights = generator.generate_product_insights(
            customer_data=sample_data,
            feature_importance=feature_importance
        )
        assert len(insights.top_churn_drivers) > 0

    def test_product_insights_contains_recommendations(self, sample_data, feature_importance):
        generator = ReportGenerator()
        insights = generator.generate_product_insights(
            customer_data=sample_data,
            feature_importance=feature_importance
        )
        assert len(insights.improvement_recommendations) > 0


class TestGovernanceReport:
    def test_generates_governance_report(self, model_metrics):
        generator = ReportGenerator()
        report = generator.generate_governance_report(
            model_metrics=model_metrics,
            data_quality_summary={"missing_rate": 0.02, "outlier_rate": 0.01}
        )
        assert isinstance(report, GovernanceReport)

    def test_governance_report_contains_performance(self, model_metrics):
        generator = ReportGenerator()
        report = generator.generate_governance_report(
            model_metrics=model_metrics,
            data_quality_summary={"missing_rate": 0.02}
        )
        assert report.model_performance is not None

    def test_governance_report_contains_retraining_recommendation(self, model_metrics):
        generator = ReportGenerator()
        report = generator.generate_governance_report(
            model_metrics=model_metrics,
            data_quality_summary={"missing_rate": 0.02},
            drift_status={"feature_drift": False, "target_drift": False}
        )
        assert report.retraining_recommendation is not None


class TestReportActionability:
    def test_ac7_22_reports_are_actionable(self, sample_data, model_metrics, feature_importance):
        generator = ReportGenerator()
        dashboard = generator.generate_executive_dashboard(
            customer_data=sample_data,
            model_metrics=model_metrics
        )
        assert len(dashboard.top_actions) > 0
