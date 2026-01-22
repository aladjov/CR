import pytest
import pandas as pd
import numpy as np
from customer_retention.analysis.business import (
    InterventionMatcher, InterventionCatalog, Intervention,
    InterventionRecommendation, RiskSegment
)


@pytest.fixture
def intervention_catalog():
    return InterventionCatalog([
        Intervention(
            name="email_campaign",
            cost=2,
            success_rate=0.10,
            channel="email",
            min_ltv=0,
            applicable_segments=[RiskSegment.LOW, RiskSegment.MEDIUM]
        ),
        Intervention(
            name="phone_call",
            cost=15,
            success_rate=0.25,
            channel="phone",
            min_ltv=100,
            applicable_segments=[RiskSegment.MEDIUM, RiskSegment.HIGH]
        ),
        Intervention(
            name="discount_10pct",
            cost=25,
            success_rate=0.30,
            channel="email",
            min_ltv=200,
            applicable_segments=[RiskSegment.HIGH, RiskSegment.CRITICAL]
        ),
        Intervention(
            name="account_manager",
            cost=150,
            success_rate=0.60,
            channel="personal",
            min_ltv=1000,
            applicable_segments=[RiskSegment.CRITICAL]
        ),
    ])


class TestInterventionCatalog:
    def test_catalog_contains_interventions(self, intervention_catalog):
        assert len(intervention_catalog.interventions) == 4

    def test_get_intervention_by_name(self, intervention_catalog):
        intervention = intervention_catalog.get("phone_call")
        assert intervention.name == "phone_call"
        assert intervention.cost == 15

    def test_filter_by_segment(self, intervention_catalog):
        interventions = intervention_catalog.filter_by_segment(RiskSegment.CRITICAL)
        names = [i.name for i in interventions]
        assert "discount_10pct" in names
        assert "account_manager" in names


class TestInterventionMatching:
    def test_matches_intervention_for_segment(self, intervention_catalog):
        matcher = InterventionMatcher(intervention_catalog)
        recommendation = matcher.match(
            risk_segment=RiskSegment.HIGH,
            customer_ltv=300
        )
        assert isinstance(recommendation, InterventionRecommendation)
        assert recommendation.intervention is not None

    def test_respects_min_ltv(self, intervention_catalog):
        matcher = InterventionMatcher(intervention_catalog)
        recommendation = matcher.match(
            risk_segment=RiskSegment.CRITICAL,
            customer_ltv=500
        )
        assert recommendation.intervention.min_ltv <= 500

    def test_no_match_for_very_low_risk(self, intervention_catalog):
        matcher = InterventionMatcher(intervention_catalog)
        recommendation = matcher.match(
            risk_segment=RiskSegment.VERY_LOW,
            customer_ltv=100
        )
        assert recommendation.intervention is None or recommendation.intervention.name == "none"


class TestInterventionReasoning:
    def test_recommendation_includes_reasoning(self, intervention_catalog):
        matcher = InterventionMatcher(intervention_catalog)
        recommendation = matcher.match(
            risk_segment=RiskSegment.HIGH,
            customer_ltv=300
        )
        assert recommendation.reasoning is not None
        assert len(recommendation.reasoning) > 0

    def test_recommendation_includes_expected_roi(self, intervention_catalog):
        matcher = InterventionMatcher(intervention_catalog)
        recommendation = matcher.match(
            risk_segment=RiskSegment.HIGH,
            customer_ltv=300,
            churn_probability=0.7
        )
        assert recommendation.expected_roi is not None


class TestMultipleRecommendations:
    def test_returns_multiple_options(self, intervention_catalog):
        matcher = InterventionMatcher(intervention_catalog)
        recommendations = matcher.match_multiple(
            risk_segment=RiskSegment.HIGH,
            customer_ltv=300,
            n=3
        )
        assert len(recommendations) >= 1

    def test_options_sorted_by_roi(self, intervention_catalog):
        matcher = InterventionMatcher(intervention_catalog)
        recommendations = matcher.match_multiple(
            risk_segment=RiskSegment.HIGH,
            customer_ltv=500,
            churn_probability=0.7,
            n=3
        )
        if len(recommendations) > 1:
            rois = [r.expected_roi for r in recommendations if r.expected_roi is not None]
            assert rois == sorted(rois, reverse=True)


class TestInterventionTiming:
    def test_intervention_has_timing(self, intervention_catalog):
        matcher = InterventionMatcher(intervention_catalog)
        recommendation = matcher.match(
            risk_segment=RiskSegment.HIGH,
            customer_ltv=300
        )
        assert recommendation.timing is not None


class TestPriorityAssignment:
    def test_priority_assigned(self, intervention_catalog):
        matcher = InterventionMatcher(intervention_catalog)
        recommendation = matcher.match(
            risk_segment=RiskSegment.HIGH,
            customer_ltv=300
        )
        assert recommendation.priority is not None
        assert 1 <= recommendation.priority <= 5

    def test_higher_priority_for_critical(self, intervention_catalog):
        matcher = InterventionMatcher(intervention_catalog)
        critical_rec = matcher.match(
            risk_segment=RiskSegment.CRITICAL,
            customer_ltv=500
        )
        medium_rec = matcher.match(
            risk_segment=RiskSegment.MEDIUM,
            customer_ltv=500
        )
        assert critical_rec.priority <= medium_rec.priority


class TestBatchMatching:
    def test_matches_for_batch(self, intervention_catalog):
        matcher = InterventionMatcher(intervention_catalog)
        customers = [
            {"risk_segment": RiskSegment.CRITICAL, "customer_ltv": 1000},
            {"risk_segment": RiskSegment.HIGH, "customer_ltv": 500},
            {"risk_segment": RiskSegment.MEDIUM, "customer_ltv": 200},
        ]
        results = matcher.match_batch(customers)
        assert len(results) == 3
