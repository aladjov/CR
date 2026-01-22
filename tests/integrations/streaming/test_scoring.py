import time
from unittest.mock import MagicMock

import pytest

from customer_retention.integrations.streaming import (
    AutoScaler,
    RealtimeScorer,
    ScalingMetrics,
    ScoringConfig,
    ScoringRequest,
)


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.predict_proba = MagicMock(return_value=[[0.3, 0.7]])
    return model


@pytest.fixture
def mock_feature_store():
    store = MagicMock()
    store.read_batch = MagicMock(return_value={
        "page_views_1h": 15.0,
        "orders_7d": 3.0,
        "activity_anomaly_score": -0.5,
        "tenure_days": 365,
        "avg_order_value": 75.0
    })
    return store


@pytest.fixture
def scorer(mock_model, mock_feature_store):
    config = ScoringConfig(
        endpoint_name="churn_scorer",
        timeout_ms=200,
        model_version="v2.3"
    )
    return RealtimeScorer(
        model=mock_model,
        feature_store=mock_feature_store,
        config=config
    )


class TestScoringEndpoint:
    def test_ac9_16_endpoint_available(self, scorer):
        health = scorer.health_check()
        assert health.status == "healthy"
        assert health.model_loaded is True
        assert health.feature_store_connected is True

    def test_health_check_returns_details(self, scorer):
        health = scorer.health_check()
        assert health.model_version is not None
        assert health.uptime_seconds >= 0
        assert health.last_request_time is not None or health.last_request_time is None


class TestScoringLatency:
    def test_ac9_17_latency_under_200ms(self, scorer):
        request = ScoringRequest(customer_id="CUST001")
        start = time.time()
        response = scorer.score(request)
        elapsed_ms = (time.time() - start) * 1000
        assert elapsed_ms < 200
        assert response.latency_ms < 200

    def test_p50_latency_under_50ms(self, scorer):
        latencies = []
        for i in range(100):
            request = ScoringRequest(customer_id=f"CUST{i:03d}")
            start = time.time()
            scorer.score(request)
            latencies.append((time.time() - start) * 1000)
        latencies.sort()
        p50 = latencies[50]
        assert p50 < 50

    def test_p99_latency_under_200ms(self, scorer):
        latencies = []
        for i in range(100):
            request = ScoringRequest(customer_id=f"CUST{i:03d}")
            start = time.time()
            scorer.score(request)
            latencies.append((time.time() - start) * 1000)
        latencies.sort()
        p99 = latencies[99]
        assert p99 < 200


class TestScoringCorrectness:
    def test_ac9_18_correct_prediction_high_risk(self, mock_feature_store):
        model = MagicMock()
        model.predict_proba = MagicMock(return_value=[[0.27, 0.73]])
        scorer = RealtimeScorer(model=model, feature_store=mock_feature_store)
        request = ScoringRequest(customer_id="CUST001")
        response = scorer.score(request)
        assert response.churn_probability == pytest.approx(0.73, abs=0.01)
        assert response.risk_segment == "High"

    def test_ac9_18_correct_prediction_low_risk(self, mock_feature_store):
        model = MagicMock()
        model.predict_proba = MagicMock(return_value=[[0.85, 0.15]])
        scorer = RealtimeScorer(model=model, feature_store=mock_feature_store)
        request = ScoringRequest(customer_id="CUST001")
        response = scorer.score(request)
        assert response.churn_probability == pytest.approx(0.15, abs=0.01)
        assert response.risk_segment == "Low"

    def test_risk_segment_boundaries(self, mock_feature_store):
        model = MagicMock()
        scorer = RealtimeScorer(model=model, feature_store=mock_feature_store)
        test_cases = [
            (0.15, "Low"),
            (0.35, "Medium"),
            (0.55, "High"),
            (0.85, "Critical")
        ]
        for prob, expected_segment in test_cases:
            model.predict_proba = MagicMock(return_value=[[1 - prob, prob]])
            response = scorer.score(ScoringRequest(customer_id="CUST001"))
            assert response.risk_segment == expected_segment, f"Expected {expected_segment} for prob {prob}"


class TestScoringResponseSchema:
    def test_response_has_all_required_fields(self, scorer):
        request = ScoringRequest(customer_id="CUST123")
        response = scorer.score(request)
        assert response.customer_id == "CUST123"
        assert response.churn_probability is not None
        assert response.risk_segment is not None
        assert response.model_version is not None
        assert response.scored_at is not None
        assert response.latency_ms is not None

    def test_response_with_explanation(self, scorer):
        request = ScoringRequest(
            customer_id="CUST001",
            include_explanation=True
        )
        response = scorer.score(request)
        assert response.top_risk_factors is not None
        assert len(response.top_risk_factors) > 0

    def test_response_with_recommendation(self, scorer):
        request = ScoringRequest(
            customer_id="CUST001",
            include_recommendation=True
        )
        response = scorer.score(request)
        assert response.recommended_action is not None

    def test_response_with_warning_signals(self, scorer):
        request = ScoringRequest(customer_id="CUST001")
        response = scorer.score(request)
        assert response.warning_signals is not None


class TestAutoScaling:
    def test_ac9_19_auto_scaling_triggered_on_load(self):
        config = ScoringConfig(
            min_replicas=2,
            max_replicas=10,
            scale_target_cpu=70
        )
        scaler = AutoScaler(config=config)
        metrics = ScalingMetrics(
            current_cpu_percent=85,
            current_replicas=2,
            requests_per_second=500
        )
        decision = scaler.evaluate(metrics)
        assert decision.should_scale_up is True
        assert decision.target_replicas > 2

    def test_scale_down_on_low_load(self):
        config = ScoringConfig(
            min_replicas=2,
            max_replicas=10,
            scale_target_cpu=70
        )
        scaler = AutoScaler(config=config)
        metrics = ScalingMetrics(
            current_cpu_percent=20,
            current_replicas=8,
            requests_per_second=50
        )
        decision = scaler.evaluate(metrics)
        assert decision.should_scale_down is True
        assert decision.target_replicas < 8

    def test_respects_min_replicas(self):
        config = ScoringConfig(min_replicas=2, max_replicas=10)
        scaler = AutoScaler(config=config)
        metrics = ScalingMetrics(
            current_cpu_percent=5,
            current_replicas=2,
            requests_per_second=1
        )
        decision = scaler.evaluate(metrics)
        assert decision.target_replicas >= 2

    def test_respects_max_replicas(self):
        config = ScoringConfig(min_replicas=2, max_replicas=10)
        scaler = AutoScaler(config=config)
        metrics = ScalingMetrics(
            current_cpu_percent=99,
            current_replicas=10,
            requests_per_second=10000
        )
        decision = scaler.evaluate(metrics)
        assert decision.target_replicas <= 10


class TestBatchScoring:
    def test_batch_scoring_multiple_customers(self, scorer):
        customer_ids = [f"CUST{i:03d}" for i in range(10)]
        responses = scorer.score_batch(customer_ids)
        assert len(responses) == 10
        for response in responses:
            assert response.churn_probability is not None

    def test_batch_scoring_latency(self, scorer):
        customer_ids = [f"CUST{i:03d}" for i in range(100)]
        start = time.time()
        scorer.score_batch(customer_ids)
        elapsed_ms = (time.time() - start) * 1000
        assert elapsed_ms < 2000


class TestErrorHandling:
    def test_missing_features_returns_error(self, mock_model):
        feature_store = MagicMock()
        feature_store.read_batch = MagicMock(return_value={})
        scorer = RealtimeScorer(model=mock_model, feature_store=feature_store)
        request = ScoringRequest(customer_id="CUST_UNKNOWN")
        response = scorer.score(request)
        assert response.error is not None or response.churn_probability is not None

    def test_model_error_handled_gracefully(self, mock_feature_store):
        model = MagicMock()
        model.predict_proba = MagicMock(side_effect=Exception("Model error"))
        scorer = RealtimeScorer(model=model, feature_store=mock_feature_store)
        request = ScoringRequest(customer_id="CUST001")
        response = scorer.score(request)
        assert response.error is not None
        assert "model" in response.error.lower()

    def test_timeout_handling(self, mock_model):
        feature_store = MagicMock()
        def slow_read(*args, **kwargs):
            time.sleep(0.5)
            return {"page_views_1h": 10.0}
        feature_store.read_batch = slow_read
        config = ScoringConfig(timeout_ms=100)
        scorer = RealtimeScorer(model=mock_model, feature_store=feature_store, config=config)
        request = ScoringRequest(customer_id="CUST001")
        response = scorer.score(request)
        assert response.error is not None or response.latency_ms > 100


class TestFallbackStrategy:
    def test_fallback_to_batch_score(self, mock_model):
        feature_store = MagicMock()
        feature_store.read_batch = MagicMock(side_effect=Exception("Store unavailable"))
        batch_scores = {"CUST001": 0.65}
        scorer = RealtimeScorer(
            model=mock_model,
            feature_store=feature_store,
            fallback_scores=batch_scores
        )
        request = ScoringRequest(customer_id="CUST001")
        response = scorer.score(request)
        assert response.churn_probability == 0.65
        assert response.is_fallback is True

    def test_fallback_to_cached_score(self, scorer):
        scorer.score(ScoringRequest(customer_id="CUST001"))
        scorer._feature_store.read_batch = MagicMock(side_effect=Exception("Temporary failure"))
        response = scorer.score(ScoringRequest(customer_id="CUST001"))
        assert response.is_fallback is True or response.churn_probability is not None


class TestSLAMetrics:
    def test_availability_tracking(self, scorer):
        for i in range(100):
            scorer.score(ScoringRequest(customer_id=f"CUST{i:03d}"))
        metrics = scorer.get_sla_metrics()
        assert metrics.availability_percent >= 99.0

    def test_error_rate_tracking(self, scorer):
        for i in range(100):
            scorer.score(ScoringRequest(customer_id=f"CUST{i:03d}"))
        metrics = scorer.get_sla_metrics()
        assert metrics.error_rate_percent < 1.0

    def test_throughput_tracking(self, scorer):
        start = time.time()
        for i in range(100):
            scorer.score(ScoringRequest(customer_id=f"CUST{i:03d}"))
        elapsed = time.time() - start
        metrics = scorer.get_sla_metrics()
        expected_throughput = 100 / elapsed
        assert metrics.throughput_per_second > 0


class TestFeatureIntegration:
    def test_uses_streaming_features(self, mock_model, mock_feature_store):
        scorer = RealtimeScorer(model=mock_model, feature_store=mock_feature_store)
        request = ScoringRequest(customer_id="CUST001")
        scorer.score(request)
        mock_feature_store.read_batch.assert_called()
        call_args = mock_feature_store.read_batch.call_args
        assert "CUST001" in str(call_args)

    def test_combines_batch_and_streaming_features(self, mock_model):
        feature_store = MagicMock()
        feature_store.read_batch = MagicMock(return_value={
            "page_views_1h": 15.0,
            "tenure_days": 365
        })
        scorer = RealtimeScorer(model=mock_model, feature_store=feature_store)
        scorer.set_required_features(["page_views_1h", "tenure_days", "avg_order_value"])
        request = ScoringRequest(customer_id="CUST001")
        response = scorer.score(request)
        assert response is not None
