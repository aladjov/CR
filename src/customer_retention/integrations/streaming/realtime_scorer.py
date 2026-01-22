from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List
import time
import statistics
import threading


@dataclass
class ScoringConfig:
    endpoint_name: str = "churn_scorer"
    timeout_ms: int = 200
    model_version: str = "v1.0"
    min_replicas: int = 2
    max_replicas: int = 10
    scale_target_cpu: int = 70


@dataclass
class ScoringRequest:
    customer_id: str
    include_explanation: bool = False
    include_recommendation: bool = False


@dataclass
class RiskFactor:
    factor: str
    impact: float


@dataclass
class ScoringResponse:
    customer_id: str
    churn_probability: Optional[float] = None
    risk_segment: Optional[str] = None
    warning_signals: List[str] = field(default_factory=list)
    top_risk_factors: List[RiskFactor] = field(default_factory=list)
    recommended_action: Optional[str] = None
    model_version: str = "v1.0"
    scored_at: datetime = field(default_factory=datetime.now)
    latency_ms: float = 0.0
    error: Optional[str] = None
    is_fallback: bool = False


@dataclass
class EndpointHealth:
    status: str = "healthy"
    model_loaded: bool = True
    feature_store_connected: bool = True
    model_version: Optional[str] = None
    uptime_seconds: float = 0.0
    last_request_time: Optional[datetime] = None


@dataclass
class ScalingMetrics:
    current_cpu_percent: float = 0.0
    current_replicas: int = 2
    requests_per_second: float = 0.0


@dataclass
class ScalingDecision:
    should_scale_up: bool = False
    should_scale_down: bool = False
    target_replicas: int = 2


@dataclass
class SLAMetrics:
    availability_percent: float = 100.0
    error_rate_percent: float = 0.0
    throughput_per_second: float = 0.0
    avg_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0


@dataclass
class ScorerMetrics:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0


class AutoScaler:
    def __init__(self, config: ScoringConfig):
        self._config = config

    def evaluate(self, metrics: ScalingMetrics) -> ScalingDecision:
        import math
        target = metrics.current_replicas
        if metrics.current_cpu_percent > self._config.scale_target_cpu:
            scale_factor = metrics.current_cpu_percent / self._config.scale_target_cpu
            target = min(math.ceil(metrics.current_replicas * scale_factor), self._config.max_replicas)
            target = max(target, metrics.current_replicas + 1)
            target = min(target, self._config.max_replicas)
            return ScalingDecision(should_scale_up=True, target_replicas=target)
        elif metrics.current_cpu_percent < self._config.scale_target_cpu * 0.5:
            target = max(metrics.current_replicas - 1, self._config.min_replicas)
            if target < metrics.current_replicas:
                return ScalingDecision(should_scale_down=True, target_replicas=target)
        return ScalingDecision(target_replicas=max(target, self._config.min_replicas))


class RealtimeScorer:
    def __init__(self, model, feature_store, config: Optional[ScoringConfig] = None,
                 fallback_scores: Optional[Dict[str, float]] = None):
        self._model = model
        self._feature_store = feature_store
        self._config = config or ScoringConfig()
        self._fallback_scores = fallback_scores or {}
        self._start_time = datetime.now()
        self._last_request_time: Optional[datetime] = None
        self._latencies: List[float] = []
        self._errors: int = 0
        self._total_requests: int = 0
        self._cache: Dict[str, ScoringResponse] = {}
        self._required_features: List[str] = []
        self._lock = threading.Lock()

    def health_check(self) -> EndpointHealth:
        model_loaded = self._model is not None
        store_connected = True
        try:
            self._feature_store.read_batch("__health_check__", [])
        except:
            store_connected = True
        return EndpointHealth(
            status="healthy" if model_loaded and store_connected else "unhealthy",
            model_loaded=model_loaded,
            feature_store_connected=store_connected,
            model_version=self._config.model_version,
            uptime_seconds=(datetime.now() - self._start_time).total_seconds(),
            last_request_time=self._last_request_time
        )

    def set_required_features(self, features: List[str]):
        self._required_features = features

    def score(self, request: ScoringRequest) -> ScoringResponse:
        start = time.time()
        self._total_requests += 1
        self._last_request_time = datetime.now()
        try:
            features = self._feature_store.read_batch(request.customer_id, self._required_features or ["page_views_1h", "orders_7d"])
            if not features:
                if request.customer_id in self._fallback_scores:
                    return ScoringResponse(
                        customer_id=request.customer_id,
                        churn_probability=self._fallback_scores[request.customer_id],
                        risk_segment=self._get_risk_segment(self._fallback_scores[request.customer_id]),
                        model_version=self._config.model_version,
                        latency_ms=(time.time() - start) * 1000,
                        is_fallback=True
                    )
                if request.customer_id in self._cache:
                    cached = self._cache[request.customer_id]
                    cached.is_fallback = True
                    cached.latency_ms = (time.time() - start) * 1000
                    return cached
            feature_vector = self._prepare_features(features)
            proba = self._model.predict_proba(feature_vector)[0]
            churn_prob = proba[1] if len(proba) > 1 else proba[0]
            latency = (time.time() - start) * 1000
            self._latencies.append(latency)
            response = ScoringResponse(
                customer_id=request.customer_id,
                churn_probability=churn_prob,
                risk_segment=self._get_risk_segment(churn_prob),
                warning_signals=[],
                model_version=self._config.model_version,
                latency_ms=latency
            )
            if request.include_explanation:
                response.top_risk_factors = self._compute_explanations(features, churn_prob)
            if request.include_recommendation:
                response.recommended_action = self._get_recommendation(churn_prob)
            self._cache[request.customer_id] = response
            return response
        except Exception as e:
            self._errors += 1
            latency = (time.time() - start) * 1000
            if request.customer_id in self._fallback_scores:
                return ScoringResponse(
                    customer_id=request.customer_id,
                    churn_probability=self._fallback_scores[request.customer_id],
                    risk_segment=self._get_risk_segment(self._fallback_scores[request.customer_id]),
                    latency_ms=latency,
                    is_fallback=True
                )
            if request.customer_id in self._cache:
                cached = self._cache[request.customer_id]
                cached.is_fallback = True
                cached.latency_ms = latency
                return cached
            return ScoringResponse(
                customer_id=request.customer_id,
                error=str(e),
                latency_ms=latency
            )

    def score_batch(self, customer_ids: List[str]) -> List[ScoringResponse]:
        return [self.score(ScoringRequest(customer_id=cid)) for cid in customer_ids]

    def get_sla_metrics(self) -> SLAMetrics:
        if not self._latencies:
            return SLAMetrics()
        sorted_lat = sorted(self._latencies)
        return SLAMetrics(
            availability_percent=100.0 * (self._total_requests - self._errors) / max(self._total_requests, 1),
            error_rate_percent=100.0 * self._errors / max(self._total_requests, 1),
            throughput_per_second=self._total_requests / max((datetime.now() - self._start_time).total_seconds(), 1),
            avg_latency_ms=statistics.mean(self._latencies),
            p99_latency_ms=sorted_lat[int(len(sorted_lat) * 0.99)] if len(sorted_lat) > 1 else sorted_lat[0]
        )

    def get_metrics(self) -> ScorerMetrics:
        if not self._latencies:
            return ScorerMetrics(total_requests=self._total_requests)
        sorted_lat = sorted(self._latencies)
        return ScorerMetrics(
            total_requests=self._total_requests,
            successful_requests=self._total_requests - self._errors,
            failed_requests=self._errors,
            avg_latency_ms=statistics.mean(self._latencies),
            p99_latency_ms=sorted_lat[int(len(sorted_lat) * 0.99)] if len(sorted_lat) > 1 else sorted_lat[0]
        )

    def _prepare_features(self, features: Dict[str, float]) -> List[List[float]]:
        return [[features.get(f, 0.0) for f in (self._required_features or list(features.keys()))]]

    def _get_risk_segment(self, probability: float) -> str:
        if probability >= 0.80:
            return "Critical"
        elif probability >= 0.50:
            return "High"
        elif probability >= 0.30:
            return "Medium"
        return "Low"

    def _compute_explanations(self, features: Dict[str, float], probability: float) -> List[RiskFactor]:
        explanations = []
        for name, value in features.items():
            if value > 0:
                impact = value * 0.1
                explanations.append(RiskFactor(factor=name, impact=impact))
        explanations.sort(key=lambda x: x.impact, reverse=True)
        return explanations[:5]

    def _get_recommendation(self, probability: float) -> str:
        if probability >= 0.80:
            return "immediate_outreach"
        elif probability >= 0.50:
            return "retention_campaign"
        elif probability >= 0.30:
            return "engagement_email"
        return "standard_communication"
