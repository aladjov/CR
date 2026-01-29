from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class RecommendationStatus(Enum):
    PENDING = "pending"
    APPLIED = "applied"
    SKIPPED = "skipped"
    FAILED = "failed"


class RecommendationType(Enum):
    CLEANING = "cleaning"
    TRANSFORM = "transform"
    FEATURE = "feature"
    ENCODING = "encoding"


@dataclass
class TrackedRecommendation:
    recommendation_id: str
    recommendation_type: RecommendationType
    source_column: str
    action: str
    description: str
    status: RecommendationStatus = RecommendationStatus.PENDING
    applied_in_iteration: Optional[str] = None
    skip_reason: Optional[str] = None
    failure_reason: Optional[str] = None
    outcome_impact: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None

    @staticmethod
    def generate_id(rec_type: RecommendationType, column: str, action: str) -> str:
        return f"{rec_type.value}_{column}_{action}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommendation_id": self.recommendation_id,
            "recommendation_type": self.recommendation_type.value,
            "source_column": self.source_column,
            "action": self.action,
            "description": self.description,
            "status": self.status.value,
            "applied_in_iteration": self.applied_in_iteration,
            "skip_reason": self.skip_reason,
            "failure_reason": self.failure_reason,
            "outcome_impact": self.outcome_impact,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrackedRecommendation":
        return cls(
            recommendation_id=data["recommendation_id"],
            recommendation_type=RecommendationType(data["recommendation_type"]),
            source_column=data["source_column"],
            action=data["action"],
            description=data.get("description", ""),
            status=RecommendationStatus(data.get("status", "pending")),
            applied_in_iteration=data.get("applied_in_iteration"),
            skip_reason=data.get("skip_reason"),
            failure_reason=data.get("failure_reason"),
            outcome_impact=data.get("outcome_impact"),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None
        )


class RecommendationTracker:
    PATTERN_SECTIONS = ["trend", "seasonality", "cohort", "recency", "categorical"]

    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.recommendations: Dict[str, TrackedRecommendation] = {}

    def add(self, recommendation: TrackedRecommendation) -> None:
        self.recommendations[recommendation.recommendation_id] = recommendation

    def _create_tracked_recommendation(
        self, rec_type: RecommendationType, source_col: str, action: str, description: str
    ) -> TrackedRecommendation:
        rec_id = TrackedRecommendation.generate_id(rec_type, source_col, action)
        tracked = TrackedRecommendation(
            recommendation_id=rec_id,
            recommendation_type=rec_type,
            source_column=source_col,
            action=action,
            description=description
        )
        self.add(tracked)
        return tracked

    def add_from_cleaning(self, cleaning_rec) -> TrackedRecommendation:
        action = f"{cleaning_rec.issue_type}_{cleaning_rec.strategy}"
        return self._create_tracked_recommendation(
            RecommendationType.CLEANING, cleaning_rec.column_name, action, cleaning_rec.description
        )

    def add_from_transform(self, transform_rec) -> TrackedRecommendation:
        return self._create_tracked_recommendation(
            RecommendationType.TRANSFORM, transform_rec.column_name,
            transform_rec.transform_type, transform_rec.reason
        )

    def add_from_feature(self, feature_rec) -> TrackedRecommendation:
        return self._create_tracked_recommendation(
            RecommendationType.FEATURE, feature_rec.source_column,
            feature_rec.feature_name, feature_rec.description
        )

    def _add_feature_list(
        self, rec_dict: Dict[str, Any], default_action: str, source_fn=None
    ) -> List[TrackedRecommendation]:
        features = rec_dict.get("features", [])
        action = rec_dict.get("action", default_action)
        reason = rec_dict.get("reason", "")
        description = f"{action}: {reason}"
        tracked_list = []
        for feature in features:
            source = source_fn(feature) if source_fn else default_action.split("_")[0]
            tracked_list.append(
                self._create_tracked_recommendation(RecommendationType.FEATURE, source, feature, description)
            )
        return tracked_list

    def add_from_recency(self, rec_dict: Dict[str, Any]) -> List[TrackedRecommendation]:
        return self._add_feature_list(rec_dict, "recency_feature", source_fn=lambda _: "recency")

    def add_from_categorical(self, rec_dict: Dict[str, Any]) -> List[TrackedRecommendation]:
        return self._add_feature_list(
            rec_dict, "categorical_feature",
            source_fn=lambda f: f.replace("_is_high_risk", "") if "_is_high_risk" in f else f,
        )

    def _should_add_recommendation(
        self, rec: TrackedRecommendation, seen_ids: set, tracked: List[TrackedRecommendation]
    ) -> bool:
        if rec.recommendation_id in seen_ids:
            return False
        seen_ids.add(rec.recommendation_id)
        self.add(rec)
        tracked.append(rec)
        return True

    def add_from_temporal_findings(self, findings: Any) -> List[TrackedRecommendation]:
        tracked: List[TrackedRecommendation] = []
        seen_ids: set = set()
        pattern_meta = findings.metadata.get("temporal_patterns", {}) if findings.metadata else {}

        def add_if_new(rec: TrackedRecommendation) -> bool:
            return self._should_add_recommendation(rec, seen_ids, tracked)

        self._process_pattern_sections(pattern_meta, add_if_new)
        self._process_temporal_features(pattern_meta, add_if_new)
        self._process_sparkline_recommendations(pattern_meta, add_if_new)
        self._process_effect_size_recommendations(pattern_meta, add_if_new)
        self._process_predictive_power_recommendations(pattern_meta, add_if_new)
        return tracked

    def _process_section_recommendations(self, pattern_meta: Dict, section: str, add_if_new, skip_actions: Optional[List[str]] = None) -> None:
        for rec in pattern_meta.get(section, {}).get("recommendations", []):
            if skip_actions and rec.get("action") in skip_actions:
                continue
            features = rec.get("features", [])
            if not features:
                continue
            action = rec.get("action", f"add_{section}_feature")
            reason = rec.get("reason", f"From {section} analysis")
            priority = rec.get("priority", "medium")
            for feature in features:
                rec_id = TrackedRecommendation.generate_id(RecommendationType.FEATURE, section, feature)
                add_if_new(TrackedRecommendation(
                    recommendation_id=rec_id, recommendation_type=RecommendationType.FEATURE,
                    source_column=section, action=feature, description=f"[{priority}] {action}: {reason}",
                ))

    def _process_pattern_sections(self, pattern_meta: Dict, add_if_new) -> None:
        for section in self.PATTERN_SECTIONS:
            self._process_section_recommendations(pattern_meta, section, add_if_new, ["skip_cohort_features"])

    def _process_temporal_features(self, pattern_meta: Dict, add_if_new) -> None:
        for section in ["velocity", "momentum", "lag"]:
            for rec in pattern_meta.get(section, {}).get("recommendations", []):
                features = rec.get("features", [])
                if not features:
                    continue
                action = rec.get("action", f"add_{section}_feature")
                description = rec.get("description", f"From {section} analysis")
                source_col = rec.get("source_column", section)
                int_priority = rec.get("priority", 2)
                priority_str = self._get_priority_label(int_priority)
                effect_size = rec.get("effect_size")
                effect_info = f" (d={effect_size:.2f})" if effect_size else ""
                for feature in features:
                    rec_id = TrackedRecommendation.generate_id(RecommendationType.FEATURE, source_col, feature)
                    add_if_new(TrackedRecommendation(
                        recommendation_id=rec_id, recommendation_type=RecommendationType.FEATURE,
                        source_column=source_col, action=feature,
                        description=f"[{priority_str}] {action}: {description}{effect_info}",
                    ))

    def _process_sparkline_recommendations(self, pattern_meta: Dict, add_if_new) -> None:
        for rec in pattern_meta.get("sparkline", {}).get("recommendations", []):
            features = rec.get("features", []) or ([rec.get("feature")] if rec.get("feature") else [])
            if not features:
                continue
            action = rec.get("action", "sparkline_feature")
            reason = rec.get("reason", "From sparkline analysis")
            priority = rec.get("priority", "medium")
            for feature in features:
                rec_id = TrackedRecommendation.generate_id(RecommendationType.FEATURE, "sparkline", feature)
                add_if_new(TrackedRecommendation(
                    recommendation_id=rec_id, recommendation_type=RecommendationType.FEATURE,
                    source_column="sparkline", action=feature, description=f"[{priority}] {action}: {reason}",
                ))

    def _process_effect_size_recommendations(self, pattern_meta: Dict, add_if_new) -> None:
        for rec in pattern_meta.get("effect_size", {}).get("recommendations", []):
            feature = rec.get("feature", "")
            if not feature or rec.get("action") == "consider_dropping":
                continue
            effect_d = rec.get("effect_size", 0)
            priority = rec.get("priority", "medium")
            reason = rec.get("reason", f"Effect size d={effect_d:.2f}")
            rec_id = TrackedRecommendation.generate_id(RecommendationType.FEATURE, "effect_size", feature)
            add_if_new(TrackedRecommendation(
                recommendation_id=rec_id, recommendation_type=RecommendationType.FEATURE,
                source_column="effect_size", action=feature, description=f"[{priority}] prioritize: {reason}",
            ))

    def _process_predictive_power_recommendations(self, pattern_meta: Dict, add_if_new) -> None:
        for rec in pattern_meta.get("predictive_power", {}).get("recommendations", []):
            feature = rec.get("feature", "")
            if not feature:
                continue
            iv, ks = rec.get("iv", 0), rec.get("ks", 0)
            priority = rec.get("priority", "medium")
            rec_id = TrackedRecommendation.generate_id(RecommendationType.FEATURE, "predictive_power", feature)
            add_if_new(TrackedRecommendation(
                recommendation_id=rec_id, recommendation_type=RecommendationType.FEATURE,
                source_column="predictive_power", action=feature,
                description=f"[{priority}] include: IV={iv:.3f}, KS={ks:.3f}",
            ))

    @staticmethod
    def _get_priority_label(int_priority: int) -> str:
        return "high" if int_priority == 1 else "medium"

    def get(self, recommendation_id: str) -> Optional[TrackedRecommendation]:
        return self.recommendations.get(recommendation_id)

    def _update_recommendation_status(self, recommendation_id: str, status: RecommendationStatus, **kwargs) -> None:
        rec = self.get(recommendation_id)
        if rec:
            rec.status = status
            rec.updated_at = datetime.now()
            for attr, value in kwargs.items():
                setattr(rec, attr, value)

    def mark_applied(self, recommendation_id: str, iteration_id: str) -> None:
        self._update_recommendation_status(
            recommendation_id, RecommendationStatus.APPLIED, applied_in_iteration=iteration_id
        )

    def mark_skipped(self, recommendation_id: str, reason: str) -> None:
        self._update_recommendation_status(
            recommendation_id, RecommendationStatus.SKIPPED, skip_reason=reason
        )

    def mark_failed(self, recommendation_id: str, reason: str) -> None:
        self._update_recommendation_status(
            recommendation_id, RecommendationStatus.FAILED, failure_reason=reason
        )

    def set_outcome_impact(self, recommendation_id: str, impact: float) -> None:
        rec = self.get(recommendation_id)
        if rec:
            rec.outcome_impact = impact
            rec.updated_at = datetime.now()

    def _get_by_status(self, status: RecommendationStatus) -> List[TrackedRecommendation]:
        return [r for r in self.recommendations.values() if r.status == status]

    def get_pending(self) -> List[TrackedRecommendation]:
        return self._get_by_status(RecommendationStatus.PENDING)

    def get_applied(self) -> List[TrackedRecommendation]:
        return self._get_by_status(RecommendationStatus.APPLIED)

    def get_skipped(self) -> List[TrackedRecommendation]:
        return self._get_by_status(RecommendationStatus.SKIPPED)

    def get_failed(self) -> List[TrackedRecommendation]:
        return self._get_by_status(RecommendationStatus.FAILED)

    def get_high_impact(self, threshold: float = 0.1) -> List[TrackedRecommendation]:
        high_impact = [
            r for r in self.recommendations.values()
            if r.outcome_impact is not None and r.outcome_impact >= threshold
        ]
        high_impact.sort(key=lambda x: x.outcome_impact or 0, reverse=True)
        return high_impact

    def get_by_type(self, rec_type: RecommendationType) -> List[TrackedRecommendation]:
        return [r for r in self.recommendations.values()
                if r.recommendation_type == rec_type]

    def get_summary(self) -> Dict[str, int]:
        return {
            "total": len(self.recommendations),
            "pending": len(self.get_pending()),
            "applied": len(self.get_applied()),
            "skipped": len(self.get_skipped()),
            "failed": len(self.get_failed()),
        }

    def save(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "recommendations": [r.to_dict() for r in self.recommendations.values()]
        }
        with open(self.storage_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def load(self) -> None:
        if not self.storage_path.exists():
            return
        with open(self.storage_path, "r") as f:
            data = yaml.safe_load(f)
        if data and "recommendations" in data:
            for rec_data in data["recommendations"]:
                rec = TrackedRecommendation.from_dict(rec_data)
                self.recommendations[rec.recommendation_id] = rec
