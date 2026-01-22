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
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.recommendations: Dict[str, TrackedRecommendation] = {}

    def add(self, recommendation: TrackedRecommendation) -> None:
        self.recommendations[recommendation.recommendation_id] = recommendation

    def add_from_cleaning(self, cleaning_rec) -> TrackedRecommendation:
        rec_id = TrackedRecommendation.generate_id(
            RecommendationType.CLEANING,
            cleaning_rec.column_name,
            f"{cleaning_rec.issue_type}_{cleaning_rec.strategy}"
        )
        tracked = TrackedRecommendation(
            recommendation_id=rec_id,
            recommendation_type=RecommendationType.CLEANING,
            source_column=cleaning_rec.column_name,
            action=f"{cleaning_rec.issue_type}_{cleaning_rec.strategy}",
            description=cleaning_rec.description
        )
        self.add(tracked)
        return tracked

    def add_from_transform(self, transform_rec) -> TrackedRecommendation:
        rec_id = TrackedRecommendation.generate_id(
            RecommendationType.TRANSFORM,
            transform_rec.column_name,
            transform_rec.transform_type
        )
        tracked = TrackedRecommendation(
            recommendation_id=rec_id,
            recommendation_type=RecommendationType.TRANSFORM,
            source_column=transform_rec.column_name,
            action=transform_rec.transform_type,
            description=transform_rec.reason
        )
        self.add(tracked)
        return tracked

    def add_from_feature(self, feature_rec) -> TrackedRecommendation:
        rec_id = TrackedRecommendation.generate_id(
            RecommendationType.FEATURE,
            feature_rec.source_column,
            feature_rec.feature_name
        )
        tracked = TrackedRecommendation(
            recommendation_id=rec_id,
            recommendation_type=RecommendationType.FEATURE,
            source_column=feature_rec.source_column,
            action=feature_rec.feature_name,
            description=feature_rec.description
        )
        self.add(tracked)
        return tracked

    def get(self, recommendation_id: str) -> Optional[TrackedRecommendation]:
        return self.recommendations.get(recommendation_id)

    def mark_applied(self, recommendation_id: str, iteration_id: str) -> None:
        rec = self.get(recommendation_id)
        if rec:
            rec.status = RecommendationStatus.APPLIED
            rec.applied_in_iteration = iteration_id
            rec.updated_at = datetime.now()

    def mark_skipped(self, recommendation_id: str, reason: str) -> None:
        rec = self.get(recommendation_id)
        if rec:
            rec.status = RecommendationStatus.SKIPPED
            rec.skip_reason = reason
            rec.updated_at = datetime.now()

    def mark_failed(self, recommendation_id: str, reason: str) -> None:
        rec = self.get(recommendation_id)
        if rec:
            rec.status = RecommendationStatus.FAILED
            rec.failure_reason = reason
            rec.updated_at = datetime.now()

    def set_outcome_impact(self, recommendation_id: str, impact: float) -> None:
        rec = self.get(recommendation_id)
        if rec:
            rec.outcome_impact = impact
            rec.updated_at = datetime.now()

    def get_pending(self) -> List[TrackedRecommendation]:
        return [r for r in self.recommendations.values()
                if r.status == RecommendationStatus.PENDING]

    def get_applied(self) -> List[TrackedRecommendation]:
        return [r for r in self.recommendations.values()
                if r.status == RecommendationStatus.APPLIED]

    def get_skipped(self) -> List[TrackedRecommendation]:
        return [r for r in self.recommendations.values()
                if r.status == RecommendationStatus.SKIPPED]

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
            "failed": len([r for r in self.recommendations.values()
                          if r.status == RecommendationStatus.FAILED])
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
