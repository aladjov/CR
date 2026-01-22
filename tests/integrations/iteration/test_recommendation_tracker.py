

class TestRecommendationStatus:
    def test_status_values(self):
        from customer_retention.integrations.iteration.recommendation_tracker import RecommendationStatus
        assert RecommendationStatus.PENDING.value == "pending"
        assert RecommendationStatus.APPLIED.value == "applied"
        assert RecommendationStatus.SKIPPED.value == "skipped"
        assert RecommendationStatus.FAILED.value == "failed"


class TestRecommendationType:
    def test_type_values(self):
        from customer_retention.integrations.iteration.recommendation_tracker import RecommendationType
        assert RecommendationType.CLEANING.value == "cleaning"
        assert RecommendationType.TRANSFORM.value == "transform"
        assert RecommendationType.FEATURE.value == "feature"
        assert RecommendationType.ENCODING.value == "encoding"


class TestTrackedRecommendation:
    def test_create_tracked_recommendation(self):
        from customer_retention.integrations.iteration.recommendation_tracker import (
            RecommendationStatus,
            RecommendationType,
            TrackedRecommendation,
        )
        rec = TrackedRecommendation(
            recommendation_id="rec_001",
            recommendation_type=RecommendationType.CLEANING,
            source_column="age",
            action="impute_median",
            description="Impute missing values with median"
        )
        assert rec.recommendation_id == "rec_001"
        assert rec.status == RecommendationStatus.PENDING
        assert rec.applied_in_iteration is None
        assert rec.skip_reason is None
        assert rec.outcome_impact is None

    def test_generate_id(self):
        from customer_retention.integrations.iteration.recommendation_tracker import (
            RecommendationType,
            TrackedRecommendation,
        )
        rec_id = TrackedRecommendation.generate_id(
            RecommendationType.CLEANING, "age", "impute_median"
        )
        assert rec_id == "cleaning_age_impute_median"

    def test_to_dict(self):
        from customer_retention.integrations.iteration.recommendation_tracker import (
            RecommendationType,
            TrackedRecommendation,
        )
        rec = TrackedRecommendation(
            recommendation_id="rec_001",
            recommendation_type=RecommendationType.TRANSFORM,
            source_column="income",
            action="log_transform",
            description="Apply log transform"
        )
        data = rec.to_dict()
        assert data["recommendation_id"] == "rec_001"
        assert data["recommendation_type"] == "transform"
        assert data["status"] == "pending"

    def test_from_dict(self):
        from customer_retention.integrations.iteration.recommendation_tracker import (
            RecommendationStatus,
            RecommendationType,
            TrackedRecommendation,
        )
        data = {
            "recommendation_id": "rec_002",
            "recommendation_type": "feature",
            "source_column": "date",
            "action": "extract_month",
            "description": "Extract month from date",
            "status": "applied",
            "applied_in_iteration": "iter_001",
            "outcome_impact": 0.15
        }
        rec = TrackedRecommendation.from_dict(data)
        assert rec.recommendation_id == "rec_002"
        assert rec.recommendation_type == RecommendationType.FEATURE
        assert rec.status == RecommendationStatus.APPLIED
        assert rec.applied_in_iteration == "iter_001"
        assert rec.outcome_impact == 0.15


class TestRecommendationTracker:
    def test_create_tracker(self, tmp_path):
        from customer_retention.integrations.iteration.recommendation_tracker import RecommendationTracker
        tracker = RecommendationTracker(str(tmp_path / "recommendations.yaml"))
        assert tracker is not None
        assert len(tracker.recommendations) == 0

    def test_add_recommendation(self, tmp_path):
        from customer_retention.integrations.iteration.recommendation_tracker import (
            RecommendationTracker,
            RecommendationType,
            TrackedRecommendation,
        )
        tracker = RecommendationTracker(str(tmp_path / "recommendations.yaml"))
        rec = TrackedRecommendation(
            recommendation_id="rec_001",
            recommendation_type=RecommendationType.CLEANING,
            source_column="age",
            action="impute_median",
            description="Impute missing values"
        )
        tracker.add(rec)
        assert len(tracker.recommendations) == 1
        assert tracker.get("rec_001") is not None

    def test_add_from_cleaning_recommendation(self, tmp_path):
        from customer_retention.analysis.auto_explorer.recommendations import CleaningRecommendation
        from customer_retention.integrations.iteration.recommendation_tracker import RecommendationTracker
        tracker = RecommendationTracker(str(tmp_path / "recommendations.yaml"))
        cleaning_rec = CleaningRecommendation(
            column_name="income",
            issue_type="outliers",
            severity="medium",
            strategy="clip_or_winsorize",
            description="Clip outliers to percentiles"
        )
        tracked = tracker.add_from_cleaning(cleaning_rec)
        assert tracked is not None
        assert tracked.source_column == "income"
        assert "outliers" in tracked.action

    def test_add_from_transform_recommendation(self, tmp_path):
        from customer_retention.analysis.auto_explorer.recommendations import TransformRecommendation
        from customer_retention.integrations.iteration.recommendation_tracker import RecommendationTracker
        tracker = RecommendationTracker(str(tmp_path / "recommendations.yaml"))
        transform_rec = TransformRecommendation(
            column_name="amount",
            transform_type="log_transform",
            reason="High skewness",
            parameters={"base": "natural"}
        )
        tracked = tracker.add_from_transform(transform_rec)
        assert tracked is not None
        assert tracked.source_column == "amount"
        assert tracked.action == "log_transform"

    def test_add_from_feature_recommendation(self, tmp_path):
        from customer_retention.analysis.auto_explorer.recommendations import FeatureRecommendation
        from customer_retention.integrations.iteration.recommendation_tracker import RecommendationTracker
        tracker = RecommendationTracker(str(tmp_path / "recommendations.yaml"))
        feature_rec = FeatureRecommendation(
            source_column="signup_date",
            feature_name="days_since_signup",
            feature_type="datetime",
            description="Days since signup"
        )
        tracked = tracker.add_from_feature(feature_rec)
        assert tracked is not None
        assert tracked.source_column == "signup_date"

    def test_mark_applied(self, tmp_path):
        from customer_retention.integrations.iteration.recommendation_tracker import (
            RecommendationStatus,
            RecommendationTracker,
            RecommendationType,
            TrackedRecommendation,
        )
        tracker = RecommendationTracker(str(tmp_path / "recommendations.yaml"))
        rec = TrackedRecommendation(
            recommendation_id="rec_001",
            recommendation_type=RecommendationType.CLEANING,
            source_column="age",
            action="impute_median",
            description="Impute missing values"
        )
        tracker.add(rec)
        tracker.mark_applied("rec_001", "iter_001")
        updated = tracker.get("rec_001")
        assert updated.status == RecommendationStatus.APPLIED
        assert updated.applied_in_iteration == "iter_001"

    def test_mark_skipped(self, tmp_path):
        from customer_retention.integrations.iteration.recommendation_tracker import (
            RecommendationStatus,
            RecommendationTracker,
            RecommendationType,
            TrackedRecommendation,
        )
        tracker = RecommendationTracker(str(tmp_path / "recommendations.yaml"))
        rec = TrackedRecommendation(
            recommendation_id="rec_002",
            recommendation_type=RecommendationType.TRANSFORM,
            source_column="income",
            action="log_transform",
            description="Apply log transform"
        )
        tracker.add(rec)
        tracker.mark_skipped("rec_002", "Not needed for this model")
        updated = tracker.get("rec_002")
        assert updated.status == RecommendationStatus.SKIPPED
        assert updated.skip_reason == "Not needed for this model"

    def test_mark_failed(self, tmp_path):
        from customer_retention.integrations.iteration.recommendation_tracker import (
            RecommendationStatus,
            RecommendationTracker,
            RecommendationType,
            TrackedRecommendation,
        )
        tracker = RecommendationTracker(str(tmp_path / "recommendations.yaml"))
        rec = TrackedRecommendation(
            recommendation_id="rec_003",
            recommendation_type=RecommendationType.FEATURE,
            source_column="date",
            action="extract_quarter",
            description="Extract quarter"
        )
        tracker.add(rec)
        tracker.mark_failed("rec_003", "Column not found")
        updated = tracker.get("rec_003")
        assert updated.status == RecommendationStatus.FAILED

    def test_set_outcome_impact(self, tmp_path):
        from customer_retention.integrations.iteration.recommendation_tracker import (
            RecommendationTracker,
            RecommendationType,
            TrackedRecommendation,
        )
        tracker = RecommendationTracker(str(tmp_path / "recommendations.yaml"))
        rec = TrackedRecommendation(
            recommendation_id="rec_001",
            recommendation_type=RecommendationType.FEATURE,
            source_column="age",
            action="age_binned",
            description="Bin age into groups"
        )
        tracker.add(rec)
        tracker.set_outcome_impact("rec_001", 0.25)
        updated = tracker.get("rec_001")
        assert updated.outcome_impact == 0.25

    def test_get_pending(self, tmp_path):
        from customer_retention.integrations.iteration.recommendation_tracker import (
            RecommendationTracker,
            RecommendationType,
            TrackedRecommendation,
        )
        tracker = RecommendationTracker(str(tmp_path / "recommendations.yaml"))
        for i in range(3):
            rec = TrackedRecommendation(
                recommendation_id=f"rec_00{i+1}",
                recommendation_type=RecommendationType.CLEANING,
                source_column=f"col_{i}",
                action="impute",
                description="Impute"
            )
            tracker.add(rec)
        tracker.mark_applied("rec_001", "iter_001")
        pending = tracker.get_pending()
        assert len(pending) == 2

    def test_get_applied(self, tmp_path):
        from customer_retention.integrations.iteration.recommendation_tracker import (
            RecommendationTracker,
            RecommendationType,
            TrackedRecommendation,
        )
        tracker = RecommendationTracker(str(tmp_path / "recommendations.yaml"))
        for i in range(3):
            rec = TrackedRecommendation(
                recommendation_id=f"rec_00{i+1}",
                recommendation_type=RecommendationType.CLEANING,
                source_column=f"col_{i}",
                action="impute",
                description="Impute"
            )
            tracker.add(rec)
        tracker.mark_applied("rec_001", "iter_001")
        tracker.mark_applied("rec_002", "iter_001")
        applied = tracker.get_applied()
        assert len(applied) == 2

    def test_get_high_impact(self, tmp_path):
        from customer_retention.integrations.iteration.recommendation_tracker import (
            RecommendationTracker,
            RecommendationType,
            TrackedRecommendation,
        )
        tracker = RecommendationTracker(str(tmp_path / "recommendations.yaml"))
        for i, impact in enumerate([0.05, 0.25, 0.15, 0.30]):
            rec = TrackedRecommendation(
                recommendation_id=f"rec_00{i+1}",
                recommendation_type=RecommendationType.FEATURE,
                source_column=f"col_{i}",
                action="feature",
                description="Feature"
            )
            tracker.add(rec)
            tracker.mark_applied(f"rec_00{i+1}", "iter_001")
            tracker.set_outcome_impact(f"rec_00{i+1}", impact)

        high_impact = tracker.get_high_impact(threshold=0.10)
        assert len(high_impact) == 3
        assert high_impact[0].outcome_impact == 0.30  # Sorted by impact

    def test_get_by_type(self, tmp_path):
        from customer_retention.integrations.iteration.recommendation_tracker import (
            RecommendationTracker,
            RecommendationType,
            TrackedRecommendation,
        )
        tracker = RecommendationTracker(str(tmp_path / "recommendations.yaml"))
        tracker.add(TrackedRecommendation("rec_001", RecommendationType.CLEANING, "a", "clean", ""))
        tracker.add(TrackedRecommendation("rec_002", RecommendationType.TRANSFORM, "b", "transform", ""))
        tracker.add(TrackedRecommendation("rec_003", RecommendationType.CLEANING, "c", "clean", ""))

        cleaning = tracker.get_by_type(RecommendationType.CLEANING)
        assert len(cleaning) == 2

    def test_save_and_load(self, tmp_path):
        from customer_retention.integrations.iteration.recommendation_tracker import (
            RecommendationStatus,
            RecommendationTracker,
            RecommendationType,
            TrackedRecommendation,
        )
        path = str(tmp_path / "recommendations.yaml")
        tracker = RecommendationTracker(path)
        rec = TrackedRecommendation(
            recommendation_id="rec_001",
            recommendation_type=RecommendationType.CLEANING,
            source_column="age",
            action="impute_median",
            description="Impute missing values"
        )
        tracker.add(rec)
        tracker.mark_applied("rec_001", "iter_001")
        tracker.set_outcome_impact("rec_001", 0.15)
        tracker.save()

        loaded = RecommendationTracker(path)
        loaded.load()
        assert len(loaded.recommendations) == 1
        loaded_rec = loaded.get("rec_001")
        assert loaded_rec.status == RecommendationStatus.APPLIED
        assert loaded_rec.outcome_impact == 0.15

    def test_get_summary(self, tmp_path):
        from customer_retention.integrations.iteration.recommendation_tracker import (
            RecommendationTracker,
            RecommendationType,
            TrackedRecommendation,
        )
        tracker = RecommendationTracker(str(tmp_path / "recommendations.yaml"))
        for i in range(5):
            rec = TrackedRecommendation(
                recommendation_id=f"rec_00{i+1}",
                recommendation_type=RecommendationType.CLEANING,
                source_column=f"col_{i}",
                action="impute",
                description="Impute"
            )
            tracker.add(rec)
        tracker.mark_applied("rec_001", "iter_001")
        tracker.mark_applied("rec_002", "iter_001")
        tracker.mark_skipped("rec_003", "Not needed")

        summary = tracker.get_summary()
        assert summary["total"] == 5
        assert summary["applied"] == 2
        assert summary["skipped"] == 1
        assert summary["pending"] == 2
        assert summary["failed"] == 0
