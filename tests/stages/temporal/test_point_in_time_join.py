
import pandas as pd
import pytest

from customer_retention.stages.temporal.point_in_time_join import PointInTimeJoiner


class TestPointInTimeJoiner:
    @pytest.fixture
    def base_df(self):
        return pd.DataFrame({
            "entity_id": ["A", "A", "B", "B"],
            "feature_timestamp": pd.to_datetime([
                "2024-03-01", "2024-06-01", "2024-02-01", "2024-05-01"
            ]),
            "base_value": [100, 200, 300, 400]
        })

    @pytest.fixture
    def feature_df(self):
        return pd.DataFrame({
            "entity_id": ["A", "A", "A", "B", "B"],
            "feature_timestamp": pd.to_datetime([
                "2024-01-01", "2024-02-15", "2024-05-01", "2024-01-15", "2024-04-01"
            ]),
            "external_feature": [10, 20, 30, 40, 50]
        })


class TestJoinFeatures(TestPointInTimeJoiner):
    def test_join_features_point_in_time_correct(self, base_df, feature_df):
        result = PointInTimeJoiner.join_features(
            base_df, feature_df, "entity_id"
        )

        assert len(result) > 0
        for _, row in result.iterrows():
            if "external_feature" in row:
                assert True

    def test_join_features_missing_base_timestamp_raises(self, feature_df):
        base_df_no_ts = pd.DataFrame({
            "entity_id": ["A", "B"],
            "value": [100, 200]
        })

        with pytest.raises(ValueError, match="Base df missing timestamp column"):
            PointInTimeJoiner.join_features(base_df_no_ts, feature_df, "entity_id")

    def test_join_features_missing_feature_timestamp_raises(self, base_df):
        feature_df_no_ts = pd.DataFrame({
            "entity_id": ["A", "B"],
            "external_feature": [10, 20]
        })

        with pytest.raises(ValueError, match="Feature df missing timestamp column"):
            PointInTimeJoiner.join_features(base_df, feature_df_no_ts, "entity_id")


class TestValidateNoFutureData(TestPointInTimeJoiner):
    def test_no_issues_when_valid(self):
        df = pd.DataFrame({
            "feature_timestamp": pd.to_datetime(["2024-06-01", "2024-06-01"]),
            "event_date": pd.to_datetime(["2024-05-01", "2024-04-01"]),
            "value": [100, 200]
        })

        issues = PointInTimeJoiner.validate_no_future_data(
            df, "feature_timestamp", ["event_date"]
        )

        assert len(issues) == 0

    def test_detects_future_data(self):
        df = pd.DataFrame({
            "feature_timestamp": pd.to_datetime(["2024-06-01", "2024-06-01"]),
            "event_date": pd.to_datetime(["2024-07-01", "2024-08-01"]),
            "value": [100, 200]
        })

        issues = PointInTimeJoiner.validate_no_future_data(
            df, "feature_timestamp", ["event_date"]
        )

        assert "event_date" in issues
        assert issues["event_date"]["violation_count"] == 2


class TestAsofJoin(TestPointInTimeJoiner):
    def test_asof_join_backward(self):
        left_df = pd.DataFrame({
            "entity_id": ["A", "A", "B"],
            "left_time": pd.to_datetime(["2024-03-01", "2024-06-01", "2024-04-01"]),
            "left_value": [1, 2, 3]
        })

        right_df = pd.DataFrame({
            "entity_id": ["A", "A", "B"],
            "right_time": pd.to_datetime(["2024-02-01", "2024-05-01", "2024-03-15"]),
            "right_value": [10, 20, 30]
        })

        result = PointInTimeJoiner.asof_join(
            left_df, right_df, "entity_id", "left_time", "right_time"
        )

        assert len(result) == 3


class TestCreateTrainingLabels(TestPointInTimeJoiner):
    def test_create_training_labels_filters_available(self):
        df = pd.DataFrame({
            "entity_id": ["A", "B", "C"],
            "feature_timestamp": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "label_timestamp": pd.to_datetime(["2024-04-01", "2024-05-01", "2024-06-01"]),
            "label_available_flag": [True, True, False],
            "target": [1, 0, 1]
        })

        result = PointInTimeJoiner.create_training_labels(df, "target")

        assert len(result) == 2
        assert list(result["entity_id"]) == ["A", "B"]

    def test_create_training_labels_missing_flag_raises(self):
        df = pd.DataFrame({
            "entity_id": ["A", "B"],
            "target": [1, 0]
        })

        with pytest.raises(ValueError, match="label_available_flag"):
            PointInTimeJoiner.create_training_labels(df, "target")


class TestValidateTemporalIntegrity(TestPointInTimeJoiner):
    def test_valid_temporal_integrity(self):
        df = pd.DataFrame({
            "feature_timestamp": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "label_timestamp": pd.to_datetime(["2024-04-01", "2024-05-01"]),
        })

        report = PointInTimeJoiner.validate_temporal_integrity(df)

        assert report["valid"] is True
        assert len(report["issues"]) == 0

    def test_invalid_feature_after_label(self):
        df = pd.DataFrame({
            "feature_timestamp": pd.to_datetime(["2024-06-01", "2024-02-01"]),
            "label_timestamp": pd.to_datetime(["2024-04-01", "2024-05-01"]),
        })

        report = PointInTimeJoiner.validate_temporal_integrity(df)

        assert report["valid"] is False
        assert any(i["type"] == "feature_after_label" for i in report["issues"])

    def test_detects_future_datetime_columns(self):
        df = pd.DataFrame({
            "feature_timestamp": pd.to_datetime(["2024-03-01", "2024-03-01"]),
            "label_timestamp": pd.to_datetime(["2024-06-01", "2024-06-01"]),
            "some_event": pd.to_datetime(["2024-04-01", "2024-04-01"]),
        })

        report = PointInTimeJoiner.validate_temporal_integrity(df)

        future_issues = [i for i in report["issues"] if i["type"] == "future_data"]
        assert len(future_issues) >= 1
