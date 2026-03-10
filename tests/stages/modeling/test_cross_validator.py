import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from customer_retention.stages.modeling import CrossValidator, CVStrategy


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 500
    X = pd.DataFrame({
        "feature1": np.random.randn(n),
        "feature2": np.random.randn(n),
    })
    y = pd.Series(np.random.choice([0, 1], n, p=[0.3, 0.7]))
    return X, y


@pytest.fixture
def sample_model():
    return LogisticRegression(max_iter=1000, random_state=42)


class TestCVStrategy:
    def test_strategy_enum_has_required_values(self):
        assert hasattr(CVStrategy, "STRATIFIED_KFOLD")
        assert hasattr(CVStrategy, "REPEATED_STRATIFIED")
        assert hasattr(CVStrategy, "TIME_SERIES")
        assert hasattr(CVStrategy, "GROUP_KFOLD")


class TestStratifiedKFold:
    def test_runs_stratified_kfold(self, sample_data, sample_model):
        X, y = sample_data
        cv = CrossValidator(strategy=CVStrategy.STRATIFIED_KFOLD, n_splits=5)
        result = cv.run(sample_model, X, y)

        assert result.cv_scores is not None
        assert len(result.cv_scores) == 5

    def test_cv_scores_in_valid_range(self, sample_data, sample_model):
        X, y = sample_data
        cv = CrossValidator(strategy=CVStrategy.STRATIFIED_KFOLD, n_splits=5)
        result = cv.run(sample_model, X, y)

        assert all(0 <= score <= 1 for score in result.cv_scores)

    def test_preserves_class_balance_in_folds(self, sample_data, sample_model):
        X, y = sample_data
        cv = CrossValidator(strategy=CVStrategy.STRATIFIED_KFOLD, n_splits=5)
        result = cv.run(sample_model, X, y)

        original_ratio = y.mean()
        for fold_info in result.fold_details:
            fold_ratio = fold_info.get("train_class_ratio", original_ratio)
            assert abs(fold_ratio - original_ratio) < 0.1


class TestRepeatedStratifiedKFold:
    def test_runs_repeated_stratified_kfold(self, sample_data, sample_model):
        X, y = sample_data
        cv = CrossValidator(
            strategy=CVStrategy.REPEATED_STRATIFIED,
            n_splits=3,
            n_repeats=2
        )
        result = cv.run(sample_model, X, y)

        assert len(result.cv_scores) == 6


class TestCVMetrics:
    def test_calculates_cv_mean(self, sample_data, sample_model):
        X, y = sample_data
        cv = CrossValidator(n_splits=5)
        result = cv.run(sample_model, X, y)

        assert result.cv_mean is not None
        assert result.cv_mean == pytest.approx(np.mean(result.cv_scores), rel=1e-6)

    def test_calculates_cv_std(self, sample_data, sample_model):
        X, y = sample_data
        cv = CrossValidator(n_splits=5)
        result = cv.run(sample_model, X, y)

        assert result.cv_std is not None
        assert result.cv_std == pytest.approx(np.std(result.cv_scores), rel=1e-6)


class TestCVResult:
    def test_result_contains_required_fields(self, sample_data, sample_model):
        X, y = sample_data
        cv = CrossValidator(n_splits=5)
        result = cv.run(sample_model, X, y)

        assert hasattr(result, "cv_scores")
        assert hasattr(result, "cv_mean")
        assert hasattr(result, "cv_std")
        assert hasattr(result, "fold_details")

    def test_fold_details_has_all_folds(self, sample_data, sample_model):
        X, y = sample_data
        cv = CrossValidator(n_splits=5)
        result = cv.run(sample_model, X, y)

        assert len(result.fold_details) == 5


class TestCVConfiguration:
    def test_custom_n_splits(self, sample_data, sample_model):
        X, y = sample_data
        cv = CrossValidator(n_splits=10)
        result = cv.run(sample_model, X, y)

        assert len(result.cv_scores) == 10

    def test_custom_scoring_metric(self, sample_data, sample_model):
        X, y = sample_data
        cv = CrossValidator(n_splits=5, scoring="roc_auc")
        result = cv.run(sample_model, X, y)

        assert result.scoring == "roc_auc"

    def test_custom_random_state(self, sample_data, sample_model):
        X, y = sample_data

        cv1 = CrossValidator(n_splits=5, random_state=42)
        cv2 = CrossValidator(n_splits=5, random_state=42)

        result1 = cv1.run(sample_model, X, y)
        result2 = cv2.run(sample_model, X, y)

        np.testing.assert_array_almost_equal(result1.cv_scores, result2.cv_scores)


class TestCVStability:
    def test_detects_high_variance(self, sample_data, sample_model):
        X, y = sample_data
        cv = CrossValidator(n_splits=5)
        result = cv.run(sample_model, X, y)

        assert hasattr(result, "is_stable")
        assert isinstance(result.is_stable, bool)

    def test_flags_unstable_cv(self):
        np.random.seed(42)
        X = pd.DataFrame({
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
        })
        y = pd.Series(np.random.choice([0, 1], 100, p=[0.5, 0.5]))

        model = LogisticRegression(max_iter=1000, random_state=42)
        cv = CrossValidator(n_splits=5, stability_threshold=0.01)
        result = cv.run(model, X, y)

        assert result.cv_std is not None


class TestAveragePrecisionScoring:
    def test_default_scoring_is_average_precision(self, sample_data, sample_model):
        X, y = sample_data
        cv = CrossValidator(n_splits=5)
        result = cv.run(sample_model, X, y)

        assert result.scoring == "average_precision"


class TestFoldDetails:
    def test_fold_contains_train_test_sizes(self, sample_data, sample_model):
        X, y = sample_data
        cv = CrossValidator(n_splits=5)
        result = cv.run(sample_model, X, y)

        for fold in result.fold_details:
            assert "train_size" in fold
            assert "test_size" in fold

    def test_fold_contains_score(self, sample_data, sample_model):
        X, y = sample_data
        cv = CrossValidator(n_splits=5)
        result = cv.run(sample_model, X, y)

        for fold in result.fold_details:
            assert "score" in fold


class TestTemporalEntityCV:
    @pytest.fixture
    def panel_data(self):
        np.random.seed(42)
        n_entities = 20
        dates = pd.date_range("2023-01-01", periods=12, freq="MS")
        rows = []
        for eid in range(n_entities):
            for d in dates:
                rows.append({
                    "entity_id": eid,
                    "as_of_date": d,
                    "feature1": np.random.randn(),
                    "feature2": np.random.randn(),
                    "target": np.random.choice([0, 1], p=[0.3, 0.7]),
                })
        df = pd.DataFrame(rows)
        X = df[["feature1", "feature2"]]
        y = df["target"]
        groups = df["entity_id"]
        temporal = df["as_of_date"]
        return X, y, groups, temporal

    @pytest.fixture
    def panel_model(self):
        return LogisticRegression(max_iter=1000, random_state=42)

    def test_no_entity_overlap_between_train_and_test(self, panel_data, panel_model):
        X, y, groups, temporal = panel_data
        from customer_retention.stages.modeling.cross_validator import TemporalEntitySplit

        splitter = TemporalEntitySplit(n_splits=5)
        for train_idx, test_idx in splitter.split(X, y, groups):
            train_entities = set(groups.iloc[train_idx].unique())
            test_entities = set(groups.iloc[test_idx].unique())
            assert train_entities.isdisjoint(test_entities)

    def test_all_rows_of_entity_in_same_set(self, panel_data, panel_model):
        X, y, groups, temporal = panel_data
        from customer_retention.stages.modeling.cross_validator import TemporalEntitySplit

        splitter = TemporalEntitySplit(n_splits=5)
        for train_idx, test_idx in splitter.split(X, y, groups):
            train_entities = set(groups.iloc[train_idx].unique())
            test_entities = set(groups.iloc[test_idx].unique())
            for eid in train_entities:
                entity_rows = groups[groups == eid].index
                assert all(idx in train_idx for idx in entity_rows)
            for eid in test_entities:
                entity_rows = groups[groups == eid].index
                assert all(idx in test_idx for idx in entity_rows)

    def test_correct_number_of_folds(self, panel_data, panel_model):
        X, y, groups, temporal = panel_data
        cv = CrossValidator(strategy=CVStrategy.TEMPORAL_ENTITY, n_splits=5, scoring="roc_auc")
        result = cv.run(panel_model, X, y, groups=groups)
        assert len(result.cv_scores) == 5

    def test_cv_scores_in_valid_range(self, panel_data, panel_model):
        X, y, groups, temporal = panel_data
        cv = CrossValidator(strategy=CVStrategy.TEMPORAL_ENTITY, n_splits=5, scoring="roc_auc")
        result = cv.run(panel_model, X, y, groups=groups)
        assert all(0 <= s <= 1 for s in result.cv_scores)

    def test_fold_details_report_entity_counts(self, panel_data, panel_model):
        X, y, groups, temporal = panel_data
        cv = CrossValidator(strategy=CVStrategy.TEMPORAL_ENTITY, n_splits=5, scoring="roc_auc")
        result = cv.run(panel_model, X, y, groups=groups)
        for fold in result.fold_details:
            assert "train_entities" in fold
            assert "test_entities" in fold
            assert fold["train_entities"] > 0
            assert fold["test_entities"] > 0

    def test_purge_gap_reduces_train_size_for_sparse_data(self):
        np.random.seed(42)
        base = pd.Timestamp("2023-01-01")
        rows = [
            {"entity_id": 0, "as_of_date": base, "feature1": 1.0, "target": 0},
            {"entity_id": 0, "as_of_date": base + pd.Timedelta(days=30), "feature1": 1.1, "target": 0},
            {"entity_id": 0, "as_of_date": base + pd.Timedelta(days=120), "feature1": 1.2, "target": 1},
            {"entity_id": 1, "as_of_date": base + pd.Timedelta(days=150), "feature1": 2.0, "target": 1},
            {"entity_id": 1, "as_of_date": base + pd.Timedelta(days=180), "feature1": 2.1, "target": 0},
        ]
        df = pd.DataFrame(rows)
        X = df[["feature1"]]
        y = df["target"]
        groups = df["entity_id"]
        temporal = df["as_of_date"]

        from customer_retention.stages.modeling.cross_validator import TemporalEntitySplit

        splitter_no_purge = TemporalEntitySplit(n_splits=2)
        splitter_purge = TemporalEntitySplit(n_splits=2, temporal_values=temporal, purge_gap_days=90)

        sizes_no_purge = [len(tr) for tr, _ in splitter_no_purge.split(X, y, groups)]
        sizes_purge = [len(tr) for tr, _ in splitter_purge.split(X, y, groups)]

        assert any(sp < snp for sp, snp in zip(sizes_purge, sizes_no_purge))

    def test_purge_gap_skipped_when_would_empty_train(self, panel_data, panel_model):
        X, y, groups, temporal = panel_data
        from customer_retention.stages.modeling.cross_validator import TemporalEntitySplit

        splitter = TemporalEntitySplit(n_splits=5, temporal_values=temporal, purge_gap_days=9999)
        for train_idx, test_idx in splitter.split(X, y, groups):
            assert len(train_idx) > 0


class TestGroupKFold:
    def test_group_kfold_keeps_groups_together(self):
        np.random.seed(42)
        n_groups = 50
        rows_per_group = 10
        data = []
        for group_id in range(n_groups):
            target = np.random.choice([0, 1])
            for _ in range(rows_per_group):
                data.append({
                    "group_id": group_id,
                    "feature1": np.random.randn(),
                    "target": target
                })

        df = pd.DataFrame(data)
        X = df[["feature1"]]
        y = df["target"]
        groups = df["group_id"]

        model = LogisticRegression(max_iter=1000, random_state=42)
        cv = CrossValidator(strategy=CVStrategy.GROUP_KFOLD, n_splits=5)
        result = cv.run(model, X, y, groups=groups)

        assert len(result.cv_scores) == 5


class TestGroupsAsDataFrame:
    @pytest.fixture
    def panel_data(self):
        np.random.seed(42)
        n_entities = 20
        dates = pd.date_range("2023-01-01", periods=12, freq="MS")
        rows = []
        for eid in range(n_entities):
            for d in dates:
                rows.append({
                    "entity_id": eid, "as_of_date": d,
                    "feature1": np.random.randn(),
                    "target": np.random.choice([0, 1], p=[0.3, 0.7]),
                })
        return pd.DataFrame(rows)

    def test_temporal_entity_cv_with_dataframe_groups(self, panel_data):
        X = panel_data[["feature1"]]
        y = panel_data["target"]
        groups_df = panel_data[["entity_id"]]
        model = LogisticRegression(max_iter=1000, random_state=42)
        cv = CrossValidator(strategy=CVStrategy.TEMPORAL_ENTITY, n_splits=5, scoring="roc_auc")
        result = cv.run(model, X, y, groups=groups_df)
        assert len(result.cv_scores) == 5

    def test_group_kfold_with_dataframe_groups(self, panel_data):
        X = panel_data[["feature1"]]
        y = panel_data["target"]
        groups_df = panel_data[["entity_id"]]
        model = LogisticRegression(max_iter=1000, random_state=42)
        cv = CrossValidator(strategy=CVStrategy.GROUP_KFOLD, n_splits=5, scoring="roc_auc")
        result = cv.run(model, X, y, groups=groups_df)
        assert len(result.cv_scores) == 5

    def test_temporal_entity_cv_with_dataframe_temporal(self, panel_data):
        X = panel_data[["feature1"]]
        y = panel_data["target"]
        groups = panel_data["entity_id"]
        temporal_df = panel_data[["as_of_date"]]
        model = LogisticRegression(max_iter=1000, random_state=42)
        cv = CrossValidator(
            strategy=CVStrategy.TEMPORAL_ENTITY, n_splits=5,
            scoring="roc_auc", purge_gap_days=30,
        )
        result = cv.run(model, X, y, groups=groups, temporal_values=temporal_df)
        assert len(result.cv_scores) == 5


class _MockSparkModel:
    """Model with .clone() that mimics SparkClassifierWrapper for test."""

    def __init__(self):
        self.classes_ = np.array([0, 1])
        self._fitted = False

    def clone(self):
        return _MockSparkModel()

    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression

        feature_cols = [c for c in X.columns if not c.startswith("__cv_")]
        self._inner = LogisticRegression(max_iter=200, random_state=42)
        self._inner.fit(X[feature_cols], y)
        self._fitted = True
        return self

    def predict(self, X):
        feature_cols = [c for c in X.columns if not c.startswith("__cv_")]
        return self._inner.predict(X[feature_cols])

    def predict_proba(self, X):
        feature_cols = [c for c in X.columns if not c.startswith("__cv_")]
        return self._inner.predict_proba(X[feature_cols])


class TestHasCvMetadata:
    def test_detects_metadata_columns(self):
        from customer_retention.stages.modeling.cross_validator import _has_cv_metadata

        df = pd.DataFrame({"f1": [1], "__cv_entity__": [0], "__cv_date__": [0]})
        assert _has_cv_metadata(df) is True

    def test_missing_entity_column(self):
        from customer_retention.stages.modeling.cross_validator import _has_cv_metadata

        df = pd.DataFrame({"f1": [1], "__cv_date__": [0]})
        assert _has_cv_metadata(df) is False

    def test_missing_date_column(self):
        from customer_retention.stages.modeling.cross_validator import _has_cv_metadata

        df = pd.DataFrame({"f1": [1], "__cv_entity__": [0]})
        assert _has_cv_metadata(df) is False

    def test_no_metadata(self):
        from customer_retention.stages.modeling.cross_validator import _has_cv_metadata

        df = pd.DataFrame({"f1": [1], "f2": [2]})
        assert _has_cv_metadata(df) is False


class TestDistributedCVWithCloneableModel:
    @pytest.fixture
    def panel_with_metadata(self):
        np.random.seed(42)
        n_entities = 20
        dates = pd.date_range("2023-01-01", periods=12, freq="MS")
        rows = []
        for eid in range(n_entities):
            for d in dates:
                rows.append({
                    "entity_id": eid,
                    "as_of_date": d,
                    "feature1": np.random.randn(),
                    "feature2": np.random.randn(),
                    "target": np.random.choice([0, 1], p=[0.3, 0.7]),
                })
        df = pd.DataFrame(rows)
        X = df[["feature1", "feature2"]].copy()
        X["__cv_entity__"] = df["entity_id"]
        X["__cv_date__"] = df["as_of_date"]
        y = df["target"]
        groups = df["entity_id"]
        temporal = df["as_of_date"]
        return X, y, groups, temporal

    def test_cloneable_model_triggers_distributed_path(self, panel_with_metadata):
        X, y, groups, temporal = panel_with_metadata
        model = _MockSparkModel()
        cv = CrossValidator(
            strategy=CVStrategy.TEMPORAL_ENTITY, n_splits=5, scoring="roc_auc",
        )
        result = cv.run(model, X, y, groups=groups, temporal_values=temporal)
        assert len(result.cv_scores) == 5
        assert all(0 <= s <= 1 for s in result.cv_scores)

    def test_fold_details_have_entity_counts(self, panel_with_metadata):
        X, y, groups, temporal = panel_with_metadata
        model = _MockSparkModel()
        cv = CrossValidator(
            strategy=CVStrategy.TEMPORAL_ENTITY, n_splits=5, scoring="roc_auc",
        )
        result = cv.run(model, X, y, groups=groups, temporal_values=temporal)
        for fold in result.fold_details:
            assert "train_entities" in fold
            assert "test_entities" in fold
            assert fold["train_entities"] > 0
            assert fold["test_entities"] > 0

    def test_metadata_columns_ignored_during_training(self, panel_with_metadata):
        X, y, groups, temporal = panel_with_metadata
        model = _MockSparkModel()
        cv = CrossValidator(
            strategy=CVStrategy.TEMPORAL_ENTITY, n_splits=5, scoring="roc_auc",
        )
        result = cv.run(model, X, y, groups=groups, temporal_values=temporal)
        assert result.cv_mean > 0

    def test_purge_gap_reduces_train_rows(self):
        np.random.seed(42)
        base = pd.Timestamp("2023-01-01")
        rows = [
            {"entity_id": 0, "as_of_date": base, "f1": 1.0, "target": 0},
            {"entity_id": 0, "as_of_date": base + pd.Timedelta(days=30), "f1": 1.1, "target": 0},
            {"entity_id": 0, "as_of_date": base + pd.Timedelta(days=120), "f1": 1.2, "target": 1},
            {"entity_id": 1, "as_of_date": base + pd.Timedelta(days=150), "f1": 2.0, "target": 1},
            {"entity_id": 1, "as_of_date": base + pd.Timedelta(days=180), "f1": 2.1, "target": 0},
        ]
        df = pd.DataFrame(rows)

        from customer_retention.stages.modeling.cross_validator import TemporalEntitySplit

        temporal = df["as_of_date"]
        groups = df["entity_id"]
        splitter_no = TemporalEntitySplit(n_splits=2)
        splitter_purge = TemporalEntitySplit(
            n_splits=2, temporal_values=temporal, purge_gap_days=90,
        )
        sizes_no = [len(tr) for tr, _ in splitter_no.split(np.zeros(len(df)), groups=groups)]
        sizes_purge = [len(tr) for tr, _ in splitter_purge.split(np.zeros(len(df)), groups=groups)]
        assert any(sp < sn for sp, sn in zip(sizes_purge, sizes_no))

    def test_cv_result_fields_complete(self, panel_with_metadata):
        X, y, groups, temporal = panel_with_metadata
        model = _MockSparkModel()
        cv = CrossValidator(
            strategy=CVStrategy.TEMPORAL_ENTITY, n_splits=5, scoring="roc_auc",
        )
        result = cv.run(model, X, y, groups=groups, temporal_values=temporal)
        assert result.scoring == "roc_auc"
        assert isinstance(result.is_stable, bool)
        assert len(result.fold_details) == 5
        assert result.cv_std == pytest.approx(np.std(result.cv_scores), rel=1e-6)

    def test_without_metadata_still_works(self):
        np.random.seed(42)
        n_entities = 20
        dates = pd.date_range("2023-01-01", periods=12, freq="MS")
        rows = []
        for eid in range(n_entities):
            for d in dates:
                rows.append({
                    "entity_id": eid, "as_of_date": d,
                    "f1": np.random.randn(), "target": np.random.choice([0, 1], p=[0.3, 0.7]),
                })
        df = pd.DataFrame(rows)
        X = df[["f1"]]
        y = df["target"]
        groups = df["entity_id"]
        temporal = df["as_of_date"]

        model = _MockSparkModel()
        cv = CrossValidator(
            strategy=CVStrategy.TEMPORAL_ENTITY, n_splits=5, scoring="roc_auc",
        )
        result = cv.run(model, X, y, groups=groups, temporal_values=temporal)
        assert len(result.cv_scores) == 5

    def test_average_precision_scoring(self, panel_with_metadata):
        X, y, groups, temporal = panel_with_metadata
        model = _MockSparkModel()
        cv = CrossValidator(
            strategy=CVStrategy.TEMPORAL_ENTITY, n_splits=5,
            scoring="average_precision",
        )
        result = cv.run(model, X, y, groups=groups, temporal_values=temporal)
        assert result.scoring == "average_precision"
        assert all(0 <= s <= 1 for s in result.cv_scores)


class TestOnFoldCompleteCallback:
    @pytest.fixture
    def panel_data(self):
        np.random.seed(42)
        n_entities = 20
        dates = pd.date_range("2023-01-01", periods=12, freq="MS")
        rows = []
        for eid in range(n_entities):
            for d in dates:
                rows.append({
                    "entity_id": eid,
                    "as_of_date": d,
                    "feature1": np.random.randn(),
                    "feature2": np.random.randn(),
                    "target": np.random.choice([0, 1], p=[0.3, 0.7]),
                })
        df = pd.DataFrame(rows)
        X = df[["feature1", "feature2"]]
        y = df["target"]
        groups = df["entity_id"]
        temporal = df["as_of_date"]
        return X, y, groups, temporal

    def test_callback_called_per_fold(self, panel_data):
        X, y, groups, temporal = panel_data
        model = LogisticRegression(max_iter=1000, random_state=42)
        cv = CrossValidator(
            strategy=CVStrategy.TEMPORAL_ENTITY, n_splits=5, scoring="roc_auc",
        )
        calls = []
        cv.run(
            model, X, y, groups=groups, temporal_values=temporal,
            on_fold_complete=lambda detail, fold_num, total_folds: calls.append(
                (detail, fold_num, total_folds)
            ),
        )
        assert len(calls) == 5
        for detail, fold_num, total_folds in calls:
            assert total_folds == 5
        assert [c[1] for c in calls] == [1, 2, 3, 4, 5]

    def test_callback_receives_elapsed_seconds(self, panel_data):
        X, y, groups, temporal = panel_data
        model = LogisticRegression(max_iter=1000, random_state=42)
        cv = CrossValidator(
            strategy=CVStrategy.TEMPORAL_ENTITY, n_splits=5, scoring="roc_auc",
        )
        details_seen = []
        cv.run(
            model, X, y, groups=groups, temporal_values=temporal,
            on_fold_complete=lambda detail, fold_num, total_folds: details_seen.append(detail),
        )
        for detail in details_seen:
            assert "elapsed_seconds" in detail
            assert isinstance(detail["elapsed_seconds"], float)
            assert detail["elapsed_seconds"] >= 0

    def test_callback_optional(self, panel_data):
        X, y, groups, temporal = panel_data
        model = LogisticRegression(max_iter=1000, random_state=42)
        cv = CrossValidator(
            strategy=CVStrategy.TEMPORAL_ENTITY, n_splits=5, scoring="roc_auc",
        )
        result = cv.run(model, X, y, groups=groups, temporal_values=temporal)
        assert len(result.cv_scores) == 5
        assert all(0 <= s <= 1 for s in result.cv_scores)

    def test_fold_details_have_elapsed_seconds(self, panel_data):
        X, y, groups, temporal = panel_data
        model = LogisticRegression(max_iter=1000, random_state=42)
        cv = CrossValidator(
            strategy=CVStrategy.TEMPORAL_ENTITY, n_splits=5, scoring="roc_auc",
        )
        result = cv.run(model, X, y, groups=groups, temporal_values=temporal)
        for fold in result.fold_details:
            assert "elapsed_seconds" in fold
            assert isinstance(fold["elapsed_seconds"], float)
            assert fold["elapsed_seconds"] >= 0

    def test_callback_with_distributed_model(self):
        np.random.seed(42)
        n_entities = 20
        dates = pd.date_range("2023-01-01", periods=12, freq="MS")
        rows = []
        for eid in range(n_entities):
            for d in dates:
                rows.append({
                    "entity_id": eid,
                    "as_of_date": d,
                    "feature1": np.random.randn(),
                    "feature2": np.random.randn(),
                    "target": np.random.choice([0, 1], p=[0.3, 0.7]),
                })
        df = pd.DataFrame(rows)
        X = df[["feature1", "feature2"]]
        y = df["target"]
        groups = df["entity_id"]
        temporal = df["as_of_date"]

        model = _MockSparkModel()
        cv = CrossValidator(
            strategy=CVStrategy.TEMPORAL_ENTITY, n_splits=5, scoring="roc_auc",
        )
        calls = []
        result = cv.run(
            model, X, y, groups=groups, temporal_values=temporal,
            on_fold_complete=lambda detail, fold_num, total_folds: calls.append(
                (detail, fold_num, total_folds)
            ),
        )
        assert len(calls) == 5
        for detail, fold_num, total_folds in calls:
            assert "elapsed_seconds" in detail
            assert total_folds == 5


class TestDistributedPathAvoidsUnnecessaryCollection:
    """Verify _run_distributed uses collect_for_sklearn (Arrow-optimized, no warning)."""

    @pytest.fixture
    def panel_with_metadata(self):
        np.random.seed(42)
        n_entities = 20
        dates = pd.date_range("2023-01-01", periods=12, freq="MS")
        rows = []
        for eid in range(n_entities):
            for d in dates:
                rows.append({
                    "entity_id": eid, "as_of_date": d,
                    "feature1": np.random.randn(), "feature2": np.random.randn(),
                    "target": np.random.choice([0, 1], p=[0.3, 0.7]),
                })
        df = pd.DataFrame(rows)
        X = df[["feature1", "feature2"]].copy()
        X["__cv_entity__"] = df["entity_id"]
        X["__cv_date__"] = df["as_of_date"]
        y = df["target"]
        groups = df["entity_id"]
        temporal = df["as_of_date"]
        return X, y, groups, temporal

    def test_to_pandas_not_imported_in_distributed_module(self):
        import inspect

        from customer_retention.stages.modeling import cross_validator as cv_mod

        source = inspect.getsource(cv_mod)
        assert "to_pandas" not in source, (
            "cross_validator.py still imports or calls to_pandas; "
            "use collect_for_sklearn for Arrow-optimized collection"
        )

    def test_collect_for_sklearn_produces_same_results(self, panel_with_metadata):
        X, y, groups, temporal = panel_with_metadata
        model = _MockSparkModel()
        cv = CrossValidator(strategy=CVStrategy.TEMPORAL_ENTITY, n_splits=5, scoring="roc_auc")
        result = cv.run(model, X, y, groups=groups, temporal_values=temporal)
        assert all(0 <= s <= 1 for s in result.cv_scores)
        assert result.cv_mean == pytest.approx(np.mean(result.cv_scores), rel=1e-6)


class TestFoldClassRatioAccuracy:
    @pytest.fixture
    def panel_data_with_known_ratio(self):
        np.random.seed(42)
        n_entities = 20
        dates = pd.date_range("2023-01-01", periods=12, freq="MS")
        rows = []
        for eid in range(n_entities):
            for d in dates:
                rows.append({
                    "entity_id": eid,
                    "as_of_date": d,
                    "feature1": np.random.randn(),
                    "feature2": np.random.randn(),
                    "target": np.random.choice([0, 1], p=[0.3, 0.7]),
                })
        df = pd.DataFrame(rows)
        X = df[["feature1", "feature2"]]
        y = df["target"]
        groups = df["entity_id"]
        temporal = df["as_of_date"]
        return X, y, groups, temporal

    def test_train_class_ratio_matches_actual_fold_data(self, panel_data_with_known_ratio):
        X, y, groups, temporal = panel_data_with_known_ratio
        model = LogisticRegression(max_iter=1000, random_state=42)
        cv = CrossValidator(
            strategy=CVStrategy.TEMPORAL_ENTITY, n_splits=5, scoring="roc_auc",
        )
        result = cv.run(model, X, y, groups=groups, temporal_values=temporal)
        overall_ratio = float(y.mean())
        for fold in result.fold_details:
            assert 0.0 <= fold["train_class_ratio"] <= 1.0
            assert abs(fold["train_class_ratio"] - overall_ratio) < 0.3

    def test_distributed_model_class_ratio_matches_actual(self, panel_data_with_known_ratio):
        X, y, groups, temporal = panel_data_with_known_ratio
        model = _MockSparkModel()
        cv = CrossValidator(
            strategy=CVStrategy.TEMPORAL_ENTITY, n_splits=5, scoring="roc_auc",
        )
        result = cv.run(model, X, y, groups=groups, temporal_values=temporal)
        overall_ratio = float(y.mean())
        for fold in result.fold_details:
            assert 0.0 <= fold["train_class_ratio"] <= 1.0
            assert abs(fold["train_class_ratio"] - overall_ratio) < 0.3

    def test_f1_weighted_scoring(self):
        np.random.seed(42)
        n_entities = 20
        dates = pd.date_range("2023-01-01", periods=12, freq="MS")
        rows = []
        for eid in range(n_entities):
            for d in dates:
                rows.append({
                    "entity_id": eid, "as_of_date": d,
                    "feature1": np.random.randn(),
                    "target": np.random.choice([0, 1, 2], p=[0.3, 0.4, 0.3]),
                })
        df = pd.DataFrame(rows)
        model = LogisticRegression(max_iter=1000, random_state=42)
        cv = CrossValidator(
            strategy=CVStrategy.TEMPORAL_ENTITY, n_splits=5, scoring="f1_weighted",
        )
        result = cv.run(
            model, df[["feature1"]], df["target"],
            groups=df["entity_id"], temporal_values=df["as_of_date"],
        )
        assert result.scoring == "f1_weighted"
        assert all(0 <= s <= 1 for s in result.cv_scores)


class TestOnFoldCompleteForAllStrategies:
    """on_fold_complete callback should fire for any strategy, not just TEMPORAL_ENTITY."""

    def test_stratified_kfold_callback_called_per_fold(self, sample_data, sample_model):
        X, y = sample_data
        cv = CrossValidator(strategy=CVStrategy.STRATIFIED_KFOLD, n_splits=5, scoring="roc_auc")
        calls = []
        result = cv.run(
            sample_model, X, y,
            on_fold_complete=lambda detail, fold_num, total_folds: calls.append(
                (detail, fold_num, total_folds)
            ),
        )
        assert len(calls) == 5
        assert [c[1] for c in calls] == [1, 2, 3, 4, 5]
        assert all(c[2] == 5 for c in calls)

    def test_stratified_kfold_callback_has_elapsed_seconds(self, sample_data, sample_model):
        X, y = sample_data
        cv = CrossValidator(strategy=CVStrategy.STRATIFIED_KFOLD, n_splits=3, scoring="roc_auc")
        details = []
        cv.run(
            sample_model, X, y,
            on_fold_complete=lambda detail, fold_num, total_folds: details.append(detail),
        )
        for detail in details:
            assert "elapsed_seconds" in detail
            assert isinstance(detail["elapsed_seconds"], float)
            assert detail["elapsed_seconds"] >= 0
            assert "score" in detail
            assert 0 <= detail["score"] <= 1

    def test_stratified_kfold_callback_result_matches_no_callback(self, sample_data, sample_model):
        X, y = sample_data
        cv = CrossValidator(strategy=CVStrategy.STRATIFIED_KFOLD, n_splits=5, scoring="roc_auc")
        result_no_cb = cv.run(sample_model, X, y)
        result_with_cb = cv.run(sample_model, X, y, on_fold_complete=lambda d, f, t: None)
        assert len(result_no_cb.cv_scores) == len(result_with_cb.cv_scores)
        assert result_no_cb.scoring == result_with_cb.scoring

    def test_no_callback_uses_original_path(self, sample_data, sample_model):
        X, y = sample_data
        cv = CrossValidator(strategy=CVStrategy.STRATIFIED_KFOLD, n_splits=5, scoring="roc_auc")
        result = cv.run(sample_model, X, y)
        assert len(result.cv_scores) == 5
        assert all(0 <= s <= 1 for s in result.cv_scores)

    def test_callback_fold_details_have_train_test_sizes(self, sample_data, sample_model):
        X, y = sample_data
        cv = CrossValidator(strategy=CVStrategy.STRATIFIED_KFOLD, n_splits=3, scoring="roc_auc")
        details = []
        cv.run(
            sample_model, X, y,
            on_fold_complete=lambda detail, fold_num, total_folds: details.append(detail),
        )
        for detail in details:
            assert "train_size" in detail
            assert "test_size" in detail
            assert detail["train_size"] > 0
            assert detail["test_size"] > 0
