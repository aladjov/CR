import numpy as np
import pandas as pd
import pytest

from customer_retention.core.compat import to_pandas
from customer_retention.stages.modeling import DataSplitter, SplitConfig, SplitStrategy


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 1000
    return pd.DataFrame({
        "feature1": np.random.randn(n),
        "feature2": np.random.randn(n),
        "feature3": np.random.randn(n),
        "target": np.random.choice([0, 1], n, p=[0.25, 0.75]),
        "custid": range(n),
        "date": pd.date_range("2024-01-01", periods=n, freq="h")
    })


class TestSplitStrategy:
    def test_strategy_enum_has_required_values(self):
        assert hasattr(SplitStrategy, "RANDOM_STRATIFIED")
        assert hasattr(SplitStrategy, "TEMPORAL")
        assert hasattr(SplitStrategy, "GROUP")
        assert hasattr(SplitStrategy, "CUSTOM")


class TestSplitConfig:
    def test_default_config_values(self):
        config = SplitConfig()
        assert config.test_size == 0.11  # ~11% test gives ~80% final training with 90/10 cutoff
        assert config.validation_size == 0.10
        assert config.stratify is True
        assert config.random_state == 42

    def test_custom_config_values(self):
        config = SplitConfig(test_size=0.30, validation_size=0.15, random_state=123)
        assert config.test_size == 0.30
        assert config.validation_size == 0.15
        assert config.random_state == 123


class TestRandomStratifiedSplit:
    def test_split_produces_correct_sizes(self, sample_df):
        splitter = DataSplitter(
            target_column="target",
            strategy=SplitStrategy.RANDOM_STRATIFIED,
            test_size=0.20
        )
        result = splitter.split(sample_df)

        expected_test = int(len(sample_df) * 0.20)
        assert abs(len(result.X_test) - expected_test) <= 1
        assert len(result.X_train) == len(sample_df) - len(result.X_test)

    def test_split_preserves_class_proportions_sp001(self, sample_df):
        splitter = DataSplitter(target_column="target", strategy=SplitStrategy.RANDOM_STRATIFIED)
        result = splitter.split(sample_df)

        train_prop = result.y_train.mean()
        test_prop = result.y_test.mean()
        original_prop = sample_df["target"].mean()

        assert abs(train_prop - original_prop) < 0.05
        assert abs(test_prop - original_prop) < 0.05

    def test_split_no_overlap_sp003(self, sample_df):
        splitter = DataSplitter(target_column="target", strategy=SplitStrategy.RANDOM_STRATIFIED)
        result = splitter.split(sample_df)

        train_indices = set(result.X_train.index)
        test_indices = set(result.X_test.index)

        assert len(train_indices & test_indices) == 0

    def test_split_reproducible_with_seed(self, sample_df):
        splitter1 = DataSplitter(target_column="target", random_state=42)
        splitter2 = DataSplitter(target_column="target", random_state=42)

        result1 = splitter1.split(sample_df)
        result2 = splitter2.split(sample_df)

        pd.testing.assert_frame_equal(result1.X_train, result2.X_train)
        pd.testing.assert_frame_equal(result1.X_test, result2.X_test)

    def test_split_different_with_different_seed(self, sample_df):
        splitter1 = DataSplitter(target_column="target", random_state=42)
        splitter2 = DataSplitter(target_column="target", random_state=123)

        result1 = splitter1.split(sample_df)
        result2 = splitter2.split(sample_df)

        assert not result1.X_train.equals(result2.X_train)


class TestStratifiedSplitRareClassFallback:
    def test_rare_class_falls_back_to_non_stratified(self):
        n = 100
        df = pd.DataFrame({
            "feature1": np.random.randn(n),
            "target": [0] * 50 + [1] * 49 + [11],
        })
        splitter = DataSplitter(
            target_column="target",
            strategy=SplitStrategy.RANDOM_STRATIFIED,
            test_size=0.2, stratify=True,
        )
        result = splitter.split(df)
        assert len(result.X_train) + len(result.X_test) == n

    def test_rare_class_with_validation_split(self):
        n = 100
        df = pd.DataFrame({
            "feature1": np.random.randn(n),
            "target": [0] * 50 + [1] * 49 + [11],
        })
        splitter = DataSplitter(
            target_column="target",
            strategy=SplitStrategy.RANDOM_STRATIFIED,
            test_size=0.2, stratify=True, include_validation=True,
        )
        result = splitter.split(df)
        total = len(result.X_train) + len(result.X_test)
        if result.X_val is not None:
            total += len(result.X_val)
        assert total == n


class TestTemporalSplit:
    def test_temporal_split_respects_time_order_sp004(self, sample_df):
        splitter = DataSplitter(
            target_column="target",
            strategy=SplitStrategy.TEMPORAL,
            temporal_column="date",
            test_size=0.20
        )
        result = splitter.split(sample_df)

        train_max_date = sample_df.loc[result.X_train.index, "date"].max()
        test_min_date = sample_df.loc[result.X_test.index, "date"].min()

        assert train_max_date < test_min_date

    def test_temporal_split_produces_correct_sizes(self, sample_df):
        splitter = DataSplitter(
            target_column="target",
            strategy=SplitStrategy.TEMPORAL,
            temporal_column="date",
            test_size=0.20
        )
        result = splitter.split(sample_df)

        expected_test = int(len(sample_df) * 0.20)
        assert abs(len(result.X_test) - expected_test) <= 1


class TestGroupSplit:
    @pytest.fixture
    def grouped_df(self):
        np.random.seed(42)
        n_groups = 100
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
        return pd.DataFrame(data)

    def test_group_split_keeps_groups_together(self, grouped_df):
        splitter = DataSplitter(
            target_column="target",
            strategy=SplitStrategy.GROUP,
            group_column="group_id",
            test_size=0.20
        )
        result = splitter.split(grouped_df)

        train_groups = set(grouped_df.loc[result.X_train.index, "group_id"])
        test_groups = set(grouped_df.loc[result.X_test.index, "group_id"])

        assert len(train_groups & test_groups) == 0


class TestValidationSplit:
    def test_split_with_validation_set(self, sample_df):
        splitter = DataSplitter(
            target_column="target",
            test_size=0.20,
            validation_size=0.10,
            include_validation=True
        )
        result = splitter.split(sample_df)

        assert result.X_val is not None
        assert result.y_val is not None
        assert len(result.X_val) > 0

    def test_validation_no_overlap_with_train_or_test(self, sample_df):
        splitter = DataSplitter(
            target_column="target",
            test_size=0.20,
            validation_size=0.10,
            include_validation=True
        )
        result = splitter.split(sample_df)

        train_idx = set(result.X_train.index)
        val_idx = set(result.X_val.index)
        test_idx = set(result.X_test.index)

        assert len(train_idx & val_idx) == 0
        assert len(train_idx & test_idx) == 0
        assert len(val_idx & test_idx) == 0


class TestMinoritySampleValidation:
    def test_warns_on_insufficient_minority_sp002(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "feature1": np.random.randn(100),
            "target": [0] * 5 + [1] * 95
        })

        splitter = DataSplitter(target_column="target", test_size=0.20)

        with pytest.warns(UserWarning, match="minority"):
            splitter.split(df)


class TestSplitResult:
    def test_result_contains_all_fields(self, sample_df):
        splitter = DataSplitter(target_column="target")
        result = splitter.split(sample_df)

        assert hasattr(result, "X_train")
        assert hasattr(result, "X_test")
        assert hasattr(result, "y_train")
        assert hasattr(result, "y_test")
        assert hasattr(result, "split_info")

    def test_split_info_contains_metadata(self, sample_df):
        splitter = DataSplitter(target_column="target")
        result = splitter.split(sample_df)

        assert "train_size" in result.split_info
        assert "test_size" in result.split_info
        assert "strategy" in result.split_info
        assert "random_state" in result.split_info


class TestFeatureExclusion:
    def test_excludes_target_from_features(self, sample_df):
        splitter = DataSplitter(target_column="target")
        result = splitter.split(sample_df)

        assert "target" not in result.X_train.columns
        assert "target" not in result.X_test.columns

    def test_excludes_specified_columns(self, sample_df):
        splitter = DataSplitter(
            target_column="target",
            exclude_columns=["custid", "date"]
        )
        result = splitter.split(sample_df)

        assert "custid" not in result.X_train.columns
        assert "date" not in result.X_train.columns

    def test_auto_excludes_datetime_columns(self, sample_df):
        splitter = DataSplitter(target_column="target")
        result = splitter.split(sample_df)

        assert "date" not in result.X_train.columns
        assert "date" not in result.X_test.columns

    def test_auto_excludes_datetime_preserves_numeric(self):
        df = pd.DataFrame({
            "num_feat": np.random.randn(100),
            "int_feat": np.random.randint(0, 10, 100),
            "dt_col": pd.date_range("2024-01-01", periods=100, freq="h"),
            "target": np.random.choice([0, 1], 100),
        })
        splitter = DataSplitter(target_column="target", test_size=0.2)
        result = splitter.split(df)

        assert "dt_col" not in result.X_train.columns
        assert set(result.X_train.columns) == {"num_feat", "int_feat"}

    def test_auto_excludes_datetime_with_tz(self):
        dates = pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC")
        df = pd.DataFrame({
            "feature": np.random.randn(100),
            "tz_date": dates,
            "target": np.random.choice([0, 1], 100),
        })
        splitter = DataSplitter(target_column="target", test_size=0.2)
        result = splitter.split(df)

        assert "tz_date" not in result.X_train.columns
        assert "feature" in result.X_train.columns

    def test_auto_excludes_timedelta_columns(self):
        df = pd.DataFrame({
            "feature": np.random.randn(100),
            "duration": pd.to_timedelta(np.random.randint(1, 100, 100), unit="D"),
            "target": np.random.choice([0, 1], 100),
        })
        splitter = DataSplitter(target_column="target", test_size=0.2)
        result = splitter.split(df)

        assert "duration" not in result.X_train.columns
        assert "feature" in result.X_train.columns


class TestTemporalSplitDistributed:
    def test_temporal_split_skips_to_pandas(self):
        from unittest.mock import patch

        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            "feature1": np.random.randn(n),
            "target": np.random.choice([0, 1], n),
            "as_of_date": pd.date_range("2023-01-01", periods=n, freq="D"),
        })
        splitter = DataSplitter(
            target_column="target", strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date", test_size=0.2,
        )
        with patch("customer_retention.stages.modeling.data_splitter.to_pandas") as mock_tp:
            result = splitter.split(df)
            mock_tp.assert_not_called()
        assert len(result.X_train) > 0
        assert len(result.X_test) > 0

    def test_temporal_split_with_purge_skips_to_pandas(self):
        from unittest.mock import patch

        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            "feature1": np.random.randn(n),
            "target": np.random.choice([0, 1], n),
            "as_of_date": pd.date_range("2023-01-01", periods=n, freq="D"),
        })
        splitter = DataSplitter(
            target_column="target", strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date", test_size=0.2, purge_gap_days=30,
        )
        with patch("customer_retention.stages.modeling.data_splitter.to_pandas") as mock_tp:
            result = splitter.split(df)
            mock_tp.assert_not_called()
        assert len(result.X_train) + len(result.X_test) < n

    def test_temporal_split_train_metadata_alignment(self):
        """Train entities/dates reconstructed via cutoff must align with X_train indices."""
        from datetime import timedelta

        np.random.seed(42)
        n = 500
        purge_gap_days = 30
        df = pd.DataFrame({
            "feature1": np.random.randn(n),
            "target": np.random.choice([0, 1], n),
            "as_of_date": pd.date_range("2023-01-01", periods=n, freq="D"),
            "entity_id": np.random.choice(["a", "b", "c", "d"], n),
        })
        df = df.sort_values("as_of_date").reset_index(drop=True)
        splitter = DataSplitter(
            target_column="target", strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date", test_size=0.2,
            purge_gap_days=purge_gap_days, exclude_columns=["as_of_date", "entity_id"],
        )
        result = splitter.split(df)

        cutoff = pd.Timestamp(result.split_info["cutoff_date"])
        purge_start = cutoff - timedelta(days=purge_gap_days)
        train_mask = df["as_of_date"] < purge_start
        reconstructed_entities = df.loc[train_mask, "entity_id"]
        reconstructed_dates = df.loc[train_mask, "as_of_date"]

        assert list(result.X_train.index) == list(reconstructed_entities.index)
        assert list(result.X_train.index) == list(reconstructed_dates.index)
        assert len(reconstructed_entities) == len(result.X_train)

    def test_stratified_split_still_calls_to_pandas(self):
        from unittest.mock import patch

        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            "feature1": np.random.randn(n),
            "target": np.random.choice([0, 1], n),
        })
        splitter = DataSplitter(
            target_column="target", strategy=SplitStrategy.RANDOM_STRATIFIED,
            test_size=0.2,
        )
        with patch("customer_retention.stages.modeling.data_splitter.to_pandas", wraps=to_pandas) as mock_tp:
            splitter.split(df)
            mock_tp.assert_called_once()


class TestTemporalPurgeGap:
    @pytest.fixture
    def temporal_df(self):
        np.random.seed(42)
        n = 1000
        return pd.DataFrame({
            "feature1": np.random.randn(n),
            "feature2": np.random.randn(n),
            "target": np.random.choice([0, 1], n, p=[0.3, 0.7]),
            "as_of_date": pd.date_range("2023-01-01", periods=n, freq="D"),
        })

    def test_purge_gap_removes_rows_between_train_and_test(self, temporal_df):
        splitter = DataSplitter(
            target_column="target",
            strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date",
            test_size=0.2,
            purge_gap_days=30,
        )
        result = splitter.split(temporal_df)

        total_kept = len(result.X_train) + len(result.X_test)
        assert total_kept < len(temporal_df)

    def test_purge_gap_zero_behaves_like_basic_temporal(self, temporal_df):
        splitter_gap = DataSplitter(
            target_column="target",
            strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date",
            test_size=0.2,
            purge_gap_days=0,
        )
        splitter_basic = DataSplitter(
            target_column="target",
            strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date",
            test_size=0.2,
        )
        result_gap = splitter_gap.split(temporal_df)
        result_basic = splitter_basic.split(temporal_df)

        assert len(result_gap.X_train) == len(result_basic.X_train)
        assert len(result_gap.X_test) == len(result_basic.X_test)

    def test_purge_gap_none_behaves_like_basic_temporal(self, temporal_df):
        splitter_gap = DataSplitter(
            target_column="target",
            strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date",
            test_size=0.2,
            purge_gap_days=None,
        )
        splitter_basic = DataSplitter(
            target_column="target",
            strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date",
            test_size=0.2,
        )
        result_gap = splitter_gap.split(temporal_df)
        result_basic = splitter_basic.split(temporal_df)

        assert len(result_gap.X_train) == len(result_basic.X_train)
        assert len(result_gap.X_test) == len(result_basic.X_test)

    def test_purge_gap_preserves_time_ordering(self, temporal_df):
        splitter = DataSplitter(
            target_column="target",
            strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date",
            test_size=0.2,
            purge_gap_days=30,
            exclude_columns=["as_of_date"],
        )
        result = splitter.split(temporal_df)

        train_max = temporal_df.loc[result.X_train.index, "as_of_date"].max()
        test_min = temporal_df.loc[result.X_test.index, "as_of_date"].min()
        gap = (test_min - train_max).days

        assert gap >= 30

    def test_purge_gap_train_plus_test_less_than_total(self, temporal_df):
        splitter = DataSplitter(
            target_column="target",
            strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date",
            test_size=0.2,
            purge_gap_days=50,
        )
        result = splitter.split(temporal_df)

        total_kept = len(result.X_train) + len(result.X_test)
        assert total_kept < len(temporal_df)
        purge_rows = result.split_info.get("purge_gap_rows", 0)
        assert purge_rows > 0
        assert total_kept + purge_rows == len(temporal_df)

    def test_purge_gap_split_info_reports_gap_metadata(self, temporal_df):
        splitter = DataSplitter(
            target_column="target",
            strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date",
            test_size=0.2,
            purge_gap_days=30,
        )
        result = splitter.split(temporal_df)

        assert "purge_gap_days" in result.split_info
        assert result.split_info["purge_gap_days"] == 30
        assert "purge_gap_rows" in result.split_info
        assert result.split_info["purge_gap_rows"] > 0
        assert "cutoff_date" in result.split_info

    def test_purge_gap_with_validation_split(self, temporal_df):
        splitter = DataSplitter(
            target_column="target",
            strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date",
            test_size=0.2,
            purge_gap_days=30,
            include_validation=True,
            validation_size=0.10,
        )
        result = splitter.split(temporal_df)

        assert result.X_val is not None
        assert len(result.X_val) > 0
        assert len(result.X_train) + len(result.X_val) + len(result.X_test) < len(temporal_df)

    def test_purge_gap_large_gap_still_has_train_and_test(self, temporal_df):
        splitter = DataSplitter(
            target_column="target",
            strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date",
            test_size=0.2,
            purge_gap_days=500,
        )
        result = splitter.split(temporal_df)

        assert len(result.X_train) > 0
        assert len(result.X_test) > 0
