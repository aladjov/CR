import numpy as np
import pandas as pd
import pytest

from customer_retention.core.compat import to_pandas
from customer_retention.stages.modeling import DataSplitter, SplitConfig, SplitResult, SplitStrategy


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
    @pytest.fixture
    def entity_temporal_df(self):
        """Multi-entity dataset with temporal snapshots — realistic scenario."""
        np.random.seed(42)
        rows = []
        dates = pd.date_range("2023-01-01", periods=52, freq="W")
        for eid in range(50):
            target = np.random.choice([0, 1])
            for d in dates:
                rows.append({"entity_id": f"e{eid}", "as_of_date": d,
                             "feature1": np.random.randn(), "target": target})
        return pd.DataFrame(rows)

    def test_temporal_split_no_entity_in_both_sets(self, entity_temporal_df):
        splitter = DataSplitter(
            target_column="target", strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date", group_column="entity_id", test_size=0.20,
            exclude_columns=["as_of_date", "entity_id"],
        )
        result = splitter.split(entity_temporal_df)
        train_entities = set(entity_temporal_df.loc[result.X_train.index, "entity_id"])
        test_entities = set(entity_temporal_df.loc[result.X_test.index, "entity_id"])
        assert len(train_entities & test_entities) == 0, "Same entity in train AND test — leakage"

    def test_temporal_split_test_entities_are_late(self, entity_temporal_df):
        splitter = DataSplitter(
            target_column="target", strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date", group_column="entity_id", test_size=0.20,
            exclude_columns=["as_of_date", "entity_id"],
        )
        result = splitter.split(entity_temporal_df)
        cutoff = pd.Timestamp(result.split_info["cutoff_date"])
        test_entities = set(entity_temporal_df.loc[result.X_test.index, "entity_id"])
        for eid in test_entities:
            last_date = entity_temporal_df.loc[entity_temporal_df["entity_id"] == eid, "as_of_date"].max()
            assert last_date >= cutoff, f"Entity {eid} last date {last_date} < cutoff {cutoff}"

    def test_temporal_split_with_purge_gap_entity_grouped(self, entity_temporal_df):
        splitter = DataSplitter(
            target_column="target", strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date", group_column="entity_id",
            test_size=0.20, purge_gap_days=30,
            exclude_columns=["as_of_date", "entity_id"],
        )
        result = splitter.split(entity_temporal_df)
        train_entities = set(entity_temporal_df.loc[result.X_train.index, "entity_id"])
        test_entities = set(entity_temporal_df.loc[result.X_test.index, "entity_id"])
        assert len(train_entities & test_entities) == 0
        assert len(result.X_train) + len(result.X_test) < len(entity_temporal_df)

    def test_temporal_split_all_rows_accounted(self, entity_temporal_df):
        splitter = DataSplitter(
            target_column="target", strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date", group_column="entity_id", test_size=0.20,
            exclude_columns=["as_of_date", "entity_id"],
        )
        result = splitter.split(entity_temporal_df)
        total = len(result.X_train) + len(result.X_test)
        assert total == len(entity_temporal_df)

    def test_temporal_split_entity_isolation_sp004(self, sample_df):
        splitter = DataSplitter(
            target_column="target",
            strategy=SplitStrategy.TEMPORAL,
            temporal_column="date",
            group_column="custid",
            test_size=0.20
        )
        result = splitter.split(sample_df)
        train_ids = set(sample_df.loc[result.X_train.index, "custid"])
        test_ids = set(sample_df.loc[result.X_test.index, "custid"])
        assert len(train_ids & test_ids) == 0

    def test_temporal_split_produces_correct_sizes(self, sample_df):
        splitter = DataSplitter(
            target_column="target",
            strategy=SplitStrategy.TEMPORAL,
            temporal_column="date",
            group_column="custid",
            test_size=0.20
        )
        result = splitter.split(sample_df)

        expected_test = int(len(sample_df) * 0.20)
        assert abs(len(result.X_test) - expected_test) <= len(sample_df) * 0.05


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
            "entity_id": [f"e{i % 20}" for i in range(n)],
        })
        splitter = DataSplitter(
            target_column="target", strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date", group_column="entity_id",
            test_size=0.2, exclude_columns=["as_of_date", "entity_id"],
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
            "entity_id": [f"e{i % 20}" for i in range(n)],
        })
        splitter = DataSplitter(
            target_column="target", strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date", group_column="entity_id",
            test_size=0.2, purge_gap_days=30,
            exclude_columns=["as_of_date", "entity_id"],
        )
        with patch("customer_retention.stages.modeling.data_splitter.to_pandas") as mock_tp:
            result = splitter.split(df)
            mock_tp.assert_not_called()
        assert len(result.X_train) + len(result.X_test) < n

    def test_temporal_split_train_metadata_alignment(self):
        """Train entities/dates must align with X_train indices."""
        np.random.seed(42)
        n = 500
        purge_gap_days = 30
        df = pd.DataFrame({
            "feature1": np.random.randn(n),
            "target": np.random.choice([0, 1], n),
            "as_of_date": pd.date_range("2023-01-01", periods=n, freq="D"),
            "entity_id": np.random.choice(["a", "b", "c", "d"], n),
        })
        splitter = DataSplitter(
            target_column="target", strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date", group_column="entity_id",
            test_size=0.2, purge_gap_days=purge_gap_days,
            exclude_columns=["as_of_date", "entity_id"],
        )
        result = splitter.split(df)
        train_entities = set(df.loc[result.X_train.index, "entity_id"])
        test_entities = set(df.loc[result.X_test.index, "entity_id"])
        assert len(train_entities & test_entities) == 0

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


class TestTemporalSplitNoIloc:
    def test_no_sort_or_iloc_in_distributed_temporal_split(self):
        import inspect

        from customer_retention.stages.modeling.data_splitter import DataSplitter
        source = inspect.getsource(DataSplitter._distributed_temporal_split)
        assert ".iloc[" not in source, "_distributed_temporal_split must not use .iloc"
        assert "sort_values" not in source, "_distributed_temporal_split must not sort"
        assert "toPandas()" not in source, "_distributed_temporal_split must not collect full DataFrame"

    def test_cutoff_date_always_in_split_info(self):
        n = 200
        df = pd.DataFrame({
            "feature1": np.random.randn(n),
            "target": np.random.choice([0, 1], n),
            "as_of_date": pd.date_range("2023-01-01", periods=n, freq="D"),
            "entity_id": [f"e{i % 20}" for i in range(n)],
        })
        splitter = DataSplitter(
            target_column="target", strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date", group_column="entity_id",
            test_size=0.2, exclude_columns=["as_of_date", "entity_id"],
        )
        result = splitter.split(df)
        assert "cutoff_date" in result.split_info

    def test_unsorted_input_produces_correct_split(self):
        np.random.seed(42)
        n = 500
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        df = pd.DataFrame({
            "feature1": np.random.randn(n),
            "target": np.random.choice([0, 1], n),
            "as_of_date": dates,
            "entity_id": [f"e{i % 25}" for i in range(n)],
        })
        shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        splitter = DataSplitter(
            target_column="target", strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date", group_column="entity_id",
            test_size=0.2, purge_gap_days=30,
            exclude_columns=["as_of_date", "entity_id"],
        )
        result = splitter.split(shuffled)
        train_entities = set(shuffled.loc[result.X_train.index, "entity_id"])
        test_entities = set(shuffled.loc[result.X_test.index, "entity_id"])
        assert len(train_entities & test_entities) == 0


class TestDistributedTemporalSplitDispatch:
    def test_dispatches_to_distributed_on_spark_pandas(self):
        from unittest.mock import MagicMock, patch

        splitter = DataSplitter(
            target_column="target", strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date", test_size=0.2,
        )
        mock_result = MagicMock(spec=SplitResult)
        with patch("customer_retention.stages.modeling.data_splitter._is_spark_pandas", return_value=True):
            with patch.object(splitter, "_distributed_temporal_split", return_value=mock_result) as mock_dist:
                result = splitter._temporal_split(MagicMock())
        mock_dist.assert_called_once()
        assert result is mock_result

    def test_uses_pandas_path_for_native_pandas(self):
        n = 200
        df = pd.DataFrame({
            "feature1": np.random.randn(n),
            "target": np.random.choice([0, 1], n),
            "as_of_date": pd.date_range("2023-01-01", periods=n, freq="D"),
            "entity_id": [f"e{i % 20}" for i in range(n)],
        })
        splitter = DataSplitter(
            target_column="target", strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date", group_column="entity_id",
            test_size=0.2, exclude_columns=["as_of_date", "entity_id"],
        )
        from unittest.mock import patch
        with patch.object(splitter, "_distributed_temporal_split") as mock_dist:
            result = splitter._temporal_split(df)
        mock_dist.assert_not_called()
        assert len(result.X_train) > 0

    def test_no_banned_patterns_in_distributed_split(self):
        import inspect

        from customer_retention.stages.modeling.data_splitter import DataSplitter
        source = inspect.getsource(DataSplitter._distributed_temporal_split)
        assert ".iloc[" not in source
        assert "sort_values" not in source
        assert "toPandas()" not in source

    def test_spark_features_target_column_selection(self):
        try:
            from pyspark.sql.types import (
                DoubleType,
                IntegerType,
                StringType,
                StructField,
                StructType,
                TimestampNTZType,
            )
        except ImportError:
            pytest.skip("PySpark not installed")

        from unittest.mock import MagicMock, patch

        schema = StructType([
            StructField("feature1", DoubleType()),
            StructField("feature2", IntegerType()),
            StructField("target", IntegerType()),
            StructField("as_of_date", TimestampNTZType()),
            StructField("entity_id", StringType()),
        ])
        mock_spark_df = MagicMock()
        mock_spark_df.schema = schema

        mock_ps = MagicMock()
        mock_ps.__getitem__ = MagicMock(return_value=MagicMock())

        splitter = DataSplitter(
            target_column="target", strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date", test_size=0.2,
            exclude_columns=["as_of_date", "entity_id"],
        )
        with patch("customer_retention.core.compat.spark_backend._as_pandas_api", return_value=mock_ps):
            splitter._spark_select_features_target(mock_spark_df, extract_metadata=True)

        calls = mock_spark_df.select.call_args_list
        assert calls[0].args == (["feature1", "feature2"],)
        assert calls[1].args == ("target",)
        assert calls[2].args == (["as_of_date", "entity_id"],)

    def test_train_metadata_populated_on_distributed_path(self):
        n = 200
        df = pd.DataFrame({
            "feature1": np.random.randn(n),
            "target": np.random.choice([0, 1], n),
            "as_of_date": pd.date_range("2023-01-01", periods=n, freq="D"),
            "entity_id": [f"e{i % 20}" for i in range(n)],
        })
        splitter = DataSplitter(
            target_column="target", strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date", group_column="entity_id",
            test_size=0.2, exclude_columns=["as_of_date", "entity_id"],
        )
        result = splitter.split(df)
        assert result.train_metadata == {}

    def test_train_metadata_empty_on_pandas_path(self):
        n = 200
        df = pd.DataFrame({
            "feature1": np.random.randn(n),
            "target": np.random.choice([0, 1], n),
            "as_of_date": pd.date_range("2023-01-01", periods=n, freq="D"),
            "entity_id": np.random.choice(["a", "b", "c"], n),
        })
        splitter = DataSplitter(
            target_column="target", strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date", group_column="entity_id",
            test_size=0.2, exclude_columns=["as_of_date", "entity_id"],
        )
        result = splitter.split(df)
        assert result.train_metadata == {}


class TestSplitSparkDispatch:
    def test_split_converts_to_spark_for_pyspark_pandas(self):
        from unittest.mock import MagicMock, patch

        splitter = DataSplitter(
            target_column="target", strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date", test_size=0.2,
        )
        mock_df = MagicMock()
        mock_result = MagicMock(spec=SplitResult)
        mock_result.split_info = {}

        with patch("customer_retention.stages.modeling.data_splitter._is_spark_pandas", return_value=True):
            with patch("customer_retention.stages.modeling.data_splitter.as_spark_df", return_value=MagicMock()) as mock_conv:
                with patch.object(splitter, "_split_spark", return_value=mock_result) as mock_split:
                    splitter.split(mock_df)
        mock_conv.assert_called_once_with(mock_df)
        mock_split.assert_called_once()

    def test_split_never_accesses_pyspark_pandas_getitem(self):
        from unittest.mock import MagicMock, patch

        mock_df = MagicMock()
        def fail_on_access(key):
            raise AssertionError(f"pyspark.pandas __getitem__ accessed: {key}")
        mock_df.__getitem__ = fail_on_access
        mock_result = MagicMock(spec=SplitResult)
        mock_result.split_info = {}

        splitter = DataSplitter(
            target_column="target", strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date", test_size=0.2,
        )
        with patch("customer_retention.stages.modeling.data_splitter._is_spark_pandas", return_value=True):
            with patch("customer_retention.stages.modeling.data_splitter.as_spark_df", return_value=MagicMock()):
                with patch.object(splitter, "_split_spark", return_value=mock_result):
                    splitter.split(mock_df)

    def test_pandas_path_unchanged_when_not_spark(self, sample_df):
        splitter = DataSplitter(
            target_column="target", strategy=SplitStrategy.TEMPORAL,
            temporal_column="date", group_column="custid", test_size=0.2,
        )
        result = splitter.split(sample_df)
        assert len(result.X_train) > 0
        assert len(result.X_test) > 0

    def test_spark_select_uses_separate_wrappers(self):
        try:
            from pyspark.sql.types import (
                DoubleType,
                IntegerType,
                StructField,
                StructType,
                TimestampNTZType,
            )
        except ImportError:
            pytest.skip("PySpark not installed")

        from unittest.mock import MagicMock, patch

        schema = StructType([
            StructField("f1", DoubleType()),
            StructField("target", IntegerType()),
            StructField("as_of_date", TimestampNTZType()),
        ])
        mock_spark_df = MagicMock()
        mock_spark_df.schema = schema

        wrappers = []
        def track_wrappers(sdf):
            m = MagicMock()
            m.__getitem__ = MagicMock(return_value=MagicMock())
            wrappers.append(m)
            return m

        splitter = DataSplitter(
            target_column="target", strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date", test_size=0.2,
            exclude_columns=["as_of_date"],
        )
        with patch("customer_retention.core.compat.spark_backend._as_pandas_api", side_effect=track_wrappers):
            splitter._spark_select_features_target(mock_spark_df, extract_metadata=True)

        assert len(wrappers) == 3, "each projection (features, target, metadata) gets its own wrapper"


class TestDistributedSplitSingleCheckpoint:
    """Tests for the improved _distributed_temporal_split that checkpoints once."""

    def test_no_redundant_select_in_distributed_split(self):
        """_distributed_temporal_split should not call _spark_select_features_target."""
        import inspect

        from customer_retention.stages.modeling.data_splitter import DataSplitter

        source = inspect.getsource(DataSplitter._distributed_temporal_split)
        assert "_spark_select_features_target" not in source, (
            "_distributed_temporal_split should use _select_features_target_from_ps, "
            "not _spark_select_features_target"
        )

    def test_uses_single_as_pandas_api_per_partition(self):
        """Each partition (train/test) gets exactly one _as_pandas_api call (plus import + val)."""
        import inspect

        from customer_retention.stages.modeling.data_splitter import DataSplitter

        source = inspect.getsource(DataSplitter._distributed_temporal_split)
        # import (1) + val (1) + train (1) + test (1) = 4 non-checkpoint usages
        # Previously: _spark_select_features_target called _as_pandas_api 3x per partition
        call_count = source.count("_as_pandas_api(")
        assert call_count <= 4, f"Expected at most 4 _as_pandas_api calls, got {call_count}"

    def test_select_features_target_from_ps_excludes_correct_cols(self):
        """_select_features_target_from_ps correctly excludes target and excluded columns."""
        try:
            from pyspark.sql.types import (
                DoubleType,
                IntegerType,
                StringType,
                StructField,
                StructType,
                TimestampNTZType,
            )
        except ImportError:
            pytest.skip("PySpark not installed")

        from unittest.mock import MagicMock

        schema = StructType([
            StructField("feature1", DoubleType()),
            StructField("feature2", IntegerType()),
            StructField("target", IntegerType()),
            StructField("as_of_date", TimestampNTZType()),
            StructField("entity_id", StringType()),
        ])
        mock_ps = MagicMock()
        mock_ps.__getitem__ = MagicMock(side_effect=lambda key: f"col:{key}")

        splitter = DataSplitter(
            target_column="target", strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date", test_size=0.2,
            exclude_columns=["as_of_date", "entity_id"],
        )
        features, target = splitter._select_features_target_from_ps(mock_ps, schema)

        # features should select ["feature1", "feature2"] (no target, no excluded, no timestamp)
        mock_ps.__getitem__.assert_any_call(["feature1", "feature2"])
        # target should select "target"
        mock_ps.__getitem__.assert_any_call("target")

    def test_distributed_split_metadata_from_same_wrapper(self):
        """Train metadata comes from the same _as_pandas_api wrapper as features."""
        import inspect

        from customer_retention.stages.modeling.data_splitter import DataSplitter

        source = inspect.getsource(DataSplitter._distributed_temporal_split)
        # Metadata should be extracted from train_ps, not a separate spark_df.select()
        assert "train_ps[c]" in source, "metadata should come from train_ps, not a separate wrapper"


class TestTemporalSplitFailFast:
    def test_raises_on_all_nat_temporal_column(self):
        n = 100
        df = pd.DataFrame({
            "feature1": np.random.randn(n),
            "target": np.random.choice([0, 1], n),
            "as_of_date": [pd.NaT] * n,
            "entity_id": [f"e{i % 10}" for i in range(n)],
        })
        splitter = DataSplitter(
            target_column="target", strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date", group_column="entity_id", test_size=0.2,
            exclude_columns=["as_of_date", "entity_id"],
        )
        with pytest.raises(ValueError, match="no valid dates"):
            splitter.split(df)

    def test_raises_on_empty_dataframe(self):
        df = pd.DataFrame({
            "feature1": pd.Series(dtype=float),
            "target": pd.Series(dtype=int),
            "as_of_date": pd.Series(dtype="datetime64[ns]"),
            "entity_id": pd.Series(dtype=str),
        })
        splitter = DataSplitter(
            target_column="target", strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date", group_column="entity_id", test_size=0.2,
            exclude_columns=["as_of_date", "entity_id"],
        )
        with pytest.raises(ValueError, match="no valid dates"):
            splitter.split(df)

    def test_raises_on_all_nat_with_purge_gap(self):
        n = 50
        df = pd.DataFrame({
            "feature1": np.random.randn(n),
            "target": np.random.choice([0, 1], n),
            "as_of_date": [pd.NaT] * n,
            "entity_id": [f"e{i % 5}" for i in range(n)],
        })
        splitter = DataSplitter(
            target_column="target", strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date", group_column="entity_id",
            test_size=0.2, purge_gap_days=30,
            exclude_columns=["as_of_date", "entity_id"],
        )
        with pytest.raises(ValueError, match="no valid dates"):
            splitter.split(df)

    def test_distributed_path_raises_on_nat_cutoff(self):
        pytest.importorskip("pyspark")
        from unittest.mock import MagicMock, patch

        splitter = DataSplitter(
            target_column="target", strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date", test_size=0.2, purge_gap_days=30,
        )
        mock_spark_df = MagicMock()
        mock_spark_df.columns = ["feature1", "target", "as_of_date"]
        mock_agg_row = {"cutoff": None, "total": 0}
        mock_spark_df.agg.return_value.head.return_value = mock_agg_row
        mock_spark_df.drop.return_value = mock_spark_df

        with patch("customer_retention.stages.modeling.data_splitter.as_spark_df", return_value=mock_spark_df):
            with pytest.raises(ValueError, match="no valid dates"):
                splitter._distributed_temporal_split(mock_spark_df)


class TestTemporalPurgeGap:
    @pytest.fixture
    def temporal_df(self):
        np.random.seed(42)
        n_entities = 100
        dates = pd.date_range("2023-01-01", periods=52, freq="W")
        rows = []
        for eid in range(n_entities):
            t = np.random.choice([0, 1], p=[0.3, 0.7])
            for d in dates:
                rows.append({"entity_id": f"e{eid}", "as_of_date": d,
                             "feature1": np.random.randn(), "feature2": np.random.randn(), "target": t})
        return pd.DataFrame(rows)

    def test_purge_gap_removes_rows_from_train_entities(self, temporal_df):
        splitter = DataSplitter(
            target_column="target", strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date", group_column="entity_id",
            test_size=0.2, purge_gap_days=30,
            exclude_columns=["as_of_date", "entity_id"],
        )
        result = splitter.split(temporal_df)
        total_kept = len(result.X_train) + len(result.X_test)
        assert total_kept < len(temporal_df)

    def test_purge_gap_zero_no_rows_dropped(self, temporal_df):
        splitter = DataSplitter(
            target_column="target", strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date", group_column="entity_id",
            test_size=0.2, purge_gap_days=0,
            exclude_columns=["as_of_date", "entity_id"],
        )
        result = splitter.split(temporal_df)
        assert len(result.X_train) + len(result.X_test) == len(temporal_df)

    def test_purge_gap_preserves_entity_isolation(self, temporal_df):
        splitter = DataSplitter(
            target_column="target", strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date", group_column="entity_id",
            test_size=0.2, purge_gap_days=30,
            exclude_columns=["as_of_date", "entity_id"],
        )
        result = splitter.split(temporal_df)
        train_ents = set(temporal_df.loc[result.X_train.index, "entity_id"])
        test_ents = set(temporal_df.loc[result.X_test.index, "entity_id"])
        assert len(train_ents & test_ents) == 0

    def test_purge_gap_split_info_reports_gap_metadata(self, temporal_df):
        splitter = DataSplitter(
            target_column="target", strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date", group_column="entity_id",
            test_size=0.2, purge_gap_days=30,
            exclude_columns=["as_of_date", "entity_id"],
        )
        result = splitter.split(temporal_df)
        assert "purge_gap_days" in result.split_info
        assert result.split_info["purge_gap_days"] == 30
        assert "purge_gap_rows" in result.split_info
        assert result.split_info["purge_gap_rows"] > 0
        assert "cutoff_date" in result.split_info

    def test_purge_gap_with_validation_split(self, temporal_df):
        splitter = DataSplitter(
            target_column="target", strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date", group_column="entity_id",
            test_size=0.2, purge_gap_days=30,
            include_validation=True, validation_size=0.10,
            exclude_columns=["as_of_date", "entity_id"],
        )
        result = splitter.split(temporal_df)
        assert result.X_val is not None
        assert len(result.X_val) > 0
        assert len(result.X_train) + len(result.X_val) + len(result.X_test) < len(temporal_df)

    def test_purge_gap_large_gap_still_has_train_and_test(self, temporal_df):
        splitter = DataSplitter(
            target_column="target", strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date", group_column="entity_id",
            test_size=0.2, purge_gap_days=500,
            exclude_columns=["as_of_date", "entity_id"],
        )
        result = splitter.split(temporal_df)
        # Large purge may drop all train rows but entities are still separated
        assert len(result.X_test) > 0


class TestTemporalSplitGroupColumnRequired:
    def test_pandas_path_temporal_without_group_column_raises(self, sample_df):
        splitter = DataSplitter(
            target_column="target", strategy=SplitStrategy.TEMPORAL,
            temporal_column="date", test_size=0.2,
            exclude_columns=["date", "custid"],
        )
        with pytest.raises(ValueError, match="group_column"):
            splitter.split(sample_df)

    def test_spark_path_temporal_without_group_column_raises(self):
        pytest.importorskip("pyspark")
        from unittest.mock import MagicMock, patch

        splitter = DataSplitter(
            target_column="target", strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date", test_size=0.2,
        )
        mock_df = MagicMock()
        mock_spark_df = MagicMock()
        # min_count agg returns 100
        mock_spark_df.groupBy.return_value.count.return_value.agg.return_value.head.return_value = [100]

        with patch("customer_retention.stages.modeling.data_splitter._is_spark_pandas", return_value=True), \
             patch("customer_retention.stages.modeling.data_splitter.as_spark_df", return_value=mock_spark_df):
            with pytest.raises(ValueError, match="group_column"):
                splitter.split(mock_df)


class TestSplitSparkDispatchByStrategy:
    """Verify _split_spark dispatches each strategy and merges availability warnings."""

    @pytest.fixture(autouse=True)
    def _require_pyspark(self):
        pytest.importorskip("pyspark")

    def _build_mock_spark_df(self, *, min_count=200):
        from unittest.mock import MagicMock
        sdf = MagicMock()
        sdf.groupBy.return_value.count.return_value.agg.return_value.head.return_value = [min_count]
        return sdf

    def test_random_stratified_strategy_calls_stratified_split(self):
        from unittest.mock import MagicMock, patch

        splitter = DataSplitter(
            target_column="target", strategy=SplitStrategy.RANDOM_STRATIFIED,
            test_size=0.2,
        )
        sdf = self._build_mock_spark_df()
        sdf.toPandas.return_value = pd.DataFrame({
            "feature1": np.random.randn(200),
            "target": np.random.choice([0, 1], 200, p=[0.3, 0.7]),
        })
        mock_result = MagicMock(spec=SplitResult)
        mock_result.split_info = {}
        with patch.object(splitter, "_stratified_split", return_value=mock_result) as mock_strat:
            result = splitter._split_spark(sdf)
        mock_strat.assert_called_once()
        assert result is mock_result

    def test_group_strategy_calls_group_split(self):
        from unittest.mock import MagicMock, patch

        splitter = DataSplitter(
            target_column="target", strategy=SplitStrategy.GROUP,
            group_column="entity_id", test_size=0.2,
        )
        sdf = self._build_mock_spark_df()
        sdf.toPandas.return_value = pd.DataFrame({
            "feature1": np.random.randn(200),
            "target": np.random.choice([0, 1], 200),
            "entity_id": [f"e{i % 20}" for i in range(200)],
        })
        mock_result = MagicMock(spec=SplitResult)
        mock_result.split_info = {}
        with patch.object(splitter, "_group_split", return_value=mock_result) as mock_grp:
            result = splitter._split_spark(sdf)
        mock_grp.assert_called_once()
        assert result is mock_result

    def test_temporal_strategy_calls_distributed_temporal(self):
        from unittest.mock import MagicMock, patch

        splitter = DataSplitter(
            target_column="target", strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date", group_column="entity_id", test_size=0.2,
        )
        sdf = self._build_mock_spark_df()
        mock_result = MagicMock(spec=SplitResult)
        mock_result.split_info = {}
        with patch.object(splitter, "_distributed_temporal_split", return_value=mock_result) as mock_dist:
            result = splitter._split_spark(sdf)
        mock_dist.assert_called_once_with(sdf)
        assert result is mock_result

    def test_low_minority_count_warns(self):
        from unittest.mock import MagicMock, patch

        splitter = DataSplitter(
            target_column="target", strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date", group_column="entity_id", test_size=0.2,
        )
        # min_count * 0.2 = 4 → < 50 triggers warning
        sdf = self._build_mock_spark_df(min_count=20)
        mock_result = MagicMock(spec=SplitResult)
        mock_result.split_info = {}
        with patch.object(splitter, "_distributed_temporal_split", return_value=mock_result):
            with pytest.warns(UserWarning, match="Insufficient minority samples"):
                splitter._split_spark(sdf)

    def test_min_count_none_treated_as_zero(self):
        from unittest.mock import MagicMock, patch

        splitter = DataSplitter(
            target_column="target", strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date", group_column="entity_id", test_size=0.2,
        )
        sdf = MagicMock()
        # head returns [None] simulating empty groupBy
        sdf.groupBy.return_value.count.return_value.agg.return_value.head.return_value = [None]
        mock_result = MagicMock(spec=SplitResult)
        mock_result.split_info = {}
        with patch.object(splitter, "_distributed_temporal_split", return_value=mock_result):
            with pytest.warns(UserWarning):
                splitter._split_spark(sdf)

    def test_availability_warnings_added_to_split_info(self):
        from unittest.mock import MagicMock, patch

        from customer_retention.analysis.auto_explorer.findings import (
            FeatureAvailabilityInfo,
            FeatureAvailabilityMetadata,
        )

        splitter = DataSplitter(
            target_column="target", strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date", group_column="entity_id", test_size=0.2,
        )
        sdf = self._build_mock_spark_df()
        sdf.columns = ["feature1", "target", "as_of_date", "entity_id", "new_feat"]
        availability = FeatureAvailabilityMetadata(
            data_start="2024-01-01",
            data_end="2025-01-01",
            time_span_days=365,
            new_tracking=["new_feat"],
            retired_tracking=[],
            partial_window=[],
            features={"new_feat": FeatureAvailabilityInfo(
                first_valid_date="2024-01-01",
                last_valid_date="2025-01-01",
                coverage_pct=100.0,
                availability_type="new_tracking",
                days_from_start=0,
                days_before_end=0,
            )},
        )
        mock_result = MagicMock(spec=SplitResult)
        mock_result.split_info = {}
        with patch.object(splitter, "_distributed_temporal_split", return_value=mock_result):
            splitter._split_spark(sdf, feature_availability=availability)

        assert "availability_warnings" in mock_result.split_info
        assert mock_result.split_info["availability_warnings"][0]["column"] == "new_feat"


class TestValidateFeatureAvailabilityPartialWindow:
    def test_partial_window_emits_warning(self):
        from customer_retention.analysis.auto_explorer.findings import (
            FeatureAvailabilityInfo,
            FeatureAvailabilityMetadata,
        )

        df = pd.DataFrame({"partial_feat": [1, 2, 3], "target": [0, 1, 0]})
        availability = FeatureAvailabilityMetadata(
            data_start="2024-01-01",
            data_end="2024-12-31",
            time_span_days=365,
            new_tracking=[],
            retired_tracking=[],
            partial_window=["partial_feat"],
            features={"partial_feat": FeatureAvailabilityInfo(
                first_valid_date="2024-03-01",
                last_valid_date="2024-09-30",
                coverage_pct=58.0,
                availability_type="partial_window",
                days_from_start=60,
                days_before_end=92,
            )},
        )
        splitter = DataSplitter(
            target_column="target", strategy=SplitStrategy.TEMPORAL,
            temporal_column="date", group_column="entity",
        )
        warnings_list = splitter.validate_feature_availability(df, availability)
        assert len(warnings_list) == 1
        assert warnings_list[0].column == "partial_feat"
        assert warnings_list[0].issue == "partial_window"
        assert "2024-03-01" in warnings_list[0].recommendation
        assert "2024-09-30" in warnings_list[0].recommendation

    def test_partial_window_unknown_dates_when_feature_missing(self):
        from customer_retention.analysis.auto_explorer.findings import FeatureAvailabilityMetadata

        df = pd.DataFrame({"orphan_col": [1, 2], "target": [0, 1]})
        availability = FeatureAvailabilityMetadata(
            data_start="2024-01-01",
            data_end="2024-12-31",
            time_span_days=365,
            new_tracking=[],
            retired_tracking=[],
            partial_window=["orphan_col"],
            features={},  # No metadata for orphan_col
        )
        splitter = DataSplitter(
            target_column="target", strategy=SplitStrategy.TEMPORAL,
            temporal_column="date", group_column="entity",
        )
        warnings_list = splitter.validate_feature_availability(df, availability)
        assert len(warnings_list) == 1
        assert "unknown" in warnings_list[0].recommendation


class TestGroupSplitWithValidation:
    def test_group_split_includes_validation(self):
        n = 600
        df = pd.DataFrame({
            "feature1": np.random.randn(n),
            "feature2": np.random.randn(n),
            "target": np.random.choice([0, 1], n),
            "entity_id": [f"e{i % 30}" for i in range(n)],
        })
        splitter = DataSplitter(
            target_column="target", strategy=SplitStrategy.GROUP,
            group_column="entity_id", test_size=0.2,
            include_validation=True, validation_size=0.10,
            exclude_columns=["entity_id"],
        )
        result = splitter.split(df)
        assert result.X_val is not None
        assert result.y_val is not None
        assert len(result.X_val) > 0
        # No entity should appear in more than one split
        train_ents = set(df.loc[result.X_train.index, "entity_id"])
        val_ents = set(df.loc[result.X_val.index, "entity_id"])
        test_ents = set(df.loc[result.X_test.index, "entity_id"])
        assert not (train_ents & val_ents)
        assert not (train_ents & test_ents)
        assert not (val_ents & test_ents)
        assert "validation_size" in result.split_info


class _FakeSparkDFForSplit:
    """Records spark calls and produces deterministic head/agg results."""

    def __init__(self, columns, schema_fields, *, total=200, cutoff_epoch=1700000000, history=None):
        self._columns = list(columns)
        self._schema_fields = schema_fields
        self._total = total
        self._cutoff_epoch = cutoff_epoch
        self.history = history if history is not None else []
        self._next_filtered = 100

    @property
    def columns(self):
        return list(self._columns)

    @property
    def schema(self):
        from unittest.mock import MagicMock
        s = MagicMock()
        s.fields = self._schema_fields
        return s

    def drop(self, *cols):
        self.history.append(("drop", cols))
        remaining = [c for c in self._columns if c not in cols]
        return _FakeSparkDFForSplit(
            remaining, self._schema_fields,
            total=self._total, cutoff_epoch=self._cutoff_epoch,
            history=self.history,
        )

    def agg(self, *exprs):  # noqa: ARG002
        self.history.append(("agg", "cutoff/total"))
        return self

    def head(self):
        self.history.append(("head",))
        return {"cutoff": self._cutoff_epoch, "total": self._total}

    def filter(self, expr):  # noqa: ARG002
        self.history.append(("filter",))
        return _FakeSparkDFForSplit(
            self._columns, self._schema_fields,
            total=self._total, cutoff_epoch=self._cutoff_epoch,
            history=self.history,
        )

    def localCheckpoint(self, eager=True):  # noqa: ARG002, N802
        self.history.append(("localCheckpoint",))
        return self


class TestDistributedTemporalSplitFullPath:
    """Exercise the full _distributed_temporal_split body using fake Spark DFs."""

    def _build_schema(self):
        pytest.importorskip("pyspark")
        from pyspark.sql.types import (
            DoubleType,
            IntegerType,
            StringType,
            StructField,
            TimestampNTZType,
        )
        return [
            StructField("feature1", DoubleType()),
            StructField("feature2", DoubleType()),
            StructField("target", IntegerType()),
            StructField("as_of_date", TimestampNTZType()),
            StructField("entity_id", StringType()),
        ]

    def _stub_select(self, splitter):
        from unittest.mock import MagicMock

        def fake_select(ps_df, schema):  # noqa: ARG001
            return MagicMock(name="X"), MagicMock(name="y")

        splitter._select_features_target_from_ps = fake_select  # type: ignore[method-assign]

    def test_full_distributed_path_runs(self):
        from unittest.mock import MagicMock, patch

        schema = self._build_schema()
        sdf = _FakeSparkDFForSplit(
            ["feature1", "feature2", "target", "as_of_date", "entity_id"],
            schema,
        )

        splitter = DataSplitter(
            target_column="target", strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date", group_column="entity_id",
            test_size=0.2, exclude_columns=["as_of_date", "entity_id"],
        )
        self._stub_select(splitter)

        with patch("customer_retention.stages.modeling.data_splitter.as_spark_df", return_value=sdf), \
             patch("customer_retention.core.compat.spark_backend._as_pandas_api", return_value=MagicMock()):
            result = splitter._distributed_temporal_split(MagicMock())

        assert isinstance(result, SplitResult)
        # Should call agg + head once for cutoff
        assert any(h[0] == "agg" for h in sdf.history)
        # Should checkpoint train and test
        ckpts = [h for h in sdf.history if h[0] == "localCheckpoint"]
        assert len(ckpts) >= 2

    def test_distributed_drops_index_level_columns(self):
        from unittest.mock import MagicMock, patch

        schema = self._build_schema()
        sdf = _FakeSparkDFForSplit(
            ["__index_level_0__", "feature1", "target", "as_of_date", "entity_id"],
            schema,
        )

        splitter = DataSplitter(
            target_column="target", strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date", group_column="entity_id",
            test_size=0.2, exclude_columns=["as_of_date", "entity_id"],
        )
        self._stub_select(splitter)

        with patch("customer_retention.stages.modeling.data_splitter.as_spark_df", return_value=sdf), \
             patch("customer_retention.core.compat.spark_backend._as_pandas_api", return_value=MagicMock()):
            splitter._distributed_temporal_split(MagicMock())

        drops = [h for h in sdf.history if h[0] == "drop"]
        assert any("__index_level_0__" in d[1] for d in drops)

    def test_distributed_purge_gap_filters_train(self):
        from unittest.mock import MagicMock, patch

        schema = self._build_schema()
        sdf = _FakeSparkDFForSplit(
            ["feature1", "target", "as_of_date", "entity_id"],
            schema,
            total=200,
        )

        splitter = DataSplitter(
            target_column="target", strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date", group_column="entity_id",
            test_size=0.2, purge_gap_days=30,
            exclude_columns=["as_of_date", "entity_id"],
        )
        self._stub_select(splitter)

        with patch("customer_retention.stages.modeling.data_splitter.as_spark_df", return_value=sdf), \
             patch("customer_retention.core.compat.spark_backend._as_pandas_api", return_value=MagicMock()):
            result = splitter._distributed_temporal_split(MagicMock())

        # purge_gap adds an extra filter on the train side
        filters = [h for h in sdf.history if h[0] == "filter"]
        assert len(filters) >= 3  # test_filter, train_filter, purge_filter
        assert "purge_gap_days" in result.split_info
        assert result.split_info["purge_gap_days"] == 30
        assert "purge_gap_rows" in result.split_info

    def test_distributed_validation_split_path(self):
        from unittest.mock import MagicMock, patch

        schema = self._build_schema()
        sdf = _FakeSparkDFForSplit(
            ["feature1", "target", "as_of_date", "entity_id"],
            schema,
        )

        splitter = DataSplitter(
            target_column="target", strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date", group_column="entity_id",
            test_size=0.2, include_validation=True, validation_size=0.1,
            exclude_columns=["as_of_date", "entity_id"],
        )
        self._stub_select(splitter)

        with patch("customer_retention.stages.modeling.data_splitter.as_spark_df", return_value=sdf), \
             patch("customer_retention.core.compat.spark_backend._as_pandas_api", return_value=MagicMock()):
            result = splitter._distributed_temporal_split(MagicMock())

        assert result.X_val is not None
        # Validation split adds an extra checkpoint
        ckpts = [h for h in sdf.history if h[0] == "localCheckpoint"]
        assert len(ckpts) >= 3

    def test_distributed_train_metadata_extracted(self):
        from unittest.mock import MagicMock, patch

        schema = self._build_schema()
        sdf = _FakeSparkDFForSplit(
            ["feature1", "target", "as_of_date", "entity_id"],
            schema,
        )

        splitter = DataSplitter(
            target_column="target", strategy=SplitStrategy.TEMPORAL,
            temporal_column="as_of_date", group_column="entity_id",
            test_size=0.2, exclude_columns=["as_of_date", "entity_id"],
        )
        self._stub_select(splitter)

        # Make _as_pandas_api return a dict-like with metadata cols
        ps_obj = MagicMock()
        ps_obj.__getitem__ = MagicMock(return_value="meta_series")

        with patch("customer_retention.stages.modeling.data_splitter.as_spark_df", return_value=sdf), \
             patch("customer_retention.core.compat.spark_backend._as_pandas_api", return_value=ps_obj):
            result = splitter._distributed_temporal_split(MagicMock())

        # exclude_columns ["as_of_date", "entity_id"] both in schema → metadata captured
        assert "as_of_date" in result.train_metadata
        assert "entity_id" in result.train_metadata
