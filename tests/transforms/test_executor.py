import numpy as np
import pandas as pd
import pytest

from customer_retention.generators.pipeline_generator.models import (
    PipelineTransformationType,
    TransformationStep,
)
from customer_retention.transforms import ops
from customer_retention.transforms.artifact_store import ArtifactStore
from customer_retention.transforms.executor import TransformExecutor


@pytest.fixture
def executor():
    return TransformExecutor()


@pytest.fixture
def artifact_store(tmp_path):
    return ArtifactStore(str(tmp_path / "artifacts"))


@pytest.fixture
def sample_df():
    np.random.seed(42)
    return pd.DataFrame({
        "age": [25.0, None, 35.0, 45.0, 200.0],
        "amount": [100.0, 200.0, 0.0, 400.0, 50000.0],
        "status": ["active", "churned", "active", "new", "churned"],
    })


def _step(type_: PipelineTransformationType, column: str, **params):
    return TransformationStep(
        type=type_, column=column, parameters=params, rationale="test"
    )


class TestBronzeDispatch:
    def test_impute_null(self, executor, sample_df):
        step = _step(PipelineTransformationType.IMPUTE_NULL, "age", value=0)
        result = executor.apply(sample_df, step)
        assert result["age"].isna().sum() == 0

    def test_impute_null_median(self, executor, sample_df):
        step = _step(PipelineTransformationType.IMPUTE_NULL, "age", value="median")
        result = executor.apply(sample_df, step)
        assert result["age"].isna().sum() == 0

    def test_cap_outlier(self, executor, sample_df):
        step = _step(PipelineTransformationType.CAP_OUTLIER, "age", lower=0, upper=100)
        result = executor.apply(sample_df, step)
        assert result["age"].dropna().max() <= 100

    def test_type_cast(self, executor, sample_df):
        df = sample_df.copy()
        df["age"] = df["age"].fillna(0)
        step = _step(PipelineTransformationType.TYPE_CAST, "age", dtype="int")
        result = executor.apply(df, step)
        assert result["age"].dtype == int

    def test_drop_column(self, executor, sample_df):
        step = _step(PipelineTransformationType.DROP_COLUMN, "status")
        result = executor.apply(sample_df, step)
        assert "status" not in result.columns

    def test_winsorize(self, executor, sample_df):
        step = _step(PipelineTransformationType.WINSORIZE, "age", lower_bound=20, upper_bound=50)
        result = executor.apply(sample_df, step)
        assert result["age"].dropna().max() <= 50

    def test_segment_aware_cap(self, executor):
        np.random.seed(42)
        df = pd.DataFrame({"v": np.concatenate([
            np.random.normal(10, 2, 50),
            np.random.normal(100, 5, 50),
            [500],
        ])})
        step = _step(PipelineTransformationType.SEGMENT_AWARE_CAP, "v", n_segments=2)
        result = executor.apply(df, step)
        assert result["v"].max() < 500


class TestGoldStatelessDispatch:
    def test_log_transform(self, executor, sample_df):
        step = _step(PipelineTransformationType.LOG_TRANSFORM, "amount")
        result = executor.apply(sample_df, step)
        assert result["amount"].max() < 50000

    def test_sqrt_transform(self, executor, sample_df):
        step = _step(PipelineTransformationType.SQRT_TRANSFORM, "amount")
        result = executor.apply(sample_df, step)
        assert result["amount"].iloc[0] == pytest.approx(np.sqrt(100))

    def test_zero_inflation(self, executor, sample_df):
        step = _step(PipelineTransformationType.ZERO_INFLATION_HANDLING, "amount")
        result = executor.apply(sample_df, step)
        assert "amount_is_zero" in result.columns

    def test_cap_then_log(self, executor, sample_df):
        step = _step(PipelineTransformationType.CAP_THEN_LOG, "amount")
        result = executor.apply(sample_df, step)
        assert result["amount"].max() < np.log1p(50000)


class TestEncodingDispatch:
    def test_one_hot(self, executor, sample_df):
        step = _step(PipelineTransformationType.ENCODE, "status", method="one_hot")
        result = executor.apply(sample_df, step)
        assert "status" not in result.columns
        assert "status_active" in result.columns

    def test_label_encode_fit(self, executor, sample_df, artifact_store):
        step = _step(PipelineTransformationType.ENCODE, "status", method="label")
        result = executor.apply(sample_df, step, fit_mode=True, artifact_store=artifact_store)
        assert result["status"].dtype in (np.int64, np.intp, int)

    def test_label_encode_transform(self, executor, sample_df, artifact_store):
        step = _step(PipelineTransformationType.ENCODE, "status", method="label")
        executor.apply(sample_df.copy(), step, fit_mode=True, artifact_store=artifact_store)
        result = executor.apply(sample_df.copy(), step, fit_mode=False, artifact_store=artifact_store)
        assert result["status"].dtype in (np.int64, np.intp, int, object)


class TestScalingDispatch:
    def test_standard_scale_fit(self, executor, artifact_store):
        np.random.seed(42)
        df = pd.DataFrame({"val": np.random.randn(100) * 10 + 50})
        step = _step(PipelineTransformationType.SCALE, "val", method="standard")
        result = executor.apply(df, step, fit_mode=True, artifact_store=artifact_store)
        assert result["val"].mean() == pytest.approx(0, abs=0.1)

    def test_minmax_scale_fit(self, executor, artifact_store):
        np.random.seed(42)
        df = pd.DataFrame({"val": np.random.randn(100) * 10 + 50})
        step = _step(PipelineTransformationType.SCALE, "val", method="minmax")
        result = executor.apply(df, step, fit_mode=True, artifact_store=artifact_store)
        assert result["val"].min() >= -0.01
        assert result["val"].max() <= 1.01


class TestYeoJohnsonDispatch:
    def test_yeo_johnson_fit(self, executor, artifact_store):
        np.random.seed(42)
        df = pd.DataFrame({"val": np.random.exponential(5, 100)})
        step = _step(PipelineTransformationType.YEO_JOHNSON, "val")
        result = executor.apply(df, step, fit_mode=True, artifact_store=artifact_store)
        assert artifact_store.has("val_power_transformer")

    def test_yeo_johnson_transform(self, executor, artifact_store):
        np.random.seed(42)
        df = pd.DataFrame({"val": np.random.exponential(5, 100)})
        step = _step(PipelineTransformationType.YEO_JOHNSON, "val")
        executor.apply(df.copy(), step, fit_mode=True, artifact_store=artifact_store)
        result = executor.apply(df.copy(), step, fit_mode=False, artifact_store=artifact_store)
        assert not result["val"].isna().any()


class TestFeatureSelectDispatch:
    def test_drops_column(self, executor, sample_df):
        step = _step(PipelineTransformationType.FEATURE_SELECT, "amount")
        result = executor.apply(sample_df, step)
        assert "amount" not in result.columns


class TestDerivedColumnDispatch:
    def test_ratio(self, executor, sample_df):
        step = _step(
            PipelineTransformationType.DERIVED_COLUMN, "age_per_amount",
            method="ratio", numerator="age", denominator="amount",
        )
        result = executor.apply(sample_df, step)
        assert "age_per_amount" in result.columns

    def test_interaction(self, executor, sample_df):
        step = _step(
            PipelineTransformationType.DERIVED_COLUMN, "age_x_amount",
            method="interaction", col_a="age", col_b="amount",
        )
        result = executor.apply(sample_df, step)
        assert "age_x_amount" in result.columns

    def test_composite(self, executor, sample_df):
        step = _step(
            PipelineTransformationType.DERIVED_COLUMN, "avg_vals",
            method="composite", columns=["age", "amount"],
        )
        result = executor.apply(sample_df, step)
        assert "avg_vals" in result.columns

    def test_unknown_method_noop(self, executor, sample_df):
        step = _step(
            PipelineTransformationType.DERIVED_COLUMN, "col",
            method="nonexistent_method",
        )
        result = executor.apply(sample_df, step)
        pd.testing.assert_frame_equal(result, sample_df)


class TestApplyAll:
    def test_chains_steps(self, executor, sample_df):
        steps = [
            _step(PipelineTransformationType.IMPUTE_NULL, "age", value=0),
            _step(PipelineTransformationType.CAP_OUTLIER, "age", lower=0, upper=100),
            _step(PipelineTransformationType.LOG_TRANSFORM, "amount"),
        ]
        result = executor.apply_all(sample_df, steps)
        assert result["age"].isna().sum() == 0
        assert result["age"].max() <= 100
        assert result["amount"].max() < 50000

    def test_many_steps_produce_correct_result(self, executor):
        df = pd.DataFrame({"v": [1.0, 2.0, 3.0, 100.0, 200.0] * 20})
        steps = [_step(PipelineTransformationType.LOG_TRANSFORM, "v") for _ in range(15)]
        result = executor.apply_all(df.copy(), steps)
        expected = df.copy()
        for _ in range(15):
            expected["v"] = np.log1p(expected["v"].clip(lower=0))
        np.testing.assert_array_almost_equal(result["v"].to_numpy(), expected["v"].to_numpy())

    def test_checkpoint_interval_respected(self, executor, sample_df):
        from unittest.mock import MagicMock, patch

        _MOD = "customer_retention.transforms.executor"
        steps = [_step(PipelineTransformationType.LOG_TRANSFORM, "amount") for _ in range(25)]
        checkpoint_count = 0

        mock_spark_df = MagicMock()
        mock_spark_df.localCheckpoint = MagicMock(return_value=mock_spark_df)

        with patch(f"{_MOD}._is_spark_pandas", return_value=True), \
             patch(f"{_MOD}.as_spark_df", return_value=mock_spark_df), \
             patch(f"{_MOD}._as_pandas_api", return_value=sample_df.copy()):
            executor.apply_all(sample_df.copy(), steps)
        assert mock_spark_df.localCheckpoint.call_count == 2

    def test_no_checkpoint_on_pandas(self, executor, sample_df):
        from unittest.mock import patch
        steps = [_step(PipelineTransformationType.LOG_TRANSFORM, "amount") for _ in range(25)]
        with patch("customer_retention.transforms.executor._is_spark_pandas", return_value=False):
            result = executor.apply_all(sample_df.copy(), steps)
        assert len(result) == len(sample_df)

    def test_batch_quantiles_match_per_column(self, executor):
        np.random.seed(42)
        df = pd.DataFrame({"a": np.random.randn(200), "b": np.random.exponential(5, 200)})
        steps = [
            _step(PipelineTransformationType.CAP_THEN_LOG, "a"),
            _step(PipelineTransformationType.CAP_THEN_LOG, "b"),
        ]
        result_sequential = executor.apply_all(df.copy(), steps)
        executor._precompute_quantiles(df, steps)
        assert "_precomputed_q99" in steps[0].parameters
        assert "_precomputed_q99" in steps[1].parameters
        result_batch = executor.apply_all(df.copy(), steps)
        np.testing.assert_array_almost_equal(
            result_batch["a"].to_numpy(), result_sequential["a"].to_numpy()
        )
        np.testing.assert_array_almost_equal(
            result_batch["b"].to_numpy(), result_sequential["b"].to_numpy()
        )

    def test_unknown_type_raises(self, executor, sample_df):
        step = _step(PipelineTransformationType.AGGREGATE, "age")
        with pytest.raises(ValueError, match="Unknown transformation"):
            executor.apply(sample_df, step)


class TestDistributedApplyAll:
    _MOD = "customer_retention.transforms.executor"

    def test_distributed_path_wraps_per_step(self, executor):
        from unittest.mock import MagicMock, patch

        mock_spark_df = MagicMock()
        mock_result_ps = MagicMock()
        steps = [_step(PipelineTransformationType.LOG_TRANSFORM, "a")]

        with patch(f"{self._MOD}._is_spark_pandas", return_value=True), \
             patch(f"{self._MOD}.as_spark_df", return_value=mock_spark_df) as mock_as, \
             patch(f"{self._MOD}._as_pandas_api", return_value=mock_result_ps) as mock_api, \
             patch.object(executor, "apply", return_value=mock_result_ps), \
             patch.object(executor, "_precompute_quantiles"):
            executor._apply_all_distributed(MagicMock(), steps)

        assert mock_api.call_count == 2
        assert mock_as.call_count == 2

    def test_clears_psseries_before_to_spark(self, executor):
        from unittest.mock import MagicMock, patch

        mock_result = MagicMock()
        mock_result._psseries = {"stale": "cache"}
        steps = [_step(PipelineTransformationType.LOG_TRANSFORM, "a")]

        with patch(f"{self._MOD}._is_spark_pandas", return_value=True), \
             patch(f"{self._MOD}.as_spark_df"), \
             patch(f"{self._MOD}._as_pandas_api", return_value=MagicMock()), \
             patch.object(executor, "apply", return_value=mock_result), \
             patch.object(executor, "_precompute_quantiles"):
            executor._apply_all_distributed(MagicMock(), steps)

        assert mock_result._psseries is None

    def test_zero_inflation_reads_before_writes(self):
        df = pd.DataFrame({"amount": [0.0, 5.0, 0.0, 10.0]})
        result = ops.apply_zero_inflation_handling(df.copy(), "amount")
        assert "amount_is_zero" in result.columns
        assert list(result["amount_is_zero"]) == [1, 0, 1, 0]
        assert result.loc[0, "amount"] == 0.0
        assert result.loc[1, "amount"] > 0.0
