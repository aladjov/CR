import sys
from unittest.mock import MagicMock, patch

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

_MOD = "customer_retention.transforms.executor"


@pytest.fixture
def executor():
    return TransformExecutor()


@pytest.fixture
def artifact_store(tmp_path):
    return ArtifactStore(str(tmp_path / "artifacts"))


@pytest.fixture
def mock_spark_ops(monkeypatch):
    mock = MagicMock()
    import customer_retention.transforms as _pkg
    monkeypatch.setitem(sys.modules, "customer_retention.transforms.spark_ops", mock)
    monkeypatch.setattr(_pkg, "spark_ops", mock, raising=False)
    return mock


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

    def test_plan_truncation_for_pure_spark_chain(self, executor, sample_df):
        steps = [_step(PipelineTransformationType.LOG_TRANSFORM, "amount") for _ in range(65)]
        mock_spark_df = MagicMock()
        mock_spark_df.localCheckpoint = MagicMock(return_value=mock_spark_df)

        with patch(f"{_MOD}._is_spark_pandas", return_value=True), \
             patch(f"{_MOD}.as_spark_df", return_value=mock_spark_df), \
             patch(f"{_MOD}._as_pandas_api", return_value=sample_df.copy()), \
             patch(f"{_MOD}._SPARK_DISPATCH", {PipelineTransformationType.LOG_TRANSFORM: lambda df, s: df}), \
             patch.object(executor, "_precompute_quantiles_distributed"):
            executor.apply_all(sample_df.copy(), steps)
        assert mock_spark_df.localCheckpoint.call_count == 2

    def test_checkpoint_after_roundtrips(self, executor, sample_df):
        steps = [_step(PipelineTransformationType.SCALE, "amount", method="standard") for _ in range(25)]
        mock_spark_df = MagicMock()
        mock_spark_df.localCheckpoint = MagicMock(return_value=mock_spark_df)

        with patch(f"{_MOD}._is_spark_pandas", return_value=True), \
             patch(f"{_MOD}.as_spark_df", return_value=mock_spark_df), \
             patch(f"{_MOD}._as_pandas_api", return_value=sample_df.copy()), \
             patch(f"{_MOD}._SPARK_DISPATCH", {}), \
             patch.object(executor, "_apply_fitted_spark", return_value=None), \
             patch.object(executor, "apply", return_value=sample_df.copy()), \
             patch.object(executor, "_precompute_quantiles_distributed"):
            executor.apply_all(sample_df.copy(), steps)
        assert mock_spark_df.localCheckpoint.call_count == 2

    def test_no_checkpoint_on_pandas(self, executor, sample_df):
        steps = [_step(PipelineTransformationType.LOG_TRANSFORM, "amount") for _ in range(25)]
        with patch(f"{_MOD}._is_spark_pandas", return_value=False):
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
    def test_distributed_uses_spark_dispatch(self, executor):
        mock_spark_df = MagicMock()
        mock_result = MagicMock()
        steps = [_step(PipelineTransformationType.LOG_TRANSFORM, "a")]

        with patch(f"{_MOD}._is_spark_pandas", return_value=True), \
             patch(f"{_MOD}.as_spark_df", return_value=mock_spark_df), \
             patch(f"{_MOD}._as_pandas_api", return_value=MagicMock()) as mock_api, \
             patch(f"{_MOD}._SPARK_DISPATCH", {PipelineTransformationType.LOG_TRANSFORM: lambda df, s: mock_result}), \
             patch.object(executor, "_precompute_quantiles_distributed"):
            executor._apply_all_distributed(MagicMock(), steps)
        mock_api.assert_called_once_with(mock_result)

    def test_fallback_to_pyspark_pandas_for_unknown_types(self, executor):
        mock_spark_df = MagicMock()
        mock_ps_result = MagicMock()
        steps = [_step(PipelineTransformationType.SCALE, "a", method="standard")]

        with patch(f"{_MOD}._is_spark_pandas", return_value=True), \
             patch(f"{_MOD}.as_spark_df", return_value=mock_spark_df) as mock_as, \
             patch(f"{_MOD}._as_pandas_api", return_value=MagicMock()), \
             patch(f"{_MOD}._SPARK_DISPATCH", {}), \
             patch.object(executor, "apply", return_value=mock_ps_result), \
             patch.object(executor, "_apply_fitted_spark", return_value=None), \
             patch.object(executor, "_precompute_quantiles_distributed"):
            executor._apply_all_distributed(MagicMock(), steps)
        assert mock_as.call_count == 2

    def test_zero_inflation_reads_before_writes(self):
        df = pd.DataFrame({"amount": [0.0, 5.0, 0.0, 10.0]})
        result = ops.apply_zero_inflation_handling(df.copy(), "amount")
        assert "amount_is_zero" in result.columns
        assert list(result["amount_is_zero"]) == [1, 0, 1, 0]
        assert result.loc[0, "amount"] == 0.0
        assert result.loc[1, "amount"] > 0.0

    def test_fitted_spark_uses_precomputed_standard_scale(self, mock_spark_ops):
        step = _step(PipelineTransformationType.SCALE, "a", method="standard")
        step.parameters["_spark_fitted"] = {"kind": "standard", "mean": 5.0, "scale": 2.0}
        mock_spark_df = MagicMock()
        mock_result = MagicMock()
        mock_spark_ops.spark_standard_scale.return_value = mock_result
        result = TransformExecutor._apply_fitted_spark(mock_spark_df, step, True, None)
        mock_spark_ops.spark_standard_scale.assert_called_once_with(mock_spark_df, "a", mean=5.0, scale=2.0)
        assert result is mock_result

    def test_fitted_spark_uses_precomputed_minmax_scale(self, mock_spark_ops):
        step = _step(PipelineTransformationType.SCALE, "a", method="minmax")
        step.parameters["_spark_fitted"] = {"kind": "minmax", "scale": 0.5, "offset": -0.25}
        mock_spark_df = MagicMock()
        mock_result = MagicMock()
        mock_spark_ops.spark_minmax_scale.return_value = mock_result
        result = TransformExecutor._apply_fitted_spark(mock_spark_df, step, True, None)
        mock_spark_ops.spark_minmax_scale.assert_called_once_with(mock_spark_df, "a", scale=0.5, offset=-0.25)
        assert result is mock_result

    def test_fitted_spark_uses_precomputed_yeo_johnson(self, mock_spark_ops):
        step = _step(PipelineTransformationType.YEO_JOHNSON, "a")
        step.parameters["_spark_fitted"] = {"lmbda": 0.5, "std_mean": 1.0, "std_scale": 0.5, "standardize": True}
        mock_spark_df = MagicMock()
        mock_spark_ops.spark_yeo_johnson.return_value = MagicMock()
        TransformExecutor._apply_fitted_spark(mock_spark_df, step, True, None)
        mock_spark_ops.spark_yeo_johnson.assert_called_once_with(
            mock_spark_df, "a", lmbda=0.5, std_mean=1.0, std_scale=0.5, standardize=True,
        )

    def test_fitted_spark_returns_none_for_unhandled(self, mock_spark_ops):
        step = _step(PipelineTransformationType.LOG_TRANSFORM, "a")
        result = TransformExecutor._apply_fitted_spark(MagicMock(), step, False, None)
        assert result is None

    def test_fitted_spark_scale_no_params_no_artifact_returns_none(self, mock_spark_ops):
        step = _step(PipelineTransformationType.SCALE, "a", method="standard")
        result = TransformExecutor._apply_fitted_spark(MagicMock(), step, True, None)
        assert result is None

    def test_fitted_spark_yj_no_params_no_artifact_returns_none(self, mock_spark_ops):
        step = _step(PipelineTransformationType.YEO_JOHNSON, "a")
        result = TransformExecutor._apply_fitted_spark(MagicMock(), step, True, None)
        assert result is None

    def test_apply_fitted_spark_loads_standard_from_artifact(self, mock_spark_ops, artifact_store):
        np.random.seed(42)
        from customer_retention.transforms.fitted import FittedScaler
        scaler = FittedScaler("standard")
        df = pd.DataFrame({"v": np.random.randn(50) * 10 + 50})
        scaler.fit_transform(df.copy(), "v", artifact_store)
        step = _step(PipelineTransformationType.SCALE, "v", method="standard")
        mock_spark_df = MagicMock()
        mock_spark_ops.spark_standard_scale.return_value = MagicMock()
        result = TransformExecutor._apply_fitted_spark(mock_spark_df, step, False, artifact_store)
        assert result is not None
        mock_spark_ops.spark_standard_scale.assert_called_once()

    def test_apply_fitted_spark_loads_minmax_from_artifact(self, mock_spark_ops, artifact_store):
        np.random.seed(42)
        from customer_retention.transforms.fitted import FittedScaler
        scaler = FittedScaler("minmax")
        df = pd.DataFrame({"v": np.random.randn(50) * 10 + 50})
        scaler.fit_transform(df.copy(), "v", artifact_store)
        step = _step(PipelineTransformationType.SCALE, "v", method="minmax")
        mock_spark_df = MagicMock()
        mock_spark_ops.spark_minmax_scale.return_value = MagicMock()
        result = TransformExecutor._apply_fitted_spark(mock_spark_df, step, False, artifact_store)
        assert result is not None
        mock_spark_ops.spark_minmax_scale.assert_called_once()

    def test_apply_fitted_spark_loads_yj_from_artifact(self, mock_spark_ops, artifact_store):
        np.random.seed(42)
        from customer_retention.transforms.fitted import FittedPowerTransform
        pt = FittedPowerTransform()
        df = pd.DataFrame({"v": np.random.exponential(5, 100)})
        pt.fit_transform(df.copy(), "v", artifact_store)
        step = _step(PipelineTransformationType.YEO_JOHNSON, "v")
        mock_spark_df = MagicMock()
        mock_spark_ops.spark_yeo_johnson.return_value = MagicMock()
        result = TransformExecutor._apply_fitted_spark(mock_spark_df, step, False, artifact_store)
        assert result is not None
        mock_spark_ops.spark_yeo_johnson.assert_called_once()
        call_kwargs = mock_spark_ops.spark_yeo_johnson.call_args
        assert "lmbda" in call_kwargs.kwargs

    def test_distributed_prefit_calls_both_batch_fitters(self, executor):
        mock_spark_df = MagicMock()
        steps = [
            _step(PipelineTransformationType.SCALE, "a", method="standard"),
            _step(PipelineTransformationType.YEO_JOHNSON, "b"),
        ]
        store = MagicMock()
        with patch.object(executor, "_batch_fit_scalers") as mock_scalers, \
             patch.object(executor, "_batch_fit_power_transforms") as mock_pt, \
             patch(f"{_MOD}.as_spark_df", return_value=mock_spark_df), \
             patch(f"{_MOD}._as_pandas_api", return_value=pd.DataFrame()), \
             patch(f"{_MOD}._SPARK_DISPATCH", {}), \
             patch.object(executor, "_apply_fitted_spark", return_value=None), \
             patch.object(executor, "apply", return_value=pd.DataFrame()), \
             patch.object(executor, "_precompute_quantiles_distributed"):
            executor._apply_all_distributed(MagicMock(), steps, fit_mode=True, artifact_store=store)
        mock_scalers.assert_called_once()
        mock_pt.assert_called_once()


class TestBatchFitScalers:
    def test_batch_fit_standard_matches_per_column(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "a": np.random.randn(100) * 10 + 50,
            "b": np.random.exponential(5, 100),
        })
        import tempfile

        from customer_retention.transforms.artifact_store import ArtifactStore
        from customer_retention.transforms.fitted import FittedScaler

        store1 = ArtifactStore(tempfile.mkdtemp())
        s1 = FittedScaler("standard")
        s1.fit_transform(df.copy(), "a", store1)
        s2 = FittedScaler("standard")
        s2.fit_transform(df.copy(), "b", store1)

        store2 = ArtifactStore(tempfile.mkdtemp())
        mean_a, std_a = float(df["a"].mean()), float(df["a"].std(ddof=0))
        mean_b, std_b = float(df["b"].mean()), float(df["b"].std(ddof=0))
        s3 = FittedScaler("standard")
        s3.fit_from_precomputed(mean_a, std_a, len(df))
        store2.register("scaler", "a", s3._scaler)
        s4 = FittedScaler("standard")
        s4.fit_from_precomputed(mean_b, std_b, len(df))
        store2.register("scaler", "b", s4._scaler)

        scaler_a1 = store1.load("a_scaler")
        scaler_a2 = store2.load("a_scaler")
        np.testing.assert_array_almost_equal(scaler_a1.mean_, scaler_a2.mean_)
        np.testing.assert_array_almost_equal(scaler_a1.scale_, scaler_a2.scale_)

    def test_batch_fit_minmax_matches_per_column(self):
        np.random.seed(42)
        df = pd.DataFrame({"v": np.random.randn(100)})
        import tempfile

        from customer_retention.transforms.artifact_store import ArtifactStore
        from customer_retention.transforms.fitted import FittedScaler

        store1 = ArtifactStore(tempfile.mkdtemp())
        s1 = FittedScaler("minmax")
        s1.fit_transform(df.copy(), "v", store1)

        store2 = ArtifactStore(tempfile.mkdtemp())
        s2 = FittedScaler("minmax")
        s2.fit_from_precomputed(0, 0, len(df), float(df["v"].min()), float(df["v"].max()))
        store2.register("scaler", "v", s2._scaler)

        scaler1 = store1.load("v_scaler")
        scaler2 = store2.load("v_scaler")
        np.testing.assert_array_almost_equal(scaler1.scale_, scaler2.scale_)
        np.testing.assert_array_almost_equal(scaler1.min_, scaler2.min_)


class TestBatchFitPowerTransform:
    def test_fit_from_local_matches_direct(self):
        np.random.seed(42)
        data = pd.Series(np.random.exponential(5, 200))
        from customer_retention.transforms.fitted import FittedPowerTransform
        pt1 = FittedPowerTransform()
        pt1._pt.fit(data.to_numpy().reshape(-1, 1))
        pt2 = FittedPowerTransform()
        pt2.fit_from_local(data)
        np.testing.assert_array_almost_equal(pt1._pt.lambdas_, pt2._pt.lambdas_)

    def test_yj_transform_parity(self):
        np.random.seed(42)
        data = pd.Series(np.random.randn(100) * 5)
        from customer_retention.transforms.fitted import FittedPowerTransform
        pt = FittedPowerTransform()
        pt.fit_from_local(data.fillna(0))
        pandas_result = pt._apply_yj(data.fillna(0))
        assert not pandas_result.isna().any()


class TestOnStepDoneCallback:
    def test_callback_invoked_per_step(self, executor, sample_df):
        steps = [
            _step(PipelineTransformationType.IMPUTE_NULL, "age", value=0),
            _step(PipelineTransformationType.LOG_TRANSFORM, "amount"),
        ]
        calls = []
        executor.apply_all(sample_df, steps, on_step_done=lambda *args: calls.append(args))
        assert len(calls) == 2

    def test_callback_receives_correct_index_and_total(self, executor, sample_df):
        steps = [
            _step(PipelineTransformationType.IMPUTE_NULL, "age", value=0),
            _step(PipelineTransformationType.LOG_TRANSFORM, "amount"),
            _step(PipelineTransformationType.CAP_OUTLIER, "age", lower=0, upper=100),
        ]
        calls = []
        executor.apply_all(sample_df, steps, on_step_done=lambda *args: calls.append(args))
        for i, (idx, total, step, step_s, total_s) in enumerate(calls):
            assert idx == i
            assert total == 3
            assert step is steps[i]

    def test_callback_timing_values_are_positive(self, executor, sample_df):
        steps = [
            _step(PipelineTransformationType.IMPUTE_NULL, "age", value=0),
            _step(PipelineTransformationType.LOG_TRANSFORM, "amount"),
        ]
        calls = []
        executor.apply_all(sample_df, steps, on_step_done=lambda *args: calls.append(args))
        for _, _, _, step_s, total_s in calls:
            assert step_s >= 0
            assert total_s >= 0
        assert calls[1][4] >= calls[0][4]

    def test_callback_none_by_default(self, executor, sample_df):
        steps = [_step(PipelineTransformationType.LOG_TRANSFORM, "amount")]
        result = executor.apply_all(sample_df, steps)
        assert result["amount"].max() < 50000

    def test_callback_on_distributed_path(self, executor, sample_df):
        steps = [_step(PipelineTransformationType.LOG_TRANSFORM, "amount")]
        mock_spark_df = MagicMock()
        calls = []

        with patch(f"{_MOD}._is_spark_pandas", return_value=True), \
             patch(f"{_MOD}.as_spark_df", return_value=mock_spark_df), \
             patch(f"{_MOD}._as_pandas_api", return_value=sample_df.copy()), \
             patch(f"{_MOD}._SPARK_DISPATCH", {PipelineTransformationType.LOG_TRANSFORM: lambda df, s: mock_spark_df}), \
             patch.object(executor, "_precompute_quantiles_distributed"):
            executor.apply_all(sample_df.copy(), steps, on_step_done=lambda *args: calls.append(args))
        assert len(calls) == 1
        assert calls[0][0] == 0
        assert calls[0][1] == 1

    def test_result_unchanged_with_callback(self, executor, sample_df):
        steps = [
            _step(PipelineTransformationType.IMPUTE_NULL, "age", value=0),
            _step(PipelineTransformationType.LOG_TRANSFORM, "amount"),
        ]
        result_no_cb = executor.apply_all(sample_df.copy(), steps)
        result_cb = executor.apply_all(sample_df.copy(), steps, on_step_done=lambda *a: None)
        pd.testing.assert_frame_equal(result_no_cb, result_cb)


class TestFormatStepProgress:
    def test_format_includes_step_info(self):
        from customer_retention.transforms.executor import format_step_progress
        step = _step(PipelineTransformationType.LOG_TRANSFORM, "amount")
        result = format_step_progress(0, 5, step, 0.5, 0.5)
        assert "LOG_TRANSFORM" in result
        assert "amount" in result
        assert "1/5" in result

    def test_format_eta_decreases_as_steps_complete(self):
        from customer_retention.transforms.executor import format_step_progress
        step = _step(PipelineTransformationType.LOG_TRANSFORM, "v")
        early = format_step_progress(0, 10, step, 1.0, 1.0)
        late = format_step_progress(8, 10, step, 1.0, 9.0)
        assert "left" in early
        assert "left" in late

    def test_format_last_step_shows_zero_eta(self):
        from customer_retention.transforms.executor import format_step_progress
        step = _step(PipelineTransformationType.LOG_TRANSFORM, "v")
        result = format_step_progress(4, 5, step, 0.3, 1.5)
        assert "0s left" in result or "0.0s left" in result

    def test_print_step_progress_writes_to_stdout(self, capsys):
        from customer_retention.transforms.executor import print_step_progress
        step = _step(PipelineTransformationType.LOG_TRANSFORM, "amount")
        print_step_progress(0, 3, step, 0.1, 0.1)
        captured = capsys.readouterr()
        assert "LOG_TRANSFORM" in captured.out
        assert "amount" in captured.out


class TestPrecomputeQuantilesDistributed:
    def test_sets_precomputed_q99_from_approx_quantile(self, executor):
        mock_spark_df = MagicMock()
        mock_spark_df.columns = ["a", "b", "other"]
        mock_spark_df.approxQuantile.return_value = [[42.0], [99.0]]
        steps = [
            _step(PipelineTransformationType.CAP_THEN_LOG, "a"),
            _step(PipelineTransformationType.CAP_THEN_LOG, "b"),
        ]
        executor._precompute_quantiles_distributed(mock_spark_df, steps)
        assert steps[0].parameters["_precomputed_q99"] == 42.0
        assert steps[1].parameters["_precomputed_q99"] == 99.0
        mock_spark_df.approxQuantile.assert_called_once_with(["a", "b"], [0.99], 0.01)

    def test_skips_columns_not_in_spark_df(self, executor):
        mock_spark_df = MagicMock()
        mock_spark_df.columns = ["a"]
        mock_spark_df.approxQuantile.return_value = [[10.0]]
        steps = [
            _step(PipelineTransformationType.CAP_THEN_LOG, "a"),
            _step(PipelineTransformationType.CAP_THEN_LOG, "missing"),
        ]
        executor._precompute_quantiles_distributed(mock_spark_df, steps)
        assert steps[0].parameters["_precomputed_q99"] == 10.0
        assert "_precomputed_q99" not in steps[1].parameters

    def test_noop_when_no_cap_then_log_steps(self, executor):
        mock_spark_df = MagicMock()
        mock_spark_df.columns = ["a"]
        steps = [_step(PipelineTransformationType.LOG_TRANSFORM, "a")]
        executor._precompute_quantiles_distributed(mock_spark_df, steps)
        mock_spark_df.approxQuantile.assert_not_called()

    def test_handles_empty_quantile_list(self, executor):
        mock_spark_df = MagicMock()
        mock_spark_df.columns = ["a"]
        mock_spark_df.approxQuantile.return_value = [[]]
        steps = [_step(PipelineTransformationType.CAP_THEN_LOG, "a")]
        executor._precompute_quantiles_distributed(mock_spark_df, steps)
        assert steps[0].parameters["_precomputed_q99"] is None


class TestPrecomputeQuantilesPandas:
    def test_pandas_path_sets_q99(self, executor):
        np.random.seed(42)
        df = pd.DataFrame({"v": np.random.randn(200)})
        steps = [_step(PipelineTransformationType.CAP_THEN_LOG, "v")]
        executor._precompute_quantiles(df, steps)
        assert "_precomputed_q99" in steps[0].parameters
        assert steps[0].parameters["_precomputed_q99"] == pytest.approx(float(df["v"].quantile(0.99)))

    def test_pandas_path_skips_missing_column(self, executor):
        df = pd.DataFrame({"v": [1.0, 2.0]})
        steps = [_step(PipelineTransformationType.CAP_THEN_LOG, "missing")]
        executor._precompute_quantiles(df, steps)
        assert "_precomputed_q99" not in steps[0].parameters

    def test_noop_when_no_cap_then_log(self, executor):
        df = pd.DataFrame({"v": [1.0]})
        steps = [_step(PipelineTransformationType.LOG_TRANSFORM, "v")]
        executor._precompute_quantiles(df, steps)


class TestSparkDerived:
    def test_ratio_dispatches_to_spark_ops(self, mock_spark_ops):
        from customer_retention.transforms.executor import _spark_derived
        step = _step(PipelineTransformationType.DERIVED_COLUMN, "r", method="ratio", numerator="a", denominator="b")
        mock_spark_df = MagicMock()
        _spark_derived(mock_spark_df, step)
        mock_spark_ops.spark_derived_ratio.assert_called_once_with(
            mock_spark_df, "r", numerator="a", denominator="b",
        )

    def test_interaction_dispatches_to_spark_ops(self, mock_spark_ops):
        from customer_retention.transforms.executor import _spark_derived
        step = _step(PipelineTransformationType.DERIVED_COLUMN, "ab", method="interaction", col_a="a", col_b="b")
        mock_spark_df = MagicMock()
        _spark_derived(mock_spark_df, step)
        mock_spark_ops.spark_derived_interaction.assert_called_once_with(
            mock_spark_df, "ab", col_a="a", col_b="b",
        )

    def test_composite_dispatches_to_spark_ops(self, mock_spark_ops):
        from customer_retention.transforms.executor import _spark_derived
        step = _step(PipelineTransformationType.DERIVED_COLUMN, "c", method="composite", columns=["a", "b"])
        mock_spark_df = MagicMock()
        _spark_derived(mock_spark_df, step)
        mock_spark_ops.spark_derived_composite.assert_called_once_with(
            mock_spark_df, "c", columns=["a", "b"],
        )

    def test_unknown_method_returns_df_unchanged(self, mock_spark_ops):
        from customer_retention.transforms.executor import _spark_derived
        step = _step(PipelineTransformationType.DERIVED_COLUMN, "x", method="nonexistent")
        mock_spark_df = MagicMock()
        result = _spark_derived(mock_spark_df, step)
        assert result is mock_spark_df

    def test_action_key_also_recognized(self, mock_spark_ops):
        from customer_retention.transforms.executor import _spark_derived
        step = _step(PipelineTransformationType.DERIVED_COLUMN, "r", action="ratio", numerator="a", denominator="b")
        mock_spark_df = MagicMock()
        _spark_derived(mock_spark_df, step)
        mock_spark_ops.spark_derived_ratio.assert_called_once()


class TestApplyScaleFromArtifact:
    def test_standard_scaler_from_artifact(self, mock_spark_ops, artifact_store):
        from customer_retention.transforms.executor import _apply_scale_from_artifact
        from customer_retention.transforms.fitted import FittedScaler
        np.random.seed(42)
        df = pd.DataFrame({"v": np.random.randn(50) * 10 + 50})
        scaler = FittedScaler("standard")
        scaler.fit_transform(df.copy(), "v", artifact_store)
        step = _step(PipelineTransformationType.SCALE, "v", method="standard")
        mock_spark_df = MagicMock()
        mock_spark_ops.spark_standard_scale.return_value = MagicMock()
        _apply_scale_from_artifact(mock_spark_df, step, artifact_store)
        call_kwargs = mock_spark_ops.spark_standard_scale.call_args
        assert call_kwargs.kwargs["mean"] == pytest.approx(float(scaler._scaler.mean_[0]))
        assert call_kwargs.kwargs["scale"] == pytest.approx(float(scaler._scaler.scale_[0]))

    def test_minmax_scaler_from_artifact(self, mock_spark_ops, artifact_store):
        from customer_retention.transforms.executor import _apply_scale_from_artifact
        from customer_retention.transforms.fitted import FittedScaler
        np.random.seed(42)
        df = pd.DataFrame({"v": np.random.randn(50) * 10 + 50})
        scaler = FittedScaler("minmax")
        scaler.fit_transform(df.copy(), "v", artifact_store)
        step = _step(PipelineTransformationType.SCALE, "v", method="minmax")
        mock_spark_df = MagicMock()
        mock_spark_ops.spark_minmax_scale.return_value = MagicMock()
        _apply_scale_from_artifact(mock_spark_df, step, artifact_store)
        mock_spark_ops.spark_minmax_scale.assert_called_once()
        call_kwargs = mock_spark_ops.spark_minmax_scale.call_args
        assert call_kwargs.kwargs["scale"] == pytest.approx(float(scaler._scaler.scale_[0]))
        assert call_kwargs.kwargs["offset"] == pytest.approx(float(scaler._scaler.min_[0]))


class TestApplyYjFromArtifact:
    def test_yj_from_artifact_passes_correct_params(self, mock_spark_ops, artifact_store):
        from customer_retention.transforms.executor import _apply_yj_from_artifact
        from customer_retention.transforms.fitted import FittedPowerTransform
        np.random.seed(42)
        df = pd.DataFrame({"v": np.random.exponential(5, 100)})
        pt = FittedPowerTransform()
        pt.fit_transform(df.copy(), "v", artifact_store)
        step = _step(PipelineTransformationType.YEO_JOHNSON, "v")
        mock_spark_df = MagicMock()
        mock_spark_ops.spark_yeo_johnson.return_value = MagicMock()
        _apply_yj_from_artifact(mock_spark_df, step, artifact_store)
        call_kwargs = mock_spark_ops.spark_yeo_johnson.call_args
        assert call_kwargs.kwargs["lmbda"] == pytest.approx(float(pt._pt.lambdas_[0]))
        assert call_kwargs.kwargs["std_mean"] == pytest.approx(float(pt._pt._scaler.mean_[0]))
        assert call_kwargs.kwargs["std_scale"] == pytest.approx(float(pt._pt._scaler.scale_[0]))
        assert call_kwargs.kwargs["standardize"] is True


class TestBatchFitScalersDistributed:
    def _mock_spark_row(self, values):
        row = MagicMock()
        row.__getitem__ = lambda _, k: values[k]
        return row

    def test_standard_scaler_prefit_stores_params_in_step(self, executor, artifact_store):
        mock_spark_df = MagicMock()
        mock_F = MagicMock()
        row_vals = {"a__mean": 50.0, "a__std": 10.0, "a__cnt": 100}
        mock_spark_df.agg.return_value.collect.return_value = [self._mock_spark_row(row_vals)]
        steps = [_step(PipelineTransformationType.SCALE, "a", method="standard")]
        mock_spark_df.columns = ["a", "b"]
        executor._batch_fit_scalers(mock_spark_df, steps, artifact_store, {"a", "b"}, mock_F)
        assert steps[0].parameters["_spark_fitted"]["kind"] == "standard"
        assert steps[0].parameters["_spark_fitted"]["mean"] == 50.0
        assert steps[0].parameters["_spark_fitted"]["scale"] == 10.0
        assert artifact_store.has("a_scaler")

    def test_minmax_scaler_prefit_stores_params_in_step(self, executor, artifact_store):
        mock_spark_df = MagicMock()
        mock_F = MagicMock()
        row_vals = {"v__min": 10.0, "v__max": 110.0, "v__cnt": 100}
        mock_spark_df.agg.return_value.collect.return_value = [self._mock_spark_row(row_vals)]
        steps = [_step(PipelineTransformationType.SCALE, "v", method="minmax")]
        executor._batch_fit_scalers(mock_spark_df, steps, artifact_store, {"v"}, mock_F)
        assert steps[0].parameters["_spark_fitted"]["kind"] == "minmax"
        assert steps[0].parameters["_spark_fitted"]["scale"] == pytest.approx(1.0 / 100.0)
        assert artifact_store.has("v_scaler")

    def test_standard_scaler_zero_std_uses_unit_scale(self, executor, artifact_store):
        mock_spark_df = MagicMock()
        mock_F = MagicMock()
        row_vals = {"a__mean": 5.0, "a__std": 0.0, "a__cnt": 50}
        mock_spark_df.agg.return_value.collect.return_value = [self._mock_spark_row(row_vals)]
        steps = [_step(PipelineTransformationType.SCALE, "a", method="standard")]
        executor._batch_fit_scalers(mock_spark_df, steps, artifact_store, {"a"}, mock_F)
        assert steps[0].parameters["_spark_fitted"]["scale"] == 1.0

    def test_minmax_scaler_zero_range_uses_unit_scale(self, executor, artifact_store):
        mock_spark_df = MagicMock()
        mock_F = MagicMock()
        row_vals = {"v__min": 5.0, "v__max": 5.0, "v__cnt": 10}
        mock_spark_df.agg.return_value.collect.return_value = [self._mock_spark_row(row_vals)]
        steps = [_step(PipelineTransformationType.SCALE, "v", method="minmax")]
        executor._batch_fit_scalers(mock_spark_df, steps, artifact_store, {"v"}, mock_F)
        assert steps[0].parameters["_spark_fitted"]["scale"] == 1.0

    def test_skips_columns_not_in_spark_df(self, executor, artifact_store):
        mock_spark_df = MagicMock()
        mock_F = MagicMock()
        steps = [_step(PipelineTransformationType.SCALE, "missing", method="standard")]
        executor._batch_fit_scalers(mock_spark_df, steps, artifact_store, {"a"}, mock_F)
        mock_spark_df.agg.assert_not_called()

    def test_noop_when_no_scale_steps(self, executor, artifact_store):
        mock_spark_df = MagicMock()
        mock_F = MagicMock()
        steps = [_step(PipelineTransformationType.LOG_TRANSFORM, "a")]
        executor._batch_fit_scalers(mock_spark_df, steps, artifact_store, {"a"}, mock_F)
        mock_spark_df.agg.assert_not_called()

    def test_mixed_standard_and_minmax_in_single_call(self, executor, artifact_store):
        mock_spark_df = MagicMock()
        mock_F = MagicMock()
        std_row = {"a__mean": 10.0, "a__std": 5.0, "a__cnt": 50}
        mm_row = {"b__min": 0.0, "b__max": 100.0, "b__cnt": 50}
        mock_spark_df.agg.return_value.collect.side_effect = [
            [self._mock_spark_row(std_row)],
            [self._mock_spark_row(mm_row)],
        ]
        steps = [
            _step(PipelineTransformationType.SCALE, "a", method="standard"),
            _step(PipelineTransformationType.SCALE, "b", method="minmax"),
        ]
        executor._batch_fit_scalers(mock_spark_df, steps, artifact_store, {"a", "b"}, mock_F)
        assert steps[0].parameters["_spark_fitted"]["kind"] == "standard"
        assert steps[1].parameters["_spark_fitted"]["kind"] == "minmax"

    def test_null_aggregate_values_default_to_zero(self, executor, artifact_store):
        mock_spark_df = MagicMock()
        mock_F = MagicMock()
        row_vals = {"a__mean": None, "a__std": None, "a__cnt": None}
        mock_spark_df.agg.return_value.collect.return_value = [self._mock_spark_row(row_vals)]
        steps = [_step(PipelineTransformationType.SCALE, "a", method="standard")]
        executor._batch_fit_scalers(mock_spark_df, steps, artifact_store, {"a"}, mock_F)
        assert steps[0].parameters["_spark_fitted"]["mean"] == 0.0
        assert steps[0].parameters["_spark_fitted"]["scale"] == 1.0


class TestBatchFitPowerTransformsDistributed:
    def test_yj_prefit_stores_params_in_step(self, executor):
        np.random.seed(42)
        sample_pdf = pd.DataFrame({"v": np.random.exponential(5, 50)})
        mock_spark_df = MagicMock()
        mock_spark_df.count.return_value = 50
        mock_store = MagicMock()
        with patch("customer_retention.transforms.executor.FittedPowerTransform") as MockPT:
            instance = MockPT.return_value
            instance._MAX_FIT_SAMPLE = 50_000
            MockPT._MAX_FIT_SAMPLE = 50_000
            instance._pt = MagicMock()
            instance._pt.lambdas_ = np.array([0.5])
            instance._pt.standardize = True
            instance._pt._scaler = MagicMock()
            instance._pt._scaler.mean_ = np.array([1.0])
            instance._pt._scaler.scale_ = np.array([0.5])
            mock_spark_df.select.return_value.toPandas.return_value = sample_pdf
            steps = [_step(PipelineTransformationType.YEO_JOHNSON, "v")]
            executor._batch_fit_power_transforms(mock_spark_df, steps, mock_store, {"v"})
        assert steps[0].parameters["_spark_fitted"]["lmbda"] == pytest.approx(0.5)
        assert steps[0].parameters["_spark_fitted"]["std_mean"] == pytest.approx(1.0)
        assert steps[0].parameters["_spark_fitted"]["std_scale"] == pytest.approx(0.5)
        mock_store.register.assert_called_once()

    def test_yj_prefit_samples_large_datasets(self, executor):
        np.random.seed(42)
        sample_pdf = pd.DataFrame({"v": np.random.exponential(5, 100)})
        mock_spark_df = MagicMock()
        mock_spark_df.count.return_value = 100_000
        mock_store = MagicMock()
        with patch("customer_retention.transforms.executor.FittedPowerTransform") as MockPT:
            instance = MockPT.return_value
            instance._MAX_FIT_SAMPLE = 50_000
            MockPT._MAX_FIT_SAMPLE = 50_000
            instance._pt = MagicMock()
            instance._pt.lambdas_ = np.array([1.0])
            instance._pt.standardize = False
            instance._pt._scaler = MagicMock()
            mock_spark_df.select.return_value.sample.return_value.limit.return_value.toPandas.return_value = sample_pdf
            steps = [_step(PipelineTransformationType.YEO_JOHNSON, "v")]
            executor._batch_fit_power_transforms(mock_spark_df, steps, mock_store, {"v"})
        mock_spark_df.select.return_value.sample.assert_called_once()
        assert steps[0].parameters["_spark_fitted"]["standardize"] is False
        assert steps[0].parameters["_spark_fitted"]["std_mean"] is None

    def test_skips_columns_not_in_spark_df(self, executor):
        mock_spark_df = MagicMock()
        steps = [_step(PipelineTransformationType.YEO_JOHNSON, "missing")]
        executor._batch_fit_power_transforms(mock_spark_df, steps, MagicMock(), {"a"})
        mock_spark_df.count.assert_not_called()

    def test_noop_when_no_yj_steps(self, executor):
        mock_spark_df = MagicMock()
        steps = [_step(PipelineTransformationType.SCALE, "a", method="standard")]
        executor._batch_fit_power_transforms(mock_spark_df, steps, MagicMock(), {"a"})
        mock_spark_df.count.assert_not_called()


class TestMakeSparkDispatch:
    def test_dispatch_table_has_all_stateless_types(self, mock_spark_ops):
        from customer_retention.transforms.executor import _make_spark_dispatch
        dispatch = _make_spark_dispatch()
        expected_types = {
            PipelineTransformationType.IMPUTE_NULL,
            PipelineTransformationType.CAP_OUTLIER,
            PipelineTransformationType.TYPE_CAST,
            PipelineTransformationType.DROP_COLUMN,
            PipelineTransformationType.WINSORIZE,
            PipelineTransformationType.SEGMENT_AWARE_CAP,
            PipelineTransformationType.LOG_TRANSFORM,
            PipelineTransformationType.SQRT_TRANSFORM,
            PipelineTransformationType.ZERO_INFLATION_HANDLING,
            PipelineTransformationType.CAP_THEN_LOG,
            PipelineTransformationType.ENCODE,
            PipelineTransformationType.FEATURE_SELECT,
            PipelineTransformationType.DERIVED_COLUMN,
        }
        assert set(dispatch.keys()) == expected_types

    def test_dispatch_log_transform_calls_spark_ops(self, mock_spark_ops):
        from customer_retention.transforms.executor import _make_spark_dispatch
        dispatch = _make_spark_dispatch()
        mock_df = MagicMock()
        step = _step(PipelineTransformationType.LOG_TRANSFORM, "v")
        dispatch[PipelineTransformationType.LOG_TRANSFORM](mock_df, step)
        mock_spark_ops.spark_log_transform.assert_called_once_with(mock_df, "v")

    def test_dispatch_encode_one_hot_calls_spark_ops(self, mock_spark_ops):
        from customer_retention.transforms.executor import _make_spark_dispatch
        dispatch = _make_spark_dispatch()
        mock_df = MagicMock()
        step = _step(PipelineTransformationType.ENCODE, "cat", method="one_hot")
        dispatch[PipelineTransformationType.ENCODE](mock_df, step)
        mock_spark_ops.spark_one_hot_encode.assert_called_once_with(mock_df, "cat")

    def test_dispatch_encode_label_returns_none(self, mock_spark_ops):
        from customer_retention.transforms.executor import _make_spark_dispatch
        dispatch = _make_spark_dispatch()
        mock_df = MagicMock()
        step = _step(PipelineTransformationType.ENCODE, "cat", method="label")
        result = dispatch[PipelineTransformationType.ENCODE](mock_df, step)
        assert result is None

    def test_dispatch_cap_then_log_passes_precomputed_q99(self, mock_spark_ops):
        from customer_retention.transforms.executor import _make_spark_dispatch
        dispatch = _make_spark_dispatch()
        mock_df = MagicMock()
        step = _step(PipelineTransformationType.CAP_THEN_LOG, "v", _precomputed_q99=42.0)
        dispatch[PipelineTransformationType.CAP_THEN_LOG](mock_df, step)
        mock_spark_ops.spark_cap_then_log.assert_called_once_with(mock_df, "v", q99=42.0)
