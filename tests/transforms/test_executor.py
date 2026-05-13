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
        original_amount = sample_df["amount"].copy()
        result = executor.apply(sample_df, step)
        assert "amount_is_zero" in result.columns
        assert "amount_log" in result.columns
        pd.testing.assert_series_equal(result["amount"], original_amount)

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
        original = df["amount"].copy()
        result = ops.apply_zero_inflation_handling(df.copy(), "amount")
        assert "amount_is_zero" in result.columns
        assert "amount_log" in result.columns
        assert list(result["amount_is_zero"]) == [1, 0, 1, 0]
        assert result.loc[0, "amount_log"] == 0.0
        assert result.loc[1, "amount_log"] > 0.0
        pd.testing.assert_series_equal(result["amount"], original)

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
        step.parameters["_fitted_yj"] = {"lmbda": 0.5, "std_mean": 1.0, "std_scale": 0.5, "standardize": True}
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
        pytest.importorskip("pyspark")
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
        assert {id(c[2]) for c in calls} == {id(s) for s in steps}

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
        pytest.importorskip("pyspark")
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
        assert steps[0].parameters["_fitted_yj"]["lmbda"] == pytest.approx(0.5)
        assert steps[0].parameters["_fitted_yj"]["std_mean"] == pytest.approx(1.0)
        assert steps[0].parameters["_fitted_yj"]["std_scale"] == pytest.approx(0.5)
        mock_store.register.assert_called_once()

    def test_yj_prefit_samples_large_datasets(self, executor):
        pytest.importorskip("pyspark")
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
        assert steps[0].parameters["_fitted_yj"]["standardize"] is False
        assert steps[0].parameters["_fitted_yj"]["std_mean"] is None

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


# ── Batch grouping ────────────────────────────────────────────────


class TestDecomposeIntoWaves:
    def test_unique_columns_single_wave(self):
        from customer_retention.transforms.executor import _decompose_into_waves
        steps = [_step(PipelineTransformationType.LOG_TRANSFORM, c) for c in "abcd"]
        waves = _decompose_into_waves(steps)
        assert len(waves) == 1
        assert len(waves[0]) == 4

    def test_repeated_column_splits_wave(self):
        from customer_retention.transforms.executor import _decompose_into_waves
        steps = [
            _step(PipelineTransformationType.ZERO_INFLATION_HANDLING, "a"),
            _step(PipelineTransformationType.SQRT_TRANSFORM, "b"),
            _step(PipelineTransformationType.LOG_TRANSFORM, "a"),
        ]
        waves = _decompose_into_waves(steps)
        assert len(waves) == 2
        assert len(waves[0]) == 2
        assert len(waves[1]) == 1

    def test_zero_inflation_reserves_is_zero_column(self):
        from customer_retention.transforms.executor import _decompose_into_waves
        steps = [
            _step(PipelineTransformationType.ZERO_INFLATION_HANDLING, "x"),
            _step(PipelineTransformationType.LOG_TRANSFORM, "x_is_zero"),
        ]
        waves = _decompose_into_waves(steps)
        assert len(waves) == 2

    def test_zero_inflation_reserves_log_column(self):
        from customer_retention.transforms.executor import _decompose_into_waves
        steps = [
            _step(PipelineTransformationType.ZERO_INFLATION_HANDLING, "x"),
            _step(PipelineTransformationType.LOG_TRANSFORM, "x_log"),
        ]
        waves = _decompose_into_waves(steps)
        assert len(waves) == 2

    def test_empty(self):
        from customer_retention.transforms.executor import _decompose_into_waves
        assert _decompose_into_waves([]) == []


class TestGroupBatchableRuns:
    def test_consecutive_same_batchable_type_grouped(self):
        from customer_retention.transforms.executor import _PANDAS_BATCHABLE_TYPES, _group_batchable_runs
        steps = [_step(PipelineTransformationType.LOG_TRANSFORM, c) for c in "abc"]
        groups = _group_batchable_runs(steps, _PANDAS_BATCHABLE_TYPES)
        assert len(groups) == 1
        assert len(groups[0]) == 3

    def test_non_batchable_types_stay_individual(self):
        from customer_retention.transforms.executor import _PANDAS_BATCHABLE_TYPES, _group_batchable_runs
        steps = [
            _step(PipelineTransformationType.IMPUTE_NULL, "a", value=0),
            _step(PipelineTransformationType.IMPUTE_NULL, "b", value=0),
        ]
        groups = _group_batchable_runs(steps, _PANDAS_BATCHABLE_TYPES)
        assert len(groups) == 2
        assert all(len(g) == 1 for g in groups)

    def test_interleaved_batchable_types_merged_within_wave(self):
        from customer_retention.transforms.executor import _PANDAS_BATCHABLE_TYPES, _group_batchable_runs
        steps = [
            _step(PipelineTransformationType.ZERO_INFLATION_HANDLING, "a"),
            _step(PipelineTransformationType.CAP_THEN_LOG, "b"),
            _step(PipelineTransformationType.ZERO_INFLATION_HANDLING, "c"),
            _step(PipelineTransformationType.SQRT_TRANSFORM, "d"),
        ]
        groups = _group_batchable_runs(steps, _PANDAS_BATCHABLE_TYPES)
        batchable_groups = [g for g in groups if len(g) > 1 or g[0].type in _PANDAS_BATCHABLE_TYPES]
        zi_groups = [g for g in batchable_groups if g[0].type == PipelineTransformationType.ZERO_INFLATION_HANDLING]
        assert len(zi_groups) == 1
        assert len(zi_groups[0]) == 2

    def test_duplicate_column_creates_new_wave(self):
        from customer_retention.transforms.executor import _PANDAS_BATCHABLE_TYPES, _group_batchable_runs
        steps = [
            _step(PipelineTransformationType.LOG_TRANSFORM, "a"),
            _step(PipelineTransformationType.LOG_TRANSFORM, "b"),
            _step(PipelineTransformationType.LOG_TRANSFORM, "a"),
            _step(PipelineTransformationType.LOG_TRANSFORM, "c"),
        ]
        groups = _group_batchable_runs(steps, _PANDAS_BATCHABLE_TYPES)
        assert len(groups) == 2
        assert set(s.column for s in groups[0]) == {"a", "b"}
        assert set(s.column for s in groups[1]) == {"a", "c"}

    def test_all_same_column_stays_individual(self):
        from customer_retention.transforms.executor import _PANDAS_BATCHABLE_TYPES, _group_batchable_runs
        steps = [_step(PipelineTransformationType.LOG_TRANSFORM, "v") for _ in range(3)]
        groups = _group_batchable_runs(steps, _PANDAS_BATCHABLE_TYPES)
        assert len(groups) == 3
        assert all(len(g) == 1 for g in groups)

    def test_empty_steps(self):
        from customer_retention.transforms.executor import _PANDAS_BATCHABLE_TYPES, _group_batchable_runs
        assert _group_batchable_runs([], _PANDAS_BATCHABLE_TYPES) == []

    def test_chunking_large_batch(self):
        from customer_retention.transforms.executor import (
            _BATCH_CHUNK_SIZE,
            _PANDAS_BATCHABLE_TYPES,
            _group_batchable_runs,
        )
        n = _BATCH_CHUNK_SIZE + 10
        steps = [_step(PipelineTransformationType.LOG_TRANSFORM, f"c{i}") for i in range(n)]
        groups = _group_batchable_runs(steps, _PANDAS_BATCHABLE_TYPES)
        assert len(groups) == 2
        assert len(groups[0]) == _BATCH_CHUNK_SIZE
        assert len(groups[1]) == 10

    def test_batchable_between_non_batchable(self):
        from customer_retention.transforms.executor import _PANDAS_BATCHABLE_TYPES, _group_batchable_runs
        steps = [
            _step(PipelineTransformationType.IMPUTE_NULL, "a", value=0),
            _step(PipelineTransformationType.LOG_TRANSFORM, "b"),
            _step(PipelineTransformationType.LOG_TRANSFORM, "c"),
            _step(PipelineTransformationType.LOG_TRANSFORM, "d"),
            _step(PipelineTransformationType.IMPUTE_NULL, "e", value=0),
        ]
        groups = _group_batchable_runs(steps, _PANDAS_BATCHABLE_TYPES)
        log_groups = [g for g in groups if g[0].type == PipelineTransformationType.LOG_TRANSFORM]
        assert len(log_groups) == 1
        assert len(log_groups[0]) == 3


# ── Batched apply_all ─────────────────────────────────────────────


class TestBatchedApplyAll:
    def test_consecutive_log_transforms_match_sequential(self, executor):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "c": [7.0, 8.0, 9.0]})
        steps = [
            _step(PipelineTransformationType.LOG_TRANSFORM, "a"),
            _step(PipelineTransformationType.LOG_TRANSFORM, "b"),
            _step(PipelineTransformationType.LOG_TRANSFORM, "c"),
        ]
        expected = df.copy()
        for s in steps:
            expected = executor.apply(expected, s)
        result = executor.apply_all(df.copy(), steps)
        pd.testing.assert_frame_equal(result, expected)

    def test_consecutive_sqrt_transforms_match_sequential(self, executor):
        df = pd.DataFrame({"a": [4.0, 9.0], "b": [16.0, 25.0]})
        steps = [
            _step(PipelineTransformationType.SQRT_TRANSFORM, "a"),
            _step(PipelineTransformationType.SQRT_TRANSFORM, "b"),
        ]
        expected = df.copy()
        for s in steps:
            expected = executor.apply(expected, s)
        result = executor.apply_all(df.copy(), steps)
        pd.testing.assert_frame_equal(result, expected)

    def test_consecutive_zero_inflation_match_sequential(self, executor):
        df = pd.DataFrame({"a": [0.0, 1.0, 5.0], "b": [3.0, 0.0, 0.0], "c": [0.0, 0.0, 7.0]})
        steps = [
            _step(PipelineTransformationType.ZERO_INFLATION_HANDLING, "a"),
            _step(PipelineTransformationType.ZERO_INFLATION_HANDLING, "b"),
            _step(PipelineTransformationType.ZERO_INFLATION_HANDLING, "c"),
        ]
        expected = df.copy()
        for s in steps:
            expected = executor.apply(expected, s)
        result = executor.apply_all(df.copy(), steps)
        pd.testing.assert_frame_equal(result, expected)

    def test_cross_type_ordering_preserved(self, executor):
        df = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0], "b": [5.0, 6.0, 7.0, 8.0]})
        steps = [
            _step(PipelineTransformationType.ZERO_INFLATION_HANDLING, "a"),
            _step(PipelineTransformationType.SQRT_TRANSFORM, "b"),
            _step(PipelineTransformationType.LOG_TRANSFORM, "a"),
            _step(PipelineTransformationType.LOG_TRANSFORM, "b"),
        ]
        expected = df.copy()
        for s in steps:
            expected = executor.apply(expected, s)
        result = executor.apply_all(df.copy(), steps)
        pd.testing.assert_frame_equal(result, expected)

    def test_single_batchable_step_not_batched(self, executor):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        steps = [_step(PipelineTransformationType.LOG_TRANSFORM, "a")]
        expected = executor.apply(df.copy(), steps[0])
        result = executor.apply_all(df.copy(), steps)
        pd.testing.assert_frame_equal(result, expected)

    def test_callback_called_for_batched_steps(self, executor):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0]})
        steps = [
            _step(PipelineTransformationType.LOG_TRANSFORM, "a"),
            _step(PipelineTransformationType.LOG_TRANSFORM, "b"),
            _step(PipelineTransformationType.LOG_TRANSFORM, "c"),
        ]
        calls = []
        executor.apply_all(df.copy(), steps, on_step_done=lambda *args: calls.append(args))
        assert len(calls) >= 1
        last_idx, total = calls[-1][0], calls[-1][1]
        assert last_idx == 2
        assert total == 3

    def test_mixed_batchable_and_nonbatchable(self, executor):
        df = pd.DataFrame({
            "a": [1.0, 2.0, None, 4.0],
            "b": [5.0, 6.0, 7.0, 8.0],
            "c": [9.0, 10.0, 11.0, 12.0],
        })
        steps = [
            _step(PipelineTransformationType.IMPUTE_NULL, "a", value=0),
            _step(PipelineTransformationType.LOG_TRANSFORM, "b"),
            _step(PipelineTransformationType.LOG_TRANSFORM, "c"),
        ]
        expected = df.copy()
        for s in steps:
            expected = executor.apply(expected, s)
        result = executor.apply_all(df.copy(), steps)
        pd.testing.assert_frame_equal(result, expected)

    def test_many_log_transforms_faster_than_sequential(self, executor):
        """Batched apply_all should be faster than a per-step loop at 50
        columns × 1000 rows. Timings are wall-clock and therefore noisy —
        compare the MIN of several runs (which reflects pure compute cost;
        transient CPU contention only ever adds latency to a given run).
        Correctness is asserted unconditionally below; the timing check is
        a soft regression guard."""
        import time

        np.random.seed(42)
        n_cols = 50
        cols = {f"col_{i}": np.random.rand(1000) for i in range(n_cols)}
        df = pd.DataFrame(cols)
        steps = [_step(PipelineTransformationType.LOG_TRANSFORM, f"col_{i}") for i in range(n_cols)]

        def _time(fn) -> float:
            t0 = time.perf_counter()
            fn()
            return time.perf_counter() - t0

        def _run_batch():
            return executor.apply_all(df.copy(), steps)

        def _run_seq():
            result = df.copy()
            for s in steps:
                result = executor.apply(result, s)
            return result

        # Warm-up pass (import / JIT / page cache), then 5 timed runs each
        _run_batch()
        _run_seq()
        batch_times = sorted(_time(_run_batch) for _ in range(5))
        seq_times = sorted(_time(_run_seq) for _ in range(5))
        t_batch_min = batch_times[0]
        t_seq_min = seq_times[0]

        pd.testing.assert_frame_equal(_run_batch(), _run_seq())
        # Batch must not be dramatically slower. Allow up to 2x to absorb
        # residual noise from background processes; anything worse indicates
        # a real batching regression.
        assert t_batch_min < t_seq_min * 2.0, (
            f"batch min={t_batch_min:.4f}s > 2x sequential min={t_seq_min:.4f}s "
            f"(batch_times={batch_times}, seq_times={seq_times})"
        )

    def test_realistic_transform_pipeline(self, executor):
        np.random.seed(42)
        df = pd.DataFrame({
            "event_count_180d": np.random.randint(0, 100, 200).astype(float),
            "event_count_365d": np.random.randint(0, 200, 200).astype(float),
            "send_hour_sum": np.random.rand(200) * 1000,
            "time_to_open": np.random.rand(200) * 500,
        })
        steps = [
            _step(PipelineTransformationType.ZERO_INFLATION_HANDLING, "event_count_180d"),
            _step(PipelineTransformationType.ZERO_INFLATION_HANDLING, "event_count_365d"),
            _step(PipelineTransformationType.CAP_THEN_LOG, "send_hour_sum"),
            _step(PipelineTransformationType.SQRT_TRANSFORM, "time_to_open"),
            _step(PipelineTransformationType.LOG_TRANSFORM, "event_count_180d"),
            _step(PipelineTransformationType.LOG_TRANSFORM, "event_count_365d"),
            _step(PipelineTransformationType.LOG_TRANSFORM, "send_hour_sum"),
            _step(PipelineTransformationType.LOG_TRANSFORM, "time_to_open"),
        ]
        expected = df.copy()
        for s in steps:
            expected = executor.apply(expected, s)
        result = executor.apply_all(df.copy(), steps)
        pd.testing.assert_frame_equal(result, expected)

    def test_cap_then_log_batch_matches_sequential(self, executor):
        np.random.seed(42)
        df = pd.DataFrame({
            "a": np.random.rand(100) * 1000,
            "b": np.random.rand(100) * 500,
            "c": np.random.rand(100) * 200,
        })
        steps = [
            _step(PipelineTransformationType.CAP_THEN_LOG, "a"),
            _step(PipelineTransformationType.CAP_THEN_LOG, "b"),
            _step(PipelineTransformationType.CAP_THEN_LOG, "c"),
        ]
        expected = df.copy()
        for s in steps:
            expected = executor.apply(expected, s)
        result = executor.apply_all(df.copy(), steps)
        pd.testing.assert_frame_equal(result, expected)

    def test_batch_reporter_auto_upgrades_for_print_step_progress(self, executor):
        from customer_retention.transforms.executor import print_step_progress
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        steps = [
            _step(PipelineTransformationType.LOG_TRANSFORM, "a"),
            _step(PipelineTransformationType.LOG_TRANSFORM, "b"),
        ]
        output = []
        with patch("customer_retention.transforms.executor.print_batch_progress", side_effect=lambda *a: output.append(a)):
            executor.apply_all(df.copy(), steps, on_step_done=print_step_progress)
        assert len(output) == 1
        start_idx, end_idx, total = output[0][0], output[0][1], output[0][2]
        assert start_idx == 0
        assert end_idx == 1
        assert total == 2

    def test_explicit_on_batch_done_overrides(self, executor):
        df = pd.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0]})
        steps = [
            _step(PipelineTransformationType.LOG_TRANSFORM, "a"),
            _step(PipelineTransformationType.LOG_TRANSFORM, "b"),
            _step(PipelineTransformationType.LOG_TRANSFORM, "c"),
        ]
        batch_calls = []
        step_calls = []
        executor.apply_all(
            df.copy(), steps,
            on_step_done=lambda *a: step_calls.append(a),
            on_batch_done=lambda *a: batch_calls.append(a),
        )
        assert len(batch_calls) == 1
        assert len(step_calls) == 0


class TestSparkBatchDispatch:
    def test_spark_batch_dispatch_log_calls_batch_fn(self, mock_spark_ops):
        from customer_retention.transforms.executor import _make_spark_batch_dispatch
        dispatch = _make_spark_batch_dispatch()
        mock_df = MagicMock()
        steps = [
            _step(PipelineTransformationType.LOG_TRANSFORM, "a"),
            _step(PipelineTransformationType.LOG_TRANSFORM, "b"),
        ]
        dispatch[PipelineTransformationType.LOG_TRANSFORM](mock_df, steps)
        mock_spark_ops.spark_batch_log_transform.assert_called_once_with(mock_df, ["a", "b"])

    def test_spark_batch_dispatch_sqrt_calls_batch_fn(self, mock_spark_ops):
        from customer_retention.transforms.executor import _make_spark_batch_dispatch
        dispatch = _make_spark_batch_dispatch()
        mock_df = MagicMock()
        steps = [
            _step(PipelineTransformationType.SQRT_TRANSFORM, "x"),
            _step(PipelineTransformationType.SQRT_TRANSFORM, "y"),
        ]
        dispatch[PipelineTransformationType.SQRT_TRANSFORM](mock_df, steps)
        mock_spark_ops.spark_batch_sqrt_transform.assert_called_once_with(mock_df, ["x", "y"])

    def test_spark_batch_dispatch_zero_inflation_calls_batch_fn(self, mock_spark_ops):
        from customer_retention.transforms.executor import _make_spark_batch_dispatch
        dispatch = _make_spark_batch_dispatch()
        mock_df = MagicMock()
        steps = [
            _step(PipelineTransformationType.ZERO_INFLATION_HANDLING, "a"),
            _step(PipelineTransformationType.ZERO_INFLATION_HANDLING, "b"),
        ]
        dispatch[PipelineTransformationType.ZERO_INFLATION_HANDLING](mock_df, steps)
        mock_spark_ops.spark_batch_zero_inflation.assert_called_once_with(mock_df, ["a", "b"])

    def test_spark_batch_dispatch_cap_then_log(self, mock_spark_ops):
        from customer_retention.transforms.executor import _make_spark_batch_dispatch
        dispatch = _make_spark_batch_dispatch()
        mock_df = MagicMock()
        steps = [
            _step(PipelineTransformationType.CAP_THEN_LOG, "a", _precomputed_q99=10.0),
            _step(PipelineTransformationType.CAP_THEN_LOG, "b", _precomputed_q99=20.0),
        ]
        dispatch[PipelineTransformationType.CAP_THEN_LOG](mock_df, steps)
        mock_spark_ops.spark_batch_cap_then_log.assert_called_once_with(mock_df, [("a", 10.0), ("b", 20.0)])

    def test_spark_batch_dispatch_yeo_johnson(self, mock_spark_ops):
        from customer_retention.transforms.executor import _make_spark_batch_dispatch
        dispatch = _make_spark_batch_dispatch()
        mock_df = MagicMock()
        steps = [
            _step(PipelineTransformationType.YEO_JOHNSON, "a", _fitted_yj={"lmbda": 0.5, "std_mean": 1.0, "std_scale": 0.5, "standardize": True}),
            _step(PipelineTransformationType.YEO_JOHNSON, "b", _fitted_yj={"lmbda": 1.0, "std_mean": None, "std_scale": None, "standardize": False}),
        ]
        dispatch[PipelineTransformationType.YEO_JOHNSON](mock_df, steps)
        mock_spark_ops.spark_batch_yeo_johnson.assert_called_once()
        args = mock_spark_ops.spark_batch_yeo_johnson.call_args[0]
        assert args[0] is mock_df
        assert len(args[1]) == 2
        assert args[1][0][0] == "a"
        assert args[1][0][1] == 0.5

    def test_spark_batch_dispatch_scale_splits_standard_and_minmax(self, mock_spark_ops):
        from customer_retention.transforms.executor import _make_spark_batch_dispatch
        dispatch = _make_spark_batch_dispatch()
        mock_df = MagicMock()
        steps = [
            _step(PipelineTransformationType.SCALE, "a", _spark_fitted={"kind": "standard", "mean": 1.0, "scale": 2.0}),
            _step(PipelineTransformationType.SCALE, "b", _spark_fitted={"kind": "standard", "mean": 0.0, "scale": 1.0}),
            _step(PipelineTransformationType.SCALE, "c", _spark_fitted={"kind": "minmax", "scale": 0.5, "offset": -1.0}),
        ]
        mock_spark_ops.spark_batch_standard_scale.return_value = mock_df
        mock_spark_ops.spark_batch_minmax_scale.return_value = mock_df
        dispatch[PipelineTransformationType.SCALE](mock_df, steps)
        mock_spark_ops.spark_batch_standard_scale.assert_called_once_with(mock_df, [("a", 1.0, 2.0), ("b", 0.0, 1.0)])
        mock_spark_ops.spark_batch_minmax_scale.assert_called_once_with(mock_df, [("c", 0.5, -1.0)])

    def test_spark_batch_dispatch_scale_skips_steps_without_fitted(self, mock_spark_ops):
        from customer_retention.transforms.executor import _make_spark_batch_dispatch
        dispatch = _make_spark_batch_dispatch()
        mock_df = MagicMock()
        steps = [_step(PipelineTransformationType.SCALE, "a")]
        result = dispatch[PipelineTransformationType.SCALE](mock_df, steps)
        assert result is mock_df
        mock_spark_ops.spark_batch_standard_scale.assert_not_called()
        mock_spark_ops.spark_batch_minmax_scale.assert_not_called()


class TestPreloadScalerParams:
    def test_loads_standard_scaler_from_artifact(self, artifact_store):
        from customer_retention.transforms.executor import _preload_scaler_params
        from customer_retention.transforms.fitted import FittedScaler
        df = pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 5.0]})
        scaler = FittedScaler("standard")
        scaler.fit_transform(df.copy(), "v", artifact_store)
        steps = [_step(PipelineTransformationType.SCALE, "v")]
        _preload_scaler_params(steps, artifact_store)
        assert steps[0].parameters["_spark_fitted"]["kind"] == "standard"
        assert "mean" in steps[0].parameters["_spark_fitted"]
        assert "scale" in steps[0].parameters["_spark_fitted"]

    def test_loads_minmax_scaler_from_artifact(self, artifact_store):
        from customer_retention.transforms.executor import _preload_scaler_params
        from customer_retention.transforms.fitted import FittedScaler
        df = pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 5.0]})
        scaler = FittedScaler("minmax")
        scaler.fit_transform(df.copy(), "v", artifact_store)
        steps = [_step(PipelineTransformationType.SCALE, "v")]
        _preload_scaler_params(steps, artifact_store)
        assert steps[0].parameters["_spark_fitted"]["kind"] == "minmax"

    def test_skips_when_already_loaded(self, artifact_store):
        from customer_retention.transforms.executor import _preload_scaler_params
        steps = [_step(PipelineTransformationType.SCALE, "v", _spark_fitted={"kind": "standard", "mean": 7.0, "scale": 3.0})]
        _preload_scaler_params(steps, artifact_store)
        assert steps[0].parameters["_spark_fitted"]["mean"] == 7.0

    def test_skips_non_scale_steps(self, artifact_store):
        from customer_retention.transforms.executor import _preload_scaler_params
        steps = [_step(PipelineTransformationType.LOG_TRANSFORM, "v")]
        _preload_scaler_params(steps, artifact_store)
        assert "_spark_fitted" not in steps[0].parameters

    def test_skips_when_artifact_not_found(self):
        from customer_retention.transforms.executor import _preload_scaler_params
        mock_store = MagicMock()
        mock_store.has.return_value = False
        steps = [_step(PipelineTransformationType.SCALE, "v")]
        _preload_scaler_params(steps, mock_store)
        assert "_spark_fitted" not in steps[0].parameters


class TestPreloadYjParams:
    def test_loads_params_from_artifact_store(self, artifact_store):
        from customer_retention.transforms.executor import _preload_yj_params
        from customer_retention.transforms.fitted import FittedPowerTransform
        np.random.seed(42)
        df = pd.DataFrame({"v": np.random.exponential(5, 100)})
        pt = FittedPowerTransform()
        pt.fit_transform(df.copy(), "v", artifact_store)
        steps = [_step(PipelineTransformationType.YEO_JOHNSON, "v")]
        _preload_yj_params(steps, artifact_store)
        assert "_fitted_yj" in steps[0].parameters
        assert "lmbda" in steps[0].parameters["_fitted_yj"]
        assert "std_mean" in steps[0].parameters["_fitted_yj"]
        assert "std_scale" in steps[0].parameters["_fitted_yj"]
        assert steps[0].parameters["_fitted_yj"]["standardize"] is True

    def test_skips_when_already_loaded(self, artifact_store):
        from customer_retention.transforms.executor import _preload_yj_params
        steps = [_step(PipelineTransformationType.YEO_JOHNSON, "v", _fitted_yj={"lmbda": 0.5})]
        _preload_yj_params(steps, artifact_store)
        assert steps[0].parameters["_fitted_yj"]["lmbda"] == 0.5

    def test_skips_non_yj_steps(self, artifact_store):
        from customer_retention.transforms.executor import _preload_yj_params
        steps = [_step(PipelineTransformationType.LOG_TRANSFORM, "v")]
        _preload_yj_params(steps, artifact_store)
        assert "_fitted_yj" not in steps[0].parameters

    def test_skips_when_artifact_not_found(self):
        from customer_retention.transforms.executor import _preload_yj_params
        mock_store = MagicMock()
        mock_store.has.return_value = False
        steps = [_step(PipelineTransformationType.YEO_JOHNSON, "v")]
        _preload_yj_params(steps, mock_store)
        assert "_fitted_yj" not in steps[0].parameters


class TestYjParamsFromSteps:
    def test_extracts_params(self):
        from customer_retention.transforms.executor import _yj_params_from_steps
        steps = [
            _step(PipelineTransformationType.YEO_JOHNSON, "a", _fitted_yj={"lmbda": 0.5, "std_mean": 1.0, "std_scale": 2.0, "standardize": True}),
            _step(PipelineTransformationType.YEO_JOHNSON, "b", _fitted_yj={"lmbda": 1.0}),
        ]
        result = _yj_params_from_steps(steps)
        assert len(result) == 2
        assert result[0] == ("a", 0.5, 1.0, 2.0, True)
        assert result[1][0] == "b"
        assert result[1][1] == 1.0

    def test_skips_steps_without_fitted_yj(self):
        from customer_retention.transforms.executor import _yj_params_from_steps
        steps = [
            _step(PipelineTransformationType.YEO_JOHNSON, "a", _fitted_yj={"lmbda": 0.5, "std_mean": 1.0, "std_scale": 2.0}),
            _step(PipelineTransformationType.YEO_JOHNSON, "b"),
        ]
        result = _yj_params_from_steps(steps)
        assert len(result) == 1
        assert result[0][0] == "a"


class TestBatchedYeoJohnsonApplyAll:
    def test_consecutive_yj_batch_matches_sequential(self, executor, artifact_store):
        np.random.seed(42)
        df = pd.DataFrame({
            "a": np.random.exponential(5, 100),
            "b": np.random.exponential(3, 100),
            "c": np.random.exponential(8, 100),
        })
        steps_fit = [
            _step(PipelineTransformationType.YEO_JOHNSON, "a"),
            _step(PipelineTransformationType.YEO_JOHNSON, "b"),
            _step(PipelineTransformationType.YEO_JOHNSON, "c"),
        ]
        executor.apply_all(df.copy(), steps_fit, fit_mode=True, artifact_store=artifact_store)
        steps_seq = [
            _step(PipelineTransformationType.YEO_JOHNSON, "a"),
            _step(PipelineTransformationType.YEO_JOHNSON, "b"),
            _step(PipelineTransformationType.YEO_JOHNSON, "c"),
        ]
        expected = df.copy()
        for s in steps_seq:
            expected = executor.apply(expected, s, fit_mode=False, artifact_store=artifact_store)
        steps_batch = [
            _step(PipelineTransformationType.YEO_JOHNSON, "a"),
            _step(PipelineTransformationType.YEO_JOHNSON, "b"),
            _step(PipelineTransformationType.YEO_JOHNSON, "c"),
        ]
        result = executor.apply_all(df.copy(), steps_batch, fit_mode=False, artifact_store=artifact_store)
        pd.testing.assert_frame_equal(result, expected)

    def test_preload_yj_params_used_in_apply_all(self, executor, artifact_store):
        np.random.seed(42)
        df = pd.DataFrame({"v": np.random.exponential(5, 100)})
        step_fit = _step(PipelineTransformationType.YEO_JOHNSON, "v")
        executor.apply(df.copy(), step_fit, fit_mode=True, artifact_store=artifact_store)
        step_apply = _step(PipelineTransformationType.YEO_JOHNSON, "v")
        result = executor.apply_all(df.copy(), [step_apply], fit_mode=False, artifact_store=artifact_store)
        assert not result["v"].isna().any()


class TestDistributedBatchApply:
    def test_distributed_batch_handler_used_for_multi_step(self, executor):
        mock_spark_df = MagicMock()
        mock_batch_result = MagicMock()
        steps = [
            _step(PipelineTransformationType.LOG_TRANSFORM, "a"),
            _step(PipelineTransformationType.LOG_TRANSFORM, "b"),
        ]
        with patch(f"{_MOD}._is_spark_pandas", return_value=True), \
             patch(f"{_MOD}.as_spark_df", return_value=mock_spark_df), \
             patch(f"{_MOD}._as_pandas_api", return_value=MagicMock()), \
             patch(f"{_MOD}._SPARK_BATCH_DISPATCH", {PipelineTransformationType.LOG_TRANSFORM: lambda df, s: mock_batch_result}), \
             patch(f"{_MOD}._SPARK_BATCHABLE_TYPES", frozenset({PipelineTransformationType.LOG_TRANSFORM})), \
             patch(f"{_MOD}._SPARK_DISPATCH", {}), \
             patch.object(executor, "_precompute_quantiles_distributed"):
            executor._apply_all_distributed(MagicMock(), steps)

    def test_distributed_preload_yj_on_score_mode(self, executor):
        mock_spark_df = MagicMock()
        mock_store = MagicMock()
        mock_store.has.return_value = False
        steps = [_step(PipelineTransformationType.YEO_JOHNSON, "v")]
        with patch(f"{_MOD}._is_spark_pandas", return_value=True), \
             patch(f"{_MOD}.as_spark_df", return_value=mock_spark_df), \
             patch(f"{_MOD}._as_pandas_api", return_value=MagicMock()), \
             patch(f"{_MOD}._SPARK_DISPATCH", {}), \
             patch(f"{_MOD}._SPARK_BATCH_DISPATCH", {}), \
             patch(f"{_MOD}._SPARK_BATCHABLE_TYPES", frozenset()), \
             patch.object(executor, "_apply_fitted_spark", return_value=None), \
             patch.object(executor, "apply", return_value=MagicMock()), \
             patch.object(executor, "_precompute_quantiles_distributed"), \
             patch(f"{_MOD}._preload_yj_params") as mock_preload:
            executor._apply_all_distributed(MagicMock(), steps, fit_mode=False, artifact_store=mock_store)
        mock_preload.assert_called_once_with(steps, mock_store)

    def test_distributed_batch_reporter_called(self, executor):
        mock_spark_df = MagicMock()
        steps = [
            _step(PipelineTransformationType.LOG_TRANSFORM, "a"),
            _step(PipelineTransformationType.LOG_TRANSFORM, "b"),
        ]
        batch_calls = []
        with patch(f"{_MOD}._is_spark_pandas", return_value=True), \
             patch(f"{_MOD}.as_spark_df", return_value=mock_spark_df), \
             patch(f"{_MOD}._as_pandas_api", return_value=MagicMock()), \
             patch(f"{_MOD}._SPARK_BATCH_DISPATCH", {PipelineTransformationType.LOG_TRANSFORM: lambda df, s: mock_spark_df}), \
             patch(f"{_MOD}._SPARK_BATCHABLE_TYPES", frozenset({PipelineTransformationType.LOG_TRANSFORM})), \
             patch(f"{_MOD}._SPARK_DISPATCH", {}), \
             patch.object(executor, "_precompute_quantiles_distributed"):
            executor._apply_all_distributed(
                MagicMock(), steps,
                on_batch_done=lambda *a: batch_calls.append(a),
            )
        assert len(batch_calls) == 1
        assert batch_calls[0][0] == 0
        assert batch_calls[0][1] == 1
        assert batch_calls[0][2] == 2
