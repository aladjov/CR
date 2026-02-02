import numpy as np
import pandas as pd
import pytest

from customer_retention.generators.pipeline_generator.models import (
    PipelineTransformationType,
    TransformationStep,
)
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

    def test_unknown_type_raises(self, executor, sample_df):
        step = _step(PipelineTransformationType.AGGREGATE, "age")
        with pytest.raises(ValueError, match="Unknown transformation"):
            executor.apply(sample_df, step)
