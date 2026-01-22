import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from customer_retention.core.config import ColumnType
from customer_retention.stages.cleaning import (
    MissingValueHandler, ImputationStrategy,
    OutlierHandler, OutlierDetectionMethod, OutlierTreatmentStrategy
)
from customer_retention.stages.transformation import (
    NumericTransformer, ScalingStrategy, PowerTransform,
    CategoricalEncoder, EncodingStrategy,
    DatetimeTransformer, BinaryHandler,
    TransformationPipeline
)


@pytest.fixture
def retail_data():
    retail_path = Path(__file__).parent.parent / "fixtures" / "customer_retention_retail.csv"
    return pd.read_csv(retail_path)


@pytest.fixture
def retail_column_types():
    return {
        "custid": ColumnType.IDENTIFIER,
        "created": ColumnType.DATETIME,
        "firstorder": ColumnType.DATETIME,
        "lastorder": ColumnType.DATETIME,
        "esent": ColumnType.NUMERIC_DISCRETE,
        "eopenrate": ColumnType.NUMERIC_CONTINUOUS,
        "eclickrate": ColumnType.NUMERIC_CONTINUOUS,
        "avgorder": ColumnType.NUMERIC_CONTINUOUS,
        "ordfreq": ColumnType.NUMERIC_CONTINUOUS,
        "paperless": ColumnType.BINARY,
        "refill": ColumnType.BINARY,
        "doorstep": ColumnType.BINARY,
        "favday": ColumnType.CATEGORICAL_CYCLICAL,
        "city": ColumnType.CATEGORICAL_NOMINAL,
        "retained": ColumnType.TARGET
    }


class TestMissingValueHandlingRetail:
    def test_median_imputation_on_numeric(self, retail_data):
        series = retail_data["avgorder"].copy()
        series.iloc[:10] = None

        handler = MissingValueHandler(strategy=ImputationStrategy.MEDIAN)
        result = handler.fit_transform(series)

        assert result.series.isna().sum() == 0
        assert result.values_imputed == 10

    def test_mode_imputation_on_categorical(self, retail_data):
        series = retail_data["city"].copy()
        series.iloc[:5] = None

        handler = MissingValueHandler(strategy=ImputationStrategy.MODE)
        result = handler.fit_transform(series)

        assert result.series.isna().sum() == 0
        assert result.values_imputed == 5


class TestOutlierTreatmentRetail:
    def test_iqr_capping_on_avgorder(self, retail_data):
        handler = OutlierHandler(
            detection_method=OutlierDetectionMethod.IQR,
            treatment_strategy=OutlierTreatmentStrategy.CAP_IQR
        )
        result = handler.fit_transform(retail_data["avgorder"])

        original_max = retail_data["avgorder"].max()
        assert result.series.max() <= original_max
        assert result.outliers_detected >= 0

    def test_percentile_capping_on_eopenrate(self, retail_data):
        handler = OutlierHandler(
            detection_method=OutlierDetectionMethod.PERCENTILE,
            treatment_strategy=OutlierTreatmentStrategy.CAP_PERCENTILE,
            percentile_lower=1, percentile_upper=99
        )
        result = handler.fit_transform(retail_data["eopenrate"])

        assert result.series.min() >= result.lower_bound
        assert result.series.max() <= result.upper_bound


class TestNumericTransformationRetail:
    def test_standard_scaling_on_avgorder(self, retail_data):
        transformer = NumericTransformer(scaling=ScalingStrategy.STANDARD)
        result = transformer.fit_transform(retail_data["avgorder"])

        assert result.series.mean() == pytest.approx(0.0, abs=1e-10)
        assert result.series.std(ddof=0) == pytest.approx(1.0, abs=1e-10)

    def test_log_transform_on_skewed_data(self, retail_data):
        positive_data = retail_data["avgorder"][retail_data["avgorder"] > 0]
        transformer = NumericTransformer(
            power_transform=PowerTransform.LOG1P,
            scaling=ScalingStrategy.STANDARD
        )
        result = transformer.fit_transform(positive_data)

        assert result.series is not None
        assert len(result.transformations_applied) == 2


class TestCategoricalEncodingRetail:
    def test_one_hot_encoding_city(self, retail_data):
        encoder = CategoricalEncoder(strategy=EncodingStrategy.ONE_HOT, drop_first=True)
        result = encoder.fit_transform(retail_data["city"])

        assert result.df is not None
        n_unique = retail_data["city"].nunique()
        assert len(result.columns_created) == n_unique - 1

    def test_cyclical_encoding_favday(self, retail_data):
        encoder = CategoricalEncoder(strategy=EncodingStrategy.CYCLICAL, period=7)
        result = encoder.fit_transform(retail_data["favday"])

        assert result.df is not None
        assert len(result.columns_created) == 2
        assert result.df[result.columns_created[0]].min() >= -1
        assert result.df[result.columns_created[0]].max() <= 1


class TestDatetimeTransformationRetail:
    def test_datetime_extraction_created(self, retail_data):
        transformer = DatetimeTransformer(
            extract_features=["year", "month", "day_of_week"],
            reference_date="2024-01-01"
        )
        result = transformer.fit_transform(retail_data["created"])

        assert "year" in result.df.columns
        assert "month" in result.df.columns
        assert "day_of_week" in result.df.columns
        assert "days_since" in result.df.columns


class TestBinaryHandlingRetail:
    def test_binary_standardization_paperless(self, retail_data):
        handler = BinaryHandler()
        result = handler.fit_transform(retail_data["paperless"])

        assert set(result.series.dropna().unique()).issubset({0, 1})

    def test_binary_standardization_refill(self, retail_data):
        handler = BinaryHandler()
        result = handler.fit_transform(retail_data["refill"])

        assert set(result.series.dropna().unique()).issubset({0, 1})


class TestFullPipelineRetail:
    def test_pipeline_end_to_end(self, retail_data, retail_column_types):
        pipeline = TransformationPipeline(column_types=retail_column_types)
        result = pipeline.fit_transform(retail_data)

        assert result.df is not None
        assert len(result.df) == len(retail_data)
        assert result.validation_passed is True

    def test_pipeline_drops_identifier(self, retail_data, retail_column_types):
        pipeline = TransformationPipeline(column_types=retail_column_types)
        result = pipeline.fit_transform(retail_data)

        assert "custid" not in result.df.columns

    def test_pipeline_preserves_target(self, retail_data, retail_column_types):
        pipeline = TransformationPipeline(column_types=retail_column_types)
        result = pipeline.fit_transform(retail_data)

        assert "retained" in result.df.columns
        pd.testing.assert_series_equal(
            result.df["retained"], retail_data["retained"], check_names=False
        )

    def test_pipeline_no_nulls(self, retail_data, retail_column_types):
        pipeline = TransformationPipeline(column_types=retail_column_types)
        result = pipeline.fit_transform(retail_data)

        non_target = result.df.drop(columns=["retained"], errors='ignore')
        assert non_target.isna().sum().sum() == 0

    def test_pipeline_no_infinities(self, retail_data, retail_column_types):
        pipeline = TransformationPipeline(column_types=retail_column_types)
        result = pipeline.fit_transform(retail_data)

        numeric_cols = result.df.select_dtypes(include=[np.number]).columns
        assert not np.isinf(result.df[numeric_cols].values).any()

    def test_pipeline_manifest_completeness(self, retail_data, retail_column_types):
        pipeline = TransformationPipeline(column_types=retail_column_types)
        result = pipeline.fit_transform(retail_data)

        assert result.manifest.timestamp is not None
        assert result.manifest.input_rows == 30801
        assert result.manifest.output_rows == 30801
        assert len(result.manifest.columns_dropped) > 0
        assert "custid" in result.manifest.columns_dropped

    def test_pipeline_execution_order(self, retail_data, retail_column_types):
        pipeline = TransformationPipeline(column_types=retail_column_types)
        result = pipeline.fit_transform(retail_data)

        expected_order = [
            "drop_columns", "handle_missing", "treat_outliers",
            "transform_datetime", "transform_numeric",
            "encode_categorical", "standardize_binary", "validate"
        ]
        assert result.manifest.execution_order == expected_order

    def test_pipeline_completes_in_reasonable_time(self, retail_data, retail_column_types):
        import time

        pipeline = TransformationPipeline(column_types=retail_column_types)

        start = time.time()
        result = pipeline.fit_transform(retail_data)
        elapsed = time.time() - start

        assert elapsed < 60
        assert result.df is not None


class TestPipelineFitTransformSeparation:
    def test_fit_on_train_transform_on_test(self, retail_data, retail_column_types):
        train = retail_data.iloc[:20000]
        test = retail_data.iloc[20000:]

        pipeline = TransformationPipeline(column_types=retail_column_types)
        pipeline.fit(train)
        result = pipeline.transform(test)

        assert result.df is not None
        assert len(result.df) == len(test)


class TestTransformationQualityGate:
    def test_tq001_no_nulls(self, retail_data, retail_column_types):
        pipeline = TransformationPipeline(column_types=retail_column_types)
        result = pipeline.fit_transform(retail_data)

        non_target = result.df.drop(columns=["retained"], errors='ignore')
        assert non_target.isna().sum().sum() == 0, "TQ001 Failed: Null values found"

    def test_tq002_no_infinities(self, retail_data, retail_column_types):
        pipeline = TransformationPipeline(column_types=retail_column_types)
        result = pipeline.fit_transform(retail_data)

        numeric = result.df.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.values).any(), "TQ002 Failed: Infinite values found"

    def test_tq007_scaling_successful(self, retail_data, retail_column_types):
        pipeline = TransformationPipeline(column_types=retail_column_types)
        result = pipeline.fit_transform(retail_data)

        for col in ["eopenrate", "eclickrate", "avgorder", "ordfreq"]:
            if col in result.df.columns:
                mean = result.df[col].mean()
                std = result.df[col].std()
                assert abs(mean) < 0.5, f"TQ007 Failed: {col} mean not ≈ 0"
                assert 0.5 < std < 1.5, f"TQ007 Failed: {col} std not ≈ 1"

    def test_tq003_no_constant_columns(self, retail_data, retail_column_types):
        pipeline = TransformationPipeline(column_types=retail_column_types)
        result = pipeline.fit_transform(retail_data)

        non_target = result.df.drop(columns=["retained"], errors='ignore')
        for col in non_target.columns:
            assert non_target[col].nunique() > 1, f"TQ003 Failed: {col} is constant"

    def test_tq004_numeric_in_valid_range(self, retail_data, retail_column_types):
        pipeline = TransformationPipeline(column_types=retail_column_types)
        result = pipeline.fit_transform(retail_data)

        numeric = result.df.select_dtypes(include=[np.number])
        for col in numeric.columns:
            max_val = numeric[col].abs().max()
            assert max_val < 1e10, f"TQ004 Failed: {col} has extreme values"

    def test_tq008_binary_values_valid(self, retail_data, retail_column_types):
        pipeline = TransformationPipeline(column_types=retail_column_types)
        result = pipeline.fit_transform(retail_data)

        for col in ["paperless", "refill", "doorstep"]:
            if col in result.df.columns:
                unique_vals = set(result.df[col].dropna().unique())
                assert unique_vals.issubset({0, 1, 0.0, 1.0}), f"TQ008 Failed: {col} has non-binary values"


class TestAcceptanceCriteria:
    def test_ac3_1_all_simple_strategies_implemented(self):
        from customer_retention.stages.cleaning import ImputationStrategy
        simple_strategies = [
            ImputationStrategy.MEAN, ImputationStrategy.MEDIAN,
            ImputationStrategy.MODE, ImputationStrategy.CONSTANT,
            ImputationStrategy.DROP_ROW
        ]
        for strategy in simple_strategies:
            handler = MissingValueHandler(strategy=strategy, fill_value=0)
            assert handler.strategy == strategy

    def test_ac3_6_iqr_detection_correct(self, retail_data):
        handler = OutlierHandler(detection_method=OutlierDetectionMethod.IQR)
        series = retail_data["avgorder"]
        result = handler.detect(series)

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        expected_lower = q1 - 1.5 * iqr
        expected_upper = q3 + 1.5 * iqr

        assert result.lower_bound == pytest.approx(expected_lower, rel=0.01)
        assert result.upper_bound == pytest.approx(expected_upper, rel=0.01)

    def test_ac3_11_standard_scaling_mean_zero(self, retail_data):
        transformer = NumericTransformer(scaling=ScalingStrategy.STANDARD)
        result = transformer.fit_transform(retail_data["avgorder"])

        assert result.series.mean() == pytest.approx(0.0, abs=1e-10)

    def test_ac3_12_minmax_scaling_range(self, retail_data):
        transformer = NumericTransformer(scaling=ScalingStrategy.MINMAX)
        result = transformer.fit_transform(retail_data["avgorder"])

        assert result.series.min() == pytest.approx(0.0)
        assert result.series.max() == pytest.approx(1.0)

    def test_ac3_16_one_hot_correct_columns(self, retail_data):
        encoder = CategoricalEncoder(strategy=EncodingStrategy.ONE_HOT, drop_first=True)
        result = encoder.fit_transform(retail_data["city"])

        n_unique = retail_data["city"].nunique()
        assert len(result.columns_created) == n_unique - 1

    def test_ac3_17_cyclical_valid_sin_cos(self, retail_data):
        encoder = CategoricalEncoder(strategy=EncodingStrategy.CYCLICAL, period=7)
        result = encoder.fit_transform(retail_data["favday"])

        sin_col = [c for c in result.df.columns if "_sin" in c][0]
        cos_col = [c for c in result.df.columns if "_cos" in c][0]

        assert result.df[sin_col].min() >= -1
        assert result.df[sin_col].max() <= 1
        assert result.df[cos_col].min() >= -1
        assert result.df[cos_col].max() <= 1

    def test_ac3_21_pipeline_correct_execution_order(self, retail_data, retail_column_types):
        pipeline = TransformationPipeline(column_types=retail_column_types)
        result = pipeline.fit_transform(retail_data)

        expected = [
            "drop_columns", "handle_missing", "treat_outliers",
            "transform_datetime", "transform_numeric",
            "encode_categorical", "standardize_binary", "validate"
        ]
        assert result.manifest.execution_order == expected

    def test_ac3_22_manifest_captures_all_transforms(self, retail_data, retail_column_types):
        pipeline = TransformationPipeline(column_types=retail_column_types)
        result = pipeline.fit_transform(retail_data)

        assert result.manifest.columns_dropped is not None
        assert result.manifest.numeric_transformations is not None
        assert result.manifest.categorical_encodings is not None
        assert result.manifest.datetime_transformations is not None
        assert result.manifest.binary_mappings is not None
        assert result.manifest.column_mapping is not None

    def test_ac3_23_pipeline_reapply_to_new_data(self, retail_data, retail_column_types):
        train = retail_data.iloc[:20000]
        test = retail_data.iloc[20000:]

        pipeline = TransformationPipeline(column_types=retail_column_types)
        pipeline.fit(train)

        train_result = pipeline.transform(train)
        test_result = pipeline.transform(test)

        assert train_result.df.columns.tolist() == test_result.df.columns.tolist()
