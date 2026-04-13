"""Tests for SegmentAwareOutlierAnalyzer Spark (distributed) path."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyspark")

pytestmark = pytest.mark.spark

from pyspark.sql import SparkSession

from customer_retention.core.compat import as_pandas_api
from customer_retention.stages.cleaning import OutlierDetectionMethod
from customer_retention.stages.profiling import SegmentAwareOutlierAnalyzer


@pytest.fixture(scope="module")
def spark():
    try:
        return (
            SparkSession.builder
            .master("local[2]")
            .appName("test_segment_aware_outlier_spark")
            .config("spark.sql.shuffle.partitions", "2")
            .getOrCreate()
        )
    except RuntimeError:
        pytest.skip("local Spark session not available")


def _to_psdf(spark, pdf):
    return as_pandas_api(spark.createDataFrame(pdf))


# ---- explicit segment: fully distributed path ----


class TestSparkExplicitSegmentBounds:
    def test_global_bounds_match_pandas(self, spark):
        np.random.seed(42)
        pdf = pd.DataFrame({
            "segment": ["A"] * 50 + ["B"] * 50,
            "value": np.concatenate([
                np.random.normal(10, 2, 50),
                np.random.normal(100, 10, 50),
            ]),
        })
        psdf = _to_psdf(spark, pdf)

        analyzer = SegmentAwareOutlierAnalyzer()
        spark_result = analyzer.analyze(psdf, feature_cols=["value"], segment_col="segment")
        pandas_result = analyzer.analyze(pdf, feature_cols=["value"], segment_col="segment")

        sg = spark_result.global_analysis["value"]
        pg = pandas_result.global_analysis["value"]
        assert abs(sg.lower_bound - pg.lower_bound) < 1.0
        assert abs(sg.upper_bound - pg.upper_bound) < 1.0
        assert sg.outliers_detected == pg.outliers_detected

    def test_segment_analysis_populated(self, spark):
        np.random.seed(42)
        pdf = pd.DataFrame({
            "segment": ["A"] * 60 + ["B"] * 40,
            "value": np.concatenate([
                np.random.normal(10, 1, 60),
                np.random.normal(100, 5, 40),
            ]),
        })
        psdf = _to_psdf(spark, pdf)
        result = SegmentAwareOutlierAnalyzer().analyze(
            psdf, feature_cols=["value"], segment_col="segment"
        )
        assert result.n_segments == 2
        assert len(result.segment_analysis) >= 1

    def test_false_outliers_match_pandas(self, spark):
        np.random.seed(42)
        pdf = pd.DataFrame({
            "segment": ["retail"] * 100 + ["enterprise"] * 10,
            "order_value": np.concatenate([
                np.random.normal(50, 10, 100),
                np.random.normal(5000, 500, 10),
            ]),
        })
        psdf = _to_psdf(spark, pdf)
        analyzer = SegmentAwareOutlierAnalyzer()
        sr = analyzer.analyze(psdf, feature_cols=["order_value"], segment_col="segment")
        pr = analyzer.analyze(pdf, feature_cols=["order_value"], segment_col="segment")
        assert sr.false_outliers["order_value"] == pr.false_outliers["order_value"]

    def test_segment_labels_length(self, spark):
        np.random.seed(42)
        pdf = pd.DataFrame({
            "seg": ["X"] * 30 + ["Y"] * 20 + ["Z"] * 10,
            "value": np.random.normal(50, 10, 60),
        })
        psdf = _to_psdf(spark, pdf)
        result = SegmentAwareOutlierAnalyzer().analyze(
            psdf, feature_cols=["value"], segment_col="seg"
        )
        assert len(result.segment_labels) == 60
        assert result.n_segments == 3

    def test_small_segment_skipped(self, spark):
        np.random.seed(42)
        pdf = pd.DataFrame({
            "segment": ["A"] * 50 + ["B"] * 5,
            "value": np.concatenate([
                np.random.normal(10, 2, 50),
                np.random.normal(100, 10, 5),
            ]),
        })
        psdf = _to_psdf(spark, pdf)
        result = SegmentAwareOutlierAnalyzer().analyze(
            psdf, feature_cols=["value"], segment_col="segment"
        )
        assert result.n_segments == 2
        assert len(result.segment_analysis) == 1


class TestSparkMultipleColumns:
    def test_batched_bounds_multiple_columns(self, spark):
        np.random.seed(42)
        n = 120
        data = {"segment": ["A"] * 60 + ["B"] * 60}
        col_names = []
        for i in range(10):
            name = f"feat_{i}"
            col_names.append(name)
            data[name] = np.random.normal(i * 10, 2, n)
        pdf = pd.DataFrame(data)
        psdf = _to_psdf(spark, pdf)

        analyzer = SegmentAwareOutlierAnalyzer()
        result = analyzer.analyze(psdf, feature_cols=col_names, segment_col="segment")
        assert set(result.global_analysis.keys()) == set(col_names)
        for col in col_names:
            assert result.false_outliers[col] >= 0


class TestSparkDetectionMethods:
    def test_zscore_method(self, spark):
        np.random.seed(42)
        pdf = pd.DataFrame({
            "segment": ["A"] * 60 + ["B"] * 40,
            "value": np.concatenate([
                np.random.normal(10, 1, 60),
                np.random.normal(100, 5, 40),
            ]),
        })
        psdf = _to_psdf(spark, pdf)
        result = SegmentAwareOutlierAnalyzer(
            detection_method=OutlierDetectionMethod.ZSCORE
        ).analyze(psdf, feature_cols=["value"], segment_col="segment")
        assert result.n_segments == 2
        assert "value" in result.global_analysis

    def test_percentile_method(self, spark):
        np.random.seed(42)
        pdf = pd.DataFrame({
            "segment": ["A"] * 60 + ["B"] * 40,
            "value": np.concatenate([
                np.random.normal(10, 1, 60),
                np.random.normal(100, 5, 40),
            ]),
        })
        psdf = _to_psdf(spark, pdf)
        result = SegmentAwareOutlierAnalyzer(
            detection_method=OutlierDetectionMethod.PERCENTILE
        ).analyze(psdf, feature_cols=["value"], segment_col="segment")
        assert result.n_segments == 2
        assert "value" in result.global_analysis

    def test_modified_zscore_method(self, spark):
        np.random.seed(42)
        pdf = pd.DataFrame({
            "segment": ["A"] * 60 + ["B"] * 40,
            "value": np.concatenate([
                np.random.normal(10, 1, 60),
                np.random.normal(100, 5, 40),
            ]),
        })
        psdf = _to_psdf(spark, pdf)
        result = SegmentAwareOutlierAnalyzer(
            detection_method=OutlierDetectionMethod.MODIFIED_ZSCORE
        ).analyze(psdf, feature_cols=["value"], segment_col="segment")
        assert result.n_segments == 2
        assert "value" in result.global_analysis


class TestSparkEdgeCases:
    def test_empty_dataframe(self, spark):
        pdf = pd.DataFrame({"segment": pd.Series([], dtype=str), "value": pd.Series([], dtype=float)})
        psdf = _to_psdf(spark, pdf)
        result = SegmentAwareOutlierAnalyzer().analyze(
            psdf, feature_cols=["value"], segment_col="segment"
        )
        assert result.n_segments == 0

    def test_no_valid_columns(self, spark):
        pdf = pd.DataFrame({"segment": ["A"] * 10, "other": range(10)})
        psdf = _to_psdf(spark, pdf)
        result = SegmentAwareOutlierAnalyzer().analyze(
            psdf, feature_cols=["nonexistent"], segment_col="segment"
        )
        assert result.n_segments == 0

    def test_all_null_column(self, spark):
        pdf = pd.DataFrame({
            "segment": ["A"] * 20 + ["B"] * 20,
            "value": [None] * 40,
        })
        psdf = _to_psdf(spark, pdf)
        result = SegmentAwareOutlierAnalyzer().analyze(
            psdf, feature_cols=["value"], segment_col="segment"
        )
        assert result.global_analysis["value"].outliers_detected == 0

    def test_recommendations_generated(self, spark):
        np.random.seed(42)
        pdf = pd.DataFrame({
            "segment": ["A"] * 60 + ["B"] * 40,
            "value": np.concatenate([
                np.random.normal(10, 1, 60),
                np.random.normal(100, 5, 40),
            ]),
        })
        psdf = _to_psdf(spark, pdf)
        result = SegmentAwareOutlierAnalyzer().analyze(
            psdf, feature_cols=["value"], segment_col="segment"
        )
        assert isinstance(result.recommendations, list)
        assert isinstance(result.rationale, list)


# ---- auto-detect: falls back to pandas for SegmentAnalyzer ----


class TestSparkAutoDetectFallback:
    def test_auto_detect_produces_result(self, spark):
        np.random.seed(42)
        pdf = pd.DataFrame({
            "value": np.concatenate([
                np.random.normal(10, 2, 50),
                np.random.normal(100, 10, 50),
            ]),
        })
        psdf = _to_psdf(spark, pdf)
        result = SegmentAwareOutlierAnalyzer().analyze(psdf, feature_cols=["value"])
        assert result.n_segments >= 1
        assert "value" in result.global_analysis

    def test_auto_detect_with_target(self, spark):
        np.random.seed(42)
        pdf = pd.DataFrame({
            "value": np.concatenate([
                np.random.normal(10, 2, 50),
                np.random.normal(100, 10, 50),
            ]),
            "target": [0] * 50 + [1] * 50,
        })
        psdf = _to_psdf(spark, pdf)
        result = SegmentAwareOutlierAnalyzer().analyze(
            psdf, feature_cols=["value"], target_col="target"
        )
        assert result.n_segments >= 1

    def test_auto_detect_wide_dataframe_collects_once(self, spark):
        """Wide DataFrames (many columns) should collect to pandas once, not per-column."""
        np.random.seed(42)
        n = 60
        data = {}
        col_names = []
        for i in range(50):
            name = f"feat_{i}"
            col_names.append(name)
            data[name] = np.random.normal(i, 2, n)
        pdf = pd.DataFrame(data)
        psdf = _to_psdf(spark, pdf)
        result = SegmentAwareOutlierAnalyzer(max_segments=3).analyze(
            psdf, feature_cols=col_names,
        )
        assert result.n_segments >= 1
        assert set(result.global_analysis.keys()) == set(col_names)

    def test_auto_detect_homogeneous_returns_single_segment(self, spark):
        np.random.seed(42)
        pdf = pd.DataFrame({
            "value": np.random.normal(50, 5, 100),
        })
        psdf = _to_psdf(spark, pdf)
        result = SegmentAwareOutlierAnalyzer().analyze(psdf, feature_cols=["value"])
        assert result.n_segments == 1
        assert len(result.segment_labels) == 100

    def test_auto_detect_uses_penultimate_slice_on_grid(self, spark):
        """Grid data (entity × as_of_date) should use penultimate time slice for IID."""
        np.random.seed(42)
        entities = ["E1", "E2", "E3"] * 10
        dates = pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"] * 10)
        pdf = pd.DataFrame({
            "entity_id": entities,
            "as_of_date": dates,
            "value": np.random.normal(50, 5, 30),
        })
        psdf = _to_psdf(spark, pdf)
        result = SegmentAwareOutlierAnalyzer().analyze(
            psdf, feature_cols=["value"],
        )
        assert result.n_segments >= 1
        assert len(result.segment_labels) == 30
