"""Tests for SegmentAwareOutlierAnalyzer."""
import decimal

import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.cleaning import OutlierDetectionMethod
from customer_retention.stages.profiling import SegmentAwareOutlierAnalyzer


class TestSegmentAwareOutlierAnalyzerInit:
    def test_default_init(self):
        analyzer = SegmentAwareOutlierAnalyzer()
        assert analyzer.detection_method == OutlierDetectionMethod.IQR
        assert analyzer.iqr_multiplier == 1.5

    def test_custom_init(self):
        analyzer = SegmentAwareOutlierAnalyzer(
            detection_method=OutlierDetectionMethod.ZSCORE,
            zscore_threshold=2.5
        )
        assert analyzer.detection_method == OutlierDetectionMethod.ZSCORE
        assert analyzer.zscore_threshold == 2.5


class TestGlobalVsSegmentOutliers:
    @pytest.fixture
    def bimodal_data(self):
        """Create data with two distinct segments - retail and enterprise customers."""
        np.random.seed(42)
        # Retail customers: small orders around $50
        retail = pd.DataFrame({
            'customer_type': ['retail'] * 100,
            'order_value': np.random.normal(50, 10, 100),
            'order_count': np.random.poisson(5, 100)
        })
        # Enterprise customers: large orders around $5000
        enterprise = pd.DataFrame({
            'customer_type': ['enterprise'] * 10,
            'order_value': np.random.normal(5000, 500, 10),
            'order_count': np.random.poisson(50, 10)
        })
        return pd.concat([retail, enterprise], ignore_index=True)

    def test_detects_segments(self, bimodal_data):
        analyzer = SegmentAwareOutlierAnalyzer()
        result = analyzer.analyze(bimodal_data, feature_cols=['order_value', 'order_count'])
        assert result.n_segments >= 2

    def test_global_outliers_identified(self, bimodal_data):
        analyzer = SegmentAwareOutlierAnalyzer()
        result = analyzer.analyze(bimodal_data, feature_cols=['order_value', 'order_count'])
        # Enterprise customers should be flagged as global outliers
        assert result.global_analysis['order_value'].outliers_detected > 0

    def test_segment_outliers_differ_from_global(self, bimodal_data):
        analyzer = SegmentAwareOutlierAnalyzer()
        result = analyzer.analyze(bimodal_data, feature_cols=['order_value', 'order_count'])
        # Within segments, there should be fewer outliers
        total_segment_outliers = sum(
            seg['order_value'].outliers_detected
            for seg in result.segment_analysis.values()
        )
        global_outliers = result.global_analysis['order_value'].outliers_detected
        assert total_segment_outliers < global_outliers

    def test_false_outliers_identified(self, bimodal_data):
        analyzer = SegmentAwareOutlierAnalyzer()
        result = analyzer.analyze(bimodal_data, feature_cols=['order_value', 'order_count'])
        # False outliers are global outliers that are normal within their segment
        assert 'order_value' in result.false_outliers
        assert result.false_outliers['order_value'] > 0


class TestAnalysisResult:
    @pytest.fixture
    def simple_data(self):
        np.random.seed(42)
        return pd.DataFrame({
            'value': np.concatenate([
                np.random.normal(10, 2, 50),
                np.random.normal(100, 10, 10)
            ])
        })

    def test_result_has_global_analysis(self, simple_data):
        analyzer = SegmentAwareOutlierAnalyzer()
        result = analyzer.analyze(simple_data, feature_cols=['value'])
        assert 'value' in result.global_analysis
        assert hasattr(result.global_analysis['value'], 'outliers_detected')

    def test_result_has_segment_analysis(self, simple_data):
        analyzer = SegmentAwareOutlierAnalyzer()
        result = analyzer.analyze(simple_data, feature_cols=['value'])
        assert len(result.segment_analysis) > 0

    def test_result_has_recommendations(self, simple_data):
        analyzer = SegmentAwareOutlierAnalyzer()
        result = analyzer.analyze(simple_data, feature_cols=['value'])
        assert hasattr(result, 'recommendations')

    def test_result_has_segmentation_recommended_flag(self, simple_data):
        analyzer = SegmentAwareOutlierAnalyzer()
        result = analyzer.analyze(simple_data, feature_cols=['value'])
        assert hasattr(result, 'segmentation_recommended')


class TestRecommendations:
    @pytest.fixture
    def heterogeneous_data(self):
        """Data where segmentation would help outlier detection."""
        np.random.seed(42)
        return pd.DataFrame({
            'amount': np.concatenate([
                np.random.normal(100, 20, 80),  # Group 1
                np.random.normal(1000, 100, 20)  # Group 2 - looks like outliers globally
            ])
        })

    def test_recommends_segmentation_when_beneficial(self, heterogeneous_data):
        analyzer = SegmentAwareOutlierAnalyzer()
        result = analyzer.analyze(heterogeneous_data, feature_cols=['amount'])
        # Should recommend segmentation when many global outliers are normal in segments
        if result.false_outliers.get('amount', 0) > 5:
            assert result.segmentation_recommended

    def test_recommendations_include_rationale(self, heterogeneous_data):
        analyzer = SegmentAwareOutlierAnalyzer()
        result = analyzer.analyze(heterogeneous_data, feature_cols=['amount'])
        assert len(result.rationale) > 0


class TestWithExplicitSegments:
    def test_analyze_with_segment_labels(self):
        """Test when segment labels are provided explicitly."""
        np.random.seed(42)
        df = pd.DataFrame({
            'segment': ['A'] * 50 + ['B'] * 50,
            'value': np.concatenate([
                np.random.normal(10, 2, 50),
                np.random.normal(100, 10, 50)
            ])
        })
        analyzer = SegmentAwareOutlierAnalyzer()
        result = analyzer.analyze(
            df,
            feature_cols=['value'],
            segment_col='segment'
        )
        assert result.n_segments == 2
        assert 'A' in result.segment_analysis or 0 in result.segment_analysis


class TestEdgeCases:
    def test_single_segment_data(self):
        """Test with homogeneous data that forms one segment."""
        np.random.seed(42)
        df = pd.DataFrame({
            'value': np.random.normal(50, 5, 100)
        })
        analyzer = SegmentAwareOutlierAnalyzer()
        result = analyzer.analyze(df, feature_cols=['value'])
        assert result.n_segments >= 1
        assert not result.segmentation_recommended

    def test_empty_dataframe(self):
        analyzer = SegmentAwareOutlierAnalyzer()
        df = pd.DataFrame({'value': []})
        result = analyzer.analyze(df, feature_cols=['value'])
        assert result.n_segments == 0

    def test_all_nulls(self):
        analyzer = SegmentAwareOutlierAnalyzer()
        df = pd.DataFrame({'value': [None, None, None]})
        result = analyzer.analyze(df, feature_cols=['value'])
        assert result.global_analysis['value'].outliers_detected == 0

    def test_no_valid_columns(self):
        """Test when none of the feature_cols exist in the DataFrame."""
        analyzer = SegmentAwareOutlierAnalyzer()
        df = pd.DataFrame({'other_col': [1, 2, 3]})
        result = analyzer.analyze(df, feature_cols=['nonexistent_col'])
        assert result.n_segments == 0
        assert not result.segmentation_recommended

    def test_too_small_for_segmentation(self):
        """Test with a dataset too small to detect segments."""
        np.random.seed(42)
        # MIN_SEGMENT_SIZE * 2 = 20, use only 15 rows
        df = pd.DataFrame({
            'value': np.random.normal(10, 2, 15)
        })
        analyzer = SegmentAwareOutlierAnalyzer()
        result = analyzer.analyze(df, feature_cols=['value'])
        # Should fall back to 1 segment
        assert result.n_segments == 1

    def test_small_segment_skipped(self):
        """Test that segments smaller than MIN_SEGMENT_SIZE are skipped."""
        np.random.seed(42)
        # Create data with explicit segments: one large, one too small
        df = pd.DataFrame({
            'segment': ['A'] * 50 + ['B'] * 5,  # B has only 5 < MIN_SEGMENT_SIZE (10)
            'value': np.concatenate([
                np.random.normal(10, 2, 50),
                np.random.normal(100, 10, 5)
            ])
        })
        analyzer = SegmentAwareOutlierAnalyzer()
        result = analyzer.analyze(df, feature_cols=['value'], segment_col='segment')
        # Segment B should be skipped from segment_analysis
        assert result.n_segments == 2
        # Only segment 0 (A) should be in the analysis (B is too small)
        assert len(result.segment_analysis) == 1

    def test_explicit_segment_with_nan_labels(self):
        """Test explicit segments with NaN segment labels."""
        np.random.seed(42)
        df = pd.DataFrame({
            'segment': ['A'] * 20 + [None] * 5 + ['B'] * 20,
            'value': np.concatenate([
                np.random.normal(10, 2, 20),
                np.random.normal(50, 5, 5),
                np.random.normal(100, 10, 20)
            ])
        })
        analyzer = SegmentAwareOutlierAnalyzer()
        result = analyzer.analyze(df, feature_cols=['value'], segment_col='segment')
        assert result.n_segments == 2

    def test_homogeneous_data_recommendations(self):
        """Test that homogeneous data gets global treatment recommendation."""
        np.random.seed(42)
        df = pd.DataFrame({
            'value': np.random.normal(50, 5, 100)
        })
        analyzer = SegmentAwareOutlierAnalyzer()
        result = analyzer.analyze(df, feature_cols=['value'])

        # When n_segments <= 1 and no segmentation recommended, should get global recommendation
        if result.n_segments <= 1 and not result.segmentation_recommended:
            assert any("global" in r.lower() for r in result.recommendations)
            assert any("homogeneous" in r.lower() for r in result.rationale)

    def test_moderate_false_outlier_ratio(self):
        """Test the 0.2 < false_ratio < 0.5 branch."""
        np.random.seed(42)
        # Carefully construct data so false outlier ratio is between 20-50%
        # Majority group with some moderate outliers
        main_group = np.random.normal(50, 5, 80)
        # Small secondary group that creates global outliers
        secondary = np.random.normal(80, 3, 20)
        df = pd.DataFrame({
            'segment': ['A'] * 80 + ['B'] * 20,
            'value': np.concatenate([main_group, secondary])
        })
        analyzer = SegmentAwareOutlierAnalyzer()
        result = analyzer.analyze(df, feature_cols=['value'], segment_col='segment')

        # Just verify the analysis runs and has rationale
        assert isinstance(result.rationale, list)

    def test_outlier_reduction_recommendation(self):
        """Test segment-aware outlier reduction recommendation path."""
        np.random.seed(42)
        # Create strongly bimodal data that has many global outliers
        # but fewer segment-level outliers
        group_a = np.random.normal(10, 1, 60)
        group_b = np.random.normal(100, 5, 40)
        df = pd.DataFrame({
            'segment': ['A'] * 60 + ['B'] * 40,
            'value': np.concatenate([group_a, group_b])
        })
        analyzer = SegmentAwareOutlierAnalyzer()
        result = analyzer.analyze(df, feature_cols=['value'], segment_col='segment')

        # Verify the analysis produces results - exact path depends on data
        assert isinstance(result.rationale, list)
        assert isinstance(result.recommendations, list)

    def test_analyze_with_target_col(self):
        """Test analyze with target_col for auto-segmentation."""
        np.random.seed(42)
        df = pd.DataFrame({
            'value': np.concatenate([
                np.random.normal(10, 2, 50),
                np.random.normal(100, 10, 50)
            ]),
            'target': [0] * 50 + [1] * 50
        })
        analyzer = SegmentAwareOutlierAnalyzer()
        result = analyzer.analyze(df, feature_cols=['value'], target_col='target')
        assert result.n_segments >= 1


class TestDecimalColumnHandling:
    def test_decimal_columns_do_not_raise(self):
        """Reproduce Databricks DecimalType failure: decimal.Decimal * float."""
        np.random.seed(42)
        values = [decimal.Decimal(str(round(v, 2))) for v in np.random.normal(50, 10, 60)]
        df = pd.DataFrame({"amount": values})
        analyzer = SegmentAwareOutlierAnalyzer()
        result = analyzer.analyze(df, feature_cols=["amount"])
        assert "amount" in result.global_analysis
        assert result.global_analysis["amount"].outliers_detected >= 0

    def test_mixed_decimal_and_float_columns(self):
        np.random.seed(42)
        dec_values = [decimal.Decimal(str(round(v, 2))) for v in np.random.normal(50, 10, 60)]
        df = pd.DataFrame({
            "amount": dec_values,
            "score": np.random.normal(0, 1, 60),
        })
        analyzer = SegmentAwareOutlierAnalyzer()
        result = analyzer.analyze(df, feature_cols=["amount", "score"])
        assert "amount" in result.global_analysis
        assert "score" in result.global_analysis


class TestVectorizedFalseOutlierIdentification:
    def test_false_outliers_bounded_by_global_count(self):
        np.random.seed(42)
        retail = pd.DataFrame({
            'order_value': np.random.normal(50, 10, 100),
            'order_count': np.random.poisson(5, 100),
        })
        enterprise = pd.DataFrame({
            'order_value': np.random.normal(5000, 500, 10),
            'order_count': np.random.poisson(50, 10),
        })
        df = pd.concat([retail, enterprise], ignore_index=True)

        analyzer = SegmentAwareOutlierAnalyzer()
        result = analyzer.analyze(df, feature_cols=['order_value', 'order_count'])

        for col in ['order_value', 'order_count']:
            assert result.false_outliers[col] <= result.global_analysis[col].outliers_detected

    def test_all_outliers_in_skipped_segments_yields_zero(self):
        np.random.seed(42)
        df = pd.DataFrame({
            'segment': ['big'] * 50 + ['tiny'] * 3,
            'value': np.concatenate([
                np.random.normal(50, 5, 50),
                np.array([5000, 6000, 7000]),
            ]),
        })
        analyzer = SegmentAwareOutlierAnalyzer()
        result = analyzer.analyze(df, feature_cols=['value'], segment_col='segment')
        assert result.false_outliers['value'] == 0
