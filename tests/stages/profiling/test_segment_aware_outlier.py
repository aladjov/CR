"""Tests for SegmentAwareOutlierAnalyzer."""
import pytest
import numpy as np
import pandas as pd

from customer_retention.stages.profiling import SegmentAwareOutlierAnalyzer, SegmentAnalyzer
from customer_retention.stages.cleaning import OutlierHandler, OutlierDetectionMethod


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
