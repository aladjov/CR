import pytest
import pandas as pd

from customer_retention.core.config.column_config import ColumnType
from customer_retention.analysis.auto_explorer.findings import ColumnFinding, ExplorationFindings
from customer_retention.analysis.recommendations.registry import RecommendationRegistry
from customer_retention.analysis.recommendations import (
    ImputeRecommendation,
    OutlierCapRecommendation,
    StandardScaleRecommendation,
    MinMaxScaleRecommendation,
    LogTransformRecommendation,
    OneHotEncodeRecommendation,
    LabelEncodeRecommendation,
    ExtractMonthRecommendation,
    ExtractDayOfWeekRecommendation,
    DaysSinceRecommendation,
)


class TestRecommendationRegistryCleaningMap:
    def test_impute_median(self):
        rec = RecommendationRegistry.create_cleaning("impute_median", ["col1"], None)
        assert isinstance(rec, ImputeRecommendation)
        assert rec.strategy == "median"

    def test_impute_mean(self):
        rec = RecommendationRegistry.create_cleaning("impute_mean", ["col1"], None)
        assert isinstance(rec, ImputeRecommendation)
        assert rec.strategy == "mean"

    def test_impute_mode(self):
        rec = RecommendationRegistry.create_cleaning("impute_mode", ["col1"], None)
        assert isinstance(rec, ImputeRecommendation)
        assert rec.strategy == "mode"

    def test_impute_zero(self):
        rec = RecommendationRegistry.create_cleaning("impute_zero", ["col1"], None)
        assert isinstance(rec, ImputeRecommendation)
        assert rec.strategy == "constant"
        assert rec.fill_value == 0

    def test_cap_outliers_99(self):
        rec = RecommendationRegistry.create_cleaning("cap_outliers_99", ["col1"], None)
        assert isinstance(rec, OutlierCapRecommendation)
        assert rec.percentile == 99

    def test_cap_outliers_95(self):
        rec = RecommendationRegistry.create_cleaning("cap_outliers_95", ["col1"], None)
        assert isinstance(rec, OutlierCapRecommendation)
        assert rec.percentile == 95

    def test_unknown_cleaning_returns_none(self):
        rec = RecommendationRegistry.create_cleaning("unknown_cleaning", ["col1"], None)
        assert rec is None


class TestRecommendationRegistryTransformMap:
    def test_standard_scale(self):
        rec = RecommendationRegistry.create_transform("standard_scale", ["col1"], None)
        assert isinstance(rec, StandardScaleRecommendation)

    def test_minmax_scale(self):
        rec = RecommendationRegistry.create_transform("minmax_scale", ["col1"], None)
        assert isinstance(rec, MinMaxScaleRecommendation)

    def test_log_transform(self):
        rec = RecommendationRegistry.create_transform("log_transform", ["col1"], None)
        assert isinstance(rec, LogTransformRecommendation)

    def test_unknown_transform_returns_none(self):
        rec = RecommendationRegistry.create_transform("unknown_transform", ["col1"], None)
        assert rec is None


class TestRecommendationRegistryEncodingMap:
    def test_onehot_encode(self):
        rec = RecommendationRegistry.create_encoding("onehot_encode", ["col1"], None)
        assert isinstance(rec, OneHotEncodeRecommendation)

    def test_label_encode(self):
        rec = RecommendationRegistry.create_encoding("label_encode", ["col1"], None)
        assert isinstance(rec, LabelEncodeRecommendation)

    def test_unknown_encoding_returns_none(self):
        rec = RecommendationRegistry.create_encoding("unknown_encoding", ["col1"], None)
        assert rec is None


class TestRecommendationRegistryDatetimeMap:
    def test_extract_month(self):
        rec = RecommendationRegistry.create_datetime("extract_month", ["date"], None)
        assert isinstance(rec, ExtractMonthRecommendation)

    def test_extract_dayofweek(self):
        rec = RecommendationRegistry.create_datetime("extract_dayofweek", ["date"], None)
        assert isinstance(rec, ExtractDayOfWeekRecommendation)

    def test_days_since(self):
        rec = RecommendationRegistry.create_datetime("days_since", ["date"], None)
        assert isinstance(rec, DaysSinceRecommendation)

    def test_unknown_datetime_returns_none(self):
        rec = RecommendationRegistry.create_datetime("unknown_datetime", ["date"], None)
        assert rec is None


@pytest.fixture
def sample_findings():
    columns = {
        "id": ColumnFinding(
            name="id", inferred_type=ColumnType.IDENTIFIER, confidence=0.99, evidence=["unique"]
        ),
        "age": ColumnFinding(
            name="age", inferred_type=ColumnType.NUMERIC_CONTINUOUS, confidence=0.95, evidence=["numeric"],
            cleaning_recommendations=["impute_median", "cap_outliers_99"],
            transformation_recommendations=["standard_scale"]
        ),
        "category": ColumnFinding(
            name="category", inferred_type=ColumnType.CATEGORICAL_NOMINAL, confidence=0.9, evidence=["categorical"],
            cleaning_recommendations=["impute_mode"],
            transformation_recommendations=["onehot_encode"]
        ),
        "signup_date": ColumnFinding(
            name="signup_date", inferred_type=ColumnType.DATETIME, confidence=0.95, evidence=["datetime"],
            transformation_recommendations=["extract_month", "days_since"]
        ),
        "target": ColumnFinding(
            name="target", inferred_type=ColumnType.TARGET, confidence=0.99, evidence=["binary"]
        ),
    }
    return ExplorationFindings(
        source_path="data.csv", source_format="csv", row_count=1000, column_count=5,
        columns=columns, target_column="target", identifier_columns=["id"]
    )


class TestRecommendationRegistryFromFindings:
    def test_creates_recommendations_from_findings(self, sample_findings):
        recs = RecommendationRegistry.from_findings(sample_findings)
        assert len(recs) > 0

    def test_skips_identifier_columns(self, sample_findings):
        recs = RecommendationRegistry.from_findings(sample_findings)
        col_names = [col for rec in recs for col in rec.columns]
        assert "id" not in col_names

    def test_skips_target_columns(self, sample_findings):
        recs = RecommendationRegistry.from_findings(sample_findings)
        col_names = [col for rec in recs for col in rec.columns]
        assert "target" not in col_names

    def test_creates_cleaning_recommendations(self, sample_findings):
        recs = RecommendationRegistry.from_findings(sample_findings)
        cleaning_recs = [r for r in recs if r.category == "cleaning"]
        assert len(cleaning_recs) >= 3

    def test_creates_transform_recommendations(self, sample_findings):
        recs = RecommendationRegistry.from_findings(sample_findings)
        transform_recs = [r for r in recs if r.category == "transform"]
        assert len(transform_recs) >= 1

    def test_creates_encoding_recommendations(self, sample_findings):
        recs = RecommendationRegistry.from_findings(sample_findings)
        encoding_recs = [r for r in recs if r.category == "encoding"]
        assert len(encoding_recs) >= 1

    def test_creates_datetime_recommendations(self, sample_findings):
        recs = RecommendationRegistry.from_findings(sample_findings)
        datetime_recs = [r for r in recs if r.category == "datetime"]
        assert len(datetime_recs) >= 2


class TestRecommendationRegistryEmptyFindings:
    def test_empty_findings(self):
        findings = ExplorationFindings(
            source_path="data.csv", source_format="csv", row_count=0, column_count=0,
            columns={}, target_column=None
        )
        recs = RecommendationRegistry.from_findings(findings)
        assert recs == []
