import logging

import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.profiling.distribution_analysis import (
    DistributionAnalyzer,
    DistributionTransformationType,
)


@pytest.fixture
def analyzer():
    return DistributionAnalyzer()


def _series_with_zero_share(share: float, n: int = 1000) -> pd.Series:
    rng = np.random.default_rng(42)
    values = rng.exponential(10, n)
    mask = rng.random(n) < share
    values[mask] = 0
    return pd.Series(values, name="X")


def test_90pct_zeros_above_threshold_emits_recommendation(analyzer):
    analysis = analyzer.analyze_distribution(_series_with_zero_share(0.90), "COL")
    rec = analyzer.recommend_transformation(analysis)
    assert rec.recommended_transform == DistributionTransformationType.ZERO_INFLATION_HANDLING


def test_50pct_zeros_below_threshold_suppresses(analyzer):
    analysis = analyzer.analyze_distribution(_series_with_zero_share(0.50), "COL")
    rec = analyzer.recommend_transformation(analysis)
    assert rec.recommended_transform != DistributionTransformationType.ZERO_INFLATION_HANDLING


def test_skip_log_emitted_when_below_threshold(analyzer, caplog):
    analysis = analyzer.analyze_distribution(_series_with_zero_share(0.50), "MYCOL")
    with caplog.at_level(logging.INFO, logger="customer_retention.stages.profiling.distribution_analysis"):
        analyzer.recommend_transformation(analysis)
    msgs = [r.getMessage() for r in caplog.records]
    assert any("zero_inflation skipped" in m and "MYCOL" in m for m in msgs)


def test_skip_log_not_emitted_when_above_threshold(analyzer, caplog):
    analysis = analyzer.analyze_distribution(_series_with_zero_share(0.90), "MYCOL")
    with caplog.at_level(logging.INFO, logger="customer_retention.stages.profiling.distribution_analysis"):
        analyzer.recommend_transformation(analysis)
    msgs = [r.getMessage() for r in caplog.records]
    assert not any("zero_inflation skipped" in m for m in msgs)


def test_skip_log_not_emitted_when_mostly_non_zero(analyzer, caplog):
    analysis = analyzer.analyze_distribution(_series_with_zero_share(0.01), "MYCOL")
    with caplog.at_level(logging.INFO, logger="customer_retention.stages.profiling.distribution_analysis"):
        analyzer.recommend_transformation(analysis)
    msgs = [r.getMessage() for r in caplog.records]
    assert not any("zero_inflation skipped" in m for m in msgs)


def test_count_zero_inflation_recommendations():
    rec_zi = type(
        "R", (), {"recommended_transform": DistributionTransformationType.ZERO_INFLATION_HANDLING}
    )()
    rec_log = type(
        "R", (), {"recommended_transform": DistributionTransformationType.LOG_TRANSFORM}
    )()
    rec_none = type(
        "R", (), {"recommended_transform": DistributionTransformationType.NONE}
    )()
    assert DistributionAnalyzer.count_zero_inflation_recommendations(
        [rec_zi, rec_zi, rec_log, rec_none, None]
    ) == 2


def test_skip_log_message_format_includes_pct_and_threshold(analyzer, caplog):
    analysis = analyzer.analyze_distribution(_series_with_zero_share(0.50), "COL")
    with caplog.at_level(logging.INFO, logger="customer_retention.stages.profiling.distribution_analysis"):
        analyzer.recommend_transformation(analysis)
    msg = next(r.getMessage() for r in caplog.records if "zero_inflation skipped" in r.getMessage())
    assert "zero_pct=" in msg
    assert "threshold=" in msg
