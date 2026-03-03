import math

import pandas as pd
import pytest

from customer_retention.analysis.auto_explorer.sampling import (
    estimate_sampling_accuracy,
    stratified_entity_sample,
)


class TestEstimateSamplingAccuracy:
    def test_correct_ci_formula(self):
        results = estimate_sampling_accuracy(10000, 0.5, [1000], n_cohorts=1)
        expected_ci = 1.96 * math.sqrt(0.5 * 0.5 / 1000)
        assert abs(results[0]["churn_rate_ci"] - expected_ci) < 1e-6

    def test_correlation_error_formula(self):
        results = estimate_sampling_accuracy(10000, 0.5, [1000], n_cohorts=1)
        expected = 1 / math.sqrt(1000)
        assert abs(results[0]["correlation_error"] - expected) < 1e-6

    def test_sample_equals_total(self):
        results = estimate_sampling_accuracy(500, 0.3, [500], n_cohorts=1)
        assert results[0]["pct_of_total"] == 1.0
        assert results[0]["sample_size"] == 500

    def test_target_rate_zero(self):
        results = estimate_sampling_accuracy(1000, 0.0, [100], n_cohorts=1)
        assert results[0]["churn_rate_ci"] == 0.0

    def test_target_rate_one(self):
        results = estimate_sampling_accuracy(1000, 1.0, [100], n_cohorts=1)
        assert results[0]["churn_rate_ci"] == 0.0

    def test_multiple_sample_sizes(self):
        results = estimate_sampling_accuracy(10000, 0.2, [500, 1000, 5000], n_cohorts=4)
        assert len(results) == 3
        assert results[0]["sample_size"] == 500
        assert results[1]["sample_size"] == 1000
        assert results[2]["sample_size"] == 5000

    def test_cohort_ok_threshold(self):
        results = estimate_sampling_accuracy(10000, 0.5, [100], n_cohorts=4)
        assert results[0]["entities_per_cohort"] == 25
        assert results[0]["cohort_ok"] is False

        results = estimate_sampling_accuracy(10000, 0.5, [120], n_cohorts=4)
        assert results[0]["entities_per_cohort"] == 30
        assert results[0]["cohort_ok"] is True

    def test_minority_expected(self):
        results = estimate_sampling_accuracy(10000, 0.1, [1000], n_cohorts=1)
        assert results[0]["minority_expected"] == 100.0

    def test_empty_sample_sizes(self):
        results = estimate_sampling_accuracy(1000, 0.5, [], n_cohorts=1)
        assert results == []

    def test_sample_capped_at_total(self):
        results = estimate_sampling_accuracy(100, 0.5, [200], n_cohorts=1)
        assert results[0]["sample_size"] == 100


class TestStratifiedEntitySample:
    @pytest.fixture
    def entity_df(self):
        return pd.DataFrame({
            "entity_id": list(range(200)),
            "churned": [1] * 40 + [0] * 160,
            "signup_date": pd.date_range("2020-01-01", periods=200, freq="D"),
            "region": (["east"] * 50 + ["west"] * 50 + ["north"] * 50 + ["south"] * 50),
        })

    def test_returns_correct_count(self, entity_df):
        ids = stratified_entity_sample(entity_df, 50, "entity_id", "churned")
        assert len(ids) == 50

    def test_target_proportions_preserved(self, entity_df):
        ids = stratified_entity_sample(entity_df, 100, "entity_id", "churned")
        sampled = entity_df[entity_df["entity_id"].isin(ids)]
        rate = sampled["churned"].mean()
        assert 0.1 <= rate <= 0.3

    def test_reproducibility(self, entity_df):
        ids1 = stratified_entity_sample(entity_df, 50, "entity_id", "churned", random_state=42)
        ids2 = stratified_entity_sample(entity_df, 50, "entity_id", "churned", random_state=42)
        assert ids1 == ids2

    def test_different_seed_different_result(self, entity_df):
        ids1 = stratified_entity_sample(entity_df, 50, "entity_id", "churned", random_state=42)
        ids2 = stratified_entity_sample(entity_df, 50, "entity_id", "churned", random_state=99)
        assert ids1 != ids2

    def test_n_entities_exceeds_total(self, entity_df):
        ids = stratified_entity_sample(entity_df, 999, "entity_id", "churned")
        assert len(ids) == 200

    def test_n_entities_zero(self, entity_df):
        ids = stratified_entity_sample(entity_df, 0, "entity_id", "churned")
        assert ids == []

    def test_rare_class_floor(self):
        df = pd.DataFrame({
            "entity_id": list(range(100)),
            "target": [1] * 3 + [0] * 97,
        })
        ids = stratified_entity_sample(df, 20, "entity_id", "target", min_rare_count=5)
        sampled = df[df["entity_id"].isin(ids)]
        rare_count = (sampled["target"] == 1).sum()
        assert rare_count == 3

    def test_cohort_coverage(self, entity_df):
        ids = stratified_entity_sample(
            entity_df, 80, "entity_id", "churned", time_col="signup_date",
        )
        sampled = entity_df[entity_df["entity_id"].isin(ids)]
        quarters = (
            pd.to_datetime(sampled["signup_date"]).dt.year.astype(str) + "-Q"
            + pd.to_datetime(sampled["signup_date"]).dt.quarter.astype(str)
        )
        assert quarters.nunique() >= 2

    def test_extra_strat_cols_respected(self, entity_df):
        ids = stratified_entity_sample(
            entity_df, 100, "entity_id", "churned",
            extra_strat_cols=["region"],
        )
        sampled = entity_df[entity_df["entity_id"].isin(ids)]
        assert set(sampled["region"].unique()) == {"east", "west", "north", "south"}

    def test_no_target_column(self, entity_df):
        ids = stratified_entity_sample(entity_df, 50, "entity_id")
        assert len(ids) == 50

    def test_duplicate_entities_deduplicated(self):
        df = pd.DataFrame({
            "entity_id": [1, 1, 2, 2, 3, 3],
            "value": [10, 20, 30, 40, 50, 60],
        })
        ids = stratified_entity_sample(df, 2, "entity_id")
        assert len(ids) == 2
        assert len(set(ids)) == 2

    def test_combined_strat_target_time_and_extra(self, entity_df):
        ids = stratified_entity_sample(
            entity_df, 80, "entity_id", "churned",
            time_col="signup_date", extra_strat_cols=["region"],
        )
        assert len(ids) == 80
        sampled = entity_df[entity_df["entity_id"].isin(ids)]
        assert set(sampled["region"].unique()) == {"east", "west", "north", "south"}
        rate = sampled["churned"].mean()
        assert 0.1 <= rate <= 0.3

    def test_no_ndarray_column_assignment(self):
        df = pd.DataFrame({
            "entity_id": list(range(50)),
            "target": [1] * 10 + [0] * 40,
            "ts": pd.date_range("2022-01-01", periods=50, freq="D"),
        })
        ids = stratified_entity_sample(df, 20, "entity_id", "target", time_col="ts")
        assert len(ids) == 20
        assert all(isinstance(i, int) for i in ids)
