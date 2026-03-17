from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.profiling.categorical_target_analyzer import (
    _classify_categorical_columns,
    _get_filter_reasons,
    filter_categorical_columns,
)


class TestBulkCategoricalFiltering:

    @pytest.fixture
    def mixed_df(self):
        rng = np.random.default_rng(42)
        n = 200
        return pd.DataFrame({
            "entity_id": [f"ent_{i}" for i in range(n)],
            "target": rng.integers(0, 2, size=n),
            "plan_type": rng.choice(["basic", "pro", "enterprise"], size=n),
            "region": rng.choice(["US", "EU", "APAC", "LATAM", "MEA"], size=n),
            "high_card_id": [f"id_{i}" for i in range(n)],
            "single_val": ["constant"] * n,
            "age": rng.integers(18, 65, size=n),
            "signup_date": pd.date_range("2024-01-01", periods=n),
        })

    def test_filter_returns_valid_columns(self, mixed_df):
        valid = filter_categorical_columns(mixed_df, "entity_id", "target")
        assert "plan_type" in valid
        assert "region" in valid
        assert "entity_id" not in valid
        assert "target" not in valid

    def test_filter_excludes_high_cardinality(self, mixed_df):
        valid = filter_categorical_columns(mixed_df, "entity_id", "target")
        assert "high_card_id" not in valid

    def test_filter_excludes_single_value(self, mixed_df):
        valid = filter_categorical_columns(mixed_df, "entity_id", "target")
        assert "single_val" not in valid

    def test_filter_excludes_datetime(self, mixed_df):
        valid = filter_categorical_columns(mixed_df, "entity_id", "target")
        assert "signup_date" not in valid

    def test_filter_excludes_numeric(self, mixed_df):
        valid = filter_categorical_columns(mixed_df, "entity_id", "target")
        assert "age" not in valid

    def test_reasons_match_filtering(self, mixed_df):
        valid = filter_categorical_columns(mixed_df, "entity_id", "target")
        reasons = _get_filter_reasons(mixed_df, "entity_id", "target")
        all_object_cols = list(mixed_df.select_dtypes(include=["object", "category"]).columns)
        for col in all_object_cols:
            if col in ("entity_id", "target"):
                continue
            if col in valid:
                assert col not in reasons
            else:
                assert col in reasons

    def test_classify_returns_both(self, mixed_df):
        valid, reasons = _classify_categorical_columns(mixed_df, "entity_id", "target", 0.5)
        assert isinstance(valid, list)
        assert isinstance(reasons, dict)
        assert "plan_type" in valid
        assert "high_card_id" in reasons

    def test_no_categorical_columns(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        valid = filter_categorical_columns(df, "a", "b")
        assert valid == []

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        valid = filter_categorical_columns(df, "entity", "target")
        assert valid == []
