from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestSafeSelect:

    def test_matches_np_select_basic(self):
        from customer_retention.core.compat import safe_select
        s = pd.Series([10, 50, 100, 200])
        conditions = [s <= 30, s <= 90]
        choices = ["low", "medium"]
        result = safe_select(conditions, choices, default="high")
        expected = np.select([c.to_numpy() for c in conditions], choices, default="high")
        np.testing.assert_array_equal(result, expected)

    def test_first_condition_wins(self):
        from customer_retention.core.compat import safe_select
        s = pd.Series([1, 2, 3])
        conditions = [s > 0, s > 1]
        choices = ["first", "second"]
        result = safe_select(conditions, choices, default="none")
        assert list(result) == ["first", "first", "first"]

    def test_default_applied_when_no_match(self):
        from customer_retention.core.compat import safe_select
        s = pd.Series([1, 2, 3])
        conditions = [s > 10]
        choices = ["match"]
        result = safe_select(conditions, choices, default="no_match")
        assert list(result) == ["no_match", "no_match", "no_match"]

    def test_handles_nan_conditions(self):
        from customer_retention.core.compat import safe_select
        s = pd.Series([0.9, None, 0.1, 0.5, None])
        conditions = [s.isna(), s >= 0.7, s >= 0.3]
        choices = ["Unknown", "High", "Medium"]
        result = safe_select(conditions, choices, default="Low")
        assert list(result) == ["High", "Unknown", "Low", "Medium", "Unknown"]

    def test_multiple_conditions_priority(self):
        from customer_retention.core.compat import safe_select
        days = pd.Series([10, None, 50, 100, 200])
        conditions = [days.isna(), days <= 30, days <= 90, days <= 180]
        choices = ["Unknown", "Active_30d", "Recent_90d", "Lapsing_180d"]
        result = safe_select(conditions, choices, default="Dormant_180d+")
        expected = ["Active_30d", "Unknown", "Recent_90d", "Lapsing_180d", "Dormant_180d+"]
        assert list(result) == expected

    def test_boolean_conditions_with_and_or(self):
        from customer_retention.core.compat import safe_select
        df = pd.DataFrame({"val": [100, 200, 50, 300], "freq": [1, 10, 8, 2]})
        high_val = df["val"] >= 150
        high_freq = df["freq"] >= 5
        conditions = [high_val & high_freq, high_val & ~high_freq, ~high_val & high_freq]
        choices = ["HV_HF", "HV_LF", "LV_HF"]
        result = safe_select(conditions, choices, default="LV_LF")
        assert list(result) == ["LV_LF", "HV_HF", "LV_HF", "HV_LF"]

    def test_empty_conditions_raises(self):
        from customer_retention.core.compat import safe_select
        with pytest.raises(ValueError):
            safe_select([], [], default="fallback")

    def test_single_row(self):
        from customer_retention.core.compat import safe_select
        s = pd.Series([42])
        conditions = [s > 100]
        result = safe_select(conditions, ["big"], default="small")
        assert list(result) == ["small"]

    def test_all_nan(self):
        from customer_retention.core.compat import safe_select
        s = pd.Series([None, None, None], dtype=float)
        conditions = [s.isna()]
        result = safe_select(conditions, ["null"], default="not_null")
        assert list(result) == ["null", "null", "null"]


@pytest.mark.databricks
class TestSafeSelectSparkPandas:

    def _to_spark_series(self, s):
        import pyspark.pandas as ps
        return ps.from_pandas(s)

    def test_spark_nan_conditions_string_labels(self):
        from customer_retention.core.compat import safe_select
        s = self._to_spark_series(pd.Series([0.9, None, 0.1, 0.5, None]))
        conditions = [s.isna(), s >= 0.7, s >= 0.3]
        choices = ["Unknown", "High", "Medium"]
        result = safe_select(conditions, choices, default="Low")
        assert list(result.to_pandas()) == ["High", "Unknown", "Low", "Medium", "Unknown"]

    def test_spark_engagement_segmentation(self):
        from customer_retention.core.compat import safe_select
        score = self._to_spark_series(pd.Series([0.8, None, 0.2, 0.5]))
        conditions = [score.isna(), score >= 0.7, score >= 0.3]
        choices = ["Unknown", "High_Engagement", "Medium_Engagement"]
        result = safe_select(conditions, choices, default="Low_Engagement")
        expected = ["High_Engagement", "Unknown", "Low_Engagement", "Medium_Engagement"]
        assert list(result.to_pandas()) == expected

    def test_spark_boolean_and_or_conditions(self):
        import pyspark.pandas as ps

        from customer_retention.core.compat import safe_select
        df = ps.from_pandas(pd.DataFrame({"val": [100, 200, 50, 300], "freq": [1, 10, 8, 2]}))
        high_val = df["val"] >= 150
        high_freq = df["freq"] >= 5
        conditions = [high_val & high_freq, high_val & ~high_freq, ~high_val & high_freq]
        choices = ["HV_HF", "HV_LF", "LV_HF"]
        result = safe_select(conditions, choices, default="LV_LF")
        assert list(result.to_pandas()) == ["LV_LF", "HV_HF", "LV_HF", "HV_LF"]

    def test_spark_all_default(self):
        from customer_retention.core.compat import safe_select
        s = self._to_spark_series(pd.Series([1, 2, 3]))
        conditions = [s > 10]
        result = safe_select(conditions, ["match"], default="no_match")
        assert list(result.to_pandas()) == ["no_match", "no_match", "no_match"]

    def test_spark_first_condition_wins(self):
        from customer_retention.core.compat import safe_select
        s = self._to_spark_series(pd.Series([1, 2, 3]))
        conditions = [s > 0, s > 1]
        choices = ["first", "second"]
        result = safe_select(conditions, choices, default="none")
        assert list(result.to_pandas()) == ["first", "first", "first"]


class TestSafeSelectInAllExports:

    def test_safe_select_in_compat_all(self):
        from customer_retention.core import compat
        assert "safe_select" in compat.__all__
