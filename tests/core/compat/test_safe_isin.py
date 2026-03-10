from __future__ import annotations

import pandas as pd
import pytest


class TestSafeIsin:
    def test_filters_matching_rows(self):
        from customer_retention.core.compat import safe_isin
        df = pd.DataFrame({"id": [1, 2, 3, 4, 5], "val": list("abcde")})
        result = safe_isin(df, "id", [2, 4])
        assert sorted(result["id"].tolist()) == [2, 4]

    def test_negate_excludes_matching(self):
        from customer_retention.core.compat import safe_isin
        df = pd.DataFrame({"id": [1, 2, 3, 4, 5], "val": list("abcde")})
        result = safe_isin(df, "id", [2, 4], negate=True)
        assert sorted(result["id"].tolist()) == [1, 3, 5]

    def test_empty_values_returns_empty(self):
        from customer_retention.core.compat import safe_isin
        df = pd.DataFrame({"id": [1, 2, 3]})
        result = safe_isin(df, "id", [])
        assert len(result) == 0

    def test_empty_values_negate_returns_all(self):
        from customer_retention.core.compat import safe_isin
        df = pd.DataFrame({"id": [1, 2, 3]})
        result = safe_isin(df, "id", [], negate=True)
        assert len(result) == 3

    def test_all_match(self):
        from customer_retention.core.compat import safe_isin
        df = pd.DataFrame({"id": [1, 2, 3]})
        result = safe_isin(df, "id", [1, 2, 3])
        assert len(result) == 3

    def test_none_match(self):
        from customer_retention.core.compat import safe_isin
        df = pd.DataFrame({"id": [1, 2, 3]})
        result = safe_isin(df, "id", [10, 20])
        assert len(result) == 0

    def test_preserves_all_columns(self):
        from customer_retention.core.compat import safe_isin
        df = pd.DataFrame({"id": [1, 2, 3], "x": [10, 20, 30], "y": ["a", "b", "c"]})
        result = safe_isin(df, "id", [2])
        assert list(result.columns) == ["id", "x", "y"]
        assert result.iloc[0]["x"] == 20

    def test_accepts_set_values(self):
        from customer_retention.core.compat import safe_isin
        df = pd.DataFrame({"id": [1, 2, 3, 4]})
        result = safe_isin(df, "id", {2, 3})
        assert sorted(result["id"].tolist()) == [2, 3]

    def test_string_column(self):
        from customer_retention.core.compat import safe_isin
        df = pd.DataFrame({"name": ["alice", "bob", "carol"], "v": [1, 2, 3]})
        result = safe_isin(df, "name", ["bob", "carol"])
        assert sorted(result["name"].tolist()) == ["bob", "carol"]

    def test_duplicate_values_in_filter(self):
        from customer_retention.core.compat import safe_isin
        df = pd.DataFrame({"id": [1, 2, 3]})
        result = safe_isin(df, "id", [2, 2, 2])
        assert result["id"].tolist() == [2]

    def test_empty_dataframe(self):
        from customer_retention.core.compat import safe_isin
        df = pd.DataFrame({"id": pd.Series([], dtype=int)})
        result = safe_isin(df, "id", [1, 2])
        assert len(result) == 0


class TestSafeIsinInAllExports:
    def test_safe_isin_in_compat_all(self):
        from customer_retention.core import compat
        assert "safe_isin" in compat.__all__
