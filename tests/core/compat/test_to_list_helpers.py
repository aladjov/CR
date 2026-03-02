import warnings

import pandas as pd

from customer_retention.core.compat import (
    distributed_case_variations,
    head_as_list,
    unique_overlap_counts,
)


class TestHeadAsList:
    def test_returns_at_most_n_elements(self):
        s = pd.Series(range(100))
        result = head_as_list(s, 5)
        assert len(result) == 5
        assert result == [0, 1, 2, 3, 4]

    def test_returns_all_when_fewer_than_n(self):
        s = pd.Series([1, 2])
        result = head_as_list(s, 10)
        assert result == [1, 2]

    def test_works_with_index(self):
        s = pd.Series([10, 20, 30], index=["a", "b", "c"])
        result = head_as_list(s.index, 2)
        assert result == ["a", "b"]

    def test_works_with_non_subscriptable_index(self):
        class SparkPandasIndex:
            """Mimics pyspark.pandas Index: no .head(), no [:n], but has .to_series()."""
            def __init__(self, data):
                self._data = data

            def to_series(self):
                return pd.Series(self._data)

        idx = SparkPandasIndex(["x", "y", "z", "w"])
        result = head_as_list(idx, 2)
        assert result == ["x", "y"]

    def test_works_with_plain_list(self):
        result = head_as_list([5, 6, 7, 8, 9], 3)
        assert result == [5, 6, 7]

    def test_suppresses_to_list_warning(self):
        class WarningSeries:
            def head(self, n):
                return self

            def to_list(self):
                warnings.warn(
                    "Use to_list to collect all data into driver memory",
                    UserWarning,
                    stacklevel=2,
                )
                return [1, 2, 3]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = head_as_list(WarningSeries(), 10)
            assert result == [1, 2, 3]
            to_list_warnings = [
                x for x in w if "to_list" in str(x.message)
            ]
            assert len(to_list_warnings) == 0


class TestUniqueOverlapCounts:
    def test_full_overlap(self):
        a = pd.Series([1, 2, 3])
        b = pd.Series([1, 2, 3])
        left, right, overlap = unique_overlap_counts(a, b)
        assert left == 3
        assert right == 3
        assert overlap == 3

    def test_no_overlap(self):
        a = pd.Series([1, 2, 3])
        b = pd.Series([4, 5, 6])
        left, right, overlap = unique_overlap_counts(a, b)
        assert left == 3
        assert right == 3
        assert overlap == 0

    def test_partial_overlap(self):
        a = pd.Series([1, 2, 3, 4])
        b = pd.Series([3, 4, 5])
        left, right, overlap = unique_overlap_counts(a, b)
        assert left == 4
        assert right == 3
        assert overlap == 2

    def test_ignores_nulls(self):
        a = pd.Series([1, 2, None, 3])
        b = pd.Series([2, None, 4])
        left, right, overlap = unique_overlap_counts(a, b)
        assert left == 3
        assert right == 2
        assert overlap == 1

    def test_handles_duplicates(self):
        a = pd.Series([1, 1, 2, 2])
        b = pd.Series([2, 2, 3, 3])
        left, right, overlap = unique_overlap_counts(a, b)
        assert left == 2
        assert right == 2
        assert overlap == 1


class TestDistributedCaseVariations:
    def test_finds_variations(self):
        s = pd.Series(["Hello", "hello", "World", "world", "Foo"])
        result = distributed_case_variations(s)
        assert len(result) == 2
        assert any("Hello" in v and "hello" in v for v in result)
        assert any("World" in v and "world" in v for v in result)

    def test_no_variations(self):
        s = pd.Series(["a", "b", "c"])
        result = distributed_case_variations(s)
        assert result == []

    def test_respects_max_results(self):
        values = []
        for i in range(20):
            values.extend([f"Var{i}", f"var{i}"])
        s = pd.Series(values)
        result = distributed_case_variations(s, max_results=5)
        assert len(result) <= 5

    def test_single_value(self):
        s = pd.Series(["only"])
        result = distributed_case_variations(s)
        assert result == []
