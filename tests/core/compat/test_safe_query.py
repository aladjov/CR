import pandas as pd

from customer_retention.core.compat import _spark_safe_query_expr, safe_query


class TestSparkSafeQueryExpr:
    def test_list_syntax_converted_to_tuple(self):
        assert _spark_safe_query_expr("col in ['a', 'b']") == "col in ('a', 'b')"

    def test_multiple_list_expressions(self):
        expr = "region in ['US', 'UK'] and status in ['active', 'pending']"
        result = _spark_safe_query_expr(expr)
        assert result == "region in ('US', 'UK') and status in ('active', 'pending')"

    def test_no_list_syntax_unchanged(self):
        assert _spark_safe_query_expr("a > 3") == "a > 3"

    def test_not_in_list_converted(self):
        assert _spark_safe_query_expr("col not in ['x']") == "col not in ('x')"

    def test_empty_list(self):
        assert _spark_safe_query_expr("col in []") == "col in ()"

    def test_numeric_list(self):
        assert _spark_safe_query_expr("id in [1, 2, 3]") == "id in (1, 2, 3)"

    def test_compound_with_list(self):
        expr = "amount > 0 and status in ['active'] and region == 'US'"
        result = _spark_safe_query_expr(expr)
        assert result == "amount > 0 and status in ('active') and region == 'US'"

    def test_tuple_syntax_unchanged(self):
        assert _spark_safe_query_expr("col in ('a', 'b')") == "col in ('a', 'b')"

    def test_equality_with_brackets_not_in_context_unchanged(self):
        assert _spark_safe_query_expr("a == 'test'") == "a == 'test'"


class TestSafeQuery:
    def test_simple_comparison(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        result = safe_query(df, "a > 3")
        assert list(result["a"]) == [4, 5]

    def test_string_equality(self):
        df = pd.DataFrame({"region": ["US", "UK", "FR"]})
        result = safe_query(df, "region == 'US'")
        assert len(result) == 1

    def test_in_with_list_syntax(self):
        df = pd.DataFrame({"status": ["active", "cancelled", "pending"]})
        result = safe_query(df, "status in ['active', 'pending']")
        assert len(result) == 2

    def test_in_with_tuple_syntax(self):
        df = pd.DataFrame({"status": ["active", "cancelled", "pending"]})
        result = safe_query(df, "status in ('active', 'pending')")
        assert len(result) == 2

    def test_compound_filter(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "x"]})
        result = safe_query(df, "a > 1 and b == 'x'")
        assert len(result) == 1
        assert result.iloc[0]["a"] == 3

    def test_not_equal(self):
        df = pd.DataFrame({"t": ["A", "B", "C"]})
        result = safe_query(df, "t != 'B'")
        assert len(result) == 2

    def test_numeric_range(self):
        df = pd.DataFrame({"v": [10, 50, 100, 200]})
        result = safe_query(df, "v >= 50 and v <= 100")
        assert list(result["v"]) == [50, 100]

    def test_empty_result(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = safe_query(df, "a > 100")
        assert len(result) == 0

    def test_not_in_list(self):
        df = pd.DataFrame({"x": ["a", "b", "c", "d"]})
        result = safe_query(df, "x not in ['a', 'b']")
        assert list(result["x"]) == ["c", "d"]

    def test_preserves_index(self):
        df = pd.DataFrame({"a": [10, 20, 30]}, index=[5, 6, 7])
        result = safe_query(df, "a > 10")
        assert list(result.index) == [6, 7]
