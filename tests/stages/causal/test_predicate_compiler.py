"""Unit tests for ``predicate_compiler``.

The SQL renderer is pure Python and runs without PySpark. The Spark
``Column`` compiler is exercised through a thin mock that captures the
``F.col`` / ``F.lit`` chain so the test suite stays CI-portable (no
PySpark on CI).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from customer_retention.stages.causal import predicate_compiler

# ---------------------------------------------------------------------------
# predicate_to_sql
# ---------------------------------------------------------------------------


class TestPredicateToSql:
    def test_empty_predicate_renders_true(self):
        assert predicate_compiler.predicate_to_sql({}) == "TRUE"

    def test_explicit_true_predicate(self):
        assert predicate_compiler.predicate_to_sql({"op": "true"}) == "TRUE"

    def test_explicit_false_predicate(self):
        assert predicate_compiler.predicate_to_sql({"op": "false"}) == "FALSE"

    def test_simple_comparison(self):
        sql = predicate_compiler.predicate_to_sql(
            {"op": ">=", "feature": "tenure_days", "value": 365}
        )
        assert sql == "`tenure_days` >= 365"

    def test_equality_uses_single_equals(self):
        sql = predicate_compiler.predicate_to_sql(
            {"op": "==", "feature": "region", "value": "US"}
        )
        assert sql == "`region` = 'US'"

    def test_string_value_is_single_quoted_and_escaped(self):
        sql = predicate_compiler.predicate_to_sql(
            {"op": "==", "feature": "name", "value": "O'Reilly"}
        )
        assert sql == "`name` = 'O''Reilly'"

    def test_and_combines_clauses(self):
        sql = predicate_compiler.predicate_to_sql(
            {
                "op": "and",
                "clauses": [
                    {"op": ">=", "feature": "x", "value": 1},
                    {"op": "<", "feature": "y", "value": 5},
                ],
            }
        )
        assert sql == "(`x` >= 1 AND `y` < 5)"

    def test_or_combines_clauses(self):
        sql = predicate_compiler.predicate_to_sql(
            {
                "op": "or",
                "clauses": [
                    {"op": "==", "feature": "tier", "value": "A"},
                    {"op": "==", "feature": "tier", "value": "B"},
                ],
            }
        )
        assert sql == "(`tier` = 'A' OR `tier` = 'B')"

    def test_empty_and_renders_true(self):
        assert predicate_compiler.predicate_to_sql({"op": "and", "clauses": []}) == "TRUE"

    def test_empty_or_renders_false(self):
        assert predicate_compiler.predicate_to_sql({"op": "or", "clauses": []}) == "FALSE"

    def test_not_wraps_clause(self):
        sql = predicate_compiler.predicate_to_sql(
            {"op": "not", "clause": {"op": "==", "feature": "x", "value": 1}}
        )
        assert sql == "NOT (`x` = 1)"

    def test_in_uses_paren_list(self):
        sql = predicate_compiler.predicate_to_sql(
            {"op": "in", "feature": "tier", "values": ["A", "B", "C"]}
        )
        assert sql == "`tier` IN ('A', 'B', 'C')"

    def test_not_in_negates(self):
        sql = predicate_compiler.predicate_to_sql(
            {"op": "not_in", "feature": "tier", "values": ["X"]}
        )
        assert sql == "`tier` NOT IN ('X')"

    def test_is_null_and_not_null(self):
        assert (
            predicate_compiler.predicate_to_sql({"op": "is_null", "feature": "x"})
            == "`x` IS NULL"
        )
        assert (
            predicate_compiler.predicate_to_sql({"op": "not_null", "feature": "x"})
            == "`x` IS NOT NULL"
        )

    def test_null_literal_renders_null(self):
        sql = predicate_compiler.predicate_to_sql(
            {"op": "==", "feature": "x", "value": None}
        )
        assert sql == "`x` = NULL"

    def test_boolean_literals_render_uppercased(self):
        true_sql = predicate_compiler.predicate_to_sql(
            {"op": "==", "feature": "x", "value": True}
        )
        false_sql = predicate_compiler.predicate_to_sql(
            {"op": "==", "feature": "x", "value": False}
        )
        assert true_sql == "`x` = TRUE"
        assert false_sql == "`x` = FALSE"

    def test_unknown_operator_raises(self):
        with pytest.raises(ValueError):
            predicate_compiler.predicate_to_sql({"op": "xor", "feature": "x", "value": 1})

    def test_identifier_with_backticks_is_doubled(self):
        sql = predicate_compiler.predicate_to_sql(
            {"op": "==", "feature": "weird`name", "value": 1}
        )
        assert sql == "`weird``name` = 1"


# ---------------------------------------------------------------------------
# collect_features
# ---------------------------------------------------------------------------


class TestCollectFeatures:
    def test_simple_comparison(self):
        assert predicate_compiler.collect_features(
            {"op": ">=", "feature": "x", "value": 1}
        ) == ["x"]

    def test_nested_and_or(self):
        predicate = {
            "op": "or",
            "clauses": [
                {
                    "op": "and",
                    "clauses": [
                        {"op": ">=", "feature": "x", "value": 1},
                        {"op": "<", "feature": "y", "value": 2},
                    ],
                },
                {"op": "==", "feature": "z", "value": 3},
            ],
        }
        result = predicate_compiler.collect_features(predicate)
        assert sorted(result) == ["x", "y", "z"]

    def test_dedupes_repeated_features(self):
        predicate = {
            "op": "and",
            "clauses": [
                {"op": ">=", "feature": "x", "value": 1},
                {"op": "<", "feature": "x", "value": 5},
            ],
        }
        assert predicate_compiler.collect_features(predicate) == ["x"]

    def test_empty_predicate_returns_empty_list(self):
        assert predicate_compiler.collect_features({}) == []

    def test_handles_not_wrapper(self):
        predicate = {"op": "not", "clause": {"op": "==", "feature": "x", "value": 1}}
        assert predicate_compiler.collect_features(predicate) == ["x"]


# ---------------------------------------------------------------------------
# compile_predicate (mocked Spark functions module)
# ---------------------------------------------------------------------------


class _FakeColumn:
    """Captures composition so tests can assert what was built without PySpark."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __and__(self, other):
        return _FakeColumn(f"({self.name} AND {other.name})")

    def __or__(self, other):
        return _FakeColumn(f"({self.name} OR {other.name})")

    def __invert__(self):
        return _FakeColumn(f"NOT({self.name})")

    def __ge__(self, other):
        return _FakeColumn(f"({self.name} >= {other.name})")

    def __le__(self, other):
        return _FakeColumn(f"({self.name} <= {other.name})")

    def __gt__(self, other):
        return _FakeColumn(f"({self.name} > {other.name})")

    def __lt__(self, other):
        return _FakeColumn(f"({self.name} < {other.name})")

    def __eq__(self, other):  # type: ignore[override]
        return _FakeColumn(f"({self.name} == {other.name})")

    def __ne__(self, other):  # type: ignore[override]
        return _FakeColumn(f"({self.name} != {other.name})")

    def isNull(self):  # noqa: N802
        return _FakeColumn(f"{self.name}.isNull()")

    def isNotNull(self):  # noqa: N802
        return _FakeColumn(f"{self.name}.isNotNull()")

    def isin(self, values):  # noqa: N802
        return _FakeColumn(f"{self.name}.isin({values})")


class _FakeFunctions:
    @staticmethod
    def col(name):
        return _FakeColumn(f"col({name})")

    @staticmethod
    def lit(value):
        return _FakeColumn(f"lit({value!r})")


@pytest.fixture
def fake_spark_module(monkeypatch):
    """Replace ``pyspark.sql.functions`` so the compiler runs without Spark."""
    fake_pyspark_sql = MagicMock(name="pyspark.sql")
    fake_pyspark_sql.functions = _FakeFunctions
    monkeypatch.setitem(__import__("sys").modules, "pyspark.sql", fake_pyspark_sql)
    fake_pyspark = MagicMock(name="pyspark")
    fake_pyspark.sql = fake_pyspark_sql
    return fake_pyspark_sql


class TestCompilePredicate:
    def test_empty_predicate_returns_lit_true(self, fake_spark_module):
        result = predicate_compiler.compile_predicate({})
        assert "lit(True)" in result.name

    def test_simple_comparison(self, fake_spark_module):
        result = predicate_compiler.compile_predicate(
            {"op": ">=", "feature": "x", "value": 5}
        )
        assert "col(x)" in result.name and "lit(5)" in result.name and ">=" in result.name

    def test_logical_and(self, fake_spark_module):
        result = predicate_compiler.compile_predicate(
            {
                "op": "and",
                "clauses": [
                    {"op": ">=", "feature": "x", "value": 1},
                    {"op": "<", "feature": "y", "value": 2},
                ],
            }
        )
        assert "AND" in result.name

    def test_logical_or(self, fake_spark_module):
        result = predicate_compiler.compile_predicate(
            {
                "op": "or",
                "clauses": [
                    {"op": "==", "feature": "x", "value": 1},
                    {"op": "==", "feature": "y", "value": 2},
                ],
            }
        )
        assert "OR" in result.name

    def test_not_inverts(self, fake_spark_module):
        result = predicate_compiler.compile_predicate(
            {"op": "not", "clause": {"op": "==", "feature": "x", "value": 1}}
        )
        assert "NOT" in result.name

    def test_is_null(self, fake_spark_module):
        result = predicate_compiler.compile_predicate({"op": "is_null", "feature": "x"})
        assert "isNull" in result.name

    def test_in_operator(self, fake_spark_module):
        result = predicate_compiler.compile_predicate(
            {"op": "in", "feature": "tier", "values": ["A", "B"]}
        )
        assert "isin" in result.name

    def test_unknown_operator_raises(self, fake_spark_module):
        with pytest.raises(ValueError):
            predicate_compiler.compile_predicate({"op": "xor", "feature": "x", "value": 1})

    def test_missing_feature_raises(self, fake_spark_module):
        with pytest.raises(ValueError):
            predicate_compiler.compile_predicate({"op": ">=", "value": 5})
