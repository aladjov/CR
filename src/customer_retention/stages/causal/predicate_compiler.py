"""Compile JSON predicate trees into Spark ``Column`` expressions.

The rule extractor (``rule_extractor.py``) emits each archetype's eligibility
predicate as a structured JSON tree of the form::

    {"op": "or", "clauses": [
        {"op": "and", "clauses": [
            {"op": ">=", "feature": "tenure_days", "value": 365},
            {"op": "<",  "feature": "nps_score",   "value": 7},
        ]},
        ...
    ]}

The same JSON is stored on ``eligibility_policy.eligibility_rules`` so the
snapshot writer (Phase 3) can replay it cheaply at scoring time without
needing the original sklearn surrogate trees. ``compile_predicate`` walks
the tree and returns a single ``pyspark.sql.Column`` that the snapshot
writer composes into a ``DataFrame.filter(...)`` call — fully distributed,
no row-by-row Python.

A second helper, ``predicate_to_sql``, renders the same tree as a
human-readable SQL string for the dashboard's "why surfaced" view. It
shares the same recursion shape so a refactor to either output stays in
lock-step with the other.

Pure functions, no Spark imports at module level — the file imports
``pyspark.sql.functions`` lazily so unit tests can exercise the SQL
renderer without PySpark installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import Column


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_COMPARISON_OPS = {">=", "<=", ">", "<", "==", "!=", "in", "not_in", "is_null", "not_null"}
_LOGICAL_OPS = {"and", "or", "not"}
_LITERAL_TRUE = {"op": "true"}


def compile_predicate(predicate: Dict[str, Any]) -> "Column":
    """Convert a JSON predicate tree to a ``pyspark.sql.Column`` expression.

    Empty predicates and ``{"op": "true"}`` evaluate to a literal ``true``
    column so callers can pass them straight to ``df.filter`` without a
    null check. Unknown operators raise ``ValueError`` immediately rather
    than silently dropping rows.
    """
    from pyspark.sql import functions as F  # noqa: N812

    if not predicate:
        return F.lit(True)
    op = predicate.get("op")
    if op == "true":
        return F.lit(True)
    if op == "false":
        return F.lit(False)
    if op in _LOGICAL_OPS:
        return _compile_logical(predicate, F)
    if op in _COMPARISON_OPS:
        return _compile_comparison(predicate, F)
    raise ValueError(f"Unknown predicate operator {op!r}; expected one of {_LOGICAL_OPS | _COMPARISON_OPS}")


def predicate_to_sql(predicate: Dict[str, Any]) -> str:
    """Render the JSON predicate tree as a SQL ``WHERE`` fragment.

    Used by the dashboard's "why surfaced" column. The string is read by
    humans, not by Spark, so it favors readability over canonical form.
    """
    if not predicate:
        return "TRUE"
    op = predicate.get("op")
    if op == "true":
        return "TRUE"
    if op == "false":
        return "FALSE"
    if op == "and":
        clauses = predicate.get("clauses") or []
        if not clauses:
            return "TRUE"
        return "(" + " AND ".join(predicate_to_sql(c) for c in clauses) + ")"
    if op == "or":
        clauses = predicate.get("clauses") or []
        if not clauses:
            return "FALSE"
        return "(" + " OR ".join(predicate_to_sql(c) for c in clauses) + ")"
    if op == "not":
        clause = predicate.get("clause")
        return f"NOT ({predicate_to_sql(clause or {})})"
    if op in {">=", "<=", ">", "<", "==", "!="}:
        symbol = "=" if op == "==" else op
        return f"{_quote_identifier(predicate['feature'])} {symbol} {_render_literal(predicate.get('value'))}"
    if op == "in":
        values = predicate.get("values") or []
        return f"{_quote_identifier(predicate['feature'])} IN ({', '.join(_render_literal(v) for v in values)})"
    if op == "not_in":
        values = predicate.get("values") or []
        return f"{_quote_identifier(predicate['feature'])} NOT IN ({', '.join(_render_literal(v) for v in values)})"
    if op == "is_null":
        return f"{_quote_identifier(predicate['feature'])} IS NULL"
    if op == "not_null":
        return f"{_quote_identifier(predicate['feature'])} IS NOT NULL"
    raise ValueError(f"Unknown predicate operator {op!r}")


def collect_features(predicate: Dict[str, Any]) -> List[str]:
    """Return the unique feature names referenced anywhere in the tree.

    Used to populate ``eligibility_policy.requires_features`` so the
    snapshot writer can validate that all referenced columns exist before
    compiling the predicate.
    """
    features: List[str] = []
    _collect_features_recursive(predicate, features)
    seen: set[str] = set()
    out: List[str] = []
    for f in features:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


# ---------------------------------------------------------------------------
# Recursion helpers
# ---------------------------------------------------------------------------


def _compile_logical(predicate: Dict[str, Any], F) -> "Column":  # noqa: N803
    op = predicate["op"]
    if op == "not":
        clause = predicate.get("clause")
        if clause is None:
            return F.lit(True)
        return ~compile_predicate(clause)
    clauses = predicate.get("clauses") or []
    if not clauses:
        return F.lit(op == "and")
    compiled = [compile_predicate(c) for c in clauses]
    combined = compiled[0]
    for col in compiled[1:]:
        combined = combined & col if op == "and" else combined | col
    return combined


def _compile_comparison(predicate: Dict[str, Any], F) -> "Column":  # noqa: N803
    feature = predicate.get("feature")
    if not feature:
        raise ValueError(f"Comparison predicate missing 'feature': {predicate}")
    column = F.col(feature)
    op = predicate["op"]
    if op == "is_null":
        return column.isNull()
    if op == "not_null":
        return column.isNotNull()
    if op == "in":
        return column.isin(list(predicate.get("values") or []))
    if op == "not_in":
        return ~column.isin(list(predicate.get("values") or []))
    value = predicate.get("value")
    if op == ">=":
        return column >= F.lit(value)
    if op == "<=":
        return column <= F.lit(value)
    if op == ">":
        return column > F.lit(value)
    if op == "<":
        return column < F.lit(value)
    if op == "==":
        return column == F.lit(value)
    if op == "!=":
        return column != F.lit(value)
    raise ValueError(f"Unknown comparison op {op!r}")


def _collect_features_recursive(predicate: Dict[str, Any], out: List[str]) -> None:
    if not isinstance(predicate, dict):
        return
    op = predicate.get("op")
    if op in {"and", "or"}:
        for clause in predicate.get("clauses") or []:
            _collect_features_recursive(clause, out)
        return
    if op == "not":
        _collect_features_recursive(predicate.get("clause") or {}, out)
        return
    feature = predicate.get("feature")
    if feature:
        out.append(feature)


def _render_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value).replace("'", "''")
    return f"'{text}'"


def _quote_identifier(name: str) -> str:
    """Backtick-quote SQL identifiers so feature names with spaces survive rendering."""
    safe = name.replace("`", "``")
    return f"`{safe}`"
