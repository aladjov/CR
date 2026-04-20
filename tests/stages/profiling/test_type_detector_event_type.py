"""Event-type classification invariants (Cycle 002).

Invariant under test: a 2-value column classifies as BINARY only when its
values match one of the known binary sets (0/1, True/False, yes/no, y/n,
true/false, case variants). A 2-value string column with non-boolean labels
(e.g. {'start', 'terminate'}) must classify as CATEGORICAL_NOMINAL so
downstream bronze aggregators emit per-value counts (`event_type_<value>_count_*`).

Regression target: run `spschurn-e4ad6e1b` produced
`contract_findings.columns.event_type.inferred_type = 'binary'` with the
evidence string "Exactly 2 unique values (non-standard): {'start'}". That
came from a catch-all fallback that mapped ANY 2-value column to BINARY,
including string columns whose values weren't in the binary-set allowlist.

All fixtures use generic names (`event_type`, `status`, `color`, etc.) — no
client identifiers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from customer_retention.core.config import ColumnType
from customer_retention.stages.profiling import TypeDetector

_KNOWN_BINARY_SETS = [
    frozenset({0, 1}),
    frozenset({0.0, 1.0}),
    frozenset({True, False}),
    frozenset({"0", "1"}),
    frozenset({"yes", "no"}),
    frozenset({"Yes", "No"}),
    frozenset({"YES", "NO"}),
    frozenset({"true", "false"}),
    frozenset({"True", "False"}),
    frozenset({"TRUE", "FALSE"}),
    frozenset({"y", "n"}),
    frozenset({"Y", "N"}),
]


def oracle_expected_two_value_type(values: set, is_numeric: bool) -> ColumnType:
    """Expected inferred_type for a 2-distinct-value column — computed from first principles.

    Contract enforced by C2 fix: BINARY iff values match a known coded set.
    Numeric 2-value columns fall through to NUMERIC_DISCRETE (distinct<=20).
    String 2-value columns fall through to CATEGORICAL_NOMINAL.
    """
    frozen = frozenset(values)
    for known in _KNOWN_BINARY_SETS:
        if frozen == known or frozen <= known:
            return ColumnType.BINARY
    return ColumnType.NUMERIC_DISCRETE if is_numeric else ColumnType.CATEGORICAL_NOMINAL


class TestIsBinaryKnownSets:
    @pytest.mark.parametrize("values", [
        [0, 1, 0, 1, 1, 0],
        [True, False, True, True, False],
        ["yes", "no", "yes", "no"],
        ["Yes", "No", "Yes", "No"],
        ["YES", "NO", "YES", "YES"],
        ["true", "false", "true", "false"],
        ["True", "False", "True"],
        ["TRUE", "FALSE", "TRUE"],
        ["y", "n", "y", "n"],
        ["Y", "N", "Y", "N"],
        ["0", "1", "0", "1"],
    ])
    def test_known_binary_still_binary(self, values):
        series = pd.Series(values)
        result = TypeDetector().detect_type(series, "active")
        assert result.inferred_type == ColumnType.BINARY


class TestNonStandardTwoValueString:
    @pytest.mark.parametrize("values,expected", [
        (["start", "terminate"], ColumnType.CATEGORICAL_NOMINAL),
        (["active", "inactive"], ColumnType.CATEGORICAL_NOMINAL),
        (["open", "closed"], ColumnType.CATEGORICAL_NOMINAL),
        (["enabled", "disabled"], ColumnType.CATEGORICAL_NOMINAL),
        (["paid", "unpaid"], ColumnType.CATEGORICAL_NOMINAL),
    ])
    def test_two_value_string_is_categorical_nominal(self, values, expected):
        series = pd.Series(values * 10)
        result = TypeDetector().detect_type(series, "status_col")
        assert result.inferred_type == expected, (
            f"expected {expected} for {values}, got {result.inferred_type} "
            f"evidence={result.evidence}"
        )


class TestNonStandardTwoValueNumeric:
    @pytest.mark.parametrize("values", [
        [0.5, 1.5, 0.5, 1.5],
        [-1, 1, -1, 1, -1],
        [1, 2, 1, 2, 1, 2],
        [10, 20, 10, 20],
    ])
    def test_two_value_numeric_falls_to_numeric_discrete(self, values):
        series = pd.Series(values)
        result = TypeDetector().detect_type(series, "metric_col")
        assert result.inferred_type == ColumnType.NUMERIC_DISCRETE


class TestOracleParity:
    @pytest.mark.parametrize("values,is_numeric", [
        ({0, 1}, True),
        ({True, False}, True),
        ({"yes", "no"}, False),
        ({"start", "terminate"}, False),
        ({"active", "inactive"}, False),
        ({-1, 1}, True),
        ({0.5, 1.5}, True),
    ])
    def test_detector_matches_oracle(self, values, is_numeric):
        series = pd.Series(list(values) * 5)
        result = TypeDetector().detect_type(series, "generic_col")
        expected = oracle_expected_two_value_type(values, is_numeric)
        assert result.inferred_type == expected


class TestContractEventTypeRegression:
    """Reproduces the exact failure shape from run spschurn-e4ad6e1b.

    `contract_findings.yaml` showed:
        columns.event_type.inferred_type = binary
        evidence = ["Exactly 2 unique values (non-standard): {'start'}"]

    Synthetic contract event log with the same structural property
    (2 string values on `event_type`) must classify as CATEGORICAL_NOMINAL.
    """

    def test_contract_event_type_shape(self):
        rng = np.random.default_rng(seed=1)
        values = rng.choice(["start", "terminate"], size=200, p=[0.6, 0.4])
        series = pd.Series(values)
        result = TypeDetector().detect_type(series, "event_type")
        assert result.inferred_type == ColumnType.CATEGORICAL_NOMINAL
        assert result.inferred_type != ColumnType.BINARY

    def test_subscription_event_type_shape(self):
        rng = np.random.default_rng(seed=2)
        values = rng.choice(["start", "terminate"], size=500, p=[0.67, 0.33])
        series = pd.Series(values)
        result = TypeDetector().detect_type(series, "event_type")
        assert result.inferred_type == ColumnType.CATEGORICAL_NOMINAL


class TestIsBinaryDirect:
    """Direct unit coverage for the TypeDetector.is_binary public method."""

    def test_is_binary_false_for_two_string_values(self):
        detector = TypeDetector()
        series = pd.Series(["start", "terminate"] * 10)
        assert detector.is_binary(series, distinct_count=2) is False

    def test_is_binary_true_for_zero_one(self):
        detector = TypeDetector()
        series = pd.Series([0, 1] * 10)
        assert detector.is_binary(series, distinct_count=2) is True

    def test_is_binary_false_for_three_distinct(self):
        detector = TypeDetector()
        series = pd.Series(["a", "b", "c"])
        assert detector.is_binary(series, distinct_count=3) is False
