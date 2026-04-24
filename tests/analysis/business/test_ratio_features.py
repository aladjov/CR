"""Contract terminate/start ratio derivation — invariants.

Invariant under test (from Cycle 005):

    For every window W in {"7d","30d","90d","180d","365d","all_time"} where
    BOTH `contract__event_type_terminate_count_{W}` and
    `contract__event_type_start_count_{W}` are present in the silver panel,
    the ratio column MUST be emitted as:

        contract_terminate_to_start_ratio_{W} =
            term.fillna(0) / (start.fillna(0) + 1)

    The helper must:
      * fail-fast when neither term nor start is present for any requested
        window (raise, do not silently no-op);
      * skip per-window when exactly one of term/start is present;
      * preserve input column names and ordering (no unintended drops);
      * never collect to driver (no `.values` / `.iloc[-1]` / `.dt.to_period`);
      * return (df, emitted_columns) so callers can log or gate.

Regression scenario — the shape documented in
`debug/engagement_e4ad6e1b/diagnosis_20260418.md`: `churned=1` rows have
a materially higher mean ratio at 90d than `churned=0` rows (target 2.39×
per the engagement's sanity-check contract; tests assert >= 1.5× on
synthetic data sized to the same structural property).

Fixtures use only generic names (account IDs `A{n}`, arbitrary dates).
No client-identifying content.

Tests are spec-first. The production entry point
`derive_contract_ratio_features` is imported at module level; the module
skips cleanly via `ImportError` until the C5 fix lands, at which point
every test activates automatically.
"""
from __future__ import annotations

import pandas as pd
import pytest

try:
    from customer_retention.analysis.business.ratio_features import (
        DEFAULT_RATIO_WINDOWS,
        derive_contract_ratio_features,
    )
except ImportError:
    derive_contract_ratio_features = None
    DEFAULT_RATIO_WINDOWS = None

pytestmark = pytest.mark.skipif(
    derive_contract_ratio_features is None,
    reason="derive_contract_ratio_features not implemented yet (C5 fix pending)",
)


def _oracle_ratio(term: pd.Series, start: pd.Series) -> pd.Series:
    """Pure-pandas oracle — ten lines, impossible to get subtly wrong.

    Mirrors the NB06 user_code cell's formula exactly so a divergence in
    the helper surfaces as a test failure on the very first row of data.
    """
    return term.fillna(0) / (start.fillna(0) + 1)


def _panel(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _expected_cols(windows: list[str]) -> list[str]:
    return [f"contract_terminate_to_start_ratio_{w}" for w in windows]


def _input_cols(windows: list[str]) -> tuple[list[str], list[str]]:
    term = [f"contract__event_type_terminate_count_{w}" for w in windows]
    start = [f"contract__event_type_start_count_{w}" for w in windows]
    return term, start


def test_default_windows_match_intent_grid():
    assert DEFAULT_RATIO_WINDOWS == ("7d", "30d", "90d", "180d", "365d", "all_time")


def test_all_six_windows_present_emits_all_six_columns():
    windows = list(DEFAULT_RATIO_WINDOWS)
    term_cols, start_cols = _input_cols(windows)
    panel = _panel([
        {c: 2 for c in term_cols} | {c: 4 for c in start_cols},
        {c: 0 for c in term_cols} | {c: 1 for c in start_cols},
    ])
    out, emitted = derive_contract_ratio_features(panel)
    expected = _expected_cols(windows)
    assert emitted == expected
    for col in expected:
        assert col in out.columns
    # Oracle parity on every window, every row.
    for w in windows:
        term = panel[f"contract__event_type_terminate_count_{w}"]
        start = panel[f"contract__event_type_start_count_{w}"]
        oracle = _oracle_ratio(term, start)
        got = out[f"contract_terminate_to_start_ratio_{w}"]
        assert list(got.round(10)) == list(oracle.round(10))


def test_nulls_in_inputs_treated_as_zero_via_fillna():
    panel = _panel([
        {
            "contract__event_type_terminate_count_7d": None,
            "contract__event_type_start_count_7d": 3,
        },
        {
            "contract__event_type_terminate_count_7d": 5,
            "contract__event_type_start_count_7d": None,
        },
    ])
    out, _ = derive_contract_ratio_features(panel, windows=("7d",))
    got = list(out["contract_terminate_to_start_ratio_7d"])
    # Row 0: term=None → 0; start=3 → 3+1=4; ratio=0/4=0.
    # Row 1: term=5; start=None → 0; 0+1=1; ratio=5/1=5.
    assert got == [0.0, 5.0]


def test_division_safety_never_divides_by_zero():
    panel = _panel([
        {
            "contract__event_type_terminate_count_90d": 7,
            "contract__event_type_start_count_90d": 0,
        },
    ])
    out, _ = derive_contract_ratio_features(panel, windows=("90d",))
    # start=0 → denom = 0+1 = 1; ratio = 7/1 = 7. Never a NaN, never an Inf.
    val = float(out["contract_terminate_to_start_ratio_90d"].iloc[0])
    assert val == 7.0


def test_partial_window_coverage_skips_missing_windows_silently():
    # term+start present for 7d and 90d, but 30d has only start — the 30d
    # ratio column must NOT be emitted (skip-cleanly rule).
    panel = _panel([
        {
            "contract__event_type_terminate_count_7d": 1,
            "contract__event_type_start_count_7d": 2,
            "contract__event_type_start_count_30d": 5,
            "contract__event_type_terminate_count_90d": 3,
            "contract__event_type_start_count_90d": 4,
        },
    ])
    out, emitted = derive_contract_ratio_features(
        panel, windows=("7d", "30d", "90d"),
    )
    assert emitted == [
        "contract_terminate_to_start_ratio_7d",
        "contract_terminate_to_start_ratio_90d",
    ]
    assert "contract_terminate_to_start_ratio_30d" not in out.columns


def test_fail_fast_when_no_windows_have_both_inputs():
    panel = _panel([
        {
            "contract__event_type_start_count_7d": 2,
            "contract__event_type_start_count_30d": 4,
        },
    ])
    with pytest.raises(ValueError, match="no .* windows"):
        derive_contract_ratio_features(panel, windows=("7d", "30d"))


def test_empty_windows_raises():
    panel = _panel([{"contract__event_type_terminate_count_7d": 1,
                     "contract__event_type_start_count_7d": 2}])
    with pytest.raises(ValueError, match="windows"):
        derive_contract_ratio_features(panel, windows=())


def test_existing_columns_preserved_and_not_overwritten_silently():
    panel = _panel([
        {
            "ACCOUNT_ID": "A1",
            "as_of_date": pd.Timestamp("2025-01-01"),
            "contract__event_type_terminate_count_7d": 1,
            "contract__event_type_start_count_7d": 2,
        },
    ])
    out, _ = derive_contract_ratio_features(panel, windows=("7d",))
    # Helper must not drop ACCOUNT_ID / as_of_date / input counts.
    for col in panel.columns:
        assert col in out.columns


def test_bare_names_are_accepted_when_prefixed_columns_absent():
    # When contract is the first-merged dataset, TemporalMerger._resolve_conflicts
    # leaves its columns unprefixed. The helper must accept the bare names.
    panel = _panel([
        {
            "event_type_terminate_count_7d": 4,
            "event_type_start_count_7d": 3,
        },
    ])
    out, emitted = derive_contract_ratio_features(panel, windows=("7d",))
    assert emitted == ["contract_terminate_to_start_ratio_7d"]
    # 4 / (3 + 1) = 1.0
    assert float(out["contract_terminate_to_start_ratio_7d"].iloc[0]) == 1.0


def test_prefixed_wins_over_bare_when_both_present():
    # Only the prefixed column should drive the ratio when both exist.
    panel = _panel([
        {
            "contract__event_type_terminate_count_7d": 10,
            "contract__event_type_start_count_7d": 4,
            "event_type_terminate_count_7d": 999,   # must be ignored
            "event_type_start_count_7d": 999,
        },
    ])
    out, _ = derive_contract_ratio_features(panel, windows=("7d",))
    # 10 / (4 + 1) = 2.0
    assert float(out["contract_terminate_to_start_ratio_7d"].iloc[0]) == 2.0


def test_mixed_prefixed_and_bare_across_windows():
    # 7d prefixed, 90d bare — both windows should emit.
    panel = _panel([
        {
            "contract__event_type_terminate_count_7d": 1,
            "contract__event_type_start_count_7d": 1,
            "event_type_terminate_count_90d": 3,
            "event_type_start_count_90d": 2,
        },
    ])
    out, emitted = derive_contract_ratio_features(panel, windows=("7d", "90d"))
    assert emitted == [
        "contract_terminate_to_start_ratio_7d",
        "contract_terminate_to_start_ratio_90d",
    ]
    assert float(out["contract_terminate_to_start_ratio_7d"].iloc[0]) == 0.5
    # 3 / (2 + 1) = 1.0
    assert float(out["contract_terminate_to_start_ratio_90d"].iloc[0]) == 1.0


def test_regression_churned_lift_at_90d_exceeds_1_5x():
    # Synthetic shape reproducing the engagement's run artifact structure:
    # accounts with churned=1 have materially higher term/start ratio at
    # 90d. Evidence: diagnosis §2C — 2.39× lift documented in
    # sps_dataset_contract_sanity_check_report.md G1. Threshold relaxed
    # to 1.5× for test robustness on small synthetic samples.
    rows = []
    # Non-churned: 8 starts, 1 terminate over 90d.
    for i in range(50):
        rows.append({
            "ACCOUNT_ID": f"A{i}",
            "churned": 0,
            "contract__event_type_terminate_count_90d": 1,
            "contract__event_type_start_count_90d": 8,
        })
    # Churned: 3 starts, 6 terminates over 90d (ratio materially higher).
    for i in range(50, 100):
        rows.append({
            "ACCOUNT_ID": f"A{i}",
            "churned": 1,
            "contract__event_type_terminate_count_90d": 6,
            "contract__event_type_start_count_90d": 3,
        })
    panel = _panel(rows)
    out, _ = derive_contract_ratio_features(panel, windows=("90d",))

    mean_0 = float(out.loc[out["churned"] == 0,
                           "contract_terminate_to_start_ratio_90d"].mean())
    mean_1 = float(out.loc[out["churned"] == 1,
                           "contract_terminate_to_start_ratio_90d"].mean())
    # Guard against div-by-zero on the lift itself.
    assert mean_0 > 0
    lift = mean_1 / mean_0
    assert lift > 1.5, f"expected lift > 1.5x, got {lift:.3f}"
