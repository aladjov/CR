"""Silver-panel lifecycle ratio derivations.

Thin, test-locked helpers the NB06 `@cr:user_code` cells delegate to.
Putting the math in framework code (not in-line in the notebook) is what
makes the invariants CI-testable — user_code cells do not round-trip
into generated production scripts, but a framework helper does round-trip
through unit tests so any future silver-prefix or event-type-token drift
fails loudly in `tests/` rather than silently on a Databricks artifact.

Contract (Cycle 005, closes G7):

    derive_contract_ratio_features(df, windows=DEFAULT_RATIO_WINDOWS)
    derive_subscription_ratio_features(df, windows=DEFAULT_RATIO_WINDOWS)

    For every W in `windows` where BOTH of the inputs

        {dataset}__event_type_terminate_count_{W}   (preferred — dataset-prefixed)
        {dataset}__event_type_start_count_{W}

    — OR the bare-name fallback —

        event_type_terminate_count_{W}
        event_type_start_count_{W}

    are present on `df`, the helper adds

        {dataset}_terminate_to_start_ratio_{W} =
            term.fillna(0) / (start.fillna(0) + 1)

    The prefixed name is preferred; the bare name is used only when the
    prefixed column is absent. This mirrors `TemporalMerger._resolve_conflicts`,
    which only prepends `{dataset}__` on collision — so the base dataset's
    columns land prefixed only when another dataset also emitted the same
    name, and bare otherwise.

    Raises `ValueError` if `windows` is empty, or if none of the
    requested windows have both term+start columns (fail-fast — the cell
    ran but would emit zero ratio columns, which is the exact silent
    failure mode this cycle closes).

    Returns `(df_with_new_cols, emitted_column_names)`.
"""
from __future__ import annotations

from typing import Iterable, Sequence

from customer_retention.core.compat import DataFrame

DEFAULT_RATIO_WINDOWS: tuple[str, ...] = (
    "7d", "30d", "90d", "180d", "365d", "all_time",
)


def _resolve_input_col(df_columns, prefixed: str, bare: str) -> str | None:
    if prefixed in df_columns:
        return prefixed
    if bare in df_columns:
        return bare
    return None


def _pairs(
    df_columns,
    windows: Iterable[str],
    dataset: str,
) -> list[tuple[str, str | None, str | None, str]]:
    resolved = []
    for w in windows:
        term = _resolve_input_col(
            df_columns,
            f"{dataset}__event_type_terminate_count_{w}",
            f"event_type_terminate_count_{w}",
        )
        start = _resolve_input_col(
            df_columns,
            f"{dataset}__event_type_start_count_{w}",
            f"event_type_start_count_{w}",
        )
        ratio = f"{dataset}_terminate_to_start_ratio_{w}"
        resolved.append((w, term, start, ratio))
    return resolved


def derive_event_ratio_features(
    df: DataFrame,
    *,
    dataset: str,
    windows: Sequence[str] = DEFAULT_RATIO_WINDOWS,
) -> tuple[DataFrame, list[str]]:
    """Generic terminate/start ratio helper for any event-level dataset.

    `dataset` selects the input-column prefix (e.g. ``"contract"``,
    ``"subscription"``) AND the output-column prefix
    (``{dataset}_terminate_to_start_ratio_{W}``).
    """
    if not dataset:
        raise ValueError("derive_event_ratio_features requires a non-empty dataset")
    if not windows:
        raise ValueError("derive_event_ratio_features requires a non-empty windows sequence")

    plan = _pairs(df.columns, windows, dataset)
    usable = [
        (w, term, start, ratio)
        for (w, term, start, ratio) in plan
        if term is not None and start is not None
    ]
    if not usable:
        missing = [
            f"({dataset}__event_type_terminate_count_{w} | event_type_terminate_count_{w}, "
            f"{dataset}__event_type_start_count_{w} | event_type_start_count_{w})"
            for (w, _, _, _) in plan
        ]
        raise ValueError(
            f"derive_event_ratio_features({dataset!r}): no requested windows have both "
            f"term+start inputs on the panel. Looked for: {missing}. "
            "Bronze per-grid-date + value_counts=event_type (Cycles 002/003/004) "
            "must land before this cell can emit any ratios."
        )

    emitted: list[str] = []
    for _, term, start, ratio in usable:
        df[ratio] = df[term].fillna(0) / (df[start].fillna(0) + 1)
        emitted.append(ratio)
    return df, emitted


def derive_contract_ratio_features(
    df: DataFrame,
    *,
    windows: Sequence[str] = DEFAULT_RATIO_WINDOWS,
) -> tuple[DataFrame, list[str]]:
    return derive_event_ratio_features(df, dataset="contract", windows=windows)


def derive_subscription_ratio_features(
    df: DataFrame,
    *,
    windows: Sequence[str] = DEFAULT_RATIO_WINDOWS,
) -> tuple[DataFrame, list[str]]:
    return derive_event_ratio_features(df, dataset="subscription", windows=windows)


__all__ = [
    "DEFAULT_RATIO_WINDOWS",
    "derive_event_ratio_features",
    "derive_contract_ratio_features",
    "derive_subscription_ratio_features",
]
