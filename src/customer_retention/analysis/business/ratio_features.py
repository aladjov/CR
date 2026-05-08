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

PA-6 — Lane-1 registration companion:

    register_event_ratio_features(registry, dataset=..., windows=...,
        source_notebook=..., rationale=...)
    register_contract_ratio_features(registry, ...)
    register_subscription_ratio_features(registry, ...)

    Emits one ``add_silver_ratio(column, numerator, denominator, ...)``
    per window using the dataset-PREFIXED column names. Codegen's
    ``apply_derived_ratio`` reads the ``numerator`` / ``denominator``
    keys to materialise the ratio at silver-transform time; passing
    them through ``add_silver_ratio`` (rather than ``add_silver_derived``)
    is what makes the rec actually fire in production — the executor
    ignores the ``expression`` / ``source_columns`` form.

    Because the production path uses ``apply_derived_ratio``
    (``numerator / denominator.replace(0, NaN)``) and the in-cell math
    helper uses a safe-divide ``term.fillna(0) / (start.fillna(0) + 1)``
    formula, NB06's canonical exploration-vs-production parity comes
    from running the registered rec through ``apply_gold_transforms``
    (which calls ``apply_derived_ratio``) — not from the math helper.
    The math helper remains useful as a one-off NB04/NB06 inspection
    aid; for full parity, register the rec and let ``apply_gold_transforms``
    materialise the column.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

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


def register_event_ratio_features(
    registry: Any,
    *,
    dataset: str,
    windows: Sequence[str] = DEFAULT_RATIO_WINDOWS,
    source_notebook: str = "06_feature_opportunities",
    rationale: Optional[str] = None,
) -> list[str]:
    """Emit one ``add_silver_ratio`` rec per window for the dataset's
    terminate-to-start ratio (PA-6).

    Uses ``add_silver_ratio`` (not ``add_silver_derived``) so the rec
    parameters carry the ``numerator`` / ``denominator`` keys that
    ``gold_transform_applicator._derived_source_columns`` reads to
    plumb sources into ``apply_derived_ratio`` at codegen time. The
    ``add_silver_derived`` path leaves source columns in
    ``parameters["source_columns"]`` only — the executor's ratio
    handler does not look at that field, so a rec emitted via
    ``add_silver_derived`` would be silently no-op'd in production.

    Column names use the dataset-PREFIXED form so the rec round-trips
    through ``TemporalMerger`` (which prepends ``{dataset}__`` on
    column collision — the production case for any multi-source
    merge).

    Returns the list of emitted ratio column names so the caller can
    print or count them.
    """
    if not dataset:
        raise ValueError("register_event_ratio_features requires a non-empty dataset")
    if not windows:
        raise ValueError("register_event_ratio_features requires a non-empty windows sequence")
    emitted: list[str] = []
    for w in windows:
        term = f"{dataset}__event_type_terminate_count_{w}"
        start = f"{dataset}__event_type_start_count_{w}"
        ratio = f"{dataset}_terminate_to_start_ratio_{w}"
        per_window_rationale = rationale or (
            f"Per-window cancellation gradient for {dataset} at {w}: "
            f"terminate / start ratio with safe-divide via "
            f"`apply_derived_ratio` (NaN on zero start, median-imputed "
            f"downstream). Source columns are per-grid-date event_type "
            f"counts emitted by the bronze aggregator."
        )
        registry.add_silver_ratio(
            column=ratio,
            numerator=term,
            denominator=start,
            rationale=per_window_rationale,
            source_notebook=source_notebook,
        )
        emitted.append(ratio)
    return emitted


def register_contract_ratio_features(
    registry: Any,
    *,
    windows: Sequence[str] = DEFAULT_RATIO_WINDOWS,
    source_notebook: str = "06_feature_opportunities",
    rationale: Optional[str] = None,
) -> list[str]:
    return register_event_ratio_features(
        registry, dataset="contract", windows=windows,
        source_notebook=source_notebook, rationale=rationale,
    )


def register_subscription_ratio_features(
    registry: Any,
    *,
    windows: Sequence[str] = DEFAULT_RATIO_WINDOWS,
    source_notebook: str = "06_feature_opportunities",
    rationale: Optional[str] = None,
) -> list[str]:
    return register_event_ratio_features(
        registry, dataset="subscription", windows=windows,
        source_notebook=source_notebook, rationale=rationale,
    )


__all__ = [
    "DEFAULT_RATIO_WINDOWS",
    "derive_event_ratio_features",
    "derive_contract_ratio_features",
    "derive_subscription_ratio_features",
    "register_event_ratio_features",
    "register_contract_ratio_features",
    "register_subscription_ratio_features",
]
