from __future__ import annotations

import logging
from typing import Optional

from customer_retention.analysis.auto_explorer.project_context import KeyResolutionStep
from customer_retention.core.compat import pd, unique_overlap_counts

logger = logging.getLogger(__name__)

_ID_SUFFIXES = ("_ID", "_KEY", "_Id", "_Key", "_id", "_key")


def resolve_entity_keys(
    loaded_frames: dict[str, pd.DataFrame],
    resolutions: dict[str, list[KeyResolutionStep]],
) -> dict[str, pd.DataFrame]:
    result = dict(loaded_frames)
    for dataset_name, steps in resolutions.items():
        df = result[dataset_name]
        for step in steps:
            df = _apply_resolution_step(df, dataset_name, step, result)
        result[dataset_name] = df
    return result


def _apply_resolution_step(
    df: pd.DataFrame,
    dataset_name: str,
    step: KeyResolutionStep,
    all_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    if step.bridge_dataset not in all_frames:
        raise KeyError(
            f"Bridge dataset '{step.bridge_dataset}' not found in loaded frames"
        )
    bridge_df = all_frames[step.bridge_dataset]

    if step.source_key not in df.columns:
        raise KeyError(
            f"Source key column '{step.source_key}' not found in dataset '{dataset_name}'"
        )
    if step.bridge_key not in bridge_df.columns:
        raise KeyError(
            f"Bridge key column '{step.bridge_key}' not found in bridge dataset '{step.bridge_dataset}'"
        )
    if step.resolve_column not in bridge_df.columns:
        raise KeyError(
            f"Resolve column '{step.resolve_column}' not found in bridge dataset '{step.bridge_dataset}'"
        )

    bridge_subset = bridge_df[[step.bridge_key, step.resolve_column]].drop_duplicates(
        subset=[step.bridge_key]
    )

    rows_before = len(df)
    merged = df.merge(bridge_subset, left_on=step.source_key, right_on=step.bridge_key, how="inner")

    if step.source_key != step.bridge_key and step.bridge_key in merged.columns:
        merged = merged.drop(columns=[step.bridge_key])

    rows_after = len(merged)
    if rows_after == 0:
        raise ValueError(
            f"Resolution of '{dataset_name}' via '{step.bridge_dataset}' produced empty result — "
            f"no matching keys between '{step.source_key}' and '{step.bridge_key}'"
        )

    dropped = rows_before - rows_after
    if dropped > 0:
        logger.warning(
            "Key resolution for '%s' via '%s': dropped %d orphan rows (%d → %d)",
            dataset_name,
            step.bridge_dataset,
            dropped,
            rows_before,
            rows_after,
        )

    return merged


def suggest_key_resolutions(
    loaded_frames: dict[str, pd.DataFrame],
    entity_column: str,
) -> dict[str, list[KeyResolutionStep]]:
    suggestions: dict[str, list[KeyResolutionStep]] = {}

    datasets_needing_resolution = {
        name: df
        for name, df in loaded_frames.items()
        if entity_column not in df.columns
    }

    bridge_datasets = {
        name: df
        for name, df in loaded_frames.items()
        if entity_column in df.columns
    }

    for source_name, source_df in datasets_needing_resolution.items():
        source_id_cols = _id_like_columns(source_df)
        if not source_id_cols:
            continue

        best_step: Optional[KeyResolutionStep] = None
        best_coverage = 0.0

        for bridge_name, bridge_df in bridge_datasets.items():
            for col in source_id_cols:
                if col not in bridge_df.columns:
                    continue
                source_count, _bridge_count, matched = unique_overlap_counts(
                    source_df[col], bridge_df[col]
                )
                if source_count == 0:
                    continue
                coverage = matched / source_count
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_step = KeyResolutionStep(
                        bridge_dataset=bridge_name,
                        source_key=col,
                        bridge_key=col,
                        resolve_column=entity_column,
                    )

        if best_step is not None:
            suggestions[source_name] = [best_step]

    return suggestions


def _id_like_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if any(col.endswith(s) for s in _ID_SUFFIXES)]
