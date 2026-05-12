from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from customer_retention.analysis.auto_explorer.project_context import KeyResolutionStep
from customer_retention.core.compat import head_as_list, pd, resolve_column_name, unique_overlap_counts
from customer_retention.parity import ApplyOpKind, apply_op

if TYPE_CHECKING:
    from customer_retention.analysis.auto_explorer.project_context import ProjectContext
    from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace

logger = logging.getLogger(__name__)

_ID_SUFFIXES = ("_ID", "_KEY", "_Id", "_Key", "_id", "_key")


_resolve_column_name = resolve_column_name


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


class _StaleStepError(KeyError):
    pass


def _validate_step_columns(
    df: pd.DataFrame,
    step: KeyResolutionStep,
    bridge_df: pd.DataFrame,
) -> None:
    missing = []
    if not _column_exists(df.columns, step.source_key):
        missing.append(f"source_key '{step.source_key}' not in source columns")
    if not _column_exists(bridge_df.columns, step.bridge_key):
        missing.append(f"bridge_key '{step.bridge_key}' not in bridge '{step.bridge_dataset}'")
    if not _column_exists(bridge_df.columns, step.resolve_column):
        missing.append(f"resolve_column '{step.resolve_column}' not in bridge '{step.bridge_dataset}'")
    if missing:
        raise _StaleStepError(
            f"KeyResolutionStep invalid — {'; '.join(missing)}. "
            f"Step: source_key='{step.source_key}', bridge='{step.bridge_dataset}', "
            f"bridge_key='{step.bridge_key}', resolve='{step.resolve_column}'"
        )


def _column_exists(columns, name: str) -> bool:
    try:
        _resolve_column_name(columns, name)
        return True
    except KeyError:
        return False


def _is_empty(df: pd.DataFrame) -> bool:
    return len(df.head(1)) == 0


def _apply_resolution_step(
    df: pd.DataFrame,
    dataset_name: str,
    step: KeyResolutionStep,
    all_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    if _column_exists(df.columns, step.resolve_column):
        return df

    if step.bridge_dataset not in all_frames:
        raise KeyError(
            f"Bridge dataset '{step.bridge_dataset}' not found in loaded frames"
        )
    bridge_df = all_frames[step.bridge_dataset]

    _validate_step_columns(df, step, bridge_df)

    source_key = _resolve_column_name(df.columns, step.source_key)
    bridge_key = _resolve_column_name(bridge_df.columns, step.bridge_key)
    resolve_col = _resolve_column_name(bridge_df.columns, step.resolve_column)

    bridge_subset = bridge_df[[bridge_key, resolve_col]].drop_duplicates(
        subset=[bridge_key]
    )

    merged = df.merge(bridge_subset, left_on=source_key, right_on=bridge_key, how="inner")

    if source_key != bridge_key and bridge_key in merged.columns:
        merged = merged.drop(columns=[bridge_key])

    if _is_empty(merged):
        raise ValueError(
            f"Resolution of '{dataset_name}' via '{step.bridge_dataset}' produced empty result — "
            f"no matching keys between '{step.source_key}' and '{step.bridge_key}'"
        )

    return merged


@apply_op(kind=ApplyOpKind.KEY_RESOLUTION)
def resolve_single_dataset_keys(
    df: pd.DataFrame,
    steps: list[KeyResolutionStep],
    namespace: "RunNamespace",
    project_ctx: "Optional[ProjectContext]" = None,
) -> pd.DataFrame:
    from customer_retention.analysis.auto_explorer.active_dataset_store import load_bridge_distributed

    for step in steps:
        if _column_exists(df.columns, step.resolve_column):
            continue
        bridge_df = load_bridge_distributed(namespace, step.bridge_dataset, project_ctx)
        try:
            df = _apply_resolution_step(df, "<inline>", step, {step.bridge_dataset: bridge_df})
        except _StaleStepError as exc:
            logger.warning("Skipping stale key resolution step: %s", exc)
    return df


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


def resolve_sample_ids_via_bridge(
    namespace: "RunNamespace",
    steps: list[KeyResolutionStep],
    sample_entity_ids: list,
    project_ctx: "Optional[ProjectContext]" = None,
) -> tuple[str, set]:
    from customer_retention.analysis.auto_explorer.active_dataset_store import load_bridge_distributed

    current_ids = set(sample_entity_ids)
    for step in reversed(steps):
        bridge_df = load_bridge_distributed(namespace, step.bridge_dataset, project_ctx)
        resolve_col = _resolve_column_name(bridge_df.columns, step.resolve_column)
        bridge_key = _resolve_column_name(bridge_df.columns, step.bridge_key)
        matched = bridge_df[bridge_df[resolve_col].isin(list(current_ids))]
        current_ids = set(head_as_list(matched[bridge_key], max(len(current_ids) * 100, 10_000)))
    source_key = steps[0].source_key
    return source_key, current_ids


def _id_like_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if any(col.endswith(s) for s in _ID_SUFFIXES)]
