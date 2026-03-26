from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from customer_retention.analysis.auto_explorer.project_context import ProjectContext
from customer_retention.core.compat import head_as_list, pd

_UNIT_ID_PATTERNS = [
    re.compile(r"contract_?id", re.IGNORECASE),
    re.compile(r"subscription_?id", re.IGNORECASE),
    re.compile(r"membership_?id", re.IGNORECASE),
    re.compile(r"policy_?id", re.IGNORECASE),
    re.compile(r"plan_?id", re.IGNORECASE),
    re.compile(r"service_?id", re.IGNORECASE),
]

_LIFECYCLE_DATE_PATTERNS = [
    re.compile(r"contract_?end_?date", re.IGNORECASE),
    re.compile(r"termination_?date", re.IGNORECASE),
    re.compile(r"terminated_?date", re.IGNORECASE),
    re.compile(r"billing_?termination_?date", re.IGNORECASE),
    re.compile(r"expiry_?date", re.IGNORECASE),
    re.compile(r"subscription_?end_?date", re.IGNORECASE),
    re.compile(r"cancellation_?date", re.IGNORECASE),
    re.compile(r"end_?date", re.IGNORECASE),
]

_START_DATE_PATTERNS = [
    re.compile(r"contract_?start_?date", re.IGNORECASE),
    re.compile(r"subscription_?start_?date", re.IGNORECASE),
    re.compile(r"start_?date", re.IGNORECASE),
    re.compile(r"activated_?date", re.IGNORECASE),
]

_STATUS_PATTERNS = [
    re.compile(r"contract_?status", re.IGNORECASE),
    re.compile(r"subscription_?status", re.IGNORECASE),
    re.compile(r"status", re.IGNORECASE),
    re.compile(r"state", re.IGNORECASE),
]

_TERMINATED_VALUE_PATTERNS = [
    re.compile(r"cancel", re.IGNORECASE),
    re.compile(r"terminat", re.IGNORECASE),
    re.compile(r"expir", re.IGNORECASE),
    re.compile(r"closed", re.IGNORECASE),
    re.compile(r"inactive", re.IGNORECASE),
    re.compile(r"churned", re.IGNORECASE),
]


@dataclass
class ServiceUnitConfig:
    dataset_name: str
    unit_id_column: str
    entity_column: str
    anchor_date_column: str
    secondary_anchor_columns: list[str] = field(default_factory=list)
    status_column: Optional[str] = None
    terminated_statuses: list[str] = field(default_factory=list)
    start_date_column: Optional[str] = None


@dataclass
class DatasetLinkage:
    tier: str  # "contract_linked" or "account_only"
    link_method: str  # "self", "direct_fk", "key_resolution", "entity_column"
    fk_column: Optional[str] = None
    note: Optional[str] = None


def _match_column(columns: list[str], patterns: list[re.Pattern]) -> Optional[str]:
    for pat in patterns:
        for col in columns:
            if pat.fullmatch(col):
                return col
    for pat in patterns:
        for col in columns:
            if pat.search(col):
                return col
    return None


def _match_all_columns(columns: list[str], patterns: list[re.Pattern]) -> list[str]:
    matched = []
    for col in columns:
        for pat in patterns:
            if pat.search(col):
                matched.append(col)
                break
    return matched


def _detect_terminated_statuses(df: pd.DataFrame, status_col: str) -> list[str]:
    if status_col not in df.columns:
        return []
    unique_vals = head_as_list(df[status_col].dropna().unique(), 200)
    return [str(v) for v in unique_vals if any(p.search(str(v)) for p in _TERMINATED_VALUE_PATTERNS)]


def detect_service_unit(
    context: ProjectContext, loaded_frames: dict[str, pd.DataFrame]
) -> Optional[ServiceUnitConfig]:
    entity_col = context.entity_column
    if not entity_col:
        return None

    best: Optional[ServiceUnitConfig] = None
    best_score = 0

    for ds_name, entry in context.datasets.items():
        if ds_name not in loaded_frames:
            continue
        df = loaded_frames[ds_name]
        cols = list(df.columns)

        unit_id = _match_column(cols, _UNIT_ID_PATTERNS)
        if not unit_id:
            continue

        has_entity_fk = entity_col in cols or any(
            entity_col.lower() == c.lower() for c in cols
        )
        if not has_entity_fk and not entry.key_resolution:
            continue

        entity_fk = next(
            (c for c in cols if c.lower() == entity_col.lower()), entity_col
        )

        anchor_candidates = _match_all_columns(cols, _LIFECYCLE_DATE_PATTERNS)
        if not anchor_candidates:
            continue

        anchor_col = anchor_candidates[0]
        secondary = anchor_candidates[1:] if len(anchor_candidates) > 1 else []
        start_col = _match_column(cols, _START_DATE_PATTERNS)
        status_col = _match_column(cols, _STATUS_PATTERNS)
        terminated = _detect_terminated_statuses(df, status_col) if status_col else []

        score = len(anchor_candidates) + (1 if status_col else 0) + (1 if start_col else 0) + (2 if terminated else 0)
        if score > best_score:
            best_score = score
            best = ServiceUnitConfig(
                dataset_name=ds_name,
                unit_id_column=unit_id,
                entity_column=entity_fk,
                anchor_date_column=anchor_col,
                secondary_anchor_columns=secondary,
                status_column=status_col,
                terminated_statuses=terminated,
                start_date_column=start_col,
            )

    return best


def classify_dataset_linkage(
    context: ProjectContext,
    service_unit_config: ServiceUnitConfig,
    loaded_frames: dict[str, pd.DataFrame],
) -> dict[str, DatasetLinkage]:
    unit_id = service_unit_config.unit_id_column
    su_dataset = service_unit_config.dataset_name
    result: dict[str, DatasetLinkage] = {}

    for ds_name, entry in context.datasets.items():
        if ds_name not in loaded_frames:
            continue
        cols = list(loaded_frames[ds_name].columns)

        if ds_name == su_dataset:
            result[ds_name] = DatasetLinkage(tier="contract_linked", link_method="self")
            continue

        if any(c.lower() == unit_id.lower() for c in cols):
            fk = next(c for c in cols if c.lower() == unit_id.lower())
            result[ds_name] = DatasetLinkage(tier="contract_linked", link_method="direct_fk", fk_column=fk)
            continue

        has_bridge_to_su = any(
            step.bridge_dataset == su_dataset for step in entry.key_resolution
        )
        if has_bridge_to_su:
            result[ds_name] = DatasetLinkage(tier="contract_linked", link_method="key_resolution")
            continue

        result[ds_name] = DatasetLinkage(
            tier="account_only",
            link_method="entity_column",
            note=f"No {unit_id} FK found; using entity column",
        )

    return result
