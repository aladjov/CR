"""JSON sidecar read/write for the business-interpretation metadata tables.

The Phase 1 writers target Delta Lake (Spark-required). Most of this
repo's consumers — exploration notebooks, local pipeline, the causal
c02 derivation — run Python-only without a live Spark session, so we
also keep a JSON projection of each table next to the run's gold output.
The sidecar:

- uses the ``RunNamespace`` paths already provisioned
  (``feature_meta_dir``, ``feature_population_stats_dir``,
  ``column_descriptions_dir``)
- is written at generation time by the same code path that constructs
  the rows — gold template for ``feature_meta``, NB08 post-split for
  ``feature_population_stats``, NB00 bootstrap for
  ``column_descriptions``
- is read at consumption time by ``enrich_archetype_from_namespace`` so
  c02 can build ``EnrichedArchetypeContext`` without touching Spark

Every reader returns an empty mapping on missing / malformed input so
the entire interpretation layer continues to degrade gracefully when the
sidecar is stale or absent.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Sequence, Union

from customer_retention.stages.causal.interpretation.quantile_phrasing import PopulationStats

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
    from customer_retention.core.compat.remote_path import RemotePath
    from customer_retention.stages.causal.column_descriptions_writer import ColumnDescriptionRow
    from customer_retention.stages.causal.feature_meta_writer import FeatureMetaRow
    from customer_retention.stages.causal.population_stats import (
        PopulationStatsRow,
    )

    PathLike = Union[Path, RemotePath]


_FEATURE_META_FILENAME = "feature_meta.json"
_POPULATION_STATS_FILENAME = "feature_population_stats.json"
_COLUMN_DESCRIPTIONS_FILENAME = "column_descriptions.json"


def write_feature_meta_sidecar(
    namespace: "RunNamespace",
    composite_name: str,
    rows: Sequence["FeatureMetaRow"],
) -> "PathLike":
    """Persist ``rows`` as ``feature_meta.json`` under the run namespace."""
    directory = namespace.feature_meta_dir
    _ensure_dir(directory)
    path = directory / _FEATURE_META_FILENAME
    payload = {
        "composite_name": composite_name,
        "rows": [_feature_meta_to_dict(row) for row in rows],
    }
    _write_json(path, payload)
    logger.info("wrote feature_meta sidecar (%d rows) to %s", len(rows), path)
    return path


def load_feature_meta_sidecar(
    namespace: "RunNamespace",
    composite_name: Optional[str] = None,
) -> Dict[str, "FeatureMetaRow"]:
    """Return ``{feature_name: FeatureMetaRow}`` from the sidecar.

    Empty dict when the file is missing, malformed, or the stored
    ``composite_name`` does not match the one requested.
    """
    from customer_retention.stages.causal.feature_meta_writer import FeatureMetaRow

    payload = _read_json(namespace.feature_meta_dir / _FEATURE_META_FILENAME)
    if not payload:
        return {}
    if composite_name is not None and payload.get("composite_name") not in (None, composite_name):
        return {}
    result: Dict[str, FeatureMetaRow] = {}
    for entry in payload.get("rows") or []:
        name = entry.get("feature_name")
        if not name:
            continue
        result[name] = FeatureMetaRow(**_coerce_dataclass_fields(FeatureMetaRow, entry))
    return result


def write_population_stats_sidecar(
    namespace: "RunNamespace",
    rows: Sequence["PopulationStatsRow"],
) -> "PathLike":
    """Persist per-feature population summaries as JSON."""
    directory = namespace.feature_population_stats_dir
    _ensure_dir(directory)
    path = directory / _POPULATION_STATS_FILENAME
    _write_json(path, {"rows": [_population_row_to_dict(r) for r in rows]})
    logger.info("wrote population_stats sidecar (%d rows) to %s", len(rows), path)
    return path


def load_population_stats_sidecar(
    namespace: "RunNamespace",
) -> Dict[str, PopulationStats]:
    """Return ``{feature_name: PopulationStats}`` ready for ``quantile_phrase``."""
    payload = _read_json(namespace.feature_population_stats_dir / _POPULATION_STATS_FILENAME)
    if not payload:
        return {}
    result: Dict[str, PopulationStats] = {}
    for entry in payload.get("rows") or []:
        name = entry.get("feature_name")
        if not name:
            continue
        result[name] = PopulationStats(
            q05=entry.get("q05"),
            q25=entry.get("q25"),
            q50=entry.get("q50"),
            q75=entry.get("q75"),
            q95=entry.get("q95"),
        )
    return result


def write_column_descriptions_sidecar(
    namespace: "RunNamespace",
    rows: Sequence["ColumnDescriptionRow"],
) -> "PathLike":
    """Persist the column-descriptions registry as JSON (root-scoped, cross-run)."""
    directory = namespace.column_descriptions_dir
    _ensure_dir(directory)
    path = directory / _COLUMN_DESCRIPTIONS_FILENAME
    _write_json(path, {"rows": [_column_desc_to_dict(r) for r in rows]})
    logger.info("wrote column_descriptions sidecar (%d rows) to %s", len(rows), path)
    return path


def load_column_descriptions_sidecar(
    namespace: "RunNamespace",
) -> Dict[str, "ColumnDescriptionRow"]:
    """Return ``{column_name: ColumnDescriptionRow}`` keyed on raw column name."""
    from customer_retention.stages.causal.column_descriptions_writer import ColumnDescriptionRow

    payload = _read_json(namespace.column_descriptions_dir / _COLUMN_DESCRIPTIONS_FILENAME)
    if not payload:
        return {}
    result: Dict[str, ColumnDescriptionRow] = {}
    for entry in payload.get("rows") or []:
        name = entry.get("column_name")
        if not name:
            continue
        result[name] = ColumnDescriptionRow(**_coerce_dataclass_fields(ColumnDescriptionRow, entry))
    return result


def _feature_meta_to_dict(row: Any) -> Dict[str, Any]:
    return {f.name: getattr(row, f.name) for f in fields(row)}


def _column_desc_to_dict(row: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for f in fields(row):
        value = getattr(row, f.name)
        if isinstance(value, datetime):
            value = value.isoformat()
        out[f.name] = value
    return out


def _population_row_to_dict(row: Any) -> Dict[str, Any]:
    payload = {f.name: getattr(row, f.name) for f in fields(row)}
    payload.pop("top_categories", None)
    if getattr(row, "top_categories", None):
        payload["top_categories"] = [asdict(t) for t in row.top_categories]
    return payload


def _coerce_dataclass_fields(cls: Any, entry: Mapping[str, Any]) -> Dict[str, Any]:
    names = {f.name for f in fields(cls)}
    out: Dict[str, Any] = {k: v for k, v in entry.items() if k in names}
    for field_spec in fields(cls):
        if field_spec.type == "Optional[datetime]" and isinstance(out.get(field_spec.name), str):
            try:
                out[field_spec.name] = datetime.fromisoformat(out[field_spec.name])
            except ValueError:
                out[field_spec.name] = None
    return out


def _ensure_dir(path: "PathLike") -> None:
    mkdir = getattr(path, "mkdir", None)
    if mkdir is not None:
        mkdir(parents=True, exist_ok=True)


def _write_json(path: "PathLike", payload: Dict[str, Any]) -> None:
    atomic = getattr(path, "atomic_write_text", None)
    if atomic is None:
        from customer_retention.core.compat.remote_path import atomic_write_text

        atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str))
        return
    atomic(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _read_json(path: "PathLike") -> Optional[Dict[str, Any]]:
    try:
        text = path.read_text()
    except FileNotFoundError:
        return None
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        logger.warning("sidecar JSON at %s is malformed; ignoring", path)
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


__all__ = [
    "write_feature_meta_sidecar",
    "load_feature_meta_sidecar",
    "write_population_stats_sidecar",
    "load_population_stats_sidecar",
    "write_column_descriptions_sidecar",
    "load_column_descriptions_sidecar",
]
