"""Single-call sidecar discovery for the interpretation layer.

The interpretation layer has three sidecars (``column_descriptions``,
``feature_meta``, ``feature_population_stats``) that consumers — c02
derivation, the dashboard prose renderer, the customer-profile app —
need to load together. Each consumer used to repeat the same boilerplate:
resolve a ``RunNamespace`` via ``from_env_or_latest``, swallow the
``None`` result, then call three separate ``load_*_sidecar`` helpers.

Two failure modes that the per-consumer pattern hides:

1. The ``RunNamespace.from_env_or_latest()`` call returns ``None`` because
   ``get_experiments_dir()`` resolved to the wrong path on this Databricks
   workspace. Cycle 013 D3 surfaced this — the consumer silently degrades
   to empty sidecars and writes ``NULL`` to the dashboard's
   ``eligibility_rules_prose`` column.
2. The namespace IS resolved but one or more sidecars are missing because
   the upstream notebook (NB00 / NB08 / gold materialization) hasn't run
   the writer cell. Same NULL-everywhere outcome.

This module centralises both: every consumer calls
``discover_interpretation_sidecars`` and gets a typed bundle that lets it
react to either failure deliberately. Crucially, when a sidecar is empty
or the namespace itself can't be resolved, the bundle attaches a
``WARNING``-level log line naming the path it tried so the operator can
fix the wiring instead of debugging a NULL column three layers downstream.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List, Mapping, Optional

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
    from customer_retention.stages.causal.column_descriptions_writer import (
        ColumnDescriptionRow,
    )
    from customer_retention.stages.causal.feature_meta_writer import FeatureMetaRow
    from customer_retention.stages.causal.interpretation.quantile_phrasing import (
        PopulationStats,
    )


@dataclass
class InterpretationSidecars:
    """Loaded sidecars + provenance.

    A consumer that gets this bundle can:
      - render predicate prose with ``feature_meta`` + ``population_stats``
        + ``column_descriptions`` directly,
      - inspect ``warnings`` to log a single human-readable line about
        which sidecar(s) were empty,
      - check ``namespace`` to know whether discovery itself failed
        (``None``) versus a resolved-but-empty run.
    """

    namespace: Optional["RunNamespace"]
    feature_meta: Mapping[str, "FeatureMetaRow"] = field(default_factory=dict)
    population_stats: Mapping[str, "PopulationStats"] = field(default_factory=dict)
    column_descriptions: Mapping[str, "ColumnDescriptionRow"] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def fully_populated(self) -> bool:
        return (
            self.namespace is not None
            and bool(self.feature_meta)
            and bool(self.population_stats)
            and bool(self.column_descriptions)
        )

    def emit_warnings(self, *, logger_: Optional[logging.Logger] = None) -> None:
        """Log every captured warning at WARNING level.

        Splits silent failure into a loud one so operators can fix the
        upstream wiring instead of debugging a NULL column three layers
        downstream. Idempotent — call from any consumer entrypoint.
        """
        target = logger_ or logger
        for line in self.warnings:
            target.warning("[interpretation] %s", line)


def discover_interpretation_sidecars(
    *,
    namespace: Optional["RunNamespace"] = None,
    experiments_root: Optional[Path] = None,
    composite_name: Optional[str] = None,
) -> InterpretationSidecars:
    """Resolve a namespace and load every interpretation sidecar.

    Resolution honours the file-tracked discovery chain:

    1. ``namespace`` (when explicitly passed) is used as-is.
    2. Otherwise ``RunNamespace.from_env_or_latest(experiments_root)``,
       which itself walks the env → explicit-root → project pointer →
       sentinel → latest-marker chain.

    Empty sidecars and missing namespaces append human-readable warnings
    to the returned bundle's ``warnings`` list — the consumer chooses
    when to surface them. ``composite_name`` is reserved for future
    feature_meta scoping; today's loaders return all rows.
    """
    from customer_retention.stages.causal.interpretation.sidecars import (
        load_column_descriptions_sidecar,
        load_feature_meta_sidecar,
        load_population_stats_sidecar,
    )

    warnings: List[str] = []

    if namespace is None:
        namespace = _resolve_namespace(experiments_root, warnings)

    if namespace is None:
        warnings.append(
            "RunNamespace discovery failed; interpretation sidecars are empty. "
            "Set `experiments_root` explicitly when constructing DerivationConfig "
            "or ensure NB00 wrote the project pointer (.cr_active_run.json)."
        )
        return InterpretationSidecars(namespace=None, warnings=warnings)

    feature_meta = load_feature_meta_sidecar(namespace) or {}
    population_stats = load_population_stats_sidecar(namespace) or {}
    column_descriptions = load_column_descriptions_sidecar(namespace) or {}

    if not feature_meta:
        warnings.append(
            f"feature_meta sidecar empty at {namespace.feature_meta_dir}. "
            "Re-run gold materialization to regenerate it."
        )
    if not population_stats:
        warnings.append(
            f"feature_population_stats sidecar empty at {namespace.feature_population_stats_dir}. "
            "Re-run NB08 post-split cell `3d126c02` to regenerate it."
        )
    if not column_descriptions:
        warnings.append(
            "column_descriptions sidecar empty (root-scoped). "
            "Re-run NB00 cell `de214303` (bootstrap_column_descriptions) — "
            "it now accepts an llm_endpoint fallback when the markdown is unreachable."
        )

    return InterpretationSidecars(
        namespace=namespace,
        feature_meta=feature_meta,
        population_stats=population_stats,
        column_descriptions=column_descriptions,
        warnings=warnings,
    )


def _resolve_namespace(
    experiments_root: Optional[Path],
    warnings: List[str],
) -> Optional["RunNamespace"]:
    try:
        from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
    except ImportError as exc:
        warnings.append(f"RunNamespace import failed: {exc}")
        return None
    try:
        return RunNamespace.from_env_or_latest(root=experiments_root)
    except Exception as exc:  # noqa: BLE001 — discovery is best-effort
        warnings.append(f"RunNamespace.from_env_or_latest raised {type(exc).__name__}: {exc}")
        return None


__all__ = [
    "InterpretationSidecars",
    "discover_interpretation_sidecars",
]
