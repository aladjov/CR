"""Auto-apply registered landing-time transforms during exploration.

The registered overrides chain — `registry.add_landing_filter` and
`registry.add_landing_lifecycle_enrichment` — is the single source of
truth for landing-time per-dataset transforms. Production codegen
replays the chain at landing time; this helper replays the same chain at
exploration-load time so the operator does not need to maintain Lane-2
mutation cells in NB00.

Invariant: any dataset whose `datasets[name]` already points at a
``global_temp.*`` view (i.e. an operator Lane-2 cell already mutated it)
is skipped. The helper never double-applies.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


def replay_registered_landing_steps(
    datasets: Dict[str, Any],
    registry: Any,
    namespace: Any,
    *,
    original_datasets: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    if registry is None or getattr(registry, "landing", None) is None:
        return datasets

    from customer_retention.core.compat import load_spark_table, register_temp_view

    originals = (
        dict(original_datasets)
        if original_datasets is not None
        else dict(getattr(namespace, "original_datasets", None) or {})
    )

    filters_by_ds: Dict[str, list] = {}
    for rec in registry.landing.filters:
        ds = rec.parameters.get("dataset")
        if ds:
            filters_by_ds.setdefault(ds, []).append(rec)
    lifecycles_by_ds: Dict[str, list] = {}
    for rec in registry.landing.lifecycle_enrichments:
        ds = rec.parameters.get("dataset")
        if ds:
            lifecycles_by_ds.setdefault(ds, []).append(rec)

    for name in list(datasets.keys()):
        src = datasets[name]
        if isinstance(src, str) and src.startswith("global_temp."):
            continue

        filters = filters_by_ds.get(name, [])
        lifecycles = lifecycles_by_ds.get(name, [])
        if not filters and not lifecycles:
            continue

        upstream = originals.get(name, src)
        if not isinstance(upstream, str):
            continue

        df = load_spark_table(upstream)

        for rec in filters:
            predicate = rec.parameters.get("predicate")
            if predicate:
                df = df.filter(predicate)

        view_name: Optional[str] = None
        for rec in lifecycles:
            from customer_retention.stages.lifecycle.config import LifecycleEnrichmentConfig
            from customer_retention.stages.lifecycle.enrich import enrich_lifecycle_dataset

            cfg_dict = rec.parameters.get("config") or {}
            cfg = LifecycleEnrichmentConfig.from_dict(cfg_dict)
            df = enrich_lifecycle_dataset(df, config=cfg)
            view_name = cfg.enriched_view_name

        if not view_name:
            view_name = f"{name}__landing_replay"

        datasets[name] = register_temp_view(
            df, view_name, purpose="exploration_landing_replay",
        )

    return datasets
