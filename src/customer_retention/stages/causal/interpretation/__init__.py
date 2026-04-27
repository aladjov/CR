"""Deterministic business-interpretation layer.

Modules here translate structured lineage and population statistics into
prose the LLM namer and dashboard can consume without inventing values.

See ``docs/business_interpretation_layer_plan.md`` for the full design.
"""
from customer_retention.stages.causal.interpretation.archetype_context import (
    ContrastFeature,
    EnrichedArchetypeContext,
    EnrichedDriver,
    EnrichedPlaybook,
    build_enriched_context,
)
from customer_retention.stages.causal.interpretation.business_phrase import (
    render_business_phrase,
    render_window_phrase,
)
from customer_retention.stages.causal.interpretation.discovery import (
    InterpretationSidecars,
    discover_interpretation_sidecars,
)
from customer_retention.stages.causal.interpretation.enrichment_pipeline import (
    enrich_archetype_from_namespace,
)
from customer_retention.stages.causal.interpretation.feature_meta_builder import (
    FeatureLineage,
    build_feature_meta_rows,
    parse_aggregation_feature_name,
)
from customer_retention.stages.causal.interpretation.llm_prompt import (
    build_enriched_prompt_messages,
)
from customer_retention.stages.causal.interpretation.markdown_bootstrap import (
    parse_table_descriptions_md,
)
from customer_retention.stages.causal.interpretation.predicate_prose import (
    compile_predicate_prose,
)
from customer_retention.stages.causal.interpretation.prose_backfill import (
    ProseBackfillResult,
    backfill_eligibility_prose,
)
from customer_retention.stages.causal.interpretation.quantile_phrasing import (
    PopulationStats,
    quantile_phrase,
)
from customer_retention.stages.causal.interpretation.sidecars import (
    load_column_descriptions_sidecar,
    load_feature_meta_sidecar,
    load_population_stats_sidecar,
    write_column_descriptions_sidecar,
    write_feature_meta_sidecar,
    write_population_stats_sidecar,
)

__all__ = [
    "render_business_phrase",
    "render_window_phrase",
    "parse_table_descriptions_md",
    "FeatureLineage",
    "build_feature_meta_rows",
    "parse_aggregation_feature_name",
    "PopulationStats",
    "quantile_phrase",
    "ContrastFeature",
    "EnrichedArchetypeContext",
    "EnrichedDriver",
    "EnrichedPlaybook",
    "build_enriched_context",
    "backfill_eligibility_prose",
    "build_enriched_prompt_messages",
    "compile_predicate_prose",
    "discover_interpretation_sidecars",
    "enrich_archetype_from_namespace",
    "InterpretationSidecars",
    "ProseBackfillResult",
    "write_feature_meta_sidecar",
    "load_feature_meta_sidecar",
    "write_population_stats_sidecar",
    "load_population_stats_sidecar",
    "write_column_descriptions_sidecar",
    "load_column_descriptions_sidecar",
]
