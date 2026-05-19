"""Causal modeling track.

Post-training stage that consumes the production churn model and produces the
Delta tables that feed the CSM-facing dashboard. Implements the data model
specified in ``docs/playbook_execution_data_model.md``:

- Definition layer (sub-layers 1, 2, 3, 4): ``playbook_catalog``,
  ``playbook_steps``, ``response_schemas``, ``vocabularies``,
  ``archetype_catalog``, ``eligibility_policy``, ``decision_policy``.
- Instance layer: ``eligibility_snapshot`` (per scoring run, per account, per
  playbook).
- Analytical-only DDL: ``assignments``, ``actions``, ``outcomes`` (defined for
  the writeback contract; not populated by this stage).

The package is organized as a thin library that the five hand-authored
notebooks under ``causal_notebooks/`` (project root, sibling of
``exploration_notebooks/``) orchestrate: ``c01_publish_definitions``,
``c02_archetype_derivation``, ``c03_approval_gate``,
``c04_batch_inference``, and ``c05_snapshot_and_dashboard``. The same
library code is also rendered as production scripts under
``generated_pipelines/{platform}/c0X_*.py`` via the ``s_c01..s_c05``
stage generators. See
``docs/playbook_execution_data_model.md`` for the full specification and
``docs/causal_track_implementation_plan.md`` for the implementation plan.
"""

from .approval_gate import (
    ApprovalGateResult,
    StabilityDecision,
    auto_promote_stable,
    cosine_similarity,
    expire_stale_pending,
    list_pending_review,
)
from .clusterer import (
    DEFAULT_FEATURE_CAP,
    ClusterCandidate,
    ClusteringResult,
    cluster_centroids_raw,
    cluster_kmeans,
    cluster_shap_centroids,
    cluster_size_stats,
    cluster_target_means,
    select_top_shap_features,
)
from .dashboard_profile_override import (
    ProfileOverrideResult,
    apply_profile_override,
    render_profile_sql,
)
from .dashboard_views import (
    DASHBOARD_DEVIATION_VIEW_NAMES,
    DASHBOARD_PROVENANCE_VIEW_NAMES,
    DASHBOARD_TEMPLATE_TABLE_NAMES,
    DASHBOARD_TEMPLATE_VIEW_NAMES,
    DASHBOARD_VIEW_NAMES,
    MaterializedViewSpec,
    materialize_view_as_table,
    publish_dashboard_views,
    refresh_dashboard_view_materializations,
)
from .derivation import (
    DEFAULT_FIT_AUTO_THRESHOLD,
    DEFAULT_FIT_REVIEW_THRESHOLD,
    FIT_TIER_AUTO,
    FIT_TIER_CATCH_ALL,
    FIT_TIER_MANUAL,
    FIT_TIER_REVIEW,
    DerivationConfig,
    DerivationResult,
    FitThresholds,
    derive_archetypes_and_policies,
)
from .llm_namer import (
    ArchetypeContext,
    ArchetypeNaming,
    DatabricksFoundationModelNamer,
    LLMNamer,
    PlaybookFitDecision,
    ProseOverlapMatcher,
    build_llm_namer,
)
from .playbook_mapper import (
    ArchetypeMapping,
    ArchetypeSummary,
    extract_features_from_text,
    map_archetypes_to_playbooks,
    prose_overlap_score,
)
from .predicate_compiler import (
    collect_features,
    compile_predicate,
    predicate_to_sql,
)
from .rule_extractor import ExtractedRule, extract_eligibility_rules
from .run_context_writer import (
    RunContextConfig,
    ensure_run_context_table,
    from_project_context,
    write_run_context,
)
from .shap_runner import (
    ShapRunResult,
    compute_shap_distributed,
    unwrap_tree_model,
)
from .snapshot_writer import (
    SnapshotConfig,
    SnapshotResult,
    apply_decision_policy,
    assign_archetype,
    build_eligibility_snapshot,
    compute_eligibility_id,
    compute_scoring_run_id,
    evaluate_eligibility,
    write_snapshot,
)
from .top_drivers_writer import (
    TopDriversConfig,
    TopDriversResult,
    compute_and_write_top_shap_drivers,
)

__all__ = [
    "DASHBOARD_DEVIATION_VIEW_NAMES",
    "DASHBOARD_PROVENANCE_VIEW_NAMES",
    "DASHBOARD_TEMPLATE_TABLE_NAMES",
    "DASHBOARD_TEMPLATE_VIEW_NAMES",
    "DASHBOARD_VIEW_NAMES",
    "DEFAULT_FEATURE_CAP",
    "ApprovalGateResult",
    "ArchetypeContext",
    "ArchetypeMapping",
    "ArchetypeNaming",
    "ArchetypeSummary",
    "ClusterCandidate",
    "ClusteringResult",
    "DatabricksFoundationModelNamer",
    "MaterializedViewSpec",
    "ProfileOverrideResult",
    "DEFAULT_FIT_AUTO_THRESHOLD",
    "DEFAULT_FIT_REVIEW_THRESHOLD",
    "FIT_TIER_AUTO",
    "FIT_TIER_CATCH_ALL",
    "FIT_TIER_MANUAL",
    "FIT_TIER_REVIEW",
    "DerivationConfig",
    "DerivationResult",
    "FitThresholds",
    "ExtractedRule",
    "LLMNamer",
    "PlaybookFitDecision",
    "RunContextConfig",
    "ShapRunResult",
    "SnapshotConfig",
    "SnapshotResult",
    "StabilityDecision",
    "TopDriversConfig",
    "TopDriversResult",
    "ProseOverlapMatcher",
    "apply_decision_policy",
    "apply_profile_override",
    "assign_archetype",
    "auto_promote_stable",
    "build_eligibility_snapshot",
    "build_llm_namer",
    "cluster_centroids_raw",
    "cluster_kmeans",
    "cluster_shap_centroids",
    "cluster_size_stats",
    "cluster_target_means",
    "collect_features",
    "compile_predicate",
    "compute_and_write_top_shap_drivers",
    "compute_eligibility_id",
    "compute_scoring_run_id",
    "compute_shap_distributed",
    "cosine_similarity",
    "derive_archetypes_and_policies",
    "ensure_run_context_table",
    "expire_stale_pending",
    "evaluate_eligibility",
    "extract_eligibility_rules",
    "extract_features_from_text",
    "from_project_context",
    "list_pending_review",
    "map_archetypes_to_playbooks",
    "materialize_view_as_table",
    "predicate_to_sql",
    "publish_dashboard_views",
    "refresh_dashboard_view_materializations",
    "render_profile_sql",
    "prose_overlap_score",
    "select_top_shap_features",
    "unwrap_tree_model",
    "write_run_context",
    "write_snapshot",
]
