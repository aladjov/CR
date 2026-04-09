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

The package is organized as a thin library that the four generated
notebooks under ``exploration_notebooks/causal_notebooks/`` orchestrate
(``c01_publish_definitions``, ``c02_archetype_derivation``,
``c03_approval_gate``, ``c04_snapshot_and_dashboard``). See
``docs/playbook_execution_data_model.md`` for the full specification and
``docs/causal_track_implementation_plan.md`` for the implementation plan.
"""

from .approval_gate import (
    ApprovalGateResult,
    StabilityDecision,
    auto_promote_stable,
    cosine_similarity,
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
from .derivation import (
    DerivationConfig,
    DerivationResult,
    derive_archetypes_and_policies,
)
from .llm_namer import (
    ArchetypeContext,
    ArchetypeNaming,
    DatabricksFoundationModelNamer,
    LLMNamer,
    PlaybookFitDecision,
    TemplateNamer,
    build_llm_namer,
)
from .playbook_mapper import (
    ArchetypeMapping,
    ArchetypeSummary,
    extract_features_from_text,
    map_archetypes_to_playbooks,
)
from .predicate_compiler import (
    collect_features,
    compile_predicate,
    predicate_to_sql,
)
from .rule_extractor import ExtractedRule, extract_eligibility_rules
from .shap_runner import (
    BackgroundSample,
    ShapRunResult,
    compute_shap_distributed,
    freeze_background,
)

__all__ = [
    "DEFAULT_FEATURE_CAP",
    "ApprovalGateResult",
    "ArchetypeContext",
    "ArchetypeMapping",
    "ArchetypeNaming",
    "ArchetypeSummary",
    "BackgroundSample",
    "ClusterCandidate",
    "ClusteringResult",
    "DatabricksFoundationModelNamer",
    "DerivationConfig",
    "DerivationResult",
    "ExtractedRule",
    "LLMNamer",
    "PlaybookFitDecision",
    "ShapRunResult",
    "StabilityDecision",
    "TemplateNamer",
    "auto_promote_stable",
    "build_llm_namer",
    "cluster_centroids_raw",
    "cluster_kmeans",
    "cluster_shap_centroids",
    "cluster_size_stats",
    "cluster_target_means",
    "collect_features",
    "compile_predicate",
    "compute_shap_distributed",
    "cosine_similarity",
    "derive_archetypes_and_policies",
    "extract_eligibility_rules",
    "extract_features_from_text",
    "freeze_background",
    "list_pending_review",
    "map_archetypes_to_playbooks",
    "predicate_to_sql",
    "select_top_shap_features",
]
