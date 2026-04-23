"""Delta table schemas for the causal modeling track.

Single source of truth for the shape of every table in the playbook execution
data model (``docs/playbook_execution_data_model.md``). Other modules in this
package import from here when reading, writing, validating, or constructing
empty DataFrames.

Each schema is a module-level function returning a fresh ``StructType`` so the
module can be imported without ``pyspark`` initialized at import time. The
``ALL_SCHEMAS`` registry at the bottom maps logical table names to their
schema functions for tooling and parametrized tests.

Design notes:

- **JSON-stringified columns** are used for highly variable or rarely-queried
  payloads (``derivation_params``, ``eligibility_rules``, ``eligibility_evidence``,
  ``suppression_rules``, etc.). Frequently-queried nested fields are typed
  with native ``StructType`` / ``MapType`` / ``ArrayType`` so SQL views can
  reach them with dot notation without ``from_json``.

- **Enums are stored as strings.** Delta does not have a native enum type.
  Validation against the allowed value set is enforced in Python loaders and
  may later be tightened with Delta CHECK constraints. The vocabularies the
  doc enumerates (assignment_mechanism, status, action_type, etc.) live in
  ``enums.py`` once that module is added in a later phase.

- **Three tables in this module are DDL-only for the initial release**:
  ``assignments``, ``actions``, ``outcomes``. Their schemas are fully defined
  so the writeback contract can be honored once the CSM-tool transport is
  agreed, but no code in this stage populates them initially. They are
  documented here so consumers (analyst notebooks, the eventual ingest job)
  have a single reference for the wire shape.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict

if TYPE_CHECKING:  # pragma: no cover - import only for type hints
    from pyspark.sql.types import StructType


def _types():
    """Lazy import of pyspark types so the module is importable without Spark."""
    from pyspark.sql.types import (
        ArrayType,
        BooleanType,
        DoubleType,
        IntegerType,
        LongType,
        MapType,
        StringType,
        StructField,
        StructType,
        TimestampType,
    )

    return {
        "ArrayType": ArrayType,
        "BooleanType": BooleanType,
        "DoubleType": DoubleType,
        "IntegerType": IntegerType,
        "LongType": LongType,
        "MapType": MapType,
        "StringType": StringType,
        "StructField": StructField,
        "StructType": StructType,
        "TimestampType": TimestampType,
    }


# ---------------------------------------------------------------------------
# Reusable nested types
# ---------------------------------------------------------------------------


def _shap_feature_struct():
    """One ranked SHAP driver. Used by archetype_catalog and eligibility_snapshot."""
    t = _types()
    return t["StructType"](
        [
            t["StructField"]("feature", t["StringType"](), False),
            t["StructField"]("mean_shap", t["DoubleType"](), True),
            t["StructField"]("mean_value", t["DoubleType"](), True),
            t["StructField"]("direction", t["StringType"](), True),  # "positive" / "negative"
        ]
    )


def _per_account_shap_driver_struct():
    """Per-row SHAP driver as surfaced on the dashboard."""
    t = _types()
    return t["StructType"](
        [
            t["StructField"]("feature", t["StringType"](), False),
            t["StructField"]("value", t["DoubleType"](), True),
            t["StructField"]("shap_contribution", t["DoubleType"](), True),
            t["StructField"]("direction", t["StringType"](), True),
        ]
    )


def _feature_threshold_struct():
    """p25 / p50 / p75 of one driver inside one cluster."""
    t = _types()
    return t["StructType"](
        [
            t["StructField"]("p25", t["DoubleType"](), True),
            t["StructField"]("p50", t["DoubleType"](), True),
            t["StructField"]("p75", t["DoubleType"](), True),
        ]
    )


# ---------------------------------------------------------------------------
# Definition layer — sub-layer 1 (intervention design, slow-changing, YAML→Delta)
# ---------------------------------------------------------------------------


def playbook_catalog_schema() -> "StructType":
    """§1.1 — Versioned registry of playbooks as intervention designs.

    Sourced from YAML files in the playbooks volume; one row per
    ``(playbook_id, version)``. Slow-changing: a new version is a new row,
    never an overwrite.
    """
    t = _types()
    return t["StructType"](
        [
            t["StructField"]("playbook_id", t["StringType"](), False),
            t["StructField"]("version", t["StringType"](), False),
            t["StructField"]("name", t["StringType"](), False),
            t["StructField"]("description", t["StringType"](), True),
            t["StructField"]("cost_per_customer_default", t["DoubleType"](), True),
            t["StructField"]("expected_uplift_pct_default", t["DoubleType"](), True),
            t["StructField"]("outcome_windows_days", t["ArrayType"](t["IntegerType"]()), True),
            t["StructField"]("outcome_definition_version", t["StringType"](), True),
            # Target-trial definition (immutable across model retrains)
            t["StructField"]("time_zero_definition", t["StringType"](), True),
            t["StructField"]("grace_period_days", t["IntegerType"](), True),
            t["StructField"]("followup_start_rule", t["StringType"](), True),
            t["StructField"]("followup_end_rule", t["StringType"](), True),
            t["StructField"]("default_estimand", t["StringType"](), True),
            t["StructField"]("analysis_population_rule", t["StringType"](), True),
            t["StructField"]("active_from", t["TimestampType"](), True),
            t["StructField"]("active_to", t["TimestampType"](), True),
            # Prose field: "when this playbook has been found useful" — feeds
            # the archetype→playbook matcher (ProseOverlapMatcher + LLM) alongside
            # description. Authored by CS leadership in the YAML. No feature
            # column names; business prose only.
            t["StructField"]("when_applicable", t["StringType"](), True),
        ]
    )


def playbook_steps_schema() -> "StructType":
    """§1.2 — Ordered action steps that define a playbook's workflow.

    One row per ``(playbook_id, playbook_version, step_id)``.
    """
    t = _types()
    return t["StructType"](
        [
            t["StructField"]("step_id", t["StringType"](), False),
            t["StructField"]("playbook_id", t["StringType"](), False),
            t["StructField"]("playbook_version", t["StringType"](), False),
            t["StructField"]("step_sequence", t["IntegerType"](), False),
            t["StructField"]("step_name", t["StringType"](), False),
            t["StructField"]("action_type", t["StringType"](), False),
            t["StructField"]("automation_level", t["StringType"](), False),
            t["StructField"]("owner_role", t["StringType"](), True),
            t["StructField"]("cadence_trigger", t["StringType"](), True),
            t["StructField"]("cadence_relative_to", t["StringType"](), True),
            t["StructField"]("cadence_offset_days", t["IntegerType"](), True),
            t["StructField"]("cadence_condition", t["StringType"](), True),
            t["StructField"]("timeout_days", t["IntegerType"](), True),
            t["StructField"]("response_schema_id", t["StringType"](), True),
            t["StructField"]("intensity_param_name", t["StringType"](), True),
            t["StructField"]("skip_conditions", t["StringType"](), True),  # JSON
            t["StructField"]("stop_conditions", t["StringType"](), True),  # JSON
        ]
    )


def response_schemas_schema() -> "StructType":
    """§1.3 — Named response payload shapes referenced by ``playbook_steps``.

    Each row defines one named shape; the ``fields`` JSON describes the
    primitive fields and which vocabularies they draw from.
    """
    t = _types()
    return t["StructType"](
        [
            t["StructField"]("schema_id", t["StringType"](), False),
            t["StructField"]("version", t["StringType"](), False),
            t["StructField"]("name", t["StringType"](), True),
            t["StructField"]("description", t["StringType"](), True),
            t["StructField"]("fields", t["StringType"](), True),  # JSON
            t["StructField"]("valid_from", t["TimestampType"](), True),
            t["StructField"]("valid_to", t["TimestampType"](), True),
        ]
    )


def vocabularies_schema() -> "StructType":
    """§1.3 — Long-format controlled vocabulary table.

    One row per ``(vocabulary_name, value, valid_from)``. Adding a value is a
    row insert; deprecating one is a ``valid_to`` update. Historical responses
    remain interpretable.
    """
    t = _types()
    return t["StructType"](
        [
            t["StructField"]("vocabulary_name", t["StringType"](), False),
            t["StructField"]("value", t["StringType"](), False),
            t["StructField"]("description", t["StringType"](), True),
            t["StructField"]("valid_from", t["TimestampType"](), False),
            t["StructField"]("valid_to", t["TimestampType"](), True),
        ]
    )


# ---------------------------------------------------------------------------
# Definition layer — sub-layer 2 (archetype catalog, fast, per retrain)
# ---------------------------------------------------------------------------


def archetype_catalog_schema() -> "StructType":
    """§1.4 — Model-derived archetype layer.

    Each row is one SHAP-space cluster paired with a human label. Scoped to
    ``(model_name, model_version)``; immutable after activation. New rows on
    each derivation run with status ``pending_review`` until promoted by the
    approval gate (auto-stable or manual).
    """
    t = _types()
    return t["StructType"](
        [
            t["StructField"]("archetype_id", t["StringType"](), False),
            t["StructField"]("archetype_version", t["StringType"](), False),
            t["StructField"]("model_name", t["StringType"](), False),
            t["StructField"]("model_version", t["StringType"](), False),
            t["StructField"]("derivation_run_id", t["StringType"](), True),
            t["StructField"]("derivation_method", t["StringType"](), True),  # kmeans / hdbscan / gmm
            t["StructField"]("derivation_params", t["StringType"](), True),  # JSON
            t["StructField"]("cluster_raw_id", t["StringType"](), True),
            t["StructField"]("cluster_size", t["LongType"](), True),
            t["StructField"]("cluster_mean_churn_probability", t["DoubleType"](), True),
            t["StructField"](
                "top_shap_features",
                t["ArrayType"](_shap_feature_struct()),
                True,
            ),
            # Two centroids: SHAP-space (for stability tracking) + raw-feature
            # (for runtime nearest-neighbor archetype assignment in s14).
            t["StructField"]("centroid_vector", t["ArrayType"](t["DoubleType"]()), True),
            t["StructField"]("centroid_vector_raw", t["ArrayType"](t["DoubleType"]()), True),
            t["StructField"]("centroid_feature_order", t["ArrayType"](t["StringType"]()), True),
            # Per-feature population std-dev, aligned 1:1 with centroid_feature_order.
            # Used by the runtime snapshot writer to compute scaled Euclidean
            # distance (x - centroid)/scale so features with different units
            # (tenure_days vs nps_score) don't let the high-cardinality feature
            # dominate the nearest-neighbor match.
            t["StructField"]("centroid_feature_scales", t["ArrayType"](t["DoubleType"]()), True),
            t["StructField"](
                "feature_thresholds",
                t["MapType"](t["StringType"](), _feature_threshold_struct()),
                True,
            ),
            t["StructField"]("name", t["StringType"](), True),
            t["StructField"]("description", t["StringType"](), True),
            t["StructField"]("rationale", t["StringType"](), True),
            t["StructField"]("llm_model_id", t["StringType"](), True),
            t["StructField"]("stability_vs_prior_version", t["DoubleType"](), True),
            t["StructField"]("status", t["StringType"](), False),  # draft/pending_review/active/superseded/archived
            t["StructField"]("approved_by", t["StringType"](), True),
            t["StructField"]("approved_at", t["TimestampType"](), True),
            t["StructField"]("valid_from", t["TimestampType"](), True),
            t["StructField"]("valid_to", t["TimestampType"](), True),
        ]
    )


# ---------------------------------------------------------------------------
# Definition layer — sub-layer 3 (eligibility policy, fast, per retrain)
# ---------------------------------------------------------------------------


def eligibility_policy_schema() -> "StructType":
    """§1.5 — Model-derived rule layer.

    For each ``(model_version, playbook_id)`` pair, at most one ``active`` row
    holding the rule predicates that decide which customers are candidates.
    Predicates reference raw feature columns only (no SHAP at runtime).
    """
    t = _types()
    return t["StructType"](
        [
            t["StructField"]("eligibility_policy_id", t["StringType"](), False),
            t["StructField"]("version", t["StringType"](), False),
            t["StructField"]("playbook_id", t["StringType"](), False),
            t["StructField"]("playbook_version", t["StringType"](), False),
            t["StructField"]("model_name", t["StringType"](), False),
            t["StructField"]("model_version", t["StringType"](), False),
            t["StructField"]("archetype_ids", t["ArrayType"](t["StringType"]()), True),
            t["StructField"]("derivation_run_id", t["StringType"](), True),
            t["StructField"]("derivation_method", t["StringType"](), True),
            t["StructField"]("eligibility_rules", t["StringType"](), True),  # JSON predicate tree
            t["StructField"]("eligibility_rules_sql", t["StringType"](), True),  # rendered SQL for dashboard
            t["StructField"]("requires_features", t["ArrayType"](t["StringType"]()), True),
            t["StructField"]("expected_uplift_pct", t["DoubleType"](), True),
            # Archetype↔playbook prose/LLM match score in [0, 1]. Review tier
            # for this row derives from thresholds in DerivationConfig.
            t["StructField"]("fit_score", t["DoubleType"](), True),
            # Classification: auto / review / manual / catch_all. Drives c03
            # auto-promotion decisions and manual-review queue display.
            t["StructField"]("fit_tier", t["StringType"](), True),
            t["StructField"]("rationale", t["StringType"](), True),
            t["StructField"]("llm_model_id", t["StringType"](), True),
            t["StructField"]("status", t["StringType"](), False),
            t["StructField"]("approved_by", t["StringType"](), True),
            t["StructField"]("approved_at", t["TimestampType"](), True),
            t["StructField"]("valid_from", t["TimestampType"](), True),
            t["StructField"]("valid_to", t["TimestampType"](), True),
        ]
    )


# ---------------------------------------------------------------------------
# Definition layer — sub-layer 4 (decision policy, slow, business cycles)
# ---------------------------------------------------------------------------


def decision_policy_schema() -> "StructType":
    """§1.6 — Versioned record of operational policy in force at decision time.

    Capacity, holdout fractions, cooldown windows, suppression rules, channel
    availability. Versions are never overwritten; old policies remain
    queryable. Every ``eligibility_snapshot`` and ``assignments`` row is
    anchored to a ``decision_policy_id``.
    """
    t = _types()
    return t["StructType"](
        [
            t["StructField"]("decision_policy_id", t["StringType"](), False),
            t["StructField"]("version", t["StringType"](), False),
            t["StructField"]("valid_from", t["TimestampType"](), False),
            t["StructField"]("valid_to", t["TimestampType"](), True),
            t["StructField"]("eligibility_logic_version", t["StringType"](), True),
            t["StructField"]("prioritization_logic_version", t["StringType"](), True),
            t["StructField"]("capacity_rule_version", t["StringType"](), True),
            t["StructField"]("holdout_rule_version", t["StringType"](), True),
            t["StructField"]("holdout_stratification_version", t["StringType"](), True),
            t["StructField"](
                "holdout_stratification_fields",
                t["ArrayType"](t["StringType"]()),
                True,
            ),
            t["StructField"]("holdout_seed_recipe", t["StringType"](), True),
            t["StructField"](
                "holdout_fractions_by_playbook",
                t["MapType"](t["StringType"](), t["DoubleType"]()),
                True,
            ),
            t["StructField"](
                "capacity_by_playbook",
                t["MapType"](t["StringType"](), t["IntegerType"]()),
                True,
            ),
            t["StructField"](
                "cooldown_days_by_playbook",
                t["MapType"](t["StringType"](), t["IntegerType"]()),
                True,
            ),
            t["StructField"]("suppression_rules", t["StringType"](), True),  # JSON
            t["StructField"]("channel_availability_rules", t["StringType"](), True),  # JSON
            # Risk-tier thresholds in force at decision time. Stored here so
            # historical risk_tier assignments are reconstructable from the
            # policy version (cell-config defaults flow into these on publish).
            t["StructField"]("risk_tier_high_threshold", t["DoubleType"](), True),
            t["StructField"]("risk_tier_medium_threshold", t["DoubleType"](), True),
            t["StructField"]("description", t["StringType"](), True),
            t["StructField"]("changed_by", t["StringType"](), True),
        ]
    )


# ---------------------------------------------------------------------------
# Instance layer — eligibility_snapshot (per scoring run, per account, per playbook)
# ---------------------------------------------------------------------------


def eligibility_snapshot_schema() -> "StructType":
    """§1.7 — The grouping pivot.

    One row per ``(scoring_run, entity, playbook)`` for every subject
    matching a playbook's eligibility predicates, including holdouts.
    Carries the four-way definition anchor and the multi-arm context.
    The ``entity_id`` key is the generic scoring subject (subscriber for
    email, account for SPS). Downstream CSM writeback tables
    (``assignments``/``actions``/``outcomes``) keep their own
    ``account_id`` column to match the CSM tool's internal model.
    """
    t = _types()
    return t["StructType"](
        [
            t["StructField"]("eligibility_id", t["StringType"](), False),  # UUID
            t["StructField"]("scoring_run_id", t["StringType"](), False),
            t["StructField"]("as_of_date", t["TimestampType"](), False),
            t["StructField"]("entity_id", t["StringType"](), False),
            # Four-way definition anchor
            t["StructField"]("playbook_id", t["StringType"](), False),
            t["StructField"]("playbook_version", t["StringType"](), False),
            t["StructField"]("archetype_id", t["StringType"](), True),
            t["StructField"]("archetype_version", t["StringType"](), True),
            t["StructField"]("eligibility_policy_id", t["StringType"](), False),
            t["StructField"]("eligibility_policy_version", t["StringType"](), False),
            t["StructField"]("decision_policy_id", t["StringType"](), False),
            t["StructField"]("model_name", t["StringType"](), False),
            t["StructField"]("model_version", t["StringType"](), False),
            # Outcome risk + value at risk (frozen at scoring time)
            t["StructField"]("churn_probability", t["DoubleType"](), False),
            t["StructField"]("value_at_risk", t["DoubleType"](), True),
            t["StructField"]("risk_tier", t["StringType"](), True),  # High/Medium/Low
            t["StructField"](
                "top_shap_features",
                t["ArrayType"](_per_account_shap_driver_struct()),
                True,
            ),
            # Multi-arm context
            t["StructField"]("eligible_playbooks_set", t["ArrayType"](t["StringType"]()), True),
            t["StructField"]("eligible_playbook_count", t["IntegerType"](), True),
            t["StructField"]("policy_rank_among_eligible", t["IntegerType"](), True),
            t["StructField"]("priority_rank_within_cohort", t["IntegerType"](), True),
            t["StructField"]("eligibility_evidence", t["StringType"](), True),  # JSON
            # Holdout assignment
            t["StructField"]("is_holdout", t["BooleanType"](), False),
            t["StructField"]("holdout_stratum", t["StringType"](), True),
            t["StructField"]("holdout_seed", t["StringType"](), True),
            # Operational policy snapshot
            t["StructField"]("playbook_suppressed_reason", t["StringType"](), True),
            t["StructField"]("recommended", t["BooleanType"](), False),
            # Dashboard visibility flag — TRUE when the row's churn_probability
            # meets the configured cutoff or the row is High-tier. Nullable so
            # rows written before the flag existed read back as NULL; the
            # dashboard views COALESCE to TRUE to keep pre-migration rows
            # visible.
            t["StructField"]("is_dashboard_visible", t["BooleanType"](), True),
            t["StructField"]("written_at", t["TimestampType"](), False),
        ]
    )


# ---------------------------------------------------------------------------
# Analytical-only DDL — assignments / actions / outcomes
#
# These three tables are defined here for the writeback contract. The causal
# track does NOT populate them initially. They become live once the CSM tool
# (Salesforce or equivalent) is wired and the writeback transport is decided.
# ---------------------------------------------------------------------------


def assignments_schema() -> "StructType":
    """§1.8 — Decision-time anchor.

    One row per decision to enroll an eligible customer in a playbook. Holds
    the fields that must be frozen at the moment of decision: outcome risk
    snapshot, four-way definition anchor, treatment classification, and
    propensity context for post-hoc estimation. **Not populated by the
    initial release** — exists as DDL so the writeback ingest can begin
    populating it once the CSM tool transport is agreed.
    """
    t = _types()
    return t["StructType"](
        [
            t["StructField"]("assignment_id", t["StringType"](), False),
            t["StructField"]("account_id", t["StringType"](), False),
            # Four-way definition anchor (nullable for manual_initiation rows)
            t["StructField"]("playbook_id", t["StringType"](), False),
            t["StructField"]("playbook_version", t["StringType"](), False),
            t["StructField"]("archetype_id", t["StringType"](), True),
            t["StructField"]("archetype_version", t["StringType"](), True),
            t["StructField"]("eligibility_policy_id", t["StringType"](), True),
            t["StructField"]("eligibility_policy_version", t["StringType"](), True),
            t["StructField"]("decision_policy_id", t["StringType"](), False),
            t["StructField"]("scoring_run_id", t["StringType"](), True),
            t["StructField"]("model_name", t["StringType"](), True),
            t["StructField"]("model_version", t["StringType"](), True),
            t["StructField"]("created_at", t["TimestampType"](), True),
            t["StructField"]("assigned_at", t["TimestampType"](), True),
            # Time-zero / grace period
            t["StructField"]("time_zero_at", t["TimestampType"](), True),
            t["StructField"]("decision_feature_asof", t["TimestampType"](), True),
            t["StructField"]("recommendation_expiry_at", t["TimestampType"](), True),
            t["StructField"]("assignment_within_grace_period", t["BooleanType"](), True),
            # Treatment classification
            t["StructField"]("assignment_mechanism", t["StringType"](), False),
            t["StructField"]("treatment_status", t["StringType"](), True),
            t["StructField"]("is_control", t["BooleanType"](), True),
            # Outcome risk snapshot (X-side)
            t["StructField"]("churn_probability_at_decision", t["DoubleType"](), True),
            t["StructField"]("value_at_risk_at_decision", t["DoubleType"](), True),
            t["StructField"]("eligibility_rank_at_decision", t["IntegerType"](), True),
            # Multi-arm context
            t["StructField"]("eligible_playbooks_set", t["ArrayType"](t["StringType"]()), True),
            t["StructField"]("chosen_playbook_id", t["StringType"](), True),
            t["StructField"]("alternative_playbooks_available", t["ArrayType"](t["StringType"]()), True),
            t["StructField"]("policy_rank_among_eligible_playbooks", t["IntegerType"](), True),
            t["StructField"]("selection_reason", t["StringType"](), True),
            # Treatment propensity context (post-hoc estimation inputs)
            t["StructField"]("treatment_propensity_at_decision", t["DoubleType"](), True),
            t["StructField"]("capacity_state_at_decision", t["StringType"](), True),  # JSON
            t["StructField"]("csm_queue_load_at_decision", t["IntegerType"](), True),
            t["StructField"]("channel_available_flags", t["StringType"](), True),  # JSON
            t["StructField"]("playbook_suppressed_reason", t["StringType"](), True),
            t["StructField"]("eligibility_but_not_recommended_reason", t["StringType"](), True),
            # Treatment history (saturation / fatigue)
            t["StructField"]("days_since_last_assignment", t["IntegerType"](), True),
            t["StructField"]("last_playbook_id", t["StringType"](), True),
            t["StructField"]("last_assignment_outcome", t["StringType"](), True),
            t["StructField"]("prior_assignments_count_30d", t["IntegerType"](), True),
            t["StructField"]("prior_assignments_count_60d", t["IntegerType"](), True),
            t["StructField"]("prior_assignments_count_90d", t["IntegerType"](), True),
            t["StructField"]("prior_channel_mix", t["StringType"](), True),  # JSON
            t["StructField"]("prior_total_intensity", t["DoubleType"](), True),
            # Outcome definition pin
            t["StructField"]("outcome_windows_days", t["ArrayType"](t["IntegerType"]()), True),
            t["StructField"]("outcome_definition_version", t["StringType"](), True),
            # Audit
            t["StructField"]("assigned_by", t["StringType"](), True),
            t["StructField"]("assignment_source", t["StringType"](), True),
            t["StructField"]("notes", t["StringType"](), True),
        ]
    )


def actions_schema() -> "StructType":
    """§1.9 — SCD Type 2 mirror of action state from the CSM tool.

    One row per version of each action's state. The CSM tool emits state
    transitions; the ingest job (not built in this release) closes the prior
    SCD row and inserts the new one. ``response_payload`` is stored as a JSON
    string in the initial release for flexibility — it can be tightened to a
    strict nested STRUCT later when action_type vocabularies stabilize.
    **Not populated by the initial release**.
    """
    t = _types()
    return t["StructType"](
        [
            t["StructField"]("action_id", t["StringType"](), False),
            t["StructField"]("action_key", t["StringType"](), False),
            t["StructField"]("version_num", t["IntegerType"](), False),
            t["StructField"]("assignment_id", t["StringType"](), False),
            t["StructField"]("account_id", t["StringType"](), False),
            t["StructField"]("playbook_id", t["StringType"](), False),
            t["StructField"]("playbook_version", t["StringType"](), False),
            t["StructField"]("step_id", t["StringType"](), False),
            t["StructField"]("step_sequence", t["IntegerType"](), True),
            t["StructField"]("action_type", t["StringType"](), True),
            t["StructField"]("automation_level", t["StringType"](), True),
            t["StructField"]("scheduled_for", t["TimestampType"](), True),
            t["StructField"]("status", t["StringType"](), False),
            t["StructField"]("started_at", t["TimestampType"](), True),
            t["StructField"]("completed_at", t["TimestampType"](), True),
            t["StructField"]("assigned_to", t["StringType"](), True),
            t["StructField"]("executed_by", t["StringType"](), True),
            t["StructField"]("intensity_value", t["DoubleType"](), True),
            t["StructField"]("intensity_unit", t["StringType"](), True),
            # Realized cost
            t["StructField"]("actual_staff_time_minutes", t["DoubleType"](), True),
            t["StructField"]("actual_discount_cost_usd", t["DoubleType"](), True),
            t["StructField"]("actual_third_party_cost_usd", t["DoubleType"](), True),
            t["StructField"]("actual_total_cost_usd", t["DoubleType"](), True),
            t["StructField"]("response_payload", t["StringType"](), True),  # JSON; tighten later
            t["StructField"]("free_text_notes", t["StringType"](), True),
            # SCD-2 validity
            t["StructField"]("valid_from", t["TimestampType"](), False),
            t["StructField"]("valid_to", t["TimestampType"](), True),
            t["StructField"]("is_current", t["BooleanType"](), False),
            t["StructField"]("changed_by", t["StringType"](), True),
            t["StructField"]("change_reason", t["StringType"](), True),
            t["StructField"]("change_source", t["StringType"](), True),
        ]
    )


def outcomes_schema() -> "StructType":
    """§1.10 — Settled measurement of what happened to an account post-playbook.

    One row per ``(assignment_id, outcome_window_days)``, supporting analysis
    at multiple horizons. Written by a scheduled resolver job that depends on
    internal label data plus actions writebacks. **Not populated by the
    initial release** — exists as DDL so the resolver can begin writing once
    the writeback contract is live.
    """
    t = _types()
    return t["StructType"](
        [
            t["StructField"]("outcome_id", t["StringType"](), False),
            t["StructField"]("assignment_id", t["StringType"](), False),
            t["StructField"]("account_id", t["StringType"](), False),
            t["StructField"]("outcome_window_days", t["IntegerType"](), False),
            t["StructField"]("window_start", t["TimestampType"](), True),
            t["StructField"]("window_end", t["TimestampType"](), True),
            t["StructField"]("measured_at", t["TimestampType"](), True),
            # Outcome and censoring
            t["StructField"]("outcome_observed_full_window", t["BooleanType"](), True),
            t["StructField"]("churn_label", t["BooleanType"](), True),
            t["StructField"]("right_censored", t["BooleanType"](), True),
            t["StructField"]("censoring_reason", t["StringType"](), True),
            t["StructField"]("churn_label_definition", t["StringType"](), True),
            t["StructField"]("churn_event_time", t["TimestampType"](), True),
            t["StructField"]("last_observed_alive_at", t["TimestampType"](), True),
            # Secondary outcomes
            t["StructField"]("value_change_in_window", t["DoubleType"](), True),
            t["StructField"]("engagement_change_in_window", t["DoubleType"](), True),
            # Treatment exposure (derived from actions)
            t["StructField"]("intent_to_treat_received", t["BooleanType"](), True),
            t["StructField"]("per_protocol_completed", t["BooleanType"](), True),
            t["StructField"]("as_treated_received", t["BooleanType"](), True),
            t["StructField"]("treatment_intensity", t["IntegerType"](), True),
            t["StructField"]("total_actual_cost_usd", t["DoubleType"](), True),
            t["StructField"]("time_to_first_contact_hours", t["DoubleType"](), True),
            t["StructField"]("data_quality_flags", t["ArrayType"](t["StringType"]()), True),
        ]
    )


# ---------------------------------------------------------------------------
# Derived helper tables (used by the causal track for caching)
# ---------------------------------------------------------------------------


def shap_background_schema() -> "StructType":
    """Frozen SHAP background sample, keyed by ``archetype_version``.

    Lundberg & Lee (2017) require a fixed background dataset for SHAP value
    comparability. We freeze 1000 stratified rows per derivation run and
    reuse them for the lifetime of the corresponding archetype version. The
    feature set is stored as a map so the schema accommodates feature drift
    across model versions without migration.
    """
    t = _types()
    return t["StructType"](
        [
            t["StructField"]("archetype_version", t["StringType"](), False),
            t["StructField"]("model_name", t["StringType"](), False),
            t["StructField"]("model_version", t["StringType"](), False),
            t["StructField"]("frozen_at", t["TimestampType"](), False),
            t["StructField"]("sample_idx", t["IntegerType"](), False),
            t["StructField"]("stratum", t["StringType"](), True),
            t["StructField"]("label", t["IntegerType"](), True),
            t["StructField"](
                "features",
                t["MapType"](t["StringType"](), t["DoubleType"]()),
                True,
            ),
        ]
    )


def top_shap_drivers_schema() -> "StructType":
    """Per-account cache of top SHAP drivers from the training cohort.

    Computed once per derivation run for the rows the model was trained on.
    Joined back into ``eligibility_snapshot`` at scoring time for the
    dashboard's "why surfaced" column. New accounts not in the training
    cohort fall back to the archetype-global drivers from
    ``archetype_catalog.top_shap_features``.
    """
    t = _types()
    return t["StructType"](
        [
            t["StructField"]("model_name", t["StringType"](), False),
            t["StructField"]("model_version", t["StringType"](), False),
            t["StructField"]("entity_id", t["StringType"](), False),
            t["StructField"]("as_of_date", t["TimestampType"](), True),
            t["StructField"](
                "top_drivers",
                t["ArrayType"](_per_account_shap_driver_struct()),
                True,
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Registry — for tooling and parametrized tests
# ---------------------------------------------------------------------------


ALL_SCHEMAS: Dict[str, Callable[[], "StructType"]] = {
    # Definition layer — sub-layer 1
    "playbook_catalog": playbook_catalog_schema,
    "playbook_steps": playbook_steps_schema,
    "response_schemas": response_schemas_schema,
    "vocabularies": vocabularies_schema,
    # Definition layer — sub-layers 2 / 3
    "archetype_catalog": archetype_catalog_schema,
    "eligibility_policy": eligibility_policy_schema,
    # Definition layer — sub-layer 4
    "decision_policy": decision_policy_schema,
    # Instance layer
    "eligibility_snapshot": eligibility_snapshot_schema,
    # Analytical-only DDL (not populated by this release)
    "assignments": assignments_schema,
    "actions": actions_schema,
    "outcomes": outcomes_schema,
    # Derived helper tables
    "shap_background": shap_background_schema,
    "top_shap_drivers": top_shap_drivers_schema,
}


def get_schema(table_name: str) -> "StructType":
    """Resolve a logical table name to its ``StructType``.

    Raises ``KeyError`` with the available names if the table is unknown.
    """
    if table_name not in ALL_SCHEMAS:
        raise KeyError(
            f"Unknown causal-track table {table_name!r}. "
            f"Known: {sorted(ALL_SCHEMAS)}"
        )
    return ALL_SCHEMAS[table_name]()
