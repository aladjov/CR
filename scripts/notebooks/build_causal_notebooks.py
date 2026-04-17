"""Builder for the four hand-authored causal-track notebooks at ``causal_notebooks/``.

The four causal notebooks are version-controlled `.ipynb` artifacts that
follow the same conventions as ``exploration_notebooks/*.ipynb``:

- A `cr:doc` markdown chapter header
- An `init_progress` code cell with `accept_workflow_params()`,
  `track_and_export_previous(...)`, and the permanent profiler block
- A markdown configuration explanation + a single `@cr:config` cell the
  user edits — one per notebook, containing only the knobs that notebook
  actually reads (no shared bloat)
- A setup cell that resolves catalog/schema/model identifiers via
  ``ScoringConfig`` (works on both Databricks and the local pipeline)
- One or more algorithmic cells calling
  ``customer_retention.stages.causal.*``
- A final `release_stage_memory` cleanup cell

This script is the source of truth for those four files. It uses
**deterministic cell IDs** (8 hex chars derived from
``sha1(stage|cell_name)``) so re-running the builder produces the same
``id=`` fields, which keeps cell-id-based notebook sync stable across
runs. Edit content here, regenerate, commit.

Run from the project root::

    python scripts/notebooks/build_causal_notebooks.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from customer_retention.core.compat.cell_profiling_hooks import PROFILER_BLOCK

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CAUSAL_DIR = _PROJECT_ROOT / "causal_notebooks"


# ---------------------------------------------------------------------------
# Cell ID derivation
# ---------------------------------------------------------------------------


def stable_id(stage: str, cell_name: str) -> str:
    """Return an 8-hex deterministic id derived from stage + cell name.

    Cell IDs are permanent under the cell-tag contract; deriving them from
    a stable seed means re-running this builder never invalidates the
    sidecar profiler history.
    """
    digest = hashlib.sha1(f"{stage}|{cell_name}".encode("utf-8")).hexdigest()
    return digest[:8]


# ---------------------------------------------------------------------------
# Cell builders
# ---------------------------------------------------------------------------


def md_cell(stage: str, name: str, body: List[str]) -> Dict[str, Any]:
    cid = stable_id(stage, f"md:{name}")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"[//]: # (cr:doc name='{name}' id={cid})\n", *body],
        "id": cid,
    }


def code_cell(stage: str, tag: str, name: str, body: List[str]) -> Dict[str, Any]:
    cid = stable_id(stage, f"{tag}:{name}")
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [f"# @cr:{tag} name='{name}' id={cid}\n", *body],
        "id": cid,
    }


def init_progress_cell(stage: str, basename: str) -> Dict[str, Any]:
    body = [
        "from customer_retention.analysis.notebook_progress import accept_workflow_params, track_and_export_previous\n",
        "\n",
        "accept_workflow_params()\n",
        f'track_and_export_previous("{basename}.ipynb")\n',
        PROFILER_BLOCK,
        "\n",
    ]
    return code_cell(stage, "code", "init_progress", body)


def release_cleanup_cell(stage: str) -> Dict[str, Any]:
    return code_cell(
        stage,
        "code",
        "release_stage_memory",
        [
            "from customer_retention.core.compat import release_stage_memory\n",
            "\n",
            "release_stage_memory()\n",
        ],
    )


def write_notebook(path: Path, cells: List[Dict[str, Any]]) -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {path.relative_to(_PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Shared setup cell — single source imported from causal_setup_cell
# ---------------------------------------------------------------------------

from customer_retention.generators.notebook_generator.stages.causal_setup_cell import (
    _SETUP_BODY_NEEDS_MODEL,
    _SETUP_BODY_PUBLISH_ONLY,
    C01_PIPELINE_SUMMARY_BODY,
    C01_RUN_PIPELINE_BODY,
    C01_RUN_PIPELINE_MD,
)


def setup_block(stage: str, *, needs_model: bool) -> List[Dict[str, Any]]:
    body = _SETUP_BODY_NEEDS_MODEL if needs_model else _SETUP_BODY_PUBLISH_ONLY
    return [
        md_cell(
            stage,
            f"{stage}_setup",
            [
                "## 0. Setup\n",
                "\n",
                "Resolves catalog / schema / model identifiers from `ScoringConfig` "
                "(reads the persisted Databricks init JSON on Databricks, or the local "
                "pipeline's `best_model_meta.json` for local runs). The composite-name-"
                "qualified gold features table name is derived here so the algorithmic "
                "cells stay free of path-construction logic.\n",
            ],
        ),
        code_cell(stage, "code", "setup_and_resolve_model", [body]),
    ]


# ---------------------------------------------------------------------------
# c01 — publish definitions
# ---------------------------------------------------------------------------


_C01_CONFIG_MD = """## Configuration

The cell below is the only place you should need to edit. Every value here is read by the publish cell and the pipeline runner below — nothing is hardcoded inside the algorithmic cells.

- **`SKIP_PUBLISH_DEFINITIONS`** — short-circuit the publish step (e.g. when the YAMLs are unchanged since the last run).
- **`RISK_TIER_HIGH_THRESHOLD` / `RISK_TIER_MEDIUM_THRESHOLD`** — risk tier cutoffs mirrored into `decision_policy` so historical assignments can be reconstructed from the policy version in force at scoring time. The publish step writes these as the on-disk default; if a YAML row in `decision_policy.yaml` already specifies them, the YAML wins.
- **`SKIP_PIPELINE_RUN`** — skip invoking the generated pipeline (e.g. when predictions are already fresh).
- **`PIPELINE_DIR`** — directory containing the generated pipeline scripts produced by `exploration_notebooks/10_spec_generation.ipynb`. Leave as `None` to resolve to `/Workspace/{workspace_path}/generated_pipelines/databricks/{pipeline_name}` via `get_workspace_path()` + `ScoringConfig`. Override if the scripts live elsewhere.
- **`PIPELINE_STAGES`** — ordered stage subdirectories to execute. Includes `scoring` so the `predictions` Delta table is populated before `c04_snapshot_and_dashboard` runs.
- **`PIPELINE_STAGE_TIMEOUT_SECONDS`** — per-notebook timeout passed to `dbutils.notebook.run()`.
"""

_C01_CONFIG_BODY = '''SKIP_PUBLISH_DEFINITIONS = False

RISK_TIER_HIGH_THRESHOLD = 0.6
RISK_TIER_MEDIUM_THRESHOLD = 0.3

SKIP_PIPELINE_RUN = False

PIPELINE_DIR = None
PIPELINE_STAGES = ["landing", "bronze", "silver", "gold", "training", "scoring"]
PIPELINE_STAGE_TIMEOUT_SECONDS = 3600
'''


_C01_PUBLISH_BODY = '''from customer_retention.stages.causal.delta_writer import overwrite_table
from customer_retention.stages.causal.playbook_loader import load_playbooks_from_dir
from customer_retention.stages.causal.policy_loader import load_policies_from_dir
from customer_retention.stages.causal.schemas import (
    decision_policy_schema,
    playbook_catalog_schema,
    playbook_steps_schema,
    response_schemas_schema,
    vocabularies_schema,
)

if SKIP_PUBLISH_DEFINITIONS:
    print("SKIPPED: SKIP_PUBLISH_DEFINITIONS=True")
elif spark is None:
    print("SKIPPED: no active Spark session (Databricks-only cell)")
else:
    catalog_rows, step_rows = load_playbooks_from_dir(PLAYBOOKS_DIR)
    policies = load_policies_from_dir(PLAYBOOKS_DIR)

    decision_rows = policies.get("decision_policy", [])
    for row in decision_rows:
        if row.get("risk_tier_high_threshold") is None:
            row["risk_tier_high_threshold"] = RISK_TIER_HIGH_THRESHOLD
        if row.get("risk_tier_medium_threshold") is None:
            row["risk_tier_medium_threshold"] = RISK_TIER_MEDIUM_THRESHOLD

    overwrite_table(spark, catalog_rows, playbook_catalog_schema(), PLAYBOOK_CATALOG_FQN)
    overwrite_table(spark, step_rows, playbook_steps_schema(), PLAYBOOK_STEPS_FQN)
    overwrite_table(spark, decision_rows, decision_policy_schema(), DECISION_POLICY_FQN)
    overwrite_table(spark, policies.get("response_schemas", []), response_schemas_schema(), RESPONSE_SCHEMAS_FQN)
    overwrite_table(spark, policies.get("vocabularies", []), vocabularies_schema(), VOCABULARIES_FQN)
    print(
        f"Published {len(catalog_rows)} playbooks, {len(step_rows)} steps, "
        f"{len(decision_rows)} decision_policy rows"
    )
'''


def build_c01() -> List[Dict[str, Any]]:
    stage = "c01_publish_definitions"
    return [
        md_cell(
            stage,
            "chapter_c01_publish_definitions",
            [
                "# Chapter c01: Publish Definition Tables (Causal Track)\n",
                "\n",
                "Reads `*.yaml` from `PLAYBOOKS_DIR/` (the playbook catalog) and `PLAYBOOKS_DIR/policies/` (the operational policies) and writes them to the corresponding Delta tables. Full overwrite on every publish — these are small + authoritative from YAML.\n",
                "\n",
                "Run this notebook **before** `c02_archetype_derivation` whenever the playbook YAMLs change.\n",
            ],
        ),
        init_progress_cell(stage, stage),
        md_cell(stage, "c01_configuration", [_C01_CONFIG_MD]),
        code_cell(stage, "config", "configuration", [_C01_CONFIG_BODY]),
        *setup_block(stage, needs_model=False),
        md_cell(stage, "c01_publish_section", ["## 1. Publish Definition Tables (YAML → Delta)\n"]),
        code_cell(stage, "code", "publish_definition_tables", [_C01_PUBLISH_BODY]),
        md_cell(stage, "c01_run_pipeline_section", [C01_RUN_PIPELINE_MD]),
        code_cell(stage, "code", "run_generated_pipeline", [C01_RUN_PIPELINE_BODY]),
        code_cell(stage, "code", "pipeline_summary", [C01_PIPELINE_SUMMARY_BODY]),
        release_cleanup_cell(stage),
    ]


# ---------------------------------------------------------------------------
# c02 — archetype derivation
# ---------------------------------------------------------------------------


_C02_CONFIG_MD = """## Configuration

The cell below is the only place you should need to edit. Every value here is read by the derivation cell — nothing is hardcoded inside the algorithm.

- **`LLM_ENDPOINT_NAME`** — Mosaic AI Foundation Model endpoint used to refine the archetype → playbook mapping and write rationales. Default `databricks-claude-sonnet-4-6` is pay-per-token and pre-configured in every Databricks workspace. Set to `""` to fall back to the deterministic feature-overlap baseline (no network calls).
- **`SHAP_BACKGROUND_SAMPLE_SIZE`** — frozen reference sample size for SHAP. 1000 is the standard tractable size from the SHAP literature (Lundberg & Lee 2017).
- **`KMEANS_K_RANGE` / `KMEANS_MAX_K`** — silhouette sweep range; the chosen `k` is capped to keep clusters human-reviewable.
- **`KMEANS_FEATURE_CAP`** — pre-select this many top features by mean absolute SHAP before clustering. Mandatory on Spark Connect: KMeans serializes the trained model and >100 columns can exceed the 1 GB serialization limit. 50 is a safe interpretable default.
- **`FORCE_DERIVATION`** — re-run derivation even when an `active` archetype already exists for the current model version. Useful after editing the playbook catalog or the surrogate-tree depth.
"""

_C02_CONFIG_BODY = '''LLM_ENDPOINT_NAME = "databricks-claude-sonnet-4-6"

SHAP_BACKGROUND_SAMPLE_SIZE = 1000

KMEANS_K_RANGE = (4, 12)
KMEANS_MAX_K = 8
KMEANS_FEATURE_CAP = 50

FORCE_DERIVATION = False
'''


_C02_DERIVE_BODY = '''from customer_retention.stages.causal import (
    DerivationConfig,
    build_llm_namer,
    derive_archetypes_and_policies,
)
from customer_retention.stages.causal.playbook_loader import load_playbooks_from_dir


def _model_version_already_derived(_spark, table_fqn, model_name, model_version):
    if _spark is None or not _spark.catalog.tableExists(table_fqn):
        return False
    df = _spark.sql(
        f"SELECT 1 FROM {table_fqn} "
        "WHERE model_name = ? AND model_version = ? AND status = 'active' LIMIT 1",
        args=[model_name, model_version],
    )
    return df.limit(1).count() > 0


derivation_result = None
if not is_databricks():
    print("SKIPPED: derivation requires a Spark cluster (Databricks-only cell)")
elif not FORCE_DERIVATION and _model_version_already_derived(
    spark, ARCHETYPE_CATALOG_FQN, MODEL_NAME, MODEL_VERSION
):
    print(f"SKIPPED: active archetypes already exist for {MODEL_NAME} v{MODEL_VERSION}")
else:
    training_df = spark.table(GOLD_FEATURES_FQN)
    feature_columns = [
        c for c in training_df.columns
        if c not in ("account_id", "entity_id", "target", "churn_probability",
                     "event_timestamp", "inference_point_in_time", "model_uri")
    ]
    join_key = "account_id" if "account_id" in training_df.columns else "entity_id"
    entity_key_cols = [join_key]
    if "event_timestamp" in training_df.columns:
        entity_key_cols.append("event_timestamp")
    elif "inference_point_in_time" in training_df.columns:
        entity_key_cols.append("inference_point_in_time")
    catalog_rows, _ = load_playbooks_from_dir(PLAYBOOKS_DIR)
    llm_namer = build_llm_namer(LLM_ENDPOINT_NAME)
    print(f"LLM namer: {llm_namer.model_id}")

    cfg = DerivationConfig(
        spark=spark,
        training_df=training_df,
        raw_feature_df=training_df,
        feature_columns=feature_columns,
        model_uri=MODEL_URI,
        target_column="target",
        entity_key_cols=entity_key_cols,
        join_key=join_key,
        archetype_catalog_fqn=ARCHETYPE_CATALOG_FQN,
        eligibility_policy_fqn=ELIGIBILITY_POLICY_FQN,
        playbooks=catalog_rows,
        gold_feature_names=feature_columns,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        background_sample_size=SHAP_BACKGROUND_SAMPLE_SIZE,
        k_range=KMEANS_K_RANGE,
        k_cap=KMEANS_MAX_K,
        feature_cap=KMEANS_FEATURE_CAP,
        llm_endpoint_name=LLM_ENDPOINT_NAME,
        llm_namer=llm_namer,
    )
    derivation_result = derive_archetypes_and_policies(cfg)
    print(derivation_result.summary())
'''


def build_c02() -> List[Dict[str, Any]]:
    stage = "c02_archetype_derivation"
    return [
        md_cell(
            stage,
            "chapter_c02_archetype_derivation",
            [
                "# Chapter c02: Archetype Derivation (Causal Track)\n",
                "\n",
                "Heart of the causal track:\n",
                "1. Loads the `@production` model + the gold features.\n",
                "2. Freezes a stratified SHAP background sample.\n",
                "3. Computes per-row SHAP via partition-wise `pandas_udf` (TreeExplainer with the frozen background).\n",
                "4. Pre-selects top features by mean |SHAP| (`KMEANS_FEATURE_CAP`).\n",
                "5. Runs a silhouette-swept Spark KMeans over the reduced SHAP space.\n",
                "6. Fits per-cluster surrogate trees → JSON predicates.\n",
                "7. Maps archetypes ↔ playbooks via feature-overlap baseline + optional LLM refinement.\n",
                "8. Writes `archetype_catalog` + `eligibility_policy` rows as `pending_review` for the c03 approval gate.\n",
            ],
        ),
        init_progress_cell(stage, stage),
        md_cell(stage, "c02_configuration", [_C02_CONFIG_MD]),
        code_cell(stage, "config", "configuration", [_C02_CONFIG_BODY]),
        *setup_block(stage, needs_model=True),
        md_cell(stage, "c02_derive_section", ["## 1. Derive Archetypes + Eligibility Policies\n"]),
        code_cell(stage, "code", "derive_archetypes_and_policies", [_C02_DERIVE_BODY]),
        release_cleanup_cell(stage),
    ]


# ---------------------------------------------------------------------------
# c03 — approval gate
# ---------------------------------------------------------------------------


_C03_CONFIG_MD = """## Configuration

The cell below is the only place you should need to edit. Every value here is read by the approval cells — nothing is hardcoded inside the algorithm.

- **`STABILITY_THRESHOLD`** — cosine similarity above which a re-derived archetype is auto-promoted to `active` without manual review. 0.95 is the recommended starting value; tune after observing 2-3 retrains. Lower the threshold if too many obviously-stable archetypes are tripping the manual gate; raise it if drifted clusters are slipping through without review.
- **`FORCE_APPROVE`** — flip every pending row to `active`. Use after a manual review.
"""

_C03_CONFIG_BODY = '''STABILITY_THRESHOLD = 0.95

FORCE_APPROVE = False
'''


_C03_RESOLVE_BODY = '''from customer_retention.stages.causal import expire_stale_pending

derivation_run_id = None
if spark is not None and spark.catalog.tableExists(ARCHETYPE_CATALOG_FQN):
    row = spark.sql(
        f"SELECT derivation_run_id FROM {ARCHETYPE_CATALOG_FQN} "
        "WHERE status = 'pending_review' AND model_name = ? AND model_version = ? "
        "ORDER BY derivation_run_id DESC LIMIT 1",
        args=[MODEL_NAME, MODEL_VERSION],
    ).head()
    derivation_run_id = row["derivation_run_id"] if row else None

if derivation_run_id is None:
    print("No pending_review derivations found for the current model version.")
else:
    expired = expire_stale_pending(
        spark, ARCHETYPE_CATALOG_FQN, ELIGIBILITY_POLICY_FQN,
        MODEL_NAME, MODEL_VERSION, derivation_run_id,
    )
    if expired:
        print(f"Expired {expired} stale pending_review rows from prior derivation runs")
    print(f"Resolved derivation_run_id: {derivation_run_id}")
'''


_C03_GATE_BODY = '''from customer_retention.stages.causal import auto_promote_stable

gate_result = None
if derivation_run_id is None:
    print("SKIPPED: nothing pending to approve")
else:
    gate_result = auto_promote_stable(
        spark=spark,
        archetype_table_fqn=ARCHETYPE_CATALOG_FQN,
        policy_table_fqn=ELIGIBILITY_POLICY_FQN,
        derivation_run_id=derivation_run_id,
        threshold=STABILITY_THRESHOLD,
        force=FORCE_APPROVE,
    )
    print(gate_result.summary())
'''


_C03_PRINT_PENDING_BODY = '''from customer_retention.stages.causal import list_pending_review

if derivation_run_id is None:
    print("(no pending queue — derivation_run_id not resolved)")
else:
    pending = list_pending_review(spark, ARCHETYPE_CATALOG_FQN, derivation_run_id)
    if not pending:
        print("All archetypes auto-promoted. Pending review queue is empty.")
    else:
        print(
            "Pending review queue (re-run this notebook with FORCE_APPROVE=True "
            "after manual review):"
        )
        for row in pending:
            print(
                f"  - {row['archetype_id']} v{row['archetype_version']} "
                f"name={row['name']!r} stability={row['stability_vs_prior_version']}"
            )
'''


def build_c03() -> List[Dict[str, Any]]:
    stage = "c03_approval_gate"
    return [
        md_cell(
            stage,
            "chapter_c03_approval_gate",
            [
                "# Chapter c03: Approval Gate (Causal Track)\n",
                "\n",
                "Compares each new draft archetype to its prior `active` row by cosine similarity of the SHAP-space centroid vectors. Auto-promotes when `cos_sim ≥ STABILITY_THRESHOLD`; otherwise leaves the row as `pending_review` and prints it on the manual queue at the bottom.\n",
                "\n",
                "Cascades the promotion to `eligibility_policy` via `arrays_overlap`.\n",
            ],
        ),
        init_progress_cell(stage, stage),
        md_cell(stage, "c03_configuration", [_C03_CONFIG_MD]),
        code_cell(stage, "config", "configuration", [_C03_CONFIG_BODY]),
        *setup_block(stage, needs_model=True),
        md_cell(stage, "c03_resolve_run_section", ["## 1. Resolve Latest Pending Derivation Run\n"]),
        code_cell(stage, "code", "resolve_derivation_run_id", [_C03_RESOLVE_BODY]),
        md_cell(stage, "c03_run_gate_section", ["## 2. Run Approval Gate\n"]),
        code_cell(stage, "code", "approval_gate", [_C03_GATE_BODY]),
        md_cell(stage, "c03_print_pending_section", ["## 3. Print Pending Review Queue\n"]),
        code_cell(stage, "code", "print_pending_queue", [_C03_PRINT_PENDING_BODY]),
        release_cleanup_cell(stage),
    ]


# ---------------------------------------------------------------------------
# c04 — snapshot + dashboard (Phase 3 placeholder, gated)
# ---------------------------------------------------------------------------


_C04_CONFIG_MD = """## Configuration

The cell below is the only place you should need to edit. Every value here is read by the snapshot writer and the dashboard publisher — nothing is hardcoded inside the algorithmic cells.

- **`SNAPSHOT_RISK_TIER_HIGH` / `SNAPSHOT_RISK_TIER_MEDIUM`** — risk-tier thresholds applied at snapshot time. Leave as `None` to fall back to the values stored on the active `decision_policy` row (the canonical source — set in `c01_publish_definitions`).
- **`SNAPSHOT_CAPACITY_PARTITION_COLUMN`** — optional partition column for capacity caps (e.g. `"csm_owner_id"`). Leave as `""` to apply caps globally per playbook.
"""

_C04_CONFIG_BODY = '''SNAPSHOT_RISK_TIER_HIGH = None
SNAPSHOT_RISK_TIER_MEDIUM = None
SNAPSHOT_CAPACITY_PARTITION_COLUMN = ""
'''


_C04_BUILD_SNAPSHOT_BODY = '''from customer_retention.stages.causal import SnapshotConfig, build_eligibility_snapshot

snapshot_result = None
if spark is None:
    print("SKIPPED: no Spark session (Databricks-only cell)")
elif not spark.catalog.tableExists(ARCHETYPE_CATALOG_FQN):
    print(f"SKIPPED: {ARCHETYPE_CATALOG_FQN} does not exist (run c01..c03 first)")
elif not spark.catalog.tableExists(PREDICTIONS_FQN):
    print(f"SKIPPED: {PREDICTIONS_FQN} not populated (run s10_batch_inference first)")
else:
    snapshot_cfg = SnapshotConfig(
        spark=spark,
        predictions_fqn=PREDICTIONS_FQN,
        archetype_catalog_fqn=ARCHETYPE_CATALOG_FQN,
        eligibility_policy_fqn=ELIGIBILITY_POLICY_FQN,
        decision_policy_fqn=DECISION_POLICY_FQN,
        snapshot_table_fqn=ELIGIBILITY_SNAPSHOT_FQN,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        risk_tier_high=SNAPSHOT_RISK_TIER_HIGH,
        risk_tier_medium=SNAPSHOT_RISK_TIER_MEDIUM,
        capacity_partition_column=SNAPSHOT_CAPACITY_PARTITION_COLUMN or None,
        top_shap_drivers_fqn=TOP_SHAP_DRIVERS_FQN or None,
    )
    snapshot_result = build_eligibility_snapshot(snapshot_cfg)
    print(snapshot_result.summary())
'''


_C04_PUBLISH_VIEWS_BODY = '''from customer_retention.stages.causal.dashboard_views import (
    DASHBOARD_VIEW_NAMES,
    publish_dashboard_views,
)

if spark is None:
    print("SKIPPED: no Spark session (Databricks-only cell)")
elif not spark.catalog.tableExists(ELIGIBILITY_SNAPSHOT_FQN):
    print(f"SKIPPED: {ELIGIBILITY_SNAPSHOT_FQN} not populated yet")
else:
    statements = publish_dashboard_views(spark, CATALOG, SCHEMA)
    print(f"Published {len(statements)} dashboard views:")
    for view_name in DASHBOARD_VIEW_NAMES:
        print(f"  - {CATALOG}.{SCHEMA}.{view_name}")
'''


_C04_SUMMARY_BODY = '''if spark is None or not spark.catalog.tableExists(ARCHETYPE_CATALOG_FQN):
    print("(no archetype_catalog yet — run c01..c03 first)")
else:
    counts = spark.sql(
        f"SELECT status, COUNT(*) AS n FROM {ARCHETYPE_CATALOG_FQN} GROUP BY status"
    ).collect()
    print("archetype_catalog row counts:")
    for row in counts:
        print(f"  {row['status']}: {row['n']}")
    print(f"Model: {MODEL_NAME} v{MODEL_VERSION}")
    if snapshot_result is not None:
        print(snapshot_result.summary())
'''


def build_c04() -> List[Dict[str, Any]]:
    stage = "c04_snapshot_and_dashboard"
    return [
        md_cell(
            stage,
            "chapter_c04_snapshot_and_dashboard",
            [
                "# Chapter c04: Snapshot + Dashboard (Causal Track)\n",
                "\n",
                "Builds the per-scoring-run `eligibility_snapshot` table, publishes the six dashboard SQL views, and prints the four-way anchor tuple in force.\n",
                "\n",
                "Reads `predictions` from `s10_batch_inference` (do **not** trigger scoring here). Reads the active rows from `archetype_catalog`, `eligibility_policy`, and `decision_policy`. Writes `eligibility_snapshot` via Delta MERGE on the natural `(scoring_run_id, account_id, playbook_id)` key — re-running with the same anchor tuple is a no-op.\n",
            ],
        ),
        init_progress_cell(stage, stage),
        md_cell(stage, "c04_configuration", [_C04_CONFIG_MD]),
        code_cell(stage, "config", "configuration", [_C04_CONFIG_BODY]),
        *setup_block(stage, needs_model=True),
        md_cell(stage, "c04_build_snapshot_section", ["## 1. Build Eligibility Snapshot\n"]),
        code_cell(stage, "code", "build_eligibility_snapshot", [_C04_BUILD_SNAPSHOT_BODY]),
        md_cell(stage, "c04_publish_views_section", ["## 2. Publish Dashboard SQL Views\n"]),
        code_cell(stage, "code", "publish_dashboard_views", [_C04_PUBLISH_VIEWS_BODY]),
        md_cell(stage, "c04_summary_section", ["## 3. Print Run Summary\n"]),
        code_cell(stage, "code", "print_run_summary", [_C04_SUMMARY_BODY]),
        release_cleanup_cell(stage),
    ]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    _CAUSAL_DIR.mkdir(parents=True, exist_ok=True)
    write_notebook(_CAUSAL_DIR / "c01_publish_definitions.ipynb", build_c01())
    write_notebook(_CAUSAL_DIR / "c02_archetype_derivation.ipynb", build_c02())
    write_notebook(_CAUSAL_DIR / "c03_approval_gate.ipynb", build_c03())
    write_notebook(_CAUSAL_DIR / "c04_snapshot_and_dashboard.ipynb", build_c04())


if __name__ == "__main__":
    main()
