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


def _chapter_number(stage: str) -> int:
    """Extract the chapter number from a ``c0N_*`` stage slug so sections can be
    numbered ``N.x`` (e.g. c04 → 4)."""
    return int(stage[1:3])


def setup_block(stage: str, *, needs_model: bool) -> List[Dict[str, Any]]:
    body = _SETUP_BODY_NEEDS_MODEL if needs_model else _SETUP_BODY_PUBLISH_ONLY
    chapter = _chapter_number(stage)
    return [
        md_cell(
            stage,
            f"{stage}_setup",
            [
                f"## {chapter}.0 Setup\n",
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
- **`PIPELINE_STAGES`** — ordered stage subdirectories to execute. Includes `scoring` so the `predictions` Delta table is populated before `c05_snapshot_and_dashboard` runs. (The standalone `c04_batch_inference` notebook is an alternative when you only want to refresh scoring without re-running the full training pipeline.)
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
        md_cell(stage, "c01_publish_section", ["## 1.1 Publish Definition Tables (YAML → Delta)\n"]),
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

- **`LLM_ENDPOINT_NAME`** — Mosaic AI Foundation Model endpoint used to propose the archetype → playbook mapping and write rationales. Default `databricks-claude-sonnet-4-6` is pay-per-token and pre-configured in every Databricks workspace. Set to `""` to run the deterministic prose-overlap matcher only (no network calls). The setup cell below lists every foundation-model endpoint your workspace exposes so you can swap.
- **`FIT_AUTO_THRESHOLD`** — prose-match fit score (0-1) at or above which an (archetype, playbook) policy row is eligible for auto-promotion in c03. Default `0.5`.
- **`FIT_REVIEW_THRESHOLD`** — fit score at or above which a policy row is written as `pending_review` (human approves). Below this it is dropped as a regular row. Default `0.2`.
- **`DEFAULT_PLAYBOOK_ID`** — catch-all playbook id. If an archetype produces no auto/review matches, one policy row is emitted pointing at this playbook so every archetype is served. Set `""` to disable; c02 then flags uncovered archetypes in the summary and `c05` will fail on them.
- **SHAP attribution** — importance vector + background means are persisted as `shap_attribution.json` on the training run (see `stages.modeling.shap_attribution`). c02 loads that artifact via `MODEL_URI` — no sample-size knob here, no rescoring.
- **`KMEANS_K_RANGE` / `KMEANS_MAX_K`** — silhouette sweep range; the chosen `k` is capped to keep clusters human-reviewable.
- **`KMEANS_FEATURE_CAP`** — pre-select this many top features by mean absolute SHAP before clustering. Mandatory on Spark Connect: KMeans serializes the trained model and >100 columns can exceed the 1 GB serialization limit. 50 is a safe interpretable default.
- **`FORCE_DERIVATION`** — re-run derivation even when an `active` archetype already exists for the current model version. Useful after editing the playbook catalog or the surrogate-tree depth.
"""

_C02_CONFIG_BODY = '''LLM_ENDPOINT_NAME = "databricks-claude-sonnet-4-6"

FIT_AUTO_THRESHOLD = 0.5
FIT_REVIEW_THRESHOLD = 0.2
DEFAULT_PLAYBOOK_ID = ""

KMEANS_K_RANGE = (4, 12)
KMEANS_MAX_K = 8
KMEANS_FEATURE_CAP = 50

FORCE_DERIVATION = False
'''


_C02_DERIVE_BODY = '''from customer_retention.stages.causal import (
    DerivationConfig,
    FitThresholds,
    build_llm_namer,
    derive_archetypes_and_policies,
)
from customer_retention.stages.causal.playbook_loader import load_playbooks_from_dir


def _list_foundation_model_endpoints():
    """Return (configured endpoint reachable?, list of available endpoint names)."""
    try:
        from mlflow.deployments import get_deploy_client
        client = get_deploy_client("databricks")
        endpoints = client.list_endpoints() or []
    except Exception as exc:
        print(f"(Could not list serving endpoints: {exc})")
        return False, []
    names = sorted({str(e.get("name", "")) for e in endpoints if e.get("name")})
    kind_hints = ("claude", "llama", "gpt", "mistral", "dbrx", "mixtral", "gemma")
    foundation = [n for n in names if any(h in n.lower() for h in kind_hints)]
    return True, foundation


def _model_version_already_derived(_spark, table_fqn, model_name, model_version):
    if _spark is None or not _spark.catalog.tableExists(table_fqn):
        return False
    df = _spark.sql(
        f"SELECT 1 FROM {table_fqn} "
        "WHERE model_name = ? AND model_version = ? AND status = 'active' LIMIT 1",
        args=[model_name, model_version],
    )
    return df.limit(1).count() > 0


# Transparency: show which endpoint is configured and what else is available.
print(f"Configured LLM endpoint: {LLM_ENDPOINT_NAME or '(deterministic prose_overlap only)'}")
_reachable, _available = _list_foundation_model_endpoints()
if _reachable:
    if LLM_ENDPOINT_NAME and LLM_ENDPOINT_NAME not in _available:
        print(
            f"WARNING: {LLM_ENDPOINT_NAME!r} is not in the reachable foundation model list. "
            "If matching silently falls back to prose_overlap, that is why."
        )
    print("Available foundation-model endpoints on this workspace:")
    for _name in _available:
        marker = " *" if _name == LLM_ENDPOINT_NAME else "  "
        print(f"{marker} {_name}")
    print("(set LLM_ENDPOINT_NAME in the config cell above to switch)")

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
    catalog_rows, _ = load_playbooks_from_dir(PLAYBOOKS_DIR)
    llm_namer = build_llm_namer(LLM_ENDPOINT_NAME)
    print(f"Resolved LLM matcher: {llm_namer.model_id}")

    cfg = DerivationConfig(
        spark=spark,
        training_df=training_df,
        raw_feature_df=training_df,
        feature_columns=feature_columns,
        model_uri=MODEL_URI,
        target_column="target",
        join_key=join_key,
        archetype_catalog_fqn=ARCHETYPE_CATALOG_FQN,
        eligibility_policy_fqn=ELIGIBILITY_POLICY_FQN,
        playbooks=catalog_rows,
        gold_feature_names=feature_columns,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        k_range=KMEANS_K_RANGE,
        k_cap=KMEANS_MAX_K,
        feature_cap=KMEANS_FEATURE_CAP,
        llm_endpoint_name=LLM_ENDPOINT_NAME,
        llm_namer=llm_namer,
        fit_thresholds=FitThresholds(
            auto=float(FIT_AUTO_THRESHOLD),
            review=float(FIT_REVIEW_THRESHOLD),
        ),
        default_playbook_id=(DEFAULT_PLAYBOOK_ID or None),
    )
    derivation_result = derive_archetypes_and_policies(cfg)
    print(derivation_result.summary())
    # Per-archetype coverage breakdown so reviewers see exactly which
    # archetypes hit auto/review/catch_all tiers or have no coverage.
    for _arch_id, _tiers in derivation_result.coverage_report().items():
        print(f"  archetype {_arch_id}: tiers={_tiers or ['(none)']}")
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
        md_cell(stage, "c02_derive_section", ["## 2.1 Derive Archetypes + Eligibility Policies\n"]),
        code_cell(stage, "code", "derive_archetypes_and_policies", [_C02_DERIVE_BODY]),
        md_cell(
            stage,
            "c02_backfill_prose_section",
            ["## 2.2 Backfill `eligibility_rules_prose`\n"],
        ),
        code_cell(stage, "code", "backfill_eligibility_rules_prose", [_C02_BACKFILL_PROSE_BODY]),
        release_cleanup_cell(stage),
    ]


_C02_BACKFILL_PROSE_BODY = '''# Re-render `eligibility_rules_prose` for every existing active policy row whose
# prose column is NULL — derivation populates the column at row-creation time
# only, so rows written before column_descriptions / feature_meta /
# feature_population_stats sidecars existed stay NULL until we rewrite them.
# Cycle 013 P4 surfaced exactly this gap; the backfill closes it idempotently.
if is_databricks() and spark is not None and spark.catalog.tableExists(ELIGIBILITY_POLICY_FQN):
    from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
    from customer_retention.stages.causal.interpretation import (
        backfill_eligibility_prose,
    )

    # Resolve the namespace locally so this cell runs standalone — the
    # derivation cell only defines `_enrich_ns` on the freshly-derive path,
    # so a backfill-only rerun (active archetypes already exist) would
    # otherwise NameError on the reference below.
    _backfill_ns = globals().get("_enrich_ns")
    if _backfill_ns is None:
        try:
            _backfill_ns = RunNamespace.from_env_or_latest(get_experiments_dir())
        except Exception as _ns_exc:
            print(f"Backfill namespace lookup failed ({type(_ns_exc).__name__}: {_ns_exc}) — proceeding without sidecar context.")
            _backfill_ns = None

    _backfill = backfill_eligibility_prose(
        spark, ELIGIBILITY_POLICY_FQN, namespace=_backfill_ns,
    )
    print(_backfill.summary())
    if _backfill.warnings:
        print("(interpretation-layer warnings — see logs for details)")
        for _w in _backfill.warnings:
            print(f"  - {_w}")
'''


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
            "Archetype pending-review queue (re-run this notebook with FORCE_APPROVE=True "
            "after manual review):"
        )
        for row in pending:
            print(
                f"  - {row['archetype_id']} v{row['archetype_version']} "
                f"name={row['name']!r} stability={row['stability_vs_prior_version']}"
            )

# Transparency: also show every pending eligibility_policy row with its fit tier
# and score. The approval gate auto-promotes only 'auto'-tier rows; 'review'
# and 'catch_all' rows stay pending for human decision.
if derivation_run_id is not None and spark.catalog.tableExists(ELIGIBILITY_POLICY_FQN):
    policy_rows = spark.sql(
        f"SELECT playbook_id, archetype_ids, fit_tier, fit_score, rationale "
        f"FROM {ELIGIBILITY_POLICY_FQN} "
        "WHERE derivation_run_id = ? AND status = 'pending_review' "
        "ORDER BY fit_tier, fit_score DESC",
        args=[derivation_run_id],
    ).collect()
    if policy_rows:
        print("\\nEligibility policy pending-review queue:")
        for r in policy_rows:
            arch = (r["archetype_ids"] or ["?"])[0]
            score = r["fit_score"]
            score_str = f"{float(score):.2f}" if score is not None else "n/a"
            rationale = (r["rationale"] or "")[:140]
            print(
                f"  - [{r['fit_tier']}] playbook={r['playbook_id']} "
                f"archetype_version={arch} fit_score={score_str} | {rationale}"
            )
    else:
        print("\\nEligibility policy pending-review queue is empty.")
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
        md_cell(stage, "c03_resolve_run_section", ["## 3.1 Resolve Latest Pending Derivation Run\n"]),
        code_cell(stage, "code", "resolve_derivation_run_id", [_C03_RESOLVE_BODY]),
        md_cell(stage, "c03_run_gate_section", ["## 3.2 Run Approval Gate\n"]),
        code_cell(stage, "code", "approval_gate", [_C03_GATE_BODY]),
        md_cell(stage, "c03_print_pending_section", ["## 3.3 Print Pending Review Queue\n"]),
        code_cell(stage, "code", "print_pending_queue", [_C03_PRINT_PENDING_BODY]),
        release_cleanup_cell(stage),
    ]


# ---------------------------------------------------------------------------
# c04 — batch inference (standalone scoring refresh)
# ---------------------------------------------------------------------------


_C04_CONFIG_MD = """## Configuration

The cell below is the only place you should need to edit. Every value here is read by the batch-inference cell — nothing is hardcoded inside the algorithmic cells.

- **`BATCH_INFERENCE_MODE`** — `"auto"` (default): run only if `predictions` is missing or stale; `"always"`: force a scoring run regardless of freshness; `"never"`: skip entirely.
- **`PREDICTIONS_STALE_AFTER_HOURS`** — in `"auto"` mode, re-score if the latest `inference_point_in_time` in `predictions` is older than this many hours.
- **`SCORING_THRESHOLD`** — probability cutoff that marks a row as a predicted churner. Must match the threshold used in `decision_policy`.
- **`RISK_TIER_HIGH` / `RISK_TIER_MEDIUM`** — risk-tier cutoffs written onto each scored row; c05's snapshot writer reads these back when `SNAPSHOT_RISK_TIER_*` are `None`.
"""

_C04_CONFIG_BODY = '''BATCH_INFERENCE_MODE = "auto"           # "auto" | "always" | "never"
PREDICTIONS_STALE_AFTER_HOURS = 24

SCORING_THRESHOLD = 0.5
RISK_TIER_HIGH = 0.6
RISK_TIER_MEDIUM = 0.3
'''


_C04_REFRESH_PREDICTIONS_BODY = '''from datetime import datetime, timedelta, timezone

from customer_retention.stages.scoring.batch_inference import (
    BatchInferenceConfig,
    run_batch_inference,
)


def _resolve_scope_filter():
    """Return the NB00 scope filter (``ProjectContext.sample_filters``) for the
    target dataset, or ``None`` if project_context is absent / empty. Scoring
    must cover only the entity subset that was in force during exploration
    and training — this keeps the inference population in lock-step with the
    feature-spec gate upstream."""
    try:
        from customer_retention.analysis.auto_explorer.project_context import ProjectContext
        from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
    except ImportError:
        return None
    _ns = RunNamespace.from_env_or_latest()
    if _ns is None:
        return None
    _path = _ns.project_context_path
    if not _path.exists():
        return None
    _ctx = ProjectContext.load(_path)
    _filters = getattr(_ctx, "sample_filters", None) or {}
    if not _filters:
        return None
    _target_name = next(
        (name for name, ds in _ctx.datasets.items()
         if getattr(ds, "role", None) == "target"),
        None,
    )
    if _target_name is None and len(_ctx.datasets) == 1:
        _target_name = next(iter(_ctx.datasets))
    if _target_name is None:
        return None
    return _filters.get(_target_name)


_scope_filter = _resolve_scope_filter()

batch_inference_result = None
_predictions_status = "UNKNOWN"

if spark is None:
    _predictions_status = "SKIPPED: no Spark session (Databricks-only cell)"
    print(_predictions_status)
elif BATCH_INFERENCE_MODE == "never":
    _predictions_status = "SKIPPED: BATCH_INFERENCE_MODE='never'"
    print(_predictions_status)
else:
    _should_run = True
    _stale_reason = "always" if BATCH_INFERENCE_MODE == "always" else None
    if BATCH_INFERENCE_MODE == "auto":
        if not spark.catalog.tableExists(PREDICTIONS_FQN):
            _stale_reason = f"{PREDICTIONS_FQN} does not exist"
        else:
            _latest_ts_row = spark.sql(
                f"SELECT max(inference_point_in_time) AS ts FROM {PREDICTIONS_FQN}"
            ).head()
            _latest_ts = _latest_ts_row["ts"] if _latest_ts_row is not None else None
            if _latest_ts is None:
                _stale_reason = f"{PREDICTIONS_FQN} is empty"
            else:
                if _latest_ts.tzinfo is None:
                    _latest_ts = _latest_ts.replace(tzinfo=timezone.utc)
                _age_hours = (datetime.now(timezone.utc) - _latest_ts).total_seconds() / 3600.0
                if _age_hours > PREDICTIONS_STALE_AFTER_HOURS:
                    _stale_reason = f"latest inference_point_in_time is {_age_hours:.1f}h old (> {PREDICTIONS_STALE_AFTER_HOURS}h)"
                else:
                    _should_run = False
                    _predictions_status = (
                        f"FRESH: latest inference_point_in_time is {_age_hours:.1f}h old "
                        f"(<= {PREDICTIONS_STALE_AFTER_HOURS}h) — skipping"
                    )
                    print(_predictions_status)
    if _should_run:
        print(f"Running batch inference ({_stale_reason})")
        if _scope_filter:
            print(f"Scope filter (from NB00 project_context): {_scope_filter}")
        else:
            print("Scope filter: (none — scoring full entity population)")
        config = BatchInferenceConfig(
            catalog=CATALOG,
            schema=SCHEMA,
            model_uri=MODEL_URI,
            customer_table=GOLD_FEATURES_FQN,
            threshold=SCORING_THRESHOLD,
            risk_tier_high=RISK_TIER_HIGH,
            risk_tier_medium=RISK_TIER_MEDIUM,
            inference_timestamp=datetime.now(timezone.utc),
            filter_expression=_scope_filter,
        )
        batch_inference_result = run_batch_inference(config)
        _predictions_status = batch_inference_result.summary()
        print(batch_inference_result.long_summary())
'''


_C04_SUMMARY_BODY = '''if batch_inference_result is None:
    print(f"Predictions status: {_predictions_status}")
else:
    print(f"Inference id: {batch_inference_result.inference_id}")
    print(f"Inference timestamp: {batch_inference_result.inference_timestamp}")
    print(f"Scored: {batch_inference_result.total_scored:,}")
    print(f"Predicted churners: {batch_inference_result.predicted_churners:,}")
    print(f"Mean probability: {batch_inference_result.avg_probability:.4f}")
    print(f"Target table: {batch_inference_result.target_table_fqn}")
'''


def build_c04() -> List[Dict[str, Any]]:
    stage = "c04_batch_inference"
    return [
        md_cell(
            stage,
            "chapter_c04_batch_inference",
            [
                "# Chapter c04: Batch Inference (Causal Track)\n",
                "\n",
                "Refreshes the `predictions` Delta table by scoring the current feature-store snapshot against the registered `@production` model. Independent from c02 (archetypes) and c05 (snapshot + dashboard) so operators can re-run scoring without recomputing archetypes.\n",
                "\n",
                "`BATCH_INFERENCE_MODE='auto'` (default) scores only when `predictions` is missing or older than `PREDICTIONS_STALE_AFTER_HOURS`. Use `'always'` to force a fresh scoring run or `'never'` to skip (useful when inspecting the model without writing).\n",
            ],
        ),
        init_progress_cell(stage, stage),
        md_cell(stage, "c04_configuration", [_C04_CONFIG_MD]),
        code_cell(stage, "config", "configuration", [_C04_CONFIG_BODY]),
        *setup_block(stage, needs_model=True),
        md_cell(stage, "c04_refresh_section", ["## 4.1 Refresh Predictions\n"]),
        code_cell(stage, "code", "refresh_predictions", [_C04_REFRESH_PREDICTIONS_BODY]),
        md_cell(stage, "c04_summary_section", ["## 4.2 Print Run Summary\n"]),
        code_cell(stage, "code", "print_run_summary", [_C04_SUMMARY_BODY]),
        release_cleanup_cell(stage),
    ]


# ---------------------------------------------------------------------------
# c05 — snapshot + dashboard
# ---------------------------------------------------------------------------


_C05_CONFIG_MD = """## Configuration

The cell below is the only place you should need to edit. Every value here is read by the snapshot writer and the dashboard publisher — nothing is hardcoded inside the algorithmic cells.

- **`SNAPSHOT_RISK_TIER_HIGH` / `SNAPSHOT_RISK_TIER_MEDIUM`** — risk-tier thresholds applied at snapshot time. Leave as `None` to fall back to the values stored on the active `decision_policy` row (the canonical source — set in `c01_publish_definitions`).
- **`SNAPSHOT_CAPACITY_PARTITION_COLUMN`** — optional partition column for capacity caps (e.g. `"csm_owner_id"`). Leave as `""` to apply caps globally per playbook.
- **`SHAP_PER_SLICE_K`** — for each `(playbook, archetype, risk_tier)` slice, compute per-row SHAP for the top-K accounts by expected loss and write them to `top_shap_drivers` so the dashboard's L4 panel can show "why surfaced" without a pandas_udf at view time. `0` disables the writer (the L3 view falls back gracefully but per-account SHAP cells stay empty for new model versions).
- **`SHAP_TOP_DRIVERS_PER_ROW`** — how many drivers to keep per row (default `5`).
"""

_C05_CONFIG_BODY = '''SNAPSHOT_RISK_TIER_HIGH = None
SNAPSHOT_RISK_TIER_MEDIUM = None
SNAPSHOT_CAPACITY_PARTITION_COLUMN = ""
SHAP_PER_SLICE_K = 50
SHAP_TOP_DRIVERS_PER_ROW = 5
'''


_C05_BUILD_SNAPSHOT_BODY = '''from customer_retention.stages.causal import SnapshotConfig, build_eligibility_snapshot

snapshot_result = None
if spark is None:
    print("SKIPPED: no Spark session (Databricks-only cell)")
elif not spark.catalog.tableExists(ARCHETYPE_CATALOG_FQN):
    print(f"SKIPPED: {ARCHETYPE_CATALOG_FQN} does not exist (run c01..c03 first)")
elif not spark.catalog.tableExists(PREDICTIONS_FQN):
    print(f"SKIPPED: {PREDICTIONS_FQN} not populated (run c04_batch_inference first)")
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
        # gold_features_fqn is the raw-feature source for predicate evaluation
        # (eligibility rules like `active_span_days >= 42` reference raw
        # feature columns that are NOT carried on the predictions table).
        gold_features_fqn=GOLD_FEATURES_FQN,
        # entity_id_column names the scoring-subject key on predictions, gold,
        # and the snapshot output. Default is "entity_id" — override only if
        # your predictions table keys on a different column (e.g. account_id).
        entity_id_column="entity_id",
        risk_tier_high=SNAPSHOT_RISK_TIER_HIGH,
        risk_tier_medium=SNAPSHOT_RISK_TIER_MEDIUM,
        capacity_partition_column=SNAPSHOT_CAPACITY_PARTITION_COLUMN or None,
        top_shap_drivers_fqn=TOP_SHAP_DRIVERS_FQN or None,
    )
    snapshot_result = build_eligibility_snapshot(snapshot_cfg)
    print(snapshot_result.summary())
'''


_C05_COMPUTE_TOP_SHAP_BODY = '''from customer_retention.stages.causal import (
    TopDriversConfig,
    compute_and_write_top_shap_drivers,
)

# Per-slice SHAP cache for the L4 "why surfaced" panel. Keyed on
# (model_name, model_version, entity_id) so v_eligible_all_playbooks can
# fall back to it via COALESCE when the snapshot row's top_shap_features
# column is NULL. Without this writer, every new model_version produces
# 0 rows here and the dashboard's L3 cohort list comes back empty (the
# view filters WHERE top_shap_features IS NOT NULL after the COALESCE).
top_drivers_result = None
if spark is None:
    print("SKIPPED: no Spark session (Databricks-only cell)")
elif snapshot_result is None:
    print("SKIPPED: snapshot_result is None — 5.1 did not produce a run")
elif int(SHAP_PER_SLICE_K) <= 0:
    print("SKIPPED: SHAP_PER_SLICE_K == 0 (per-slice SHAP enrichment disabled)")
elif MODEL_URI is None:
    print("SKIPPED: MODEL_URI is None — local runs without an MLflow attribution artifact cannot replay SHAP")
elif not spark.catalog.tableExists(GOLD_FEATURES_FQN):
    print(f"SKIPPED: {GOLD_FEATURES_FQN} does not exist — gold features required for SHAP replay")
else:
    top_drivers_cfg = TopDriversConfig(
        spark=spark,
        snapshot_table_fqn=ELIGIBILITY_SNAPSHOT_FQN,
        gold_features_fqn=GOLD_FEATURES_FQN,
        top_shap_drivers_fqn=TOP_SHAP_DRIVERS_FQN,
        model_name=MODEL_NAME,
        model_version=str(MODEL_VERSION),
        scoring_run_id=snapshot_result.scoring_run_id,
        as_of_date=snapshot_result.as_of_date,
        model_uri=MODEL_URI,
        per_slice_k=int(SHAP_PER_SLICE_K),
        top_drivers_per_row=int(SHAP_TOP_DRIVERS_PER_ROW),
    )
    top_drivers_result = compute_and_write_top_shap_drivers(top_drivers_cfg)
    print(top_drivers_result.summary())
'''


_C05_WRITE_RUN_CONTEXT_BODY = '''from customer_retention.stages.causal import (
    from_project_context,
    write_run_context,
)

RUN_CONTEXT_FQN = f"{CATALOG}.{SCHEMA}.run_context"


def _load_project_context():
    """Best-effort ProjectContext load. Returns None when YAML is unreachable so
    the writer still emits a row with model metadata only."""
    try:
        from customer_retention.analysis.auto_explorer.project_context import ProjectContext
        from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace
    except ImportError:
        return None
    try:
        _ns = RunNamespace.from_env_or_latest()
    except Exception:
        _ns = None
    if _ns is None:
        return None
    _path = _ns.project_context_path
    if not _path.exists():
        return None
    try:
        return ProjectContext.load(_path)
    except Exception:
        return None


def _resolve_model_type():
    """Try to pull the MLflow flavor off the registered model (e.g. "xgboost")."""
    try:
        import mlflow  # noqa: F401
        if MODEL_URI is None:
            return None
        from mlflow.models import Model as _MlflowModel
        _info = _MlflowModel.load(MODEL_URI)
        _flavors = list((_info.flavors or {}).keys())
        for preferred in ("xgboost", "lightgbm", "catboost", "pytorch", "tensorflow", "sklearn"):
            if preferred in _flavors:
                return preferred
        _flavors = [f for f in _flavors if f != "python_function"]
        return _flavors[0] if _flavors else None
    except Exception:
        return None


if spark is None:
    print("SKIPPED: no Spark session (Databricks-only cell)")
elif snapshot_result is None:
    print("SKIPPED: snapshot_result is None — 5.1 did not produce a run")
else:
    _ctx = _load_project_context()
    _cfg = from_project_context(
        project_context=_ctx,
        spark=spark,
        table_fqn=RUN_CONTEXT_FQN,
        scoring_run_id=snapshot_result.scoring_run_id,
        as_of_date=snapshot_result.as_of_date,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        model_type=_resolve_model_type(),
    )
    write_run_context(_cfg)
    print(f"Wrote run_context row for scoring_run_id={snapshot_result.scoring_run_id}")
    if _ctx is None:
        print("  (project_context.yaml not reachable — context fields are NULL)")
    else:
        print(f"  horizon_days:       {_cfg.horizon_days}")
        print(f"  primary_objective:  {_cfg.primary_objective}")
        print(f"  temporal_posture:   {_cfg.temporal_posture}")
        print(f"  model_type:         {_cfg.model_type or '(unresolved)'}")
'''


_C05_PUBLISH_VIEWS_BODY = '''from customer_retention.stages.causal.dashboard_views import (
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


_C05_SUMMARY_BODY = '''if spark is None or not spark.catalog.tableExists(ARCHETYPE_CATALOG_FQN):
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


def build_c05() -> List[Dict[str, Any]]:
    stage = "c05_snapshot_and_dashboard"
    return [
        md_cell(
            stage,
            "chapter_c05_snapshot_and_dashboard",
            [
                "# Chapter c05: Snapshot + Dashboard (Causal Track)\n",
                "\n",
                "Builds the per-scoring-run `eligibility_snapshot` table, publishes the six dashboard SQL views, and prints the four-way anchor tuple in force.\n",
                "\n",
                "Reads `predictions` from `c04_batch_inference` (or the `s10_batch_inference` stage of the generated pipeline) — do **not** trigger scoring here. Reads the active rows from `archetype_catalog`, `eligibility_policy`, and `decision_policy`. Writes `eligibility_snapshot` via Delta MERGE on the natural `(scoring_run_id, account_id, playbook_id)` key — re-running with the same anchor tuple is a no-op.\n",
            ],
        ),
        init_progress_cell(stage, stage),
        md_cell(stage, "c05_configuration", [_C05_CONFIG_MD]),
        code_cell(stage, "config", "configuration", [_C05_CONFIG_BODY]),
        *setup_block(stage, needs_model=True),
        md_cell(stage, "c05_build_snapshot_section", ["## 5.1 Build Eligibility Snapshot\n"]),
        code_cell(stage, "code", "build_eligibility_snapshot", [_C05_BUILD_SNAPSHOT_BODY]),
        md_cell(stage, "c05_compute_top_shap_section", ["## 5.2 Compute Per-slice Top SHAP Drivers\n"]),
        code_cell(stage, "code", "compute_top_shap_drivers", [_C05_COMPUTE_TOP_SHAP_BODY]),
        md_cell(stage, "c05_write_run_context_section", ["## 5.3 Write Run Context (app masthead projection)\n"]),
        code_cell(stage, "code", "write_run_context", [_C05_WRITE_RUN_CONTEXT_BODY]),
        md_cell(stage, "c05_publish_views_section", ["## 5.4 Publish Dashboard SQL Views\n"]),
        code_cell(stage, "code", "publish_dashboard_views", [_C05_PUBLISH_VIEWS_BODY]),
        md_cell(stage, "c05_summary_section", ["## 5.5 Print Run Summary\n"]),
        code_cell(stage, "code", "print_run_summary", [_C05_SUMMARY_BODY]),
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
    write_notebook(_CAUSAL_DIR / "c04_batch_inference.ipynb", build_c04())
    write_notebook(_CAUSAL_DIR / "c05_snapshot_and_dashboard.ipynb", build_c05())


if __name__ == "__main__":
    main()
