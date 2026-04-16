"""Shared cell builders for the four causal-track stage generators (c01..c04).

Each causal stage produces three cell groups:

1. **Configuration block** — markdown explanation + a single `@cr:config`
   code cell. Per-notebook content; only the knobs that notebook actually
   reads are exposed (no shared bloat).
2. **Setup block** — code cell that resolves catalog / schema / model
   identifiers via ``ScoringConfig``. Two flavors: the full
   ``setup_with_model`` flavor (c02 / c03 / c04) and the lightweight
   ``setup_publish_only`` flavor (c01) which skips model URI lookup.
3. **Algorithmic cells** — supplied by the individual stage generator.

Both the script generator (which produces ``generated_pipelines/{platform}/
c0X_*.py``) and the hand-authored notebook builder
(``scripts/notebooks/build_causal_notebooks.py``) consume these cell
templates so the two layers stay in lock-step.
"""

from __future__ import annotations

from typing import List

import nbformat

from ..cell_builder import CellBuilder

# ---------------------------------------------------------------------------
# Configuration cells — per stage, only the knobs the stage reads
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

PIPELINE_DIR = None  # None → /Workspace/{workspace_path}/generated_pipelines/databricks/{pipeline_name}
PIPELINE_STAGES = ["landing", "bronze", "silver", "gold", "training", "scoring"]
PIPELINE_STAGE_TIMEOUT_SECONDS = 3600
'''


_C02_CONFIG_MD = """## Configuration

The cell below is the only place you should need to edit. Every value here is read by the derivation cell — nothing is hardcoded inside the algorithm.

- **`LLM_ENDPOINT_NAME`** — Mosaic AI Foundation Model endpoint used to refine the archetype → playbook mapping and write rationales. Default `databricks-claude-sonnet-4-6` is pay-per-token and pre-configured in every Databricks workspace. Set to `""` to fall back to the deterministic feature-overlap baseline.
- **`SHAP_BACKGROUND_SAMPLE_SIZE`** — frozen reference sample size for SHAP. 1000 is the standard tractable size from the SHAP literature.
- **`KMEANS_K_RANGE` / `KMEANS_MAX_K`** — silhouette sweep range; the chosen `k` is capped to keep clusters human-reviewable.
- **`KMEANS_FEATURE_CAP`** — pre-select this many top features by mean absolute SHAP before clustering. Mandatory on Spark Connect: KMeans serializes the trained model and >100 columns can exceed the 1 GB serialization limit.
- **`FORCE_DERIVATION`** — re-run derivation even when an `active` archetype already exists for the current model version.
"""

_C02_CONFIG_BODY = '''LLM_ENDPOINT_NAME = "databricks-claude-sonnet-4-6"

SHAP_BACKGROUND_SAMPLE_SIZE = 1000

KMEANS_K_RANGE = (4, 12)
KMEANS_MAX_K = 8
KMEANS_FEATURE_CAP = 50

FORCE_DERIVATION = False
'''


_C03_CONFIG_MD = """## Configuration

The cell below is the only place you should need to edit. Every value here is read by the approval cells — nothing is hardcoded inside the algorithm.

- **`STABILITY_THRESHOLD`** — cosine similarity above which a re-derived archetype is auto-promoted to `active` without manual review. 0.95 is the recommended starting value.
- **`FORCE_APPROVE`** — flip every pending row to `active`. Use after a manual review.
"""

_C03_CONFIG_BODY = '''STABILITY_THRESHOLD = 0.95

FORCE_APPROVE = False
'''


_C04_CONFIG_MD = """## Configuration

The cell below is the only place you should need to edit. Every value here is read by the snapshot writer and the dashboard publisher — nothing is hardcoded inside the algorithmic cells.

- **`SNAPSHOT_RISK_TIER_HIGH` / `SNAPSHOT_RISK_TIER_MEDIUM`** — risk-tier thresholds applied at snapshot time. Leave as `None` to fall back to the values stored on the active `decision_policy` row (the canonical source — set in `c01_publish_definitions`).
- **`SNAPSHOT_CAPACITY_PARTITION_COLUMN`** — optional partition column for capacity caps (e.g. `"csm_owner_id"`). Leave as `""` to apply caps globally per playbook.
"""

_C04_CONFIG_BODY = '''SNAPSHOT_RISK_TIER_HIGH = None
SNAPSHOT_RISK_TIER_MEDIUM = None
SNAPSHOT_CAPACITY_PARTITION_COLUMN = ""
'''


# ---------------------------------------------------------------------------
# Setup cells — full (with model) and lightweight (publish-only)
# ---------------------------------------------------------------------------


_SETUP_BODY_NEEDS_MODEL = '''from customer_retention.core.compat.detection import get_spark_session, is_databricks
from customer_retention.core.config import get_playbooks_dir
from customer_retention.core.config.experiments import get_experiments_dir
from customer_retention.stages.scoring import ScoringConfig

spark = get_spark_session()
PLAYBOOKS_DIR = get_playbooks_dir()

if is_databricks():
    scoring_config = ScoringConfig.from_databricks()
    CATALOG = scoring_config.catalog
    SCHEMA = scoring_config.schema
    MODEL_NAME = scoring_config.registered_model_name
    import mlflow

    mlflow_client = mlflow.tracking.MlflowClient()
    production_version = mlflow_client.get_model_version_by_alias(
        f"{CATALOG}.{SCHEMA}.{MODEL_NAME}", "production"
    )
    MODEL_VERSION = production_version.version
    MODEL_URI = f"models:/{CATALOG}.{SCHEMA}.{MODEL_NAME}@production"
else:
    scoring_config = ScoringConfig.from_local_config(get_experiments_dir())
    CATALOG = "local"
    SCHEMA = "local"
    MODEL_NAME = scoring_config.best_model_name or "local_model"
    MODEL_VERSION = "local"
    MODEL_URI = None

COMPOSITE_NAME = scoring_config.composite_name
GOLD_FEATURES_FQN = (
    f"{CATALOG}.{SCHEMA}.gold_features_{COMPOSITE_NAME}"
    if COMPOSITE_NAME
    else f"{CATALOG}.{SCHEMA}.gold_features"
)

ARCHETYPE_CATALOG_FQN = f"{CATALOG}.{SCHEMA}.archetype_catalog"
ELIGIBILITY_POLICY_FQN = f"{CATALOG}.{SCHEMA}.eligibility_policy"
DECISION_POLICY_FQN = f"{CATALOG}.{SCHEMA}.decision_policy"
ELIGIBILITY_SNAPSHOT_FQN = f"{CATALOG}.{SCHEMA}.eligibility_snapshot"
PREDICTIONS_FQN = f"{CATALOG}.{SCHEMA}.predictions"
TOP_SHAP_DRIVERS_FQN = f"{CATALOG}.{SCHEMA}.top_shap_drivers"

print(f"Resolved playbooks_dir: {PLAYBOOKS_DIR}")
print(f"Catalog/schema:         {CATALOG}.{SCHEMA}")
print(f"Composite name:         {COMPOSITE_NAME or '(unset)'}")
print(f"Gold features table:    {GOLD_FEATURES_FQN}")
print(f"Model URI:              {MODEL_URI or '(local)'}")
print(f"Model version:          {MODEL_VERSION}")
'''


_SETUP_BODY_PUBLISH_ONLY = '''from customer_retention.core.compat.detection import get_spark_session, is_databricks
from customer_retention.core.config import get_playbooks_dir
from customer_retention.core.config.experiments import get_experiments_dir
from customer_retention.stages.scoring import ScoringConfig

spark = get_spark_session()
PLAYBOOKS_DIR = get_playbooks_dir()

if is_databricks():
    scoring_config = ScoringConfig.from_databricks()
    CATALOG = scoring_config.catalog
    SCHEMA = scoring_config.schema
else:
    scoring_config = ScoringConfig.from_local_config(get_experiments_dir())
    CATALOG = "local"
    SCHEMA = "local"

PLAYBOOK_CATALOG_FQN = f"{CATALOG}.{SCHEMA}.playbook_catalog"
PLAYBOOK_STEPS_FQN = f"{CATALOG}.{SCHEMA}.playbook_steps"
DECISION_POLICY_FQN = f"{CATALOG}.{SCHEMA}.decision_policy"
RESPONSE_SCHEMAS_FQN = f"{CATALOG}.{SCHEMA}.response_schemas"
VOCABULARIES_FQN = f"{CATALOG}.{SCHEMA}.vocabularies"

print(f"Resolved playbooks_dir: {PLAYBOOKS_DIR}")
print(f"Catalog/schema:         {CATALOG}.{SCHEMA}")
'''


# ---------------------------------------------------------------------------
# Public block builders consumed by the four stage generators
# ---------------------------------------------------------------------------


def _config_block(md_text: str, code_body: str) -> List[nbformat.NotebookNode]:
    return [CellBuilder.markdown(md_text), CellBuilder.code(code_body)]


def _setup_block(needs_model: bool) -> List[nbformat.NotebookNode]:
    body = _SETUP_BODY_NEEDS_MODEL if needs_model else _SETUP_BODY_PUBLISH_ONLY
    return [
        CellBuilder.section("0. Setup + Resolve @production Model"),
        CellBuilder.code(body),
    ]


def c01_setup_block() -> List[nbformat.NotebookNode]:
    return _config_block(_C01_CONFIG_MD, _C01_CONFIG_BODY) + _setup_block(needs_model=False)


def c02_setup_block() -> List[nbformat.NotebookNode]:
    return _config_block(_C02_CONFIG_MD, _C02_CONFIG_BODY) + _setup_block(needs_model=True)


def c03_setup_block() -> List[nbformat.NotebookNode]:
    return _config_block(_C03_CONFIG_MD, _C03_CONFIG_BODY) + _setup_block(needs_model=True)


def c04_setup_block() -> List[nbformat.NotebookNode]:
    return _config_block(_C04_CONFIG_MD, _C04_CONFIG_BODY) + _setup_block(needs_model=True)


# ---------------------------------------------------------------------------
# c01 pipeline-runner cell bodies (shared by hand-authored + generated layers)
# ---------------------------------------------------------------------------


C01_RUN_PIPELINE_MD = """## 2. Run the Generated Pipeline (s01 → s10)

Before `c02_archetype_derivation` and `c04_snapshot_and_dashboard` can run, three artifacts must exist on the cluster:

1. **Gold features table** — `{CATALOG}.{SCHEMA}.customer_features` registered as a feature table
2. **Registered `@production` model** — `models:/{CATALOG}.{SCHEMA}.{MODEL_NAME}@production`
3. **Predictions table** — `{CATALOG}.{SCHEMA}.predictions` (one row per scored customer)

The cells below invoke the generated pipeline scripts produced by `exploration_notebooks/10_spec_generation.ipynb`, in the order **landing → bronze → silver → gold → training → scoring**. Each stage runs as a Databricks notebook task via `dbutils.notebook.run()` — the same pattern NB10 uses for the training leg, extended here to include `scoring` (s10) so the `predictions` table is populated.

**Progress is captured per stage**: elapsed time, any JSON result returned via `dbutils.notebook.exit(...)`, and a rolling total. A failure in any stage raises immediately; nothing downstream runs. Set `SKIP_PIPELINE_RUN = True` in the configuration cell to bypass this block when predictions are already fresh.
"""


C01_RUN_PIPELINE_BODY = '''import json as _json
import time as _time
from pathlib import Path as _Path

from customer_retention.core.compat.detection import get_dbutils, get_spark_session
from customer_retention.core.config.experiments import get_workspace_path
from customer_retention.stages.scoring import ScoringConfig


def _resolve_default_pipeline_dir() -> _Path:
    workspace_path = get_workspace_path()
    if not workspace_path:
        raise RuntimeError(
            "Cannot resolve default PIPELINE_DIR: CR_WORKSPACE_PATH is not set. "
            "Either run databricks_init(workspace_path=...) first, or set PIPELINE_DIR "
            "in the configuration cell to an absolute workspace path."
        )
    scoring_config = ScoringConfig.from_databricks()
    pipeline_name = scoring_config.pipeline_name or scoring_config.composite_name
    if not pipeline_name:
        raise RuntimeError(
            "Cannot resolve default PIPELINE_DIR: ScoringConfig has no pipeline_name. "
            "Set PIPELINE_DIR in the configuration cell to the pipeline folder directly, "
            "e.g. '/Workspace/{workspace_path}/generated_pipelines/databricks/<pipeline_name>'."
        )
    return _Path(f"/Workspace/{workspace_path}/generated_pipelines/databricks/{pipeline_name}")


_g = globals()
_skip = _g.get("SKIP_PIPELINE_RUN", False)
_pipeline_dir_cfg = _g.get("PIPELINE_DIR", None)
_stages = _g.get(
    "PIPELINE_STAGES",
    ["landing", "bronze", "silver", "gold", "training", "scoring"],
)
_timeout = _g.get("PIPELINE_STAGE_TIMEOUT_SECONDS", 3600)
_spark = _g.get("spark") or get_spark_session()

_pipeline_results = {}
_pipeline_errors = []

_dbutils = get_dbutils()

if _skip:
    print("SKIPPED: SKIP_PIPELINE_RUN=True")
elif _spark is None:
    print("SKIPPED: no active Spark session (Databricks-only cell)")
elif _dbutils is None:
    print("SKIPPED: dbutils unavailable (not running on Databricks)")
else:
    _pipeline_dir = _Path(_pipeline_dir_cfg) if _pipeline_dir_cfg else _resolve_default_pipeline_dir()
    if not _pipeline_dir.exists():
        raise FileNotFoundError(
            f"Generated pipeline directory not found at {_pipeline_dir}. "
            "Run exploration_notebooks/10_spec_generation.ipynb first to produce it, "
            "or set PIPELINE_DIR in the configuration cell."
        )

    print(f"PIPELINE_DIR: {_pipeline_dir}")
    print(f"STAGES:       {_stages}")
    print("=" * 70)

    _total_start = _time.time()
    for _stage in _stages:
        _stage_dir = _pipeline_dir / _stage
        if not _stage_dir.exists():
            print(f"[{_stage.upper():<9}] (no scripts in {_stage_dir.name}/ - skipping)")
            continue
        _notebooks = sorted(f.stem for f in _stage_dir.iterdir() if f.suffix == ".py")
        if not _notebooks:
            print(f"[{_stage.upper():<9}] (empty)")
            continue
        for _nb in _notebooks:
            _path = str(_stage_dir / _nb)
            _start = _time.time()
            print(f"[{_stage.upper():<9}] {_nb} ... ", end="", flush=True)
            try:
                _result = _dbutils.notebook.run(_path, _timeout, {})
                _elapsed = _time.time() - _start
                print(f"{_elapsed:>7.1f}s")
                if _result:
                    try:
                        _pipeline_results[_nb] = _json.loads(_result)
                    except (ValueError, TypeError):
                        _pipeline_results[_nb] = {"raw": _result}
            except Exception as _exc:
                _elapsed = _time.time() - _start
                print(f"FAILED after {_elapsed:.1f}s")
                _pipeline_errors.append({"stage": _stage, "notebook": _nb, "error": str(_exc)})
                raise

    print("=" * 70)
    print(f"Total elapsed: {_time.time() - _total_start:.1f}s")
    print(f"Notebooks with returned results: {len(_pipeline_results)}")
'''


C01_PIPELINE_SUMMARY_BODY = '''# Summary: surface training metrics from the training stage (if it returned JSON)
# and confirm the predictions Delta table is populated with a risk-tier distribution.

from customer_retention.core.compat.detection import get_spark_session
from customer_retention.stages.scoring import ScoringConfig

# Resolve upstream state defensively (see run_generated_pipeline cell comment).
_g = globals()
_skip = _g.get("SKIP_PIPELINE_RUN", False)
_spark = _g.get("spark") or get_spark_session()
_results = _g.get("_pipeline_results", {})
_catalog = _g.get("CATALOG")
_schema = _g.get("SCHEMA")
if _catalog is None or _schema is None:
    try:
        _sc = ScoringConfig.from_databricks()
        _catalog = _catalog or _sc.catalog
        _schema = _schema or _sc.schema
    except Exception:
        pass

if _skip:
    print("SKIPPED: SKIP_PIPELINE_RUN=True")
elif _spark is None:
    print("SKIPPED: no active Spark session")
elif not _catalog or not _schema:
    print("SKIPPED: CATALOG/SCHEMA not resolved (setup cell did not run)")
else:
    _predictions_fqn = f"{_catalog}.{_schema}.predictions"

    print("=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)

    _training_result = next(
        (v for k, v in _results.items() if "train" in k.lower() and isinstance(v, dict)),
        None,
    )
    if _training_result:
        print("\\nTraining results:")
        _models = _training_result.get("models")
        if _models:
            print(f"  {'Model':<25} {'AUC':>8} {'PR-AUC':>8} {'F1':>8}")
            print(f"  {'-' * 25} {'-' * 8} {'-' * 8} {'-' * 8}")
            for _name, _metrics in _models.items():
                print(
                    f"  {_name:<25} {_metrics.get('roc_auc', 0):>8.4f} "
                    f"{_metrics.get('pr_auc', 0):>8.4f} {_metrics.get('f1', 0):>8.4f}"
                )
        _best = _training_result.get("best_model")
        if _best:
            print(f"  Best: {_best} (AUC={_training_result.get('best_roc_auc', 0):.4f})")

    try:
        _pred = _spark.table(_predictions_fqn)
        _total = _pred.count()
        print(f"\\nPredictions table: {_predictions_fqn}")
        print(f"  rows: {_total:,}")
        if _total > 0:
            print("\\n  Risk-tier distribution:")
            _pred.groupBy("risk_tier").count().orderBy("risk_tier").show(truncate=False)
            print("  Model URI(s) in this table:")
            _pred.select("model_uri").distinct().show(truncate=False)
            print("  Sample rows:")
            _pred.limit(10).show(truncate=False)
        else:
            print("  WARNING: predictions table is empty. Check scoring stage logs above.")
    except Exception as _exc:
        print(f"\\nERROR reading {_predictions_fqn}: {_exc}")
        print("Pipeline may have failed before scoring. Inspect the output above.")
'''
