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

The cell below is the only place you should need to edit. Every value here is read by the publish cell — nothing is hardcoded inside the algorithm.

- **`SKIP_PUBLISH_DEFINITIONS`** — short-circuit the publish step (e.g. when the YAMLs are unchanged since the last run).
- **`RISK_TIER_HIGH_THRESHOLD` / `RISK_TIER_MEDIUM_THRESHOLD`** — risk tier cutoffs mirrored into `decision_policy` so historical assignments can be reconstructed from the policy version in force at scoring time. The publish step writes these as the on-disk default; if a YAML row in `decision_policy.yaml` already specifies them, the YAML wins.
"""

_C01_CONFIG_BODY = '''SKIP_PUBLISH_DEFINITIONS = False

RISK_TIER_HIGH_THRESHOLD = 0.6
RISK_TIER_MEDIUM_THRESHOLD = 0.3
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
