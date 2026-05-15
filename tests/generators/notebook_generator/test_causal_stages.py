"""Smoke tests for the five causal-track stage generators (s_c01..s_c05).

These tests guard the layer-2 (generated_pipelines) side of the causal
track. They verify that each stage generator emits a notebook with the
expected cell layout, the right setup block flavor, and the algorithmic
cell that calls the framework library — without spinning up a Spark
cluster.

The hand-authored layer (``causal_notebooks/c0X_*.ipynb``) is verified
separately by ``scripts/notebooks/build_causal_notebooks.py`` and a
small structural check in this file.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from customer_retention.generators.notebook_generator import (
    DatabricksNotebookGenerator,
    LocalNotebookGenerator,
    NotebookConfig,
    NotebookStage,
)
from customer_retention.generators.notebook_generator.stages import (
    ApprovalGateStage,
    ArchetypeDerivationStage,
    BatchInferenceCausalStage,
    PublishDefinitionsStage,
    SnapshotAndDashboardStage,
)

_CAUSAL_STAGES = [
    (NotebookStage.PUBLISH_DEFINITIONS, PublishDefinitionsStage, "c01"),
    (NotebookStage.ARCHETYPE_DERIVATION, ArchetypeDerivationStage, "c02"),
    (NotebookStage.APPROVAL_GATE, ApprovalGateStage, "c03"),
    (NotebookStage.BATCH_INFERENCE_CAUSAL, BatchInferenceCausalStage, "c04"),
    (NotebookStage.SNAPSHOT_AND_DASHBOARD, SnapshotAndDashboardStage, "c05"),
]


def _all_code(nb) -> str:
    return "\n".join(c.source for c in nb.cells if c.cell_type == "code")


def _has_cell_starting_with(nb, prefix: str) -> bool:
    for c in nb.cells:
        if c.cell_type == "code" and c.source.lstrip().startswith(prefix):
            return True
    return False


# ---------------------------------------------------------------------------
# Stage generator output
# ---------------------------------------------------------------------------


class TestStageRegistration:
    @pytest.mark.parametrize("stage,_cls,_label", _CAUSAL_STAGES)
    def test_local_generator_registers_stage(self, stage, _cls, _label):
        gen = LocalNotebookGenerator(NotebookConfig(), None)
        assert stage in gen.stage_generators

    @pytest.mark.parametrize("stage,_cls,_label", _CAUSAL_STAGES)
    def test_databricks_generator_registers_stage(self, stage, _cls, _label):
        gen = DatabricksNotebookGenerator(NotebookConfig(), None)
        assert stage in gen.stage_generators

    @pytest.mark.parametrize("stage,_cls,_label", _CAUSAL_STAGES)
    def test_enum_value_is_flat(self, stage, _cls, _label):
        # Causal-stage enum values must NOT contain a path prefix; the
        # hand-authored notebooks live at /causal_notebooks/, the script
        # output goes flat into generated_pipelines/{platform}/.
        assert "/" not in stage.value
        assert stage.value.startswith(_label + "_")


class TestNotebookEmission:
    @pytest.mark.parametrize("stage,_cls,label", _CAUSAL_STAGES)
    def test_local_emits_valid_notebook(self, stage, _cls, label):
        gen = LocalNotebookGenerator(NotebookConfig(), None)
        nb = gen.generate_stage(stage)
        assert nb.nbformat == 4
        assert len(nb.cells) > 0
        # Setup cell uses ScoringConfig (the local-config branch fix)
        if label != "c01":
            code = _all_code(nb)
            assert "ScoringConfig" in code
            assert "from_local_config(get_experiments_dir())" in code

    @pytest.mark.parametrize("stage,_cls,label", _CAUSAL_STAGES)
    def test_databricks_emits_valid_notebook(self, stage, _cls, label):
        gen = DatabricksNotebookGenerator(NotebookConfig(), None)
        nb = gen.generate_stage(stage)
        assert nb.nbformat == 4
        assert len(nb.cells) > 0

    @pytest.mark.parametrize("stage,_cls,_label", _CAUSAL_STAGES)
    def test_compiles(self, stage, _cls, _label):
        gen = LocalNotebookGenerator(NotebookConfig(), None)
        nb = gen.generate_stage(stage)
        compile(_all_code(nb), f"<{stage.name}>", "exec")


class TestPerStageContent:
    """Each stage's algorithmic cell calls the right framework function."""

    def test_c01_calls_publishers(self):
        gen = LocalNotebookGenerator(NotebookConfig(), None)
        code = _all_code(gen.generate_stage(NotebookStage.PUBLISH_DEFINITIONS))
        assert "load_playbooks_from_dir" in code
        assert "load_policies_from_dir" in code
        assert "overwrite_table" in code
        assert "RISK_TIER_HIGH_THRESHOLD" in code
        # No model-URI lookup in the publish-only setup
        assert "MODEL_URI" not in code
        assert "build_llm_namer" not in code

    def test_c02_calls_derive_and_includes_composite_name(self):
        gen = LocalNotebookGenerator(NotebookConfig(), None)
        code = _all_code(gen.generate_stage(NotebookStage.ARCHETYPE_DERIVATION))
        assert "derive_archetypes_and_policies" in code
        assert "DerivationConfig" in code
        # GOLD_FEATURES_FQN must include the composite_name (regression guard)
        assert "gold_features_{COMPOSITE_NAME}" in code
        assert "COMPOSITE_NAME" in code
        assert "build_llm_namer" in code  # only c02 builds the namer
        assert "feature_cap=KMEANS_FEATURE_CAP" in code

    def test_c02_passes_model_uri_not_loaded_model(self):
        """Regression for the recurring MLflow / FE model-loading failure modes:
          1. PyFuncModel exposes no featureImportances (MLflow wrapper change)
          2. KeyError: 'spark' from mlflow.spark.load_model on FE-wrapped models
          3. Any sub-artifact path guess that ages out with FE versions

        The durable fix: c02 passes ``model_uri=MODEL_URI`` (a string) to
        ``DerivationConfig``; ``shap_runner`` / ``shap_attribution`` load the
        training-time attribution artifact from that URI. No model loading,
        no model introspection, no MLflow/FE internal surface exposed to c02."""
        gen = LocalNotebookGenerator(NotebookConfig(), None)
        code = _all_code(gen.generate_stage(NotebookStage.ARCHETYPE_DERIVATION))
        # Durable primitives present
        assert "model_uri=MODEL_URI" in code
        # Fragile primitives absent
        assert "mlflow.pyfunc.load_model" not in code
        assert "mlflow.spark.load_model" not in code
        assert "unwrap_tree_model" not in code
        # No direct model-object passing
        assert "model=unwrap" not in code
        assert "model=mlflow" not in code

    def test_c02_excludes_key_columns_from_feature_set(self):
        """Key columns (``event_timestamp``, ``inference_point_in_time``,
        ``account_id``, ``entity_id``, ``target``, ``churn_probability``,
        ``model_uri``) must not appear in ``feature_columns`` — they are
        lookup keys or metadata, not features the model was trained on.

        Attribution is loaded from the MLflow run via ``MODEL_URI`` so c02
        no longer needs to reconstruct FE lookup keys — but the feature
        filter still has to exclude these columns or ``compute_shap_distributed``
        will fail schema validation."""
        gen = LocalNotebookGenerator(NotebookConfig(), None)
        code = _all_code(gen.generate_stage(NotebookStage.ARCHETYPE_DERIVATION))
        for col in (
            '"account_id"', '"entity_id"', '"target"', '"churn_probability"',
            '"event_timestamp"', '"inference_point_in_time"', '"model_uri"',
        ):
            assert col in code, f"{col!r} must appear in the feature-column exclusion tuple"

    def test_c02_no_longer_emits_entity_key_cols(self):
        """Attribution is persisted at training time; there is no
        ``fe.score_batch`` round-trip at derivation time, so ``entity_key_cols``
        and its ``event_timestamp`` append branch are no longer emitted. This
        test guards against regressions that would reintroduce the obsolete
        code path."""
        gen = LocalNotebookGenerator(NotebookConfig(), None)
        code = _all_code(gen.generate_stage(NotebookStage.ARCHETYPE_DERIVATION))
        assert "entity_key_cols=" not in code
        assert "entity_key_cols.append" not in code
        assert "background_sample_size=" not in code

    def test_c03_calls_approval_gate(self):
        gen = LocalNotebookGenerator(NotebookConfig(), None)
        code = _all_code(gen.generate_stage(NotebookStage.APPROVAL_GATE))
        assert "auto_promote_stable" in code
        assert "list_pending_review" in code
        assert "STABILITY_THRESHOLD" in code
        # c03 doesn't build the LLM namer
        assert "build_llm_namer" not in code

    def test_c04_calls_run_batch_inference(self):
        """c04 is the standalone scoring refresh — it must call
        ``run_batch_inference`` against the registered ``@production`` model
        and must NOT touch archetypes / snapshot / dashboard publishing
        (those belong to c02 and c05)."""
        gen = LocalNotebookGenerator(NotebookConfig(), None)
        nb = gen.generate_stage(NotebookStage.BATCH_INFERENCE_CAUSAL)
        code = _all_code(nb)
        assert "run_batch_inference" in code
        assert "BatchInferenceConfig" in code
        # Freshness gate + three modes
        assert "BATCH_INFERENCE_MODE" in code
        assert "PREDICTIONS_STALE_AFTER_HOURS" in code
        assert '"auto"' in code
        assert '"always"' in code
        assert '"never"' in code
        # Must NOT cross-call into archetype / snapshot / dashboard modules
        assert "build_eligibility_snapshot" not in code
        assert "publish_dashboard_views" not in code
        assert "derive_archetypes_and_policies" not in code

    def test_c04_applies_nb00_scope_filter(self):
        """The scoring population must match NB00's ``ProjectContext.sample_filters``
        for the target dataset — otherwise batch inference scores entities
        that were excluded from training / exploration, poisoning every
        downstream archetype assignment with out-of-scope rows.

        The notebook resolves the filter from ``RunNamespace.project_context_path``
        and threads it into ``BatchInferenceConfig.filter_expression``.
        Entity-level already-positive exclusion is now a separate channel
        (``exclude_already_positive_*`` fields) — see
        ``test_c04_always_excludes_already_positive_target``."""
        gen = LocalNotebookGenerator(NotebookConfig(), None)
        nb = gen.generate_stage(NotebookStage.BATCH_INFERENCE_CAUSAL)
        code = _all_code(nb)
        assert "ProjectContext" in code
        assert "project_context_path" in code
        assert "sample_filters" in code
        assert "_resolve_scope_filter" in code
        assert "filter_expression=_scope_filter" in code

    def test_c04_always_excludes_already_positive_target(self):
        """Scoring must drop any entity whose target label has EVER been 1
        in landing — entity-level wholesale, not row-level. The trained
        model has nothing useful to say about an already-positive row, and
        the CSM-facing dashboard wastes headcount on accounts the business
        can no longer recover.

        The exclusion is derived from ``ProjectContext.target_column`` +
        the target dataset's ``entity_column`` and threaded into three
        dedicated ``BatchInferenceConfig`` fields. The framework applies a
        ``left_anti`` join against
        ``landing_<target>.filter(<target_col>=1)`` — a row-level
        ``<target_col> = 0`` predicate in ``filter_expression`` is wrong
        by construction because an entity with both 0-rows and 1-rows
        survives ``.distinct(entity_id)`` and slips through."""
        gen = LocalNotebookGenerator(NotebookConfig(), None)
        nb = gen.generate_stage(NotebookStage.BATCH_INFERENCE_CAUSAL)
        code = _all_code(nb)
        assert "_resolve_already_positive_exclusion" in code
        assert "target_column" in code
        # The three exclusion fields must reach BatchInferenceConfig.
        assert "exclude_already_positive_target_column=_excl_target_col" in code
        assert "exclude_already_positive_via_table=_excl_landing_fqn" in code
        assert "exclude_already_positive_entity_key=_excl_entity_key" in code
        # The scope filter still flows through filter_expression — separately.
        assert "filter_expression=_scope_filter" in code
        # The old row-level composition path must be gone: the helper that
        # AND-joined ``churned = 0`` into filter_expression silently failed
        # against snapshot-panel landing tables and is the bug we're fixing.
        assert "_compose_scoring_filter" not in code
        assert "filter_expression=_scoring_filter" not in code
        # Operator-visible diagnostics: the cell must print which clauses
        # are in force so a confused operator can see why their account
        # population dropped.
        assert "Already-positive exclusion" in code

    def test_c04_passes_gold_features_fqn_as_customer_table(self):
        """Regression: the default ``gold_customers`` table doesn't exist
        in projects that use composite-name-qualified gold tables. c04 must
        route the already-resolved ``GOLD_FEATURES_FQN`` into
        ``BatchInferenceConfig.customer_table`` so fe.score_batch can find
        the entity universe."""
        gen = LocalNotebookGenerator(NotebookConfig(), None)
        nb = gen.generate_stage(NotebookStage.BATCH_INFERENCE_CAUSAL)
        code = _all_code(nb)
        assert "customer_table=GOLD_FEATURES_FQN" in code

    def test_c04_passes_model_uri_not_model_name(self):
        """Regression: ``ScoringConfig.registered_model_name`` is already a 3-part
        FQN (``{catalog}.{schema}.model_{CN}``). Passing it as ``model_name``
        makes the framework re-prefix ``{catalog}.{schema}`` and produce a
        broken URI like ``models:/cat.sch.cat.sch.model_abc@production``.
        The setup block already computes the correct ``MODEL_URI`` — c04 must
        thread that constant through ``BatchInferenceConfig.model_uri``."""
        gen = LocalNotebookGenerator(NotebookConfig(), None)
        nb = gen.generate_stage(NotebookStage.BATCH_INFERENCE_CAUSAL)
        code = _all_code(nb)
        assert "model_uri=MODEL_URI" in code
        assert "model_name=MODEL_NAME" not in code

    def test_c05_calls_snapshot_writer_and_dashboard_views(self):
        gen = LocalNotebookGenerator(NotebookConfig(), None)
        nb = gen.generate_stage(NotebookStage.SNAPSHOT_AND_DASHBOARD)
        code = _all_code(nb)
        # Phase 3 wires up the live snapshot writer and dashboard publisher
        assert "build_eligibility_snapshot" in code
        assert "SnapshotConfig" in code
        assert "publish_dashboard_views" in code
        assert "DASHBOARD_VIEW_NAMES" in code
        # The four Delta tables the snapshot writer reads must be referenced
        assert "ELIGIBILITY_SNAPSHOT_FQN" in code
        assert "DECISION_POLICY_FQN" in code
        assert "PREDICTIONS_FQN" in code
        # NotImplementedError / RUN_PHASE3 placeholder must be gone
        assert "RUN_PHASE3" not in code
        assert "NotImplementedError" not in code
        # The summary cell exists and queries archetype_catalog
        assert "archetype_catalog row counts" in code
        # c05 must point operators to c04, not the old s10-only wording
        assert "run c04_batch_inference first" in code

    def test_c05_materializes_provenance_prereqs_before_publish(self):
        """The provenance prereq materializer (5.3.5) must precede 5.4
        publish so v_feature_provenance gets its feature_meta +
        column_descriptions UC tables in place; otherwise
        publish_dashboard_views silently skips the provenance view and
        the dashboard's per-row lineage disclosure goes dark.

        The materializer pins the namespace via ENGAGEMENT_RUN_ID +
        ENGAGEMENT_EXPERIMENTS_DIR rather than RunNamespace.from_env_or_latest()
        because the latter resolves by mtime and drifts on clusters with
        multiple runs sharing an experiments dir (the regression that
        motivated v3 §U on setup_and_resolve_model).
        """
        gen = LocalNotebookGenerator(NotebookConfig(), None)
        nb = gen.generate_stage(NotebookStage.SNAPSHOT_AND_DASHBOARD)
        code = _all_code(nb)
        # The materialize step is present and emits the [mt4r] trace prefix.
        assert "[mt4r] feature_meta" in code
        assert "[mt4r] column_descriptions" in code
        assert "[mt4r] feature_population_stats" in code
        # It pins the namespace explicitly via the ENGAGEMENT constants
        # (the v3 §U pattern). ``from_env_or_latest`` is fine elsewhere in
        # c05 -- e.g. in ``write_run_context`` -- but the materialize cell
        # itself must not rely on it; the surrounding assert string is
        # specific enough that drift back to ``from_env_or_latest`` for the
        # provenance loaders would fail this assertion.
        assert "RunNamespace(root=Path(ENGAGEMENT_EXPERIMENTS_DIR)" in code
        assert "RunNamespace(root=Path(_ENG_CD_ROOT)" in code
        # column_descriptions is root-scoped -- the override knob is honored.
        assert "ENGAGEMENT_COLUMN_DESCRIPTIONS_ROOT" in code
        # Idempotency guard: skip when UC table already exists.
        assert "spark.catalog.tableExists(_fm_fqn)" in code
        # Order check: materialize must come BEFORE publish_dashboard_views
        # so the framework's _provenance_prerequisites_present() check at
        # publish time finds the UC tables already in place.
        idx_materialize = code.find("[mt4r] feature_meta")
        idx_publish     = code.find("publish_dashboard_views(spark, CATALOG, SCHEMA)")
        assert idx_materialize > 0
        assert idx_publish > 0
        assert idx_materialize < idx_publish, (
            "materialize_provenance_prereqs must precede publish_dashboard_views"
        )


class TestSetupBlockSeparation:
    def test_c01_setup_does_not_load_model(self):
        gen = LocalNotebookGenerator(NotebookConfig(), None)
        code = _all_code(gen.generate_stage(NotebookStage.PUBLISH_DEFINITIONS))
        assert "MODEL_URI" not in code
        assert "MODEL_VERSION" not in code
        assert "mlflow.tracking.MlflowClient" not in code

    @pytest.mark.parametrize("stage", [
        NotebookStage.ARCHETYPE_DERIVATION,
        NotebookStage.APPROVAL_GATE,
        NotebookStage.BATCH_INFERENCE_CAUSAL,
        NotebookStage.SNAPSHOT_AND_DASHBOARD,
    ])
    def test_c02_c03_c04_c05_setup_loads_model(self, stage):
        gen = LocalNotebookGenerator(NotebookConfig(), None)
        code = _all_code(gen.generate_stage(stage))
        assert "MODEL_URI" in code
        assert "MODEL_VERSION" in code
        # Local branch must use ScoringConfig.from_local_config (regression guard)
        assert "ScoringConfig.from_local_config(get_experiments_dir())" in code
        # The deleted hardcoded MODEL_NAME = "local_model" assignment must NOT
        # appear standalone — only as a fallback `or "local_model"` after the
        # ScoringConfig lookup.
        assert 'scoring_config.best_model_name or "local_model"' in code


# ---------------------------------------------------------------------------
# Hand-authored notebooks at causal_notebooks/
# ---------------------------------------------------------------------------


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_CAUSAL_DIR = _PROJECT_ROOT / "causal_notebooks"

_CR_TAG = re.compile(r"^# @cr:(\w+)\s+name='([^']+)'\s+id=([0-9a-f]{8})$")
_DOC_TAG = re.compile(r"^\[//\]: # \(cr:doc name='([^']+)' id=([0-9a-f]{8})\)$")


@pytest.mark.parametrize("basename", [
    "c01_publish_definitions",
    "c02_archetype_derivation",
    "c03_approval_gate",
    "c04_batch_inference",
    "c05_snapshot_and_dashboard",
])
class TestCausalNotebookFiles:
    def _load(self, basename: str):
        path = _CAUSAL_DIR / f"{basename}.ipynb"
        assert path.exists(), f"missing {path}"
        return json.loads(path.read_text(encoding="utf-8"))

    def test_file_exists(self, basename):
        assert (_CAUSAL_DIR / f"{basename}.ipynb").exists()

    def test_every_cell_has_valid_tag(self, basename):
        nb = self._load(basename)
        for i, cell in enumerate(nb["cells"]):
            src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            first = src.split("\n", 1)[0]
            if cell["cell_type"] == "code":
                m = _CR_TAG.match(first)
                assert m, f"{basename} cell {i}: bad code tag {first!r}"
                assert m.group(3) == cell["id"], (
                    f"{basename} cell {i}: tag id {m.group(3)} != json id {cell['id']}"
                )
            else:
                m = _DOC_TAG.match(first)
                assert m, f"{basename} cell {i}: bad markdown tag {first!r}"
                assert m.group(2) == cell["id"]

    def test_unique_cell_ids(self, basename):
        nb = self._load(basename)
        ids = [c["id"] for c in nb["cells"]]
        assert len(ids) == len(set(ids))

    def test_has_profiler_block(self, basename):
        nb = self._load(basename)
        found = False
        for c in nb["cells"]:
            if c["cell_type"] != "code":
                continue
            src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
            if "# --- cr:profiler ---" in src and "# --- /cr:profiler ---" in src:
                found = True
                break
        assert found

    def test_has_release_stage_memory_cleanup(self, basename):
        nb = self._load(basename)
        last = nb["cells"][-1]
        assert last["cell_type"] == "code"
        src = "".join(last["source"]) if isinstance(last["source"], list) else last["source"]
        assert "release_stage_memory()" in src
