"""Completeness gate: every framework apply primitive is decorated.

Two layers of enforcement:

1. **Positive list** (`test_expected_primitives_decorated`): every primitive
   we have decided is an apply primitive must be present in `APPLY_REGISTRY`
   with the right `ApplyOpKind`. New primitives added to the codebase get a
   corresponding entry here. CI fails on missing entries.

2. **Negative list** (`test_no_orphan_apply_functions`): a static AST sweep
   of `stages/`, `analysis/auto_explorer/`, and `transforms/` finds every
   public function named `apply_*` or `derive_*` returning a DataFrame.
   Any such function not in the positive list and not in
   `_NOT_APPLY_OP_ALLOWLIST` is flagged. CI fails on growth of the
   uncovered list.

The negative-list allowlist requires a one-line rationale per entry so
reviewers see the reason in PRs.
"""
from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from customer_retention.parity import APPLY_REGISTRY
from customer_retention.parity.kinds import ApplyOpKind

_FRAMEWORK_MODULES = (
    "customer_retention.analysis.auto_explorer.sampling",
    "customer_retention.stages.lifecycle.enrich",
    "customer_retention.stages.profiling.target_validator",
    "customer_retention.stages.profiling.time_window_aggregator",
    "customer_retention.stages.temporal.temporal_merger",
    "customer_retention.stages.modeling.data_splitter",
    "customer_retention.transforms.ops",
    "customer_retention.transforms.fitted",
)


_EXPECTED = {
    # Landing
    "customer_retention.analysis.auto_explorer.sampling.apply_temporal_lookback": ApplyOpKind.TEMPORAL_LOOKBACK,
    "customer_retention.analysis.auto_explorer.sampling.apply_sample_filters": ApplyOpKind.SAMPLE_FILTER,
    "customer_retention.stages.lifecycle.enrich.enrich_lifecycle_dataset": ApplyOpKind.LIFECYCLE_ENRICH,
    # Bronze
    "customer_retention.stages.profiling.time_window_aggregator.TimeWindowAggregator.aggregate": ApplyOpKind.BRONZE_AGGREGATE,
    "customer_retention.stages.profiling.time_window_aggregator.derive_extra_datetime_features": ApplyOpKind.DATETIME_DERIVE,
    "customer_retention.stages.profiling.time_window_aggregator.derive_entity_datetime_features": ApplyOpKind.DATETIME_DERIVE,
    # Silver
    "customer_retention.stages.temporal.temporal_merger.TemporalMerger.merge_all": ApplyOpKind.SILVER_TEMPORAL_MERGE,
    # Silver derived (single-column stateless ops in transforms/ops.py)
    "customer_retention.transforms.ops.apply_derived_ratio": ApplyOpKind.SILVER_DERIVED_FEATURE,
    "customer_retention.transforms.ops.apply_derived_interaction": ApplyOpKind.SILVER_DERIVED_FEATURE,
    "customer_retention.transforms.ops.apply_derived_composite": ApplyOpKind.SILVER_DERIVED_FEATURE,
    "customer_retention.transforms.ops.apply_derived_recency": ApplyOpKind.SILVER_DERIVED_FEATURE,
    "customer_retention.transforms.ops.apply_derived_duration": ApplyOpKind.SILVER_DERIVED_FEATURE,
    "customer_retention.transforms.ops.apply_derived_cyclical": ApplyOpKind.SILVER_DERIVED_FEATURE,
    "customer_retention.transforms.ops.apply_derived_tenure": ApplyOpKind.SILVER_DERIVED_FEATURE,
    "customer_retention.transforms.ops.apply_derived_extraction_is_weekend": ApplyOpKind.SILVER_DERIVED_FEATURE,
    # Gold stateless transforms
    "customer_retention.transforms.ops.apply_impute_null": ApplyOpKind.GOLD_TRANSFORMATION,
    "customer_retention.transforms.ops.apply_cap_outlier": ApplyOpKind.GOLD_TRANSFORMATION,
    "customer_retention.transforms.ops.apply_type_cast": ApplyOpKind.GOLD_TRANSFORMATION,
    "customer_retention.transforms.ops.apply_drop_column": ApplyOpKind.GOLD_TRANSFORMATION,
    "customer_retention.transforms.ops.apply_winsorize": ApplyOpKind.GOLD_TRANSFORMATION,
    "customer_retention.transforms.ops.apply_segment_aware_cap": ApplyOpKind.GOLD_TRANSFORMATION,
    "customer_retention.transforms.ops.apply_log_transform": ApplyOpKind.GOLD_TRANSFORMATION,
    "customer_retention.transforms.ops.apply_sqrt_transform": ApplyOpKind.GOLD_TRANSFORMATION,
    "customer_retention.transforms.ops.apply_zero_inflation_handling": ApplyOpKind.GOLD_TRANSFORMATION,
    "customer_retention.transforms.ops.apply_cap_then_log": ApplyOpKind.GOLD_TRANSFORMATION,
    "customer_retention.transforms.ops.apply_one_hot_encode": ApplyOpKind.GOLD_ENCODING,
    "customer_retention.transforms.ops.apply_feature_select": ApplyOpKind.GOLD_FEATURE_SPEC_GATE,
    # Gold batch ops
    "customer_retention.transforms.ops.apply_batch_log_transform": ApplyOpKind.GOLD_TRANSFORMATION,
    "customer_retention.transforms.ops.apply_batch_sqrt_transform": ApplyOpKind.GOLD_TRANSFORMATION,
    "customer_retention.transforms.ops.apply_batch_zero_inflation": ApplyOpKind.GOLD_TRANSFORMATION,
    "customer_retention.transforms.ops.apply_batch_cap_then_log": ApplyOpKind.GOLD_TRANSFORMATION,
    "customer_retention.transforms.ops.apply_batch_yeo_johnson": ApplyOpKind.GOLD_TRANSFORMATION,
    # Fitted (each class has fit_transform + transform)
    "customer_retention.transforms.fitted.FittedScaler.fit_transform": ApplyOpKind.GOLD_TRANSFORMATION,
    "customer_retention.transforms.fitted.FittedScaler.transform": ApplyOpKind.GOLD_TRANSFORMATION,
    "customer_retention.transforms.fitted.FittedEncoder.fit_transform": ApplyOpKind.GOLD_ENCODING,
    "customer_retention.transforms.fitted.FittedEncoder.transform": ApplyOpKind.GOLD_ENCODING,
    "customer_retention.transforms.fitted.FittedPowerTransform.fit_transform": ApplyOpKind.GOLD_TRANSFORMATION,
    "customer_retention.transforms.fitted.FittedPowerTransform.transform": ApplyOpKind.GOLD_TRANSFORMATION,
    # Silver target encoding (mirrored by the silver template's CASE WHEN)
    "customer_retention.stages.profiling.target_validator.apply_target_encoding": ApplyOpKind.SILVER_TARGET_LABEL_MAP,
    # Training
    "customer_retention.stages.modeling.data_splitter.DataSplitter.split": ApplyOpKind.TRAINING_SPLIT,
}


_NOT_APPLY_OP_ALLOWLIST = {
    "customer_retention.analysis.auto_explorer.findings.apply_zero_inflation_opt_in": (
        "config helper: mutates a findings dict in place, returns None"
    ),
    "customer_retention.analysis.auto_explorer.objective_support_communicator.apply_signal_rules": (
        "rule reducer: returns (int, list[str]), not a DataFrame"
    ),
    "customer_retention.analysis.auto_explorer.prediction_objective_detector.derive_objective_support": (
        "analysis: returns ObjectiveSupport dataclass, not a DataFrame"
    ),
    "customer_retention.analysis.auto_explorer.snapshot_grid.SnapshotGrid.apply_votes": (
        "config aggregator: merges per-dataset votes into the grid struct"
    ),
    "customer_retention.stages.causal.dashboard_profile_override.apply_profile_override": (
        "causal track: out of audit scope (Spark-only, omitted from coverage)"
    ),
    "customer_retention.stages.causal.derivation.derive_archetypes_and_policies": (
        "causal track: out of audit scope"
    ),
    "customer_retention.stages.causal.snapshot_writer.apply_decision_policy": (
        "causal track: out of audit scope"
    ),
    "customer_retention.stages.ingestion.loaders.DataLoader.apply_sample": (
        "loader internal: head(N) only, never invoked in the pipeline emission path"
    ),
    "customer_retention.stages.modeling.feature_profile.SelectionTraceRecorder.apply_to_profile": (
        "trace recorder: mutates a profile struct, not a DataFrame transform"
    ),
    "customer_retention.stages.profiling.temporal_coverage.derive_drift_implications": (
        "analysis: returns drift summary dict, not a DataFrame"
    ),
    "customer_retention.stages.scoring.batch_inference.apply_risk_tiers_pandas": (
        "scoring stage: downstream of training, outside parity-audit scope"
    ),
    "customer_retention.stages.scoring.batch_inference.apply_risk_tiers_spark": (
        "scoring stage: downstream of training, outside parity-audit scope"
    ),
    "customer_retention.stages.temporal.timestamp_discovery.DatetimeOrderAnalyzer.derive_last_action_date": (
        "analyzer: returns a single timestamp, not a DataFrame"
    ),
    "customer_retention.transforms.executor.TransformExecutor.apply_all": (
        "dispatcher: routes to individual decorated apply_* leaves; per the spec "
        "we decorate leaves, not dispatchers"
    ),
}


@pytest.fixture(scope="module", autouse=True)
def _ensure_modules_imported():
    for mod in _FRAMEWORK_MODULES:
        importlib.import_module(mod)
    yield


class TestPositiveDecorationManifest:
    def test_every_expected_primitive_registered(self):
        missing = sorted(set(_EXPECTED) - set(APPLY_REGISTRY))
        assert not missing, (
            "Apply primitives are expected in APPLY_REGISTRY but missing: "
            f"{missing}"
        )

    def test_every_expected_primitive_has_correct_kind(self):
        wrong: list[tuple[str, str, str]] = []
        for qn, expected_kind in _EXPECTED.items():
            if qn not in APPLY_REGISTRY:
                continue
            actual = APPLY_REGISTRY[qn].kind
            if actual is not expected_kind:
                wrong.append((qn, expected_kind.name, actual.name))
        assert not wrong, (
            "ApplyOpKind mismatches: "
            + "; ".join(f"{qn}: expected={ek} actual={ak}" for qn, ek, ak in wrong)
        )

    def test_no_unexpected_primitives(self):
        unexpected = [qn for qn in APPLY_REGISTRY if qn not in _EXPECTED]
        unexpected = [
            qn for qn in unexpected
            if qn.startswith("customer_retention.")
            and not qn.startswith("tests.")
        ]
        assert not unexpected, (
            "APPLY_REGISTRY contains entries not on the positive list. "
            "Add them to _EXPECTED with the correct kind, or remove the "
            f"@apply_op decoration: {sorted(unexpected)}"
        )


def _candidates_in_source(path: Path) -> list[str]:
    """Return public function/method qualnames that look like apply primitives."""
    tree = ast.parse(path.read_text())
    rel_module = (
        str(path.with_suffix("").relative_to(Path("src")))
        .replace("/", ".")
    )
    found: list[str] = []

    class _Walker(ast.NodeVisitor):
        def __init__(self):
            self.class_stack: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef):
            self.class_stack.append(node.name)
            self.generic_visit(node)
            self.class_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef):
            self._check(node)
            self.generic_visit(node)

        def _check(self, node):
            name = node.name
            if name.startswith("_"):
                return
            if not (name.startswith("apply_") or name.startswith("derive_")):
                return
            qual_prefix = ".".join(self.class_stack)
            qn = f"{rel_module}.{qual_prefix}.{name}".replace("..", ".")
            found.append(qn)

    _Walker().visit(tree)
    return found


class TestNegativeAllowlistOrphans:
    def test_no_orphan_apply_functions(self):
        roots = [
            Path("src/customer_retention/stages"),
            Path("src/customer_retention/analysis/auto_explorer"),
            Path("src/customer_retention/transforms"),
        ]
        candidates: set[str] = set()
        for root in roots:
            for py in root.rglob("*.py"):
                if py.name == "__init__.py":
                    continue
                candidates.update(_candidates_in_source(py))

        decorated = set(APPLY_REGISTRY.keys())
        allowlisted = set(_NOT_APPLY_OP_ALLOWLIST.keys())
        orphans = sorted(c for c in candidates if c not in decorated and c not in allowlisted)
        assert not orphans, (
            "Functions named apply_* or derive_* are not decorated and not on the "
            "_NOT_APPLY_OP_ALLOWLIST. Either decorate them or add an allowlist entry "
            f"with a one-line rationale: {orphans}"
        )
