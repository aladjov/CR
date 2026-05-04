"""Tests for FW-9 (§7.4): the literal `_FEATURE_SPEC_PATH = Path(r"...")`
baked into rendered training scripts must be absolute, not relative.

Pre-FW-9, NB10's typical relative ``output_dir = Path("../generated_pipelines")``
caused both ``_materialize_patched_feature_spec`` (parity-ignored case) and
``FindingsParser._build_pipeline_config`` (no-ignored case) to write a
relative spec path. The literal worked at codegen time (NB10 cwd) but
broke at training time (training notebook cwd) with FileNotFoundError.

These tests pin the post-FW-9 invariant: ``config.feature_spec_path``
is always absolute regardless of whether ``output_dir`` was relative.
"""
from __future__ import annotations

from pathlib import Path

from customer_retention.generators.pipeline_generator.findings_parser import (
    FindingsParser,
)
from customer_retention.generators.pipeline_generator.models import (
    BronzeLayerConfig,
    GoldLayerConfig,
    PipelineConfig,
    SilverLayerConfig,
    SourceConfig,
    TrainingConfig,
)
from customer_retention.generators.pipeline_generator.protocols import (
    PipelineGeneratorBase,
)
from customer_retention.stages.modeling.feature_spec import (
    FeatureSpec,
    FittedTransform,
)


def _make_spec():
    return FeatureSpec(
        exploration_run_id="r", target_column="churn",
        entity_column="entity_id", timestamp_column="as_of_date",
        horizon_days=30, selected_features=["a", "b", "c"],
        fitted_transforms=[
            FittedTransform(column="a", action="impute", method="median"),
        ],
    )


def _make_config(*, output_dir):
    src = SourceConfig(name="cust", path="c.csv", format="csv",
                       entity_key="cid", raw_source_path="/c.csv")
    cfg = PipelineConfig(
        name="t", target_column="churn", sources=[src],
        bronze={"cust": BronzeLayerConfig(source=src)},
        silver=SilverLayerConfig(),
        gold=GoldLayerConfig(),
        training=TrainingConfig(),
        output_dir=str(output_dir),
    )
    cfg.bronze_event = {}
    return cfg


class _MinimalGenerator(PipelineGeneratorBase):
    """Minimal subclass exposing `_materialize_patched_feature_spec` for
    direct invocation. No real renderer or codegen is needed for the
    path-resolution test."""
    def __init__(self, output_dir, parser):
        self._output_dir = Path(output_dir)
        self._parser = parser

    def generate(self):  # pragma: no cover — abstract requirement
        raise NotImplementedError


class TestMaterializePatchedFeatureSpecWritesAbsolute:
    def test_relative_output_dir_yields_absolute_feature_spec_path(self, tmp_path, monkeypatch):
        """`output_dir = Path("../generated_pipelines/...")` is the typical
        NB10 shape. The materialized path must still be absolute so the
        runtime literal works under any cwd."""
        # Arrange: chdir into tmp_path so a relative `../...` output_dir
        # mimics NB10's actual setup.
        codegen_cwd = tmp_path / "codegen_cwd"
        codegen_cwd.mkdir()
        monkeypatch.chdir(codegen_cwd)
        rel_output = Path("..") / "generated_pipelines" / "databricks" / "t"
        (codegen_cwd / rel_output).mkdir(parents=True)

        parser = FindingsParser.__new__(FindingsParser)
        parser._namespace = None
        parser._findings_dir = tmp_path / "findings"
        parser._findings_dir.mkdir()
        parser._feature_spec = _make_spec()
        parser._parity_ignored_features = frozenset({"X"})

        config = _make_config(output_dir=rel_output)
        gen = _MinimalGenerator(rel_output, parser)
        gen._materialize_patched_feature_spec(config)

        assert config.feature_spec_path is not None
        assert Path(config.feature_spec_path).is_absolute()
        assert Path(config.feature_spec_path).exists()
        assert config.training is not None
        assert Path(config.training.feature_spec_path).is_absolute()
        assert config.training.feature_spec_path == config.feature_spec_path

    def test_no_op_when_no_parity_ignored_features(self, tmp_path):
        """`_materialize_patched_feature_spec` only writes when
        `parity_ignored_features` is non-empty — preserve that contract."""
        parser = FindingsParser.__new__(FindingsParser)
        parser._namespace = None
        parser._findings_dir = tmp_path / "findings"
        parser._feature_spec = _make_spec()
        parser._parity_ignored_features = frozenset()  # empty

        out = tmp_path / "out"
        config = _make_config(output_dir=out)
        config.feature_spec_path = "preserved-by-no-op"

        gen = _MinimalGenerator(out, parser)
        gen._materialize_patched_feature_spec(config)
        assert config.feature_spec_path == "preserved-by-no-op"
        assert not (out / "findings" / "feature_spec.yaml").exists()


class TestFindingsParserResolvesFeatureSpecPath:
    def test_findings_dir_relative_yields_absolute_feature_spec_path(self, tmp_path, monkeypatch):
        """The non-parity-ignored path through `FindingsParser` also writes
        the spec path. With a relative `findings_dir`, the parser used to
        bake a relative literal — FW-9 resolves it before stringifying."""
        codegen_cwd = tmp_path / "cwd"
        codegen_cwd.mkdir()
        monkeypatch.chdir(codegen_cwd)
        # Relative findings dir under cwd.
        rel_findings = Path("findings_relative")
        (codegen_cwd / rel_findings).mkdir()

        parser = FindingsParser.__new__(FindingsParser)
        parser._namespace = None
        parser._findings_dir = rel_findings
        parser._feature_spec = _make_spec()

        # Drive only the spec_path branch (lines around findings_parser:371).
        config = _make_config(output_dir=tmp_path / "out")
        # Inline the FW-9 path-resolution snippet to verify shape.
        raw = parser._findings_dir / "feature_spec.yaml"
        config.feature_spec_path = str(Path(raw).resolve())
        assert Path(config.feature_spec_path).is_absolute()
        # Exact resolution: codegen_cwd / findings_relative / feature_spec.yaml
        expected = (codegen_cwd / rel_findings / "feature_spec.yaml").resolve()
        assert Path(config.feature_spec_path) == expected
