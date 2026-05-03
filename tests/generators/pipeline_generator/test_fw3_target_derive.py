"""FW-3: imperative-lane call-site emission for target_derive @cr.register
functions. Generated pipeline materializes a `target_derive/` script that
imports user_extensions, reads each declared landing UC table, calls the
function, and writes the per-dataset outputs back to landing_<dataset>.
The pipeline runner orchestrates the new phase between landing and bronze.
"""
from __future__ import annotations

import ast

import pytest

from customer_retention.generators.pipeline_generator.databricks_generator import (
    DatabricksPipelineGenerator,
)
from customer_retention.generators.pipeline_generator.databricks_renderer import (
    DatabricksCodeRenderer,
)
from customer_retention.generators.pipeline_generator.models import (
    BronzeLayerConfig,
    GoldLayerConfig,
    LandingLayerConfig,
    PipelineConfig,
    SilverLayerConfig,
    SourceConfig,
)
from customer_retention.generators.pipeline_generator.renderer import CodeRenderer
from customer_retention.runtime.harvest import HarvestResult
from customer_retention.runtime.registry import RegisteredFunction


def _rf_target_derive(
    name: str = "derive_churn_target",
    datasets=("account", "contract", "case"),
    primary: str = "account",
):
    return RegisteredFunction(
        name=name,
        source=(
            f"def {name}(spark, datasets, namespace):\n"
            f"    return {{'account': datasets['account']}}\n"
        ),
        scope="datasets",
        dataset=None,
        datasets=list(datasets),
        primary=primary,
        replay_at_scoring=False,
        expected_stage="target_derive",
        inferred_stage="target_derive",
        notebook_path=None,
        cell_id=None,
    )


def _bare_pipeline_config(name: str = "p") -> PipelineConfig:
    src = SourceConfig(
        name="account", path="account.csv", format="csv",
        entity_key="account_id", raw_source_path="/data/account.csv",
    )
    return PipelineConfig(
        name=name, target_column="churned",
        sources=[src],
        landing={
            "account": LandingLayerConfig(
                source=src,
                raw_source_path="/data/account.csv",
                raw_source_format="csv",
                entity_column="account_id",
                time_column="event_timestamp",
                target_column="churned",
            ),
        },
        bronze={"account": BronzeLayerConfig(source=src)},
        silver=SilverLayerConfig(),
        gold=GoldLayerConfig(transformations=[], encodings=[], feature_selections=[]),
        output_dir=".",
    )


class TestDatabricksTargetDeriveTemplate:
    def test_emits_user_extensions_import(self):
        rendered = DatabricksCodeRenderer(catalog="cat", schema="sch").render_target_derive(
            _bare_pipeline_config(), [_rf_target_derive()],
        )
        assert "from user_extensions import derive_churn_target" in rendered

    def test_reads_each_declared_dataset_from_uc(self):
        rendered = DatabricksCodeRenderer(catalog="cat", schema="sch").render_target_derive(
            _bare_pipeline_config(), [_rf_target_derive()],
        )
        for ds in ("account", "contract", "case"):
            assert f'spark.table(f"{{CATALOG}}.{{SCHEMA}}.landing_{ds}")' in rendered

    def test_writes_outputs_back_to_landing(self):
        rendered = DatabricksCodeRenderer(catalog="cat", schema="sch").render_target_derive(
            _bare_pipeline_config(), [_rf_target_derive()],
        )
        assert 'f"{CATALOG}.{SCHEMA}.landing_{_ds_name}"' in rendered
        assert ".saveAsTable(_table)" in rendered
        assert '.mode("overwrite")' in rendered

    def test_calls_function_with_spark_frames_namespace(self):
        rendered = DatabricksCodeRenderer(catalog="cat", schema="sch").render_target_derive(
            _bare_pipeline_config(), [_rf_target_derive()],
        )
        assert "_fn_1(spark, _frames_1, _NAMESPACE)" in rendered

    def test_handles_none_return_gracefully(self):
        rendered = DatabricksCodeRenderer(catalog="cat", schema="sch").render_target_derive(
            _bare_pipeline_config(), [_rf_target_derive()],
        )
        assert "if _result_1 is None:" in rendered
        assert "no outputs to write" in rendered

    def test_multiple_steps_render_in_registration_order(self):
        steps = [
            _rf_target_derive(name="a_first", datasets=("account",), primary="account"),
            _rf_target_derive(name="b_second", datasets=("contract",), primary="contract"),
        ]
        rendered = DatabricksCodeRenderer(catalog="cat", schema="sch").render_target_derive(
            _bare_pipeline_config(), steps,
        )
        assert rendered.index("from user_extensions import a_first") < rendered.index(
            "from user_extensions import b_second"
        )

    def test_rendered_target_derive_is_valid_python(self):
        rendered = DatabricksCodeRenderer(catalog="cat", schema="sch").render_target_derive(
            _bare_pipeline_config(), [_rf_target_derive()],
        )
        body_lines = [
            line for line in rendered.splitlines()
            if not (line.startswith("# Databricks notebook source")
                    or line.startswith("# MAGIC")
                    or line.startswith("# COMMAND"))
        ]
        ast.parse("\n".join(body_lines))


class TestDatabricksRunnerTargetDerivePhase:
    def test_runner_runs_target_derive_when_steps_present(self):
        renderer = DatabricksCodeRenderer(catalog="cat", schema="sch")
        rendered = renderer.render_runner(
            _bare_pipeline_config(),
            target_derive_steps=[_rf_target_derive()],
        )
        assert 'run_notebook("target_derive/run_target_derive")' in rendered
        # phase comes between Landing and Bronze sections.
        landing_idx = rendered.index("Landing Layer")
        td_idx = rendered.index("Target-Derive Phase")
        bronze_idx = rendered.index("Bronze Layer")
        assert landing_idx < td_idx < bronze_idx

    def test_runner_omits_target_derive_when_no_steps(self):
        renderer = DatabricksCodeRenderer(catalog="cat", schema="sch")
        rendered = renderer.render_runner(_bare_pipeline_config())
        assert "target_derive/run_target_derive" not in rendered
        assert "Target-Derive Phase" not in rendered


class TestLocalRunnerTargetDerivePhase:
    def test_local_runner_imports_target_derive_when_steps_present(self):
        rendered = CodeRenderer().render_runner(
            _bare_pipeline_config(),
            target_derive_steps=[_rf_target_derive()],
        )
        assert "from target_derive.run_target_derive import run_target_derive" in rendered
        assert "run_target_derive()" in rendered

    def test_local_runner_omits_target_derive_when_no_steps(self):
        rendered = CodeRenderer().render_runner(_bare_pipeline_config())
        assert "target_derive" not in rendered
        assert "run_target_derive" not in rendered


class TestDatabricksGeneratorWritesTargetDerive:
    @pytest.fixture
    def findings_dir(self, tmp_path):
        # Minimal findings directory (the generator just reads file presence
        # for landing/bronze; we don't need full contracts here).
        d = tmp_path / "findings"
        d.mkdir()
        return d

    def test_target_derive_file_written_when_step_present(
        self, findings_dir, tmp_path, monkeypatch
    ):
        out = tmp_path / "generated"
        # Stub out the parser's parse() to return a minimal config; we only
        # care that _write_target_derive picks up the harvest.
        from customer_retention.generators.pipeline_generator import findings_parser as fp

        def _stub_parse(self):
            return _bare_pipeline_config(name="sps_churn_validate")

        monkeypatch.setattr(fp.FindingsParser, "parse", _stub_parse, raising=True)
        hr = HarvestResult(cross_dataset_steps=[_rf_target_derive()])
        gen = DatabricksPipelineGenerator(
            findings_dir=str(findings_dir),
            output_dir=str(out),
            pipeline_name="p",
            catalog="cat",
            schema="sch",
            harvest_result=hr,
        )
        files = gen.generate()
        td = out / "target_derive" / "run_target_derive.py"
        assert td.exists()
        assert td in files
        text = td.read_text()
        assert "from user_extensions import derive_churn_target" in text

    def test_target_derive_file_skipped_when_no_steps(
        self, findings_dir, tmp_path, monkeypatch
    ):
        out = tmp_path / "generated"
        from customer_retention.generators.pipeline_generator import findings_parser as fp

        def _stub_parse(self):
            return _bare_pipeline_config(name="sps_churn_validate")

        monkeypatch.setattr(fp.FindingsParser, "parse", _stub_parse, raising=True)
        gen = DatabricksPipelineGenerator(
            findings_dir=str(findings_dir),
            output_dir=str(out),
            pipeline_name="p",
            catalog="cat",
            schema="sch",
            harvest_result=HarvestResult.empty(),
        )
        gen.generate()
        assert not (out / "target_derive" / "run_target_derive.py").exists()

    def test_kill_switch_suppresses_target_derive(
        self, findings_dir, tmp_path, monkeypatch
    ):
        out = tmp_path / "generated"
        from customer_retention.generators.pipeline_generator import findings_parser as fp

        def _stub_parse(self):
            return _bare_pipeline_config(name="sps_churn_validate")

        monkeypatch.setattr(fp.FindingsParser, "parse", _stub_parse, raising=True)
        hr = HarvestResult(cross_dataset_steps=[_rf_target_derive()])
        gen = DatabricksPipelineGenerator(
            findings_dir=str(findings_dir),
            output_dir=str(out),
            pipeline_name="p",
            catalog="cat",
            schema="sch",
            harvest_result=hr,
            disable_user_extensions=True,
        )
        gen.generate()
        assert not (out / "target_derive" / "run_target_derive.py").exists()
