from __future__ import annotations

import ast
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from customer_retention.generators.pipeline_generator.generation_manifest import MANIFEST_FILENAME
from customer_retention.generators.pipeline_generator.generator import PipelineGenerator
from customer_retention.generators.pipeline_generator.user_extensions_emitter import (
    USER_EXTENSIONS_FILENAME,
    write_user_extensions,
)
from customer_retention.runtime.harvest import HarvestResult
from customer_retention.runtime.registry import RegisteredFunction


def _rf(name, **over):
    kwargs = dict(
        name=name,
        source=over.pop("source", f"def {name}(df):\n    return df\n"),
        scope=over.pop("scope", "dataset"),
        dataset=over.pop("dataset", "request"),
        datasets=over.pop("datasets", None),
        primary=over.pop("primary", None),
        replay_at_scoring=over.pop("replay_at_scoring", False),
        expected_stage=over.pop("expected_stage", None),
        notebook_path=over.pop("notebook_path", None),
        cell_id=over.pop("cell_id", None),
    )
    kwargs.update(over)
    return RegisteredFunction(**kwargs)


class TestWriteUserExtensionsStandalone:
    def test_empty_harvest_writes_no_file(self, tmp_path):
        out = write_user_extensions(tmp_path, HarvestResult.empty())
        assert out is None
        assert not (tmp_path / USER_EXTENSIONS_FILENAME).exists()

    def test_none_harvest_writes_no_file(self, tmp_path):
        assert write_user_extensions(tmp_path, None) is None

    def test_populated_harvest_emits_source_and_banner(self, tmp_path):
        rf = _rf("filter_bad", dataset="request", inferred_stage="landing_post")
        hr = HarvestResult(functions_by_target={("landing_post", "request"): [rf]})
        fixed = datetime(2026, 4, 17, 13, 45, 0, tzinfo=timezone.utc)
        path = write_user_extensions(tmp_path, hr, now=fixed)
        assert path == tmp_path / USER_EXTENSIONS_FILENAME
        text = path.read_text()
        assert "# === generated — do not edit ===" in text
        assert "# Generated: 2026-04-17T13:45:00Z" in text
        assert "def filter_bad(df):" in text
        assert "__all__ = [" in text
        assert "'filter_bad'" in text

    def test_emitted_file_is_python_parseable(self, tmp_path):
        rf = _rf("f", source="def f(df):\n    return df\n", inferred_stage="landing_post")
        hr = HarvestResult(functions_by_target={("landing_post", "request"): [rf]})
        path = write_user_extensions(tmp_path, hr)
        ast.parse(path.read_text())

    def test_multiple_functions_ordered_by_stage_rank(self, tmp_path):
        rf_silver = _rf("second_silver", dataset="account", inferred_stage="silver_post")
        rf_landing = _rf("first_landing", dataset="request", inferred_stage="landing_post")
        rf_target = _rf("third_target", scope="datasets", dataset=None, datasets=["a"], primary="a",
                        inferred_stage="target_derive")
        hr = HarvestResult(
            functions_by_target={
                ("silver_post", "account"): [rf_silver],
                ("landing_post", "request"): [rf_landing],
            },
            cross_dataset_steps=[rf_target],
        )
        path = write_user_extensions(tmp_path, hr)
        text = path.read_text()
        landing_idx = text.index("def first_landing")
        silver_idx = text.index("def second_silver")
        target_idx = text.index("def third_target")
        assert landing_idx < silver_idx < target_idx

    def test_source_comment_lists_notebook_and_cell_ids(self, tmp_path):
        rf = _rf(
            "tracked",
            inferred_stage="landing_post",
            notebook_path=Path("exploration_notebooks/00_start_here.ipynb"),
            cell_id="cell-abc",
        )
        hr = HarvestResult(functions_by_target={("landing_post", "request"): [rf]})
        text = write_user_extensions(tmp_path, hr).read_text()
        assert "tracked  (exploration_notebooks/00_start_here.ipynb:cell-abc)" in text


class TestGeneratorIntegration:
    """End-to-end: PipelineGenerator wires harvest_result through
    __init__ and emits user_extensions.py only when non-empty."""

    @pytest.fixture
    def findings_dir(self, tmp_path):
        sps = Path(__file__).parent.parent.parent / "fixtures" / "user_extensions" / "sps_mini"
        dst = tmp_path / "findings"
        import shutil
        shutil.copytree(sps, dst)
        return dst

    def test_no_harvest_emits_no_user_extensions_file(self, findings_dir, tmp_path):
        out = tmp_path / "generated"
        PipelineGenerator(
            findings_dir=str(findings_dir),
            output_dir=str(out),
            pipeline_name="p",
        ).generate()
        assert not (out / USER_EXTENSIONS_FILENAME).exists()

    def test_empty_harvest_emits_no_user_extensions_file(self, findings_dir, tmp_path):
        out = tmp_path / "generated"
        PipelineGenerator(
            findings_dir=str(findings_dir),
            output_dir=str(out),
            pipeline_name="p",
            harvest_result=HarvestResult.empty(),
        ).generate()
        assert not (out / USER_EXTENSIONS_FILENAME).exists()

    def test_populated_harvest_emits_file(self, findings_dir, tmp_path):
        out = tmp_path / "generated"
        rf = _rf("derive_churn", source="def derive_churn(df):\n    return df\n",
                 inferred_stage="target_derive")
        hr = HarvestResult(cross_dataset_steps=[rf])
        PipelineGenerator(
            findings_dir=str(findings_dir),
            output_dir=str(out),
            pipeline_name="p",
            harvest_result=hr,
        ).generate()
        ue = out / USER_EXTENSIONS_FILENAME
        assert ue.exists()
        assert "def derive_churn" in ue.read_text()

    def test_kill_switch_suppresses_emission_even_with_populated_harvest(
        self, findings_dir, tmp_path
    ):
        out = tmp_path / "generated"
        rf = _rf("derive_churn", source="def derive_churn(df):\n    return df\n",
                 inferred_stage="target_derive")
        hr = HarvestResult(cross_dataset_steps=[rf])
        PipelineGenerator(
            findings_dir=str(findings_dir),
            output_dir=str(out),
            pipeline_name="p",
            harvest_result=hr,
            disable_user_extensions=True,
        ).generate()
        assert not (out / USER_EXTENSIONS_FILENAME).exists()

    def test_manifest_lists_harvested_function_names(self, findings_dir, tmp_path):
        out = tmp_path / "generated"
        rf = _rf("derive_churn", source="def derive_churn(df):\n    return df\n",
                 inferred_stage="target_derive")
        hr = HarvestResult(cross_dataset_steps=[rf])
        PipelineGenerator(
            findings_dir=str(findings_dir),
            output_dir=str(out),
            pipeline_name="p",
            harvest_result=hr,
        ).generate()
        data = json.loads((out / MANIFEST_FILENAME).read_text())
        assert data["harvested_functions"] == ["derive_churn"]

    def test_manifest_harvested_functions_empty_when_kill_switch_on(
        self, findings_dir, tmp_path
    ):
        out = tmp_path / "generated"
        rf = _rf("derive_churn", inferred_stage="target_derive",
                 source="def derive_churn(df):\n    return df\n")
        hr = HarvestResult(cross_dataset_steps=[rf])
        PipelineGenerator(
            findings_dir=str(findings_dir),
            output_dir=str(out),
            pipeline_name="p",
            harvest_result=hr,
            disable_user_extensions=True,
        ).generate()
        data = json.loads((out / MANIFEST_FILENAME).read_text())
        assert data["harvested_functions"] == []


class TestByteParityOnEmptyHarvest:
    """Plan § 6.6: empty harvest must leave generated pipelines byte-
    identical to the pre-Phase-5 baseline."""

    @pytest.fixture
    def findings_dir(self, tmp_path):
        sps = Path(__file__).parent.parent.parent / "fixtures" / "user_extensions" / "sps_mini"
        dst = tmp_path / "findings"
        import shutil
        shutil.copytree(sps, dst)
        return dst

    def test_generated_pipeline_identical_with_and_without_empty_harvest_kwarg(
        self, findings_dir, tmp_path
    ):
        baseline = tmp_path / "baseline"
        with_empty = tmp_path / "with_empty"

        PipelineGenerator(
            findings_dir=str(findings_dir),
            output_dir=str(baseline),
            pipeline_name="p",
        ).generate()
        PipelineGenerator(
            findings_dir=str(findings_dir),
            output_dir=str(with_empty),
            pipeline_name="p",
            harvest_result=HarvestResult.empty(),
        ).generate()

        # Compare the generated landing script (the one touched by Phase 1 extensions).
        a = (baseline / "landing" / "landing_request.py").read_text()
        b = (with_empty / "landing" / "landing_request.py").read_text()
        assert a == b
