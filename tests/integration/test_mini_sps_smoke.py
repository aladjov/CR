"""Mini-SPS smoke test — Release A rail (plan § 0a.4.3).

Exercises PipelineGenerator end-to-end against a frozen 3-dataset
SPS-shaped fixture. Catches regressions in registry → parser → renderer
plumbing on every PR. Must stay under the 10-minute cap per plan —
today's tier-1 findings-only fixture runs in well under a second.

When NB08 has finished and real SPS data is available, the fixture
under `tests/fixtures/user_extensions/sps_mini/` can be upgraded to
tier 2 (5K-row trimmed CSV/parquet per dataset) without any change to
this test file.
"""
from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

import pytest

from customer_retention.generators.pipeline_generator.generation_manifest import (
    BASELINE_TAG,
    MANIFEST_FILENAME,
    RELEASE,
)
from customer_retention.generators.pipeline_generator.generator import PipelineGenerator

_FIXTURE_ROOT = Path(__file__).parent.parent / "fixtures" / "user_extensions" / "sps_mini"


@pytest.fixture
def sps_mini_findings_dir(tmp_path):
    """Copy the committed sps_mini fixture into a tmp findings dir so the
    generator can write alongside without touching the repo."""
    findings_dir = tmp_path / "findings"
    shutil.copytree(_FIXTURE_ROOT, findings_dir)
    return findings_dir


def _run_generator(findings_dir: Path, output_dir: Path, **kwargs) -> PipelineGenerator:
    gen = PipelineGenerator(
        findings_dir=str(findings_dir),
        output_dir=str(output_dir),
        pipeline_name="sps_mini",
        **kwargs,
    )
    gen.generate()
    return gen


class TestMiniSpsSmoke:
    def test_generator_runs_end_to_end_without_errors(self, sps_mini_findings_dir, tmp_path):
        out = tmp_path / "generated"
        _run_generator(sps_mini_findings_dir, out)
        assert (out / "config.py").exists()
        assert (out / "landing" / "landing_request.py").exists()
        assert (out / MANIFEST_FILENAME).exists()

    def test_all_generated_python_is_ast_parseable(self, sps_mini_findings_dir, tmp_path):
        out = tmp_path / "generated"
        _run_generator(sps_mini_findings_dir, out)
        py_files = list(out.rglob("*.py"))
        assert py_files, "generator produced no .py files"
        for p in py_files:
            ast.parse(p.read_text(), filename=str(p))

    def test_generation_manifest_has_expected_shape(self, sps_mini_findings_dir, tmp_path):
        out = tmp_path / "generated"
        _run_generator(sps_mini_findings_dir, out)
        data = json.loads((out / MANIFEST_FILENAME).read_text())
        assert data["baseline_tag"] == BASELINE_TAG
        assert data["release"] == RELEASE
        assert data["kill_switch_active"] is False
        assert len(data["file_checksums"]) > 0
        assert data["template_versions"]
        assert all(v.startswith("sha256:") for v in data["template_versions"].values())

    def test_landing_filter_flows_through_to_manifest(self, sps_mini_findings_dir, tmp_path):
        out = tmp_path / "generated"
        _run_generator(sps_mini_findings_dir, out)
        data = json.loads((out / MANIFEST_FILENAME).read_text())
        assert data["landing_filters"] == [
            {"dataset": "request", "predicate": "amount > 0"}
        ]
        assert data["lifecycle_enrichments"] == []

    def test_landing_filter_line_emitted_in_generated_landing_script(
        self, sps_mini_findings_dir, tmp_path
    ):
        out = tmp_path / "generated"
        _run_generator(sps_mini_findings_dir, out)
        landing_script = (out / "landing" / "landing_request.py").read_text()
        assert "df = df.query('amount > 0')" in landing_script

    def test_kill_switch_drops_landing_filter_from_manifest_and_script(
        self, sps_mini_findings_dir, tmp_path
    ):
        out = tmp_path / "generated"
        _run_generator(sps_mini_findings_dir, out, disable_user_extensions=True)

        data = json.loads((out / MANIFEST_FILENAME).read_text())
        assert data["kill_switch_active"] is True
        assert data["landing_filters"] == []

        landing_script = (out / "landing" / "landing_request.py").read_text()
        assert "df.query" not in landing_script

    def test_kill_switch_preserves_byte_parity_on_landing_script(
        self, sps_mini_findings_dir, tmp_path
    ):
        """Generated landing script with flag-on + populated registry must
        equal the script produced when no landing block is present at all."""
        flag_on_out = tmp_path / "flag_on"
        _run_generator(sps_mini_findings_dir, flag_on_out, disable_user_extensions=True)

        recs_path = sps_mini_findings_dir / "recommendations.yaml"
        import yaml
        raw = yaml.safe_load(recs_path.read_text())
        raw.pop("landing", None)
        recs_path.write_text(yaml.dump(raw))

        no_landing_out = tmp_path / "no_landing"
        _run_generator(sps_mini_findings_dir, no_landing_out)

        flag_on_script = (flag_on_out / "landing" / "landing_request.py").read_text()
        no_landing_script = (no_landing_out / "landing" / "landing_request.py").read_text()
        assert flag_on_script == no_landing_script


class TestMiniSpsSmokeBudget:
    """The rail is 'every PR runs this'. Guard the ceiling at something an
    order of magnitude tighter than plan § 0a.4.3's 10-minute cap so a
    drift is visible long before the cap is reached."""

    @pytest.mark.timeout(60)
    def test_smoke_completes_within_one_minute(self, sps_mini_findings_dir, tmp_path):
        out = tmp_path / "generated"
        _run_generator(sps_mini_findings_dir, out)
        assert (out / MANIFEST_FILENAME).exists()
