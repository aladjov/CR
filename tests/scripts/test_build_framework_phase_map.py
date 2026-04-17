from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import nbformat
import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "build_framework_phase_map.py"


@pytest.fixture(scope="module")
def phase_map_module():
    spec = importlib.util.spec_from_file_location("build_framework_phase_map", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_framework_phase_map"] = module
    spec.loader.exec_module(module)
    return module


def _nb_with_cells(cells):
    nb = nbformat.v4.new_notebook()
    for c in cells:
        nb.cells.append(c)
    return nb


def _md(src):
    return nbformat.v4.new_markdown_cell(src)


def _code(src):
    return nbformat.v4.new_code_cell(src)


def _write_nb(path: Path, cells) -> Path:
    nbformat.write(_nb_with_cells(cells), path)
    return path


class TestSectionHeaderKeywordMatching:
    def test_landing_keyword_in_section_maps_to_landing_post(self, phase_map_module, tmp_path):
        nb = _write_nb(tmp_path / "00_start_here.ipynb", [
            _md("[//]: # (cr:doc name='0_4_6_request_landing_filter' id=abc123)"),
            _code("# @cr:code name='filter_request' id=def456\nprint('x')"),
        ])
        pm = phase_map_module.build_phase_map([nb])
        entry = pm["sections"]["00_start_here#0_4_6_request_landing_filter"]
        assert entry["stage"] == "landing_post"
        assert entry["source"] == "keyword_match"

    def test_merge_keyword_maps_to_bronze_merge(self, phase_map_module, tmp_path):
        nb = _write_nb(tmp_path / "03_dataset_merge.ipynb", [
            _md("[//]: # (cr:doc name='3_1_dataset_merge_pipeline' id=abc)"),
            _code("pass"),
        ])
        pm = phase_map_module.build_phase_map([nb])
        entry = pm["sections"]["03_dataset_merge#3_1_dataset_merge_pipeline"]
        assert entry["stage"] == "bronze_merge"
        assert entry["source"] == "keyword_match"

    def test_target_keyword_maps_to_target_derive(self, phase_map_module, tmp_path):
        nb = _write_nb(tmp_path / "00_start_here.ipynb", [
            _md("[//]: # (cr:doc name='0_2_6_derive_churn_target' id=abc)"),
            _code("pass"),
        ])
        pm = phase_map_module.build_phase_map([nb])
        entry = pm["sections"]["00_start_here#0_2_6_derive_churn_target"]
        assert entry["stage"] == "target_derive"

    def test_feature_keyword_maps_to_silver_post(self, phase_map_module, tmp_path):
        nb = _write_nb(tmp_path / "06_feature_opportunities.ipynb", [
            _md("[//]: # (cr:doc name='6_5_derived_features' id=abc)"),
            _code("pass"),
        ])
        pm = phase_map_module.build_phase_map([nb])
        entry = pm["sections"]["06_feature_opportunities#6_5_derived_features"]
        assert entry["stage"] == "silver_post"


class TestNotebookFallback:
    def test_no_keyword_match_falls_back_to_notebook_number(self, phase_map_module, tmp_path):
        nb = _write_nb(tmp_path / "05_relationship_analysis.ipynb", [
            _md("[//]: # (cr:doc name='5_3_opaque_analysis' id=abc)"),
            _code("pass"),
        ])
        pm = phase_map_module.build_phase_map([nb])
        # "opaque" and "analysis" aren't in KNOWN_STAGE_KEYWORDS — but
        # "relationship" keyword isn't in the section name either → fallback
        entry = pm["sections"]["05_relationship_analysis#5_3_opaque_analysis"]
        assert entry["stage"] == "silver_post"
        assert entry["source"] == "notebook_fallback"


class TestAnnotationOverride:
    def test_annotation_trumps_keyword_match(self, phase_map_module, tmp_path):
        nb = _write_nb(tmp_path / "00_start_here.ipynb", [
            _md("[//]: # (cr:doc name='0_1_training_setup' id=abc)"),
            _code("# @cr:code name='x' id=y\n# @cr:code phase=bronze_post\npass"),
        ])
        pm = phase_map_module.build_phase_map([nb])
        entry = pm["sections"]["00_start_here#0_1_training_setup"]
        assert entry["stage"] == "bronze_post"
        assert entry["source"] == "annotation"


class TestUnmappable:
    def test_code_cell_without_any_section_is_unmappable(self, phase_map_module, tmp_path):
        nb = _write_nb(tmp_path / "02_source_integrity.ipynb", [
            _code("# @cr:code name='orphan' id=orphan-id\npass"),
        ])
        pm = phase_map_module.build_phase_map([nb])
        assert pm["sections"] == {}
        assert len(pm["unmappable_cells"]) == 1
        u = pm["unmappable_cells"][0]
        assert u["notebook"] == "02_source_integrity"
        assert u["cell_id"] == "orphan-id"
        assert "no preceding section header" in u["reason"]


class TestDeterminism:
    def test_sections_sorted_alphabetically_for_byte_equality(self, phase_map_module, tmp_path):
        nb = _write_nb(tmp_path / "00_start_here.ipynb", [
            _md("[//]: # (cr:doc name='z_last_section' id=1)"),
            _code("pass"),
            _md("[//]: # (cr:doc name='a_first_section' id=2)"),
            _code("pass"),
        ])
        pm = phase_map_module.build_phase_map([nb])
        keys = list(pm["sections"].keys())
        assert keys == sorted(keys)

    def test_unmappable_sorted_for_stability(self, phase_map_module, tmp_path):
        nb = _write_nb(tmp_path / "02_source_integrity.ipynb", [
            _code("# @cr:code name='a' id=z-later\npass"),
            _code("# @cr:code name='b' id=a-earlier\npass"),
        ])
        pm = phase_map_module.build_phase_map([nb])
        ids = [u["cell_id"] for u in pm["unmappable_cells"]]
        assert ids == sorted(ids)

    def test_source_fingerprint_stable_across_calls(self, phase_map_module, tmp_path):
        nb_path = _write_nb(tmp_path / "00_start_here.ipynb", [
            _md("[//]: # (cr:doc name='0_1_x' id=a)"),
            _code("pass"),
        ])
        fp1 = phase_map_module.build_phase_map([nb_path])["source_fingerprint"]
        fp2 = phase_map_module.build_phase_map([nb_path])["source_fingerprint"]
        assert fp1 == fp2
        assert fp1.startswith("sha256:")

    def test_fingerprint_changes_when_notebook_content_changes(self, phase_map_module, tmp_path):
        nb_path = tmp_path / "00_start_here.ipynb"
        _write_nb(nb_path, [
            _md("[//]: # (cr:doc name='0_1_x' id=a)"),
            _code("pass"),
        ])
        fp1 = phase_map_module.build_phase_map([nb_path])["source_fingerprint"]
        _write_nb(nb_path, [
            _md("[//]: # (cr:doc name='0_1_x' id=a)"),
            _code("pass  # changed"),
        ])
        fp2 = phase_map_module.build_phase_map([nb_path])["source_fingerprint"]
        assert fp1 != fp2


class TestCurationList:
    def test_find_notebooks_skips_unknown_names(self, phase_map_module, tmp_path):
        known = tmp_path / "00_start_here.ipynb"
        unknown = tmp_path / "-1_sample_datasets.ipynb"
        _write_nb(known, [])
        _write_nb(unknown, [])
        found = phase_map_module.find_notebooks(tmp_path)
        stems = {p.stem for p in found}
        assert "00_start_here" in stems
        assert "-1_sample_datasets" not in stems


class TestRealExplorationNotebooks:
    def test_committed_framework_phase_map_is_up_to_date(self, phase_map_module):
        repo = Path(__file__).resolve().parents[2]
        committed = repo / "framework" / "phase_map.yaml"
        if not committed.exists():
            pytest.skip("framework/phase_map.yaml not checked in yet")
        fresh = phase_map_module.build_phase_map(
            phase_map_module.find_notebooks(repo / "exploration_notebooks")
        )
        assert committed.read_text() == phase_map_module.serialize(fresh), (
            "framework/phase_map.yaml is stale; run scripts/build_framework_phase_map.py"
        )

    def test_real_notebooks_produce_zero_unmappable_cells(self, phase_map_module):
        repo = Path(__file__).resolve().parents[2]
        pm = phase_map_module.build_phase_map(
            phase_map_module.find_notebooks(repo / "exploration_notebooks")
        )
        assert pm["unmappable_cells"] == [], (
            "Some user-anchorable cells have no section header. "
            "Add a cr:doc marker above them or tolerate the gap in PR review."
        )
