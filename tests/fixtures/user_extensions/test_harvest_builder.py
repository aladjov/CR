"""Smoke tests for the fixture builders themselves.

Fixtures that build wrong shapes silently break every downstream test —
catch shape drift here instead of in every consumer.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from customer_retention.analysis.auto_explorer.layered_recommendations import (
    RecommendationRegistry,
)
from tests.fixtures.user_extensions.harvest_builder import (
    make_harvest,
    make_notebook,
    make_phase_map,
    make_registered_function,
)

_FIXTURE_DIR = Path(__file__).parent


class TestMakeRegisteredFunction:
    def test_dataset_scope_default_source(self):
        rf = make_registered_function("enrich_req", dataset="request")
        assert rf.scope == "dataset"
        assert rf.dataset == "request"
        assert "def enrich_req" in rf.source

    def test_wildcard_scope(self):
        rf = make_registered_function("aug_all", scope="wildcard", dataset=None)
        assert rf.scope == "wildcard"
        assert rf.dataset is None

    def test_notebook_path_coerced_to_pathlib(self):
        rf = make_registered_function(
            "x", dataset="a", notebook_path="/tmp/nb.ipynb"
        )
        assert rf.notebook_path == Path("/tmp/nb.ipynb")


class TestMakeHarvest:
    def test_empty_spec_yields_empty_result(self):
        hr = make_harvest()
        assert hr.is_empty()

    def test_dataset_scoped_goes_into_functions_by_target(self):
        hr = make_harvest(dataset_scoped=[
            {"name": "f1", "dataset": "request", "inferred_stage": "landing_post"},
        ])
        assert ("landing_post", "request") in hr.functions_by_target
        assert hr.functions_by_target[("landing_post", "request")][0].name == "f1"

    def test_cross_dataset_goes_into_cross_dataset_steps(self):
        hr = make_harvest(cross_dataset=[
            {"name": "tgt", "scope": "datasets",
             "datasets": ["a", "b"], "primary": "a", "dataset": None},
        ])
        assert len(hr.cross_dataset_steps) == 1
        assert hr.cross_dataset_steps[0].datasets == ["a", "b"]

    def test_multiple_per_target_preserves_order(self):
        hr = make_harvest(dataset_scoped=[
            {"name": "a", "dataset": "x", "inferred_stage": "landing_post"},
            {"name": "b", "dataset": "x", "inferred_stage": "landing_post"},
            {"name": "c", "dataset": "x", "inferred_stage": "landing_post"},
        ])
        names = [rf.name for rf in hr.functions_by_target[("landing_post", "x")]]
        assert names == ["a", "b", "c"]


class TestMakePhaseMap:
    def test_entries_populate_sections(self):
        pm = make_phase_map({"cell_a": "landing_post", "cell_b": "silver_post"})
        assert pm.sections["cell_a"]["stage"] == "landing_post"
        assert pm.sections["cell_b"]["stage"] == "silver_post"


class TestMakeNotebook:
    def test_markdown_and_code_cells_round_trip(self):
        nb = make_notebook([
            {"type": "markdown", "source": "## Header"},
            {"type": "code", "source": "# @cr:code name='x' id=abc\nprint('hi')",
             "id": "abc12345"},
        ])
        assert len(nb.cells) == 2
        assert nb.cells[0].cell_type == "markdown"
        assert nb.cells[1].cell_type == "code"
        assert nb.cells[1].id == "abc12345"


class TestYamlFixturesLoad:
    def test_recommendations_with_landing_yaml_round_trips(self):
        with (_FIXTURE_DIR / "recommendations_with_landing.yaml").open() as f:
            data = yaml.safe_load(f)
        registry = RecommendationRegistry.from_dict(data)
        assert registry.landing is not None
        assert len(registry.landing.filters) == 1
        assert registry.landing.filters[0].parameters["dataset"] == "request"
        assert len(registry.landing.lifecycle_enrichments) == 1
        assert registry.landing.lifecycle_enrichments[0].parameters["dataset"] == "contract"

    def test_recommendations_fixture_has_sql_form_predicate(self):
        with (_FIXTURE_DIR / "recommendations_with_landing.yaml").open() as f:
            data = yaml.safe_load(f)
        predicate = data["landing"]["filters"][0]["parameters"]["predicate"]
        assert "IS NOT NULL" in predicate

    def test_phase_map_sps_yaml_has_expected_sps_entries(self):
        with (_FIXTURE_DIR / "phase_map_sps.yaml").open() as f:
            data = yaml.safe_load(f)
        sections = data["sections"]
        assert "00_start_here#0_4_6_request_quality_filter" in sections
        assert sections["00_start_here#0_4_6_request_quality_filter"]["stage"] == "landing_post"
        assert sections["00_start_here#0_2_6_target_derivation"]["stage"] == "target_derive"
