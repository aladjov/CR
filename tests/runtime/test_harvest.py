from __future__ import annotations

from pathlib import Path

import pytest

from customer_retention.runtime.harvest import (
    Harvester,
    PhaseMap,
)
from customer_retention.runtime.registry import RegisteredFunction, Registry


def _rf(
    name: str,
    *,
    scope: str = "dataset",
    dataset=None,
    datasets=None,
    primary=None,
    expected_stage=None,
    notebook_path=None,
    cell_id=None,
    replay_at_scoring=False,
) -> RegisteredFunction:
    return RegisteredFunction(
        name=name,
        source=f"def {name}(df): return df",
        scope=scope,  # type: ignore[arg-type]
        dataset=dataset,
        datasets=datasets,
        primary=primary,
        replay_at_scoring=replay_at_scoring,
        expected_stage=expected_stage,
        notebook_path=Path(notebook_path) if notebook_path else None,
        cell_id=cell_id,
    )


@pytest.fixture
def reg():
    return Registry()


class TestExplicitStageWins:
    def test_expected_stage_overrides_fallback_and_phase_map(self, reg):
        reg.register(_rf(
            "explicit",
            dataset="request",
            expected_stage="custom_stage",
            notebook_path="00_start_here.ipynb",
            cell_id="abc",
        ))
        phase_map = PhaseMap(sections={"abc": {"stage": "landing_post"}})
        result = Harvester(phase_map, reg).harvest()
        assert ("custom_stage", "request") in result.functions_by_target


class TestPhaseMapLookup:
    def test_cell_id_resolves_in_phase_map(self, reg):
        reg.register(_rf(
            "from_map",
            dataset="account",
            notebook_path="02_source_integrity.ipynb",
            cell_id="cell-xyz",
        ))
        phase_map = PhaseMap(sections={"cell-xyz": {"stage": "bronze_post"}})
        result = Harvester(phase_map, reg).harvest()
        assert ("bronze_post", "account") in result.functions_by_target

    def test_phase_map_entry_without_stage_falls_through(self, reg):
        reg.register(_rf(
            "no_stage_in_map",
            dataset="account",
            notebook_path="00_start_here.ipynb",
            cell_id="cell-without-stage",
        ))
        phase_map = PhaseMap(sections={"cell-without-stage": {"notebook": "00"}})
        result = Harvester(phase_map, reg).harvest()
        assert ("landing_post", "account") in result.functions_by_target


class TestNotebookFallback:
    @pytest.mark.parametrize("nb,stage", [
        ("00_start_here.ipynb", "landing_post"),
        ("01_data_discovery.ipynb", "bronze_post"),
        ("01d_event_aggregation.ipynb", "bronze_post"),
        ("03_dataset_merge.ipynb", "bronze_merge"),
        ("05_relationship_analysis.ipynb", "silver_post"),
        ("08_baseline_experiments.ipynb", "training"),
    ])
    def test_notebook_number_maps_to_expected_stage(self, reg, nb, stage):
        reg.register(_rf("f", dataset="account", notebook_path=nb))
        result = Harvester(PhaseMap.empty(), reg).harvest()
        assert (stage, "account") in result.functions_by_target


class TestUnmappable:
    def test_function_with_no_location_and_no_expected_stage_is_unmappable(self, reg):
        reg.register(_rf("orphan", dataset="account"))
        result = Harvester(PhaseMap.empty(), reg).harvest()
        assert len(result.unmappable) == 1
        assert result.unmappable[0][0].name == "orphan"
        assert "no notebook path" in result.unmappable[0][1]

    def test_unknown_notebook_without_map_is_unmappable(self, reg):
        reg.register(_rf("f", dataset="account", notebook_path="99_future.ipynb"))
        result = Harvester(PhaseMap.empty(), reg).harvest()
        assert len(result.unmappable) == 1


class TestScopeBucketing:
    def test_datasets_scope_goes_to_cross_dataset_steps(self, reg):
        reg.register(_rf(
            "derive_target",
            scope="datasets",
            datasets=["account", "contract", "case"],
            primary="account",
            expected_stage="target_derive",
        ))
        result = Harvester(PhaseMap.empty(), reg).harvest()
        assert result.functions_by_target == {}
        assert len(result.cross_dataset_steps) == 1
        assert result.cross_dataset_steps[0].name == "derive_target"
        assert result.cross_dataset_steps[0].inferred_stage == "target_derive"

    def test_wildcard_scope_goes_to_cross_dataset_steps(self, reg):
        reg.register(_rf(
            "augment_all",
            scope="wildcard",
            expected_stage="bronze_post",
        ))
        result = Harvester(PhaseMap.empty(), reg).harvest()
        assert result.functions_by_target == {}
        assert len(result.cross_dataset_steps) == 1

    def test_dataset_scope_goes_to_functions_by_target(self, reg):
        reg.register(_rf(
            "filter_req",
            dataset="request",
            expected_stage="landing_post",
        ))
        result = Harvester(PhaseMap.empty(), reg).harvest()
        assert result.cross_dataset_steps == []
        assert ("landing_post", "request") in result.functions_by_target


class TestDeclarationOrderPreserved:
    def test_multiple_at_same_target_preserve_insertion_order(self, reg):
        reg.register(_rf("first", dataset="request", expected_stage="landing_post"))
        reg.register(_rf("second", dataset="request", expected_stage="landing_post"))
        reg.register(_rf("third", dataset="request", expected_stage="landing_post"))
        result = Harvester(PhaseMap.empty(), reg).harvest()
        names = [rf.name for rf in result.functions_by_target[("landing_post", "request")]]
        assert names == ["first", "second", "third"]


class TestHarvestResultHelpers:
    def test_is_empty_on_fresh_registry(self, reg):
        result = Harvester(PhaseMap.empty(), reg).harvest()
        assert result.is_empty() is True

    def test_all_functions_flattens_both_buckets(self, reg):
        reg.register(_rf("ds", dataset="request", expected_stage="landing_post"))
        reg.register(_rf(
            "cross",
            scope="datasets",
            datasets=["a", "b"],
            primary="a",
            expected_stage="target_derive",
        ))
        result = Harvester(PhaseMap.empty(), reg).harvest()
        names = {rf.name for rf in result.all_functions()}
        assert names == {"ds", "cross"}


class TestValidationPipeline:
    def test_validation_stub_produces_empty_errors_today(self, reg):
        reg.register(_rf("f", dataset="request", expected_stage="landing_post"))
        result = Harvester(PhaseMap.empty(), reg).harvest()
        assert result.validation_errors == []
