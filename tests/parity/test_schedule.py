from __future__ import annotations

import json
from pathlib import Path

from customer_retention.parity.schedule import (
    GeneratedStage,
    JobSchedule,
    ScheduledNotebook,
    parse_inner_schedule,
    parse_outer_schedule,
)


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    return path


def _make_ipynb(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "cells": [],
            "metadata": {"kernelspec": {"name": "python3"}},
            "nbformat": 4,
            "nbformat_minor": 5,
        })
    )
    return path


class TestOuterSchedule:
    def test_numeric_prefix_inferred_in_order(self, tmp_path):
        for name in [
            "00_start_here.ipynb",
            "01_data_discovery.ipynb",
            "02_source_integrity.ipynb",
            "10_spec_generation.ipynb",
        ]:
            _make_ipynb(tmp_path / name)
        out = parse_outer_schedule(tmp_path)
        names = [n.path.name for n in out]
        assert names == [
            "00_start_here.ipynb",
            "01_data_discovery.ipynb",
            "02_source_integrity.ipynb",
            "10_spec_generation.ipynb",
        ]

    def test_negative_prefix_sorts_before_zero(self, tmp_path):
        for name in [
            "00_start_here.ipynb",
            "01_data_discovery.ipynb",
            "-1_parity_contract.ipynb",
        ]:
            _make_ipynb(tmp_path / name)
        out = parse_outer_schedule(tmp_path)
        names = [n.path.name for n in out]
        assert names[0] == "-1_parity_contract.ipynb"
        assert names[1] == "00_start_here.ipynb"
        assert names[2] == "01_data_discovery.ipynb"

    def test_alphabetic_suffix_after_numeric(self, tmp_path):
        for name in [
            "01_data_discovery.ipynb",
            "01a_temporal_deep_dive.ipynb",
            "01a_a_temporal_text.ipynb",
            "01b_temporal_quality.ipynb",
            "02_source_integrity.ipynb",
        ]:
            _make_ipynb(tmp_path / name)
        out = parse_outer_schedule(tmp_path)
        names = [n.path.name for n in out]
        assert names == [
            "01_data_discovery.ipynb",
            "01a_temporal_deep_dive.ipynb",
            "01a_a_temporal_text.ipynb",
            "01b_temporal_quality.ipynb",
            "02_source_integrity.ipynb",
        ]

    def test_non_numeric_files_skipped(self, tmp_path):
        for name in ["00_start.ipynb", "README.md", "experiments.log"]:
            _touch(tmp_path / name) if name != "00_start.ipynb" else _make_ipynb(tmp_path / name)
        out = parse_outer_schedule(tmp_path)
        assert len(out) == 1
        assert out[0].path.name == "00_start.ipynb"

    def test_subdirs_not_recursed(self, tmp_path):
        _make_ipynb(tmp_path / "00_start.ipynb")
        _make_ipynb(tmp_path / "subdir" / "99_buried.ipynb")
        out = parse_outer_schedule(tmp_path)
        assert {n.path.name for n in out} == {"00_start.ipynb"}

    def test_explicit_schedule_file_overrides_inference(self, tmp_path):
        for name in ["00_start.ipynb", "01_b.ipynb", "02_c.ipynb"]:
            _make_ipynb(tmp_path / name)
        schedule_file = tmp_path / "workflow.yml"
        schedule_file.write_text(
            "tasks:\n"
            "  - notebook: 02_c.ipynb\n"
            "  - notebook: 00_start.ipynb\n"
        )
        out = parse_outer_schedule(tmp_path, explicit_schedule_file=schedule_file)
        names = [n.path.name for n in out]
        assert names == ["02_c.ipynb", "00_start.ipynb"]

    def test_returns_scheduled_notebook_records(self, tmp_path):
        _make_ipynb(tmp_path / "00_start.ipynb")
        out = parse_outer_schedule(tmp_path)
        assert isinstance(out[0], ScheduledNotebook)
        assert out[0].prefix == "00"

    def test_empty_directory_returns_empty(self, tmp_path):
        out = parse_outer_schedule(tmp_path)
        assert out == ()


class TestInnerSchedule:
    def _write_runner(self, tmp_path: Path, body: str) -> Path:
        runner = tmp_path / "pipeline_runner.py"
        runner.write_text(body)
        return runner

    def test_imports_grouped_by_stage(self, tmp_path):
        runner = self._write_runner(
            tmp_path,
            "from landing.landing_request import run_landing_request\n"
            "from bronze.bronze_event_request import run_bronze_event_request\n"
            "from bronze.bronze_entity_account import run_bronze_entity_account\n"
            "from silver.silver_featureset_cn import run_silver_merge\n"
            "from gold.gold_features_cn import run_gold_features\n"
            "from training.ml_experiment import run_experiment\n"
        )
        stages = parse_inner_schedule(runner)
        stage_names = [s.name for s in stages]
        assert stage_names == ["landing", "bronze", "silver", "gold", "training"]

    def test_per_stage_notebooks_ordered(self, tmp_path):
        runner = self._write_runner(
            tmp_path,
            "from bronze.bronze_entity_account import run_bronze_entity_account\n"
            "from bronze.bronze_event_request import run_bronze_event_request\n"
            "from bronze.bronze_entity_contract import run_bronze_entity_contract\n"
        )
        stages = parse_inner_schedule(runner)
        bronze = next(s for s in stages if s.name == "bronze")
        names = [p.name for p in bronze.notebooks]
        assert "bronze_entity_account" in names
        assert "bronze_event_request" in names
        assert "bronze_entity_contract" in names

    def test_target_derive_stage_recognized(self, tmp_path):
        runner = self._write_runner(
            tmp_path,
            "from target_derive.run_target_derive import run_target_derive\n"
            "from landing.landing_x import run_landing_x\n"
        )
        stages = parse_inner_schedule(runner)
        names = [s.name for s in stages]
        assert "target_derive" in names

    def test_unrelated_imports_ignored(self, tmp_path):
        runner = self._write_runner(
            tmp_path,
            "import argparse\n"
            "from pathlib import Path\n"
            "from concurrent.futures import ThreadPoolExecutor\n"
            "from config import PIPELINE_NAME\n"
            "from landing.landing_x import run_landing_x\n"
        )
        stages = parse_inner_schedule(runner)
        names = [s.name for s in stages]
        assert names == ["landing"]

    def test_missing_runner_file_returns_empty(self, tmp_path):
        stages = parse_inner_schedule(tmp_path / "does_not_exist.py")
        assert stages == ()


class TestJobScheduleAggregate:
    def test_constructs_from_outer_and_inner(self, tmp_path):
        outer = (ScheduledNotebook(path=tmp_path / "00.ipynb", prefix="00", sort_key=(0, "")),)
        inner = (GeneratedStage(name="landing", notebooks=(tmp_path / "landing/landing_x.py",)),)
        s = JobSchedule(outer=outer, inner=inner)
        assert s.outer == outer
        assert s.inner == inner
