"""Cycle 3 (aggregation config wiring) — intent-window invariant tests.

Closes gaps G4 (bronze used only 2 of 7 intent-declared aggregation windows)
and G5 (bronze_aggregation_overrides per_grid_date_mode did not reach the
generated bronze). Pre-fix the tests skip via ImportError on the new
``resolve_aggregation_windows`` helper; post-fix they activate.

Invariant under test:
    event_cfg.aggregation.windows is resolved by precedence
        1. bronze_aggregation_overrides[dataset]["windows"]     (explicit)
        2. multi.aggregation_windows                            (intent)
        3. findings.time_series_metadata.aggregation_windows_used  (material)

Never the reverse. Intent is not silently narrowed by a coverage-based
recommendation in NB01a/NB01d.
"""
from __future__ import annotations

import ast

import pytest
import yaml

try:
    from customer_retention.generators.pipeline_generator.findings_parser import (
        resolve_aggregation_windows,
    )

    _FIX_LANDED = True
except ImportError:
    _FIX_LANDED = False

pytestmark = pytest.mark.skipif(
    not _FIX_LANDED,
    reason="cycle 3 fix (resolve_aggregation_windows) not landed yet",
)


_INTENT_WINDOWS = ["24h", "7d", "30d", "90d", "180d", "365d", "all_time"]
_COVERAGE_NARROWED = ["365d", "all_time"]


def oracle_expected_windows(intent, material, override):
    """Pure-python reference for the post-fix precedence rule.

    Cycle-4 widening rule: when ``override`` is a (non-strict) subset of
    ``intent``, treat the override as a redundant restatement and keep
    intent breadth. Genuine narrowing requires the override to contain
    at least one window NOT in intent.
    """
    if override:
        if intent and set(override).issubset(set(intent)):
            return list(intent)
        return list(override)
    if intent:
        return list(intent)
    if material:
        return list(material)
    return []


# ---------------------------------------------------------------------------
# Unit coverage for the pure helper
# ---------------------------------------------------------------------------


class TestResolveAggregationWindowsHelper:
    @pytest.mark.parametrize(
        "intent,material,override,expected",
        [
            (_INTENT_WINDOWS, _COVERAGE_NARROWED, None, _INTENT_WINDOWS),
            (_INTENT_WINDOWS, _INTENT_WINDOWS, None, _INTENT_WINDOWS),
            ([], _COVERAGE_NARROWED, None, _COVERAGE_NARROWED),
            ([], [], None, []),
            # Override "30d" IS a subset of _INTENT_WINDOWS — widen to intent.
            (_INTENT_WINDOWS, _COVERAGE_NARROWED, ["30d"], _INTENT_WINDOWS),
            # Genuinely narrowing override (contains a window not in intent) wins.
            (_INTENT_WINDOWS, _COVERAGE_NARROWED, ["1h"], ["1h"]),
            ([], _COVERAGE_NARROWED, ["7d", "30d"], ["7d", "30d"]),
            (_INTENT_WINDOWS, [], None, _INTENT_WINDOWS),
        ],
    )
    def test_precedence(self, intent, material, override, expected):
        assert (
            resolve_aggregation_windows(intent=intent, material=material, override=override)
            == expected
        )
        assert oracle_expected_windows(intent, material, override) == expected

    def test_override_coerces_to_list(self):
        assert resolve_aggregation_windows(
            intent=[], material=[], override=("7d", "30d")
        ) == ["7d", "30d"]

    def test_does_not_mutate_inputs(self):
        intent = list(_INTENT_WINDOWS)
        material = list(_COVERAGE_NARROWED)
        out = resolve_aggregation_windows(intent=intent, material=material, override=None)
        out.append("SENTINEL")
        assert intent == _INTENT_WINDOWS
        assert material == _COVERAGE_NARROWED


# ---------------------------------------------------------------------------
# Integration coverage via FindingsParser.parse()
# ---------------------------------------------------------------------------


def _write_findings_dir(tmp_path, *, material_windows, intent_windows):
    """Minimal findings tree with one event dataset and intent vs material
    aggregation-window divergence. Mirrors the engagement_e4ad6e1b run shape.
    """
    findings_dir = tmp_path / "findings"
    findings_dir.mkdir()

    multi = {
        "datasets": {
            "customers": {
                "name": "customers",
                "findings_path": str(findings_dir / "customers_findings.yaml"),
                "source_path": "/data/customers.csv",
                "granularity": "entity_level",
                "row_count": 100,
                "column_count": 3,
                "excluded": False,
            },
            "orders": {
                "name": "orders",
                "findings_path": str(findings_dir / "orders_findings.yaml"),
                "source_path": "/data/orders.parquet",
                "granularity": "event_level",
                "row_count": 1000,
                "column_count": 4,
                "excluded": False,
                "entity_column": "customer_id",
                "time_column": "order_date",
            },
        },
        "relationships": [
            {
                "left_dataset": "customers",
                "right_dataset": "orders",
                "left_column": "customer_id",
                "right_column": "customer_id",
                "relationship_type": "one_to_many",
                "confidence": 1.0,
            }
        ],
        "primary_entity_dataset": "customers",
        "event_datasets": ["orders"],
        "excluded_datasets": [],
        "aggregation_windows": intent_windows,
    }
    (findings_dir / "multi_dataset_findings.yaml").write_text(yaml.dump(multi))

    customers = {
        "source_path": "/data/customers.csv",
        "source_format": "csv",
        "row_count": 100,
        "column_count": 3,
        "columns": {
            "customer_id": {
                "name": "customer_id",
                "inferred_type": "identifier",
                "confidence": 0.95,
                "evidence": [],
                "quality_score": 100,
                "cleaning_needed": False,
                "cleaning_recommendations": [],
            },
            "churn": {
                "name": "churn",
                "inferred_type": "binary",
                "confidence": 0.99,
                "evidence": [],
                "quality_score": 100,
                "cleaning_needed": False,
                "cleaning_recommendations": [],
            },
        },
        "target_column": "churn",
        "identifier_columns": ["customer_id"],
    }
    (findings_dir / "customers_findings.yaml").write_text(yaml.dump(customers))

    orders = {
        "source_path": "/data/orders.parquet",
        "source_format": "parquet",
        "row_count": 1000,
        "column_count": 4,
        "columns": {
            "order_id": {
                "name": "order_id",
                "inferred_type": "identifier",
                "confidence": 0.95,
                "evidence": [],
                "quality_score": 100,
                "cleaning_needed": False,
                "cleaning_recommendations": [],
            },
            "customer_id": {
                "name": "customer_id",
                "inferred_type": "identifier",
                "confidence": 0.95,
                "evidence": [],
                "quality_score": 100,
                "cleaning_needed": False,
                "cleaning_recommendations": [],
            },
            "amount": {
                "name": "amount",
                "inferred_type": "numeric_continuous",
                "confidence": 0.9,
                "evidence": [],
                "quality_score": 90,
                "cleaning_needed": False,
                "cleaning_recommendations": [],
            },
            "event_type": {
                "name": "event_type",
                "inferred_type": "categorical_nominal",
                "confidence": 0.95,
                "evidence": [],
                "quality_score": 100,
                "cleaning_needed": False,
                "cleaning_recommendations": [],
            },
            "order_date": {
                "name": "order_date",
                "inferred_type": "datetime",
                "confidence": 0.95,
                "evidence": [],
                "quality_score": 100,
                "cleaning_needed": False,
                "cleaning_recommendations": [],
            },
        },
        "identifier_columns": ["order_id"],
        "datetime_columns": ["order_date"],
        "time_series_metadata": {
            "granularity": "event_level",
            "entity_column": "customer_id",
            "time_column": "order_date",
            "aggregation_windows_used": material_windows,
        },
    }
    (findings_dir / "orders_findings.yaml").write_text(yaml.dump(orders))
    return findings_dir


class TestParseHonorsIntentWindows:
    def test_intent_wins_when_findings_narrowed_by_coverage(self, tmp_path):
        findings_dir = _write_findings_dir(
            tmp_path, material_windows=_COVERAGE_NARROWED, intent_windows=_INTENT_WINDOWS
        )
        from customer_retention.generators.pipeline_generator.findings_parser import (
            FindingsParser,
        )

        config = FindingsParser(str(findings_dir)).parse()
        windows = config.bronze_event["orders"].aggregation.windows
        assert windows == _INTENT_WINDOWS, (
            "intent (7 windows) must win over coverage-narrowed findings "
            f"(2 windows). got: {windows}"
        )

    def test_intent_equals_findings_is_noop(self, tmp_path):
        findings_dir = _write_findings_dir(
            tmp_path, material_windows=_INTENT_WINDOWS, intent_windows=_INTENT_WINDOWS
        )
        from customer_retention.generators.pipeline_generator.findings_parser import (
            FindingsParser,
        )

        config = FindingsParser(str(findings_dir)).parse()
        assert config.bronze_event["orders"].aggregation.windows == _INTENT_WINDOWS

    def test_empty_intent_falls_back_to_findings(self, tmp_path):
        findings_dir = _write_findings_dir(
            tmp_path, material_windows=_COVERAGE_NARROWED, intent_windows=[]
        )
        from customer_retention.generators.pipeline_generator.findings_parser import (
            FindingsParser,
        )

        config = FindingsParser(str(findings_dir)).parse()
        assert config.bronze_event["orders"].aggregation.windows == _COVERAGE_NARROWED

    def test_override_windows_beat_intent_when_genuine_narrowing(self, tmp_path):
        # Cycle-4 widening rule: override must contain at least one window
        # NOT in intent for it to genuinely narrow. ["30d"] is a subset of
        # _INTENT_WINDOWS so it widens — use ["1h"] which IS a strict
        # narrowing (1h not in intent).
        findings_dir = _write_findings_dir(
            tmp_path, material_windows=_COVERAGE_NARROWED, intent_windows=_INTENT_WINDOWS
        )
        from customer_retention.generators.pipeline_generator.findings_parser import (
            FindingsParser,
        )

        config = FindingsParser(
            str(findings_dir),
            bronze_aggregation_overrides={"orders": {"windows": ["1h"]}},
        ).parse()
        assert config.bronze_event["orders"].aggregation.windows == ["1h"]

    def test_override_windows_widen_to_intent_when_subset(self, tmp_path):
        findings_dir = _write_findings_dir(
            tmp_path, material_windows=_COVERAGE_NARROWED, intent_windows=_INTENT_WINDOWS
        )
        from customer_retention.generators.pipeline_generator.findings_parser import (
            FindingsParser,
        )

        config = FindingsParser(
            str(findings_dir),
            bronze_aggregation_overrides={"orders": {"windows": ["30d"]}},
        ).parse()
        assert config.bronze_event["orders"].aggregation.windows == _INTENT_WINDOWS


# ---------------------------------------------------------------------------
# Render-grep — post-fix generated bronze_event script must contain the
# intent-breadth AGGREGATION_WINDOWS literal + per-grid-date branch literals.
# ---------------------------------------------------------------------------


class TestGeneratedBronzeLiterals:
    def test_generated_bronze_event_contains_intent_windows_and_per_grid_literals(
        self, tmp_path
    ):
        findings_dir = _write_findings_dir(
            tmp_path, material_windows=_COVERAGE_NARROWED, intent_windows=_INTENT_WINDOWS
        )
        from customer_retention.generators.pipeline_generator.findings_parser import (
            FindingsParser,
        )
        from customer_retention.generators.pipeline_generator.renderer import (
            CodeRenderer,
        )

        config = FindingsParser(
            str(findings_dir),
            bronze_aggregation_overrides={
                "orders": {
                    "per_grid_date_mode": True,
                    "value_counts_columns": ("event_type",),
                }
            },
        ).parse()
        renderer = CodeRenderer()
        rendered = renderer.render_bronze_event(
            "orders", config.bronze_event["orders"], config
        )
        ast.parse(rendered)  # raises if template-level bug

        for window in _INTENT_WINDOWS:
            assert f"'{window}'" in rendered or f'"{window}"' in rendered, (
                f"window literal {window!r} not found in generated bronze_event"
            )
        assert "AGGREGATION_WINDOWS" in rendered
        assert "VALUE_COUNTS_COLUMNS" in rendered
        assert "'event_type'" in rendered
        assert "apply_event_aggregation_per_grid_date" in rendered


# ---------------------------------------------------------------------------
# Regression — reproduces the engagement_e4ad6e1b run failure shape.
# Pre-fix this gives 2 windows in the generated bronze; post-fix it gives 7.
# Fail-loud with the exact orphan-breadth delta in the error message.
# ---------------------------------------------------------------------------


class TestEngagementRunRegression:
    """Synthetic findings with the exact structural shape of
    `debug/engagement_e4ad6e1b/run/`:
      - multi.aggregation_windows = 7 intent windows
      - contract.ts_meta.aggregation_windows_used = [365d, all_time]   (coverage narrowed)
      - bronze_aggregation_overrides has per_grid_date_mode + value_counts_columns
        BUT NO explicit `windows` key (the common user mistake the run tripped on).

    Pre-fix: windows come from findings material → 2. Post-fix: from intent → 7.
    """

    def test_windows_breadth_matches_intent_not_material(self, tmp_path):
        findings_dir = _write_findings_dir(
            tmp_path, material_windows=_COVERAGE_NARROWED, intent_windows=_INTENT_WINDOWS
        )
        from customer_retention.generators.pipeline_generator.findings_parser import (
            FindingsParser,
        )

        config = FindingsParser(
            str(findings_dir),
            bronze_aggregation_overrides={
                "orders": {
                    "per_grid_date_mode": True,
                    "value_counts_columns": ("event_type",),
                }
            },
        ).parse()
        event_cfg = config.bronze_event["orders"]
        windows = event_cfg.aggregation.windows
        missing = sorted(set(_INTENT_WINDOWS) - set(windows))
        assert not missing, (
            f"engagement_e4ad6e1b regression: intent windows missing from bronze "
            f"event_cfg = {missing}. got windows = {windows}."
        )
        assert event_cfg.per_grid_date_mode is True
        assert event_cfg.value_counts_columns == ("event_type",)
