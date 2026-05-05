"""Tests for the user-code assimilation summary mechanism."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from customer_retention.analysis.auto_explorer.layered_recommendations import (
    RecommendationRegistry,
)
from customer_retention.runtime import summary as summary_mod
from customer_retention.runtime.registry import RegisteredFunction
from customer_retention.runtime.registry import registry as live_registry
from customer_retention.runtime.summary import (
    AssimilationReport,
    AssimilationRow,
    print_user_code_summary,
    render_markdown,
    snapshot_landing_state,
    summarize_user_code,
)


@dataclass
class StubNamespace:
    """Minimal duck-typed RunNamespace stand-in for tests."""
    merged_recommendations_path: Path


@pytest.fixture(autouse=True)
def reset_registry_and_snapshot():
    live_registry.clear()
    summary_mod._landing_snapshot = {}
    yield
    live_registry.clear()
    summary_mod._landing_snapshot = {}


@pytest.fixture
def empty_namespace(tmp_path):
    return StubNamespace(merged_recommendations_path=tmp_path / "recommendations.yaml")


@pytest.fixture
def saved_recommendations(tmp_path):
    """Return a (namespace, registry) pair where registry has been
    populated and saved to disk so summarize_user_code can load it."""
    def _build(populate):
        rec_path = tmp_path / "recommendations.yaml"
        reg = RecommendationRegistry(disable_user_extensions=False)
        populate(reg)
        reg.save(rec_path)
        return StubNamespace(merged_recommendations_path=rec_path), reg
    return _build


class TestImperativeRows:
    def test_empty_registry_yields_no_rows(self, empty_namespace):
        report = summarize_user_code({}, empty_namespace)
        assert report.rows == []
        assert report.all_ok is True

    def test_registered_unmarked_is_missing(self, empty_namespace):
        live_registry.register(RegisteredFunction(
            name="derive_churn_target",
            source="def derive_churn_target(df): pass",
            scope="dataset",
            dataset="account",
            cell_id="00026001",
        ))
        report = summarize_user_code({"account": "/tmp/account.csv"}, empty_namespace)
        assert len(report.rows) == 1
        row = report.rows[0]
        assert row.name == "derive_churn_target"
        assert row.track == "imperative (@cr.register)"
        assert row.target == "account"
        assert row.lane2_status == "missing"
        assert row.cell_id == "00026001"
        assert row.emoji == "❌"
        assert report.all_ok is False
        assert report.failing == [row]

    def test_registered_and_marked_is_ok(self, empty_namespace):
        live_registry.register(RegisteredFunction(
            name="derive_churn_target",
            source="def derive_churn_target(df): pass",
            scope="datasets",
            datasets=["account", "contract"],
            primary="account",
        ))
        live_registry.mark_lane2_executed("derive_churn_target")
        report = summarize_user_code({"account": "/tmp/account.csv"}, empty_namespace)
        row = report.rows[0]
        assert row.lane2_status == "ok"
        assert row.target == "account, contract"
        assert row.emoji == "✅"
        assert report.all_ok is True

    def test_wildcard_scope_target(self, empty_namespace):
        live_registry.register(RegisteredFunction(
            name="global_filter",
            source="def global_filter(df): pass",
            scope="wildcard",
        ))
        report = summarize_user_code({}, empty_namespace)
        assert report.rows[0].target == "*"


class TestLandingDeclarativeRows:
    def test_filter_with_rebind_is_ok(self, saved_recommendations):
        def populate(reg):
            reg.add_landing_filter(
                dataset="account",
                predicate="status != 'TEST'",
                rationale="exclude internal test accounts",
                source_notebook="00_start_here.ipynb",
            )
        ns, _ = saved_recommendations(populate)

        # Snapshot before — datasets[account] is a path string.
        snapshot_landing_state({"account": "/tmp/account.csv"})
        # Lane-2 rebind happened — `account` now points at a temp view object.
        live = {"account": object()}
        report = summarize_user_code(live, ns)

        row = next(r for r in report.rows if r.track == "declarative landing.filter")
        assert row.lane2_status == "ok"
        assert row.target == "account"
        assert "rebound" in row.lane2_detail

    def test_filter_without_rebind_is_missing(self, saved_recommendations):
        def populate(reg):
            reg.add_landing_filter(
                dataset="account",
                predicate="status != 'TEST'",
                rationale="exclude internal test accounts",
                source_notebook="00_start_here.ipynb",
            )
        ns, _ = saved_recommendations(populate)

        snapshot_landing_state({"account": "/tmp/account.csv"})
        live = {"account": "/tmp/account.csv"}  # unchanged
        report = summarize_user_code(live, ns)

        row = next(r for r in report.rows if r.track == "declarative landing.filter")
        assert row.lane2_status == "missing"
        assert "still equals the upstream snapshot" in row.lane2_detail

    def test_lifecycle_enrichment_with_rebind_is_ok(self, saved_recommendations):
        def populate(reg):
            reg.add_landing_lifecycle_enrichment(
                dataset="parents",
                config={"left": "child", "right": "parent_history"},
                rationale="augment parents with SCD history",
                source_notebook="00_start_here.ipynb",
            )
        ns, _ = saved_recommendations(populate)

        snapshot_landing_state({"parents": "/tmp/parents.csv"})
        live = {"parents": object()}
        report = summarize_user_code(live, ns)

        row = next(r for r in report.rows if r.track == "declarative landing.lifecycle_enrichment")
        assert row.lane2_status == "ok"

    def test_dataset_not_in_snapshot_is_pending(self, saved_recommendations):
        def populate(reg):
            reg.add_landing_filter(
                dataset="late_join",
                predicate="active = true",
                rationale="newly added",
                source_notebook="00_start_here.ipynb",
            )
        ns, _ = saved_recommendations(populate)

        snapshot_landing_state({"account": "/tmp/account.csv"})  # late_join missing
        report = summarize_user_code({"account": "/tmp/account.csv", "late_join": "/tmp/lj.csv"}, ns)

        row = next(r for r in report.rows if r.track == "declarative landing.filter")
        assert row.lane2_status == "pending"

    def test_no_snapshot_yields_pending(self, saved_recommendations):
        def populate(reg):
            reg.add_landing_filter(
                dataset="account",
                predicate="x = 1",
                rationale="r",
                source_notebook="00.ipynb",
            )
        ns, _ = saved_recommendations(populate)
        # No snapshot taken.
        report = summarize_user_code({"account": "/tmp/account.csv"}, ns)
        row = report.rows[0]
        assert row.lane2_status == "pending"


class TestBronzeOverrideRows:
    def test_bronze_value_counts_is_config_only(self, saved_recommendations):
        def populate(reg):
            reg.add_bronze_value_counts(
                dataset="contract",
                columns=["status", "tier"],
                rationale="categorical aggregation",
                source_notebook="00_start_here.ipynb",
            )
        ns, _ = saved_recommendations(populate)
        report = summarize_user_code({}, ns)
        rows = [r for r in report.rows if r.track == "declarative bronze_aggregations"]
        assert len(rows) == 1
        row = rows[0]
        assert row.lane2_status == "config-only"
        assert row.target == "contract"
        assert row.emoji == "·"


class TestSilverDerivedRows:
    def test_silver_derived_pending_without_features(self, saved_recommendations):
        def populate(reg):
            reg.init_silver(entity_column="entity_id", time_column="event_timestamp")
            reg.add_silver_ratio(
                column="contract_value_ratio",
                numerator="value",
                denominator="parent_value",
                rationale="ratio derivation",
                source_notebook="00_start_here.ipynb",
            )
        ns, _ = saved_recommendations(populate)
        report = summarize_user_code({}, ns)
        row = next(r for r in report.rows if r.track == "declarative silver.derived")
        assert row.lane2_status == "pending"
        assert row.target == "contract_value_ratio"

    def test_silver_derived_ok_when_column_present(self, saved_recommendations):
        def populate(reg):
            reg.init_silver(entity_column="entity_id")
            reg.add_silver_ratio(
                column="my_ratio",
                numerator="x", denominator="y",
                rationale="r",
                source_notebook="00.ipynb",
            )
        ns, _ = saved_recommendations(populate)

        class _DF:
            columns = ["entity_id", "my_ratio", "other"]
        report = summarize_user_code({}, ns, df_features=_DF())
        row = next(r for r in report.rows if r.track == "declarative silver.derived")
        assert row.lane2_status == "ok"

    def test_silver_derived_missing_when_column_absent(self, saved_recommendations):
        def populate(reg):
            reg.init_silver(entity_column="entity_id")
            reg.add_silver_ratio(
                column="my_ratio",
                numerator="x", denominator="y",
                rationale="r",
                source_notebook="00.ipynb",
            )
        ns, _ = saved_recommendations(populate)

        class _DF:
            columns = ["entity_id", "other_col"]
        report = summarize_user_code({}, ns, df_features=_DF())
        row = next(r for r in report.rows if r.track == "declarative silver.derived")
        assert row.lane2_status == "missing"


class TestSnapshot:
    def test_snapshot_captures_path_strings(self):
        snap = snapshot_landing_state({"a": "/tmp/a.csv", "b": "/tmp/b.csv"})
        assert snap == {"a": "/tmp/a.csv", "b": "/tmp/b.csv"}

    def test_snapshot_captures_object_identity(self):
        df1 = object()
        snap = snapshot_landing_state({"a": df1})
        sig = snap["a"]
        assert sig.startswith("<object#")
        # Same reference → same signature.
        assert summary_mod._signature(df1) == sig

    def test_snapshot_overwrites_on_recall(self):
        snapshot_landing_state({"a": "/tmp/v1.csv"})
        snap2 = snapshot_landing_state({"a": "/tmp/v2.csv"})
        assert snap2 == {"a": "/tmp/v2.csv"}


class TestRendering:
    def test_empty_report_message(self):
        md = render_markdown(AssimilationReport(rows=[]))
        assert "No registered user_code cells" in md

    def test_summary_counts_in_header(self):
        rows = [
            AssimilationRow(cell_id="A", name="f1", track="imperative (@cr.register)",
                            target="account", lane1_present=True, lane2_status="ok",
                            lane2_detail="ok"),
            AssimilationRow(cell_id="B", name="f2", track="imperative (@cr.register)",
                            target="contract", lane1_present=True, lane2_status="missing",
                            lane2_detail="bad"),
            AssimilationRow(cell_id=None, name="b1", track="declarative bronze_aggregations",
                            target="contract", lane1_present=True, lane2_status="config-only",
                            lane2_detail="cfg"),
            AssimilationRow(cell_id=None, name="s1", track="declarative silver.derived",
                            target="ratio_x", lane1_present=True, lane2_status="pending",
                            lane2_detail="wait"),
        ]
        md = render_markdown(AssimilationReport(rows=rows))
        assert "1 applied" in md
        assert "1 config-only" in md
        assert "1 pending" in md
        assert "**1 missing**" in md
        assert "Action required" in md
        assert "`f1`" in md
        assert "`f2`" in md

    def test_ok_only_omits_action_callout(self):
        rows = [AssimilationRow(cell_id="A", name="f1", track="imperative (@cr.register)",
                                target="account", lane1_present=True, lane2_status="ok",
                                lane2_detail="ok")]
        md = render_markdown(AssimilationReport(rows=rows))
        assert "Action required" not in md


class TestPrintUserCodeSummary:
    def test_returns_report(self, empty_namespace):
        live_registry.register(RegisteredFunction(
            name="f", source="", scope="dataset", dataset="a"))
        report = print_user_code_summary({"a": "/tmp/a.csv"}, empty_namespace)
        assert isinstance(report, AssimilationReport)
        assert len(report.rows) == 1

    def test_namespace_none_yields_imperative_only(self):
        live_registry.register(RegisteredFunction(
            name="f", source="", scope="dataset", dataset="a"))
        live_registry.mark_lane2_executed("f")
        report = summarize_user_code({"a": "/tmp/a.csv"}, namespace=None)
        assert len(report.rows) == 1
        assert report.rows[0].track == "imperative (@cr.register)"
