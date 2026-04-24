"""Tests for ``feature_meta_builder`` — reverse-parse + row assembly."""
from __future__ import annotations

from customer_retention.stages.causal.column_descriptions_writer import ColumnDescriptionRow
from customer_retention.stages.causal.interpretation.feature_meta_builder import (
    FeatureLineage,
    build_feature_meta_rows,
    parse_aggregation_feature_name,
)


class TestParseAggregationFeatureName:
    def test_canonical_col_func_window(self):
        lineage = parse_aggregation_feature_name(
            "revenue_sum_30d",
            value_columns=["revenue"],
            agg_funcs=["sum"],
            source_table="bronze_orders",
        )
        assert lineage is not None
        assert lineage.feature_name == "revenue_sum_30d"
        assert lineage.source_columns == ["revenue"]
        assert lineage.source_table == "bronze_orders"
        assert lineage.aggregation_kind == "sum"
        assert lineage.window_days == 30

    def test_mean_is_aliased_to_avg(self):
        lineage = parse_aggregation_feature_name(
            "nps_mean_90d",
            value_columns=["nps"],
            agg_funcs=["mean"],
        )
        assert lineage is not None
        assert lineage.aggregation_kind == "avg"

    def test_event_count_special(self):
        lineage = parse_aggregation_feature_name("event_count_30d")
        assert lineage is not None
        assert lineage.aggregation_kind == "count"
        assert lineage.window_days == 30
        assert lineage.source_columns == []

    def test_days_since_last_event_special(self):
        lineage = parse_aggregation_feature_name("days_since_last_event")
        assert lineage is not None
        assert lineage.aggregation_kind == "recency_days"
        assert lineage.window_days is None
        assert lineage.source_columns == []

    def test_days_since_first_event_special(self):
        lineage = parse_aggregation_feature_name("days_since_first_event")
        assert lineage is not None
        assert lineage.aggregation_kind == "recency_days"

    def test_unknown_pattern_returns_none(self):
        assert parse_aggregation_feature_name("totally_unrelated_column") is None

    def test_windowed_with_unknown_col_and_func_returns_none(self):
        assert parse_aggregation_feature_name(
            "foo_bar_30d",
            value_columns=["revenue"],
            agg_funcs=["sum"],
        ) is None

    def test_longest_column_wins_over_shorter_prefix(self):
        lineage = parse_aggregation_feature_name(
            "customer_revenue_sum_30d",
            value_columns=["customer", "customer_revenue"],
            agg_funcs=["sum"],
        )
        assert lineage is not None
        assert lineage.source_columns == ["customer_revenue"]

    def test_falls_through_to_func_suffix_when_no_col_match(self):
        lineage = parse_aggregation_feature_name(
            "some_weird_col_sum_30d",
            value_columns=[],
            agg_funcs=["sum"],
        )
        assert lineage is not None
        assert lineage.source_columns == ["some_weird_col"]
        assert lineage.aggregation_kind == "sum"


class TestBuildFeatureMetaRows:
    def test_renders_business_phrase_from_column_descriptions(self):
        lineage = FeatureLineage(
            feature_name="nps_mean_90d",
            source_columns=["nps"],
            source_table="bronze_account",
            aggregation_kind="avg",
            window_days=90,
        )
        descriptions = {
            "nps": ColumnDescriptionRow(
                table="account",
                column_name="nps",
                business_name="Net Promoter Score",
            )
        }
        rows = build_feature_meta_rows(
            composite_name="cn1",
            lineages=[lineage],
            column_descriptions=descriptions,
        )
        assert rows[0].composite_name == "cn1"
        assert rows[0].business_phrase == "average Net Promoter Score over last 90 days"
        assert rows[0].window_phrase == "last 90 days"

    def test_falls_back_to_feature_name_without_description(self):
        lineage = FeatureLineage(
            feature_name="nps_mean_90d",
            source_columns=["nps"],
            aggregation_kind="avg",
            window_days=90,
        )
        rows = build_feature_meta_rows(composite_name="cn1", lineages=[lineage])
        assert "nps" in rows[0].business_phrase

    def test_ignores_description_without_business_name(self):
        lineage = FeatureLineage(
            feature_name="x_sum_30d",
            source_columns=["x"],
            aggregation_kind="sum",
            window_days=30,
        )
        descriptions = {"x": ColumnDescriptionRow(table="t", column_name="x")}
        rows = build_feature_meta_rows(
            composite_name="cn1", lineages=[lineage], column_descriptions=descriptions,
        )
        assert rows[0].business_phrase == "sum of x_sum_30d over last 30 days"

    def test_preserves_all_lineage_fields(self):
        lineage = FeatureLineage(
            feature_name="revenue_sum_30d",
            source_columns=["revenue"],
            source_table="bronze_orders",
            aggregation_kind="sum",
            window_days=30,
            mask_future=True,
            target_dependency=False,
            polarity="high_is_good",
        )
        rows = build_feature_meta_rows(composite_name="cn1", lineages=[lineage])
        r = rows[0]
        assert r.source_columns == ["revenue"]
        assert r.source_table == "bronze_orders"
        assert r.mask_future is True
        assert r.target_dependency is False
        assert r.polarity == "high_is_good"

    def test_empty_lineages_returns_empty_list(self):
        assert build_feature_meta_rows(composite_name="cn1", lineages=[]) == []
