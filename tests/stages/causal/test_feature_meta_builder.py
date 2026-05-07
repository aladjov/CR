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
        assert lineage.source_columns == ["event"]

    def test_event_count_all_time_taxonomy(self):
        lineage = parse_aggregation_feature_name("event_count_all_time")
        assert lineage is not None
        assert lineage.aggregation_kind == "count"
        assert lineage.source_columns == ["event"]
        assert lineage.window_days is None

    def test_active_span_days_taxonomy(self):
        lineage = parse_aggregation_feature_name("active_span_days")
        assert lineage is not None
        assert lineage.aggregation_kind == "derived_datetime"
        assert lineage.source_columns == ["event"]

    def test_event_frequency_taxonomy(self):
        lineage = parse_aggregation_feature_name("event_frequency")
        assert lineage is not None
        assert lineage.aggregation_kind == "ratio"

    def test_regularity_score_taxonomy(self):
        lineage = parse_aggregation_feature_name("regularity_score")
        assert lineage is not None
        assert lineage.aggregation_kind == "passthrough"

    def test_inter_event_gap_mean_taxonomy(self):
        lineage = parse_aggregation_feature_name("inter_event_gap_mean")
        assert lineage is not None
        assert lineage.aggregation_kind == "avg"
        assert lineage.source_columns == ["event_gap"]

    def test_inter_event_gap_max_taxonomy(self):
        lineage = parse_aggregation_feature_name("inter_event_gap_max")
        assert lineage is not None
        assert lineage.aggregation_kind == "max"

    def test_lifecycle_quadrant_prefix_taxonomy(self):
        lineage = parse_aggregation_feature_name("lifecycle_quadrant_intense_brief_lifecycle")
        assert lineage is not None
        assert lineage.aggregation_kind == "passthrough"
        assert lineage.source_columns == ["lifecycle"]

    def test_recency_bucket_prefix_taxonomy(self):
        lineage = parse_aggregation_feature_name("recency_bucket_31_90d")
        assert lineage is not None
        assert lineage.source_columns == ["recency"]

    def test_days_since_last_event_special(self):
        lineage = parse_aggregation_feature_name("days_since_last_event")
        assert lineage is not None
        assert lineage.aggregation_kind == "recency_days"
        assert lineage.window_days is None
        assert lineage.source_columns == ["event"]

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

    def test_empty_source_columns_backfills_to_feature_name(self):
        """Cycle 013 D1 fix — never emit a row with empty source_columns.

        compile_predicate_prose has nothing to look up in column_descriptions
        when source_columns is empty, so prose falls back to raw SQL. The
        backfill guarantees at least the feature_name itself is queryable.
        """
        lineage = FeatureLineage(
            feature_name="active_span_days",
            source_columns=None,  # caller couldn't determine real lineage
            aggregation_kind=None,
        )
        rows = build_feature_meta_rows(composite_name="cn1", lineages=[lineage])
        assert rows[0].source_columns == ["active_span_days"]
        assert rows[0].aggregation_kind == "passthrough"

    def test_empty_list_source_columns_also_backfills(self):
        """Same backfill when source_columns is an explicit empty list."""
        lineage = FeatureLineage(
            feature_name="custom_feature_x",
            source_columns=[],
            aggregation_kind=None,
        )
        rows = build_feature_meta_rows(composite_name="cn1", lineages=[lineage])
        assert rows[0].source_columns == ["custom_feature_x"]

    def test_explicit_source_columns_not_overwritten_by_backfill(self):
        lineage = FeatureLineage(
            feature_name="revenue_sum_30d",
            source_columns=["revenue"],
            aggregation_kind="sum",
            window_days=30,
        )
        rows = build_feature_meta_rows(composite_name="cn1", lineages=[lineage])
        assert rows[0].source_columns == ["revenue"]
        assert rows[0].aggregation_kind == "sum"

    def test_humanizes_column_name_when_business_name_missing(self):
        # Curated row exists but ``business_name`` is null — the resolver
        # falls back to a humanized version of the source column name
        # rather than recursing into the feature name itself. Without this,
        # the rendered phrase would read "sum of x_sum_30d over last 30 days"
        # (the feature name in its own description), which is what the
        # dashboard's feature dictionary used to surface for every feature
        # whose source column lacked a curated business_name.
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
        assert rows[0].business_phrase == "sum of x over last 30 days"

    def test_humanizes_underscored_column_name(self):
        lineage = FeatureLineage(
            feature_name="open_count_180d",
            source_columns=["email_id"],
            aggregation_kind="count",
            window_days=180,
        )
        rows = build_feature_meta_rows(composite_name="cn1", lineages=[lineage])
        assert rows[0].business_phrase == "count of email id over last 180 days"

    def test_event_placeholder_renders_as_engagement_event(self):
        # ``parse_aggregation_feature_name`` emits source_columns=["event"]
        # for event_count_* features. There is no "event" column in
        # column_descriptions to look up, so the resolver maps the
        # placeholder to the human phrase "engagement event".
        lineage = FeatureLineage(
            feature_name="event_count_365d",
            source_columns=["event"],
            aggregation_kind="count",
            window_days=365,
        )
        rows = build_feature_meta_rows(composite_name="cn1", lineages=[lineage])
        assert rows[0].business_phrase == "count of engagement event over last 365 days"

    def test_event_gap_placeholder_renders_as_gap_between_events(self):
        lineage = FeatureLineage(
            feature_name="inter_event_gap_max",
            source_columns=["event_gap"],
            aggregation_kind="max",
        )
        rows = build_feature_meta_rows(composite_name="cn1", lineages=[lineage])
        assert rows[0].business_phrase == "maximum gap between events over lifetime"

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

    def test_polarity_inherited_from_column_descriptions(self):
        lineage = FeatureLineage(
            feature_name="nps_mean_90d",
            source_columns=["nps"],
            aggregation_kind="avg",
            window_days=90,
        )
        descriptions = {
            "nps": ColumnDescriptionRow(
                table="account", column_name="nps",
                business_name="NPS score", polarity="high_is_good",
            )
        }
        rows = build_feature_meta_rows(
            composite_name="cn1", lineages=[lineage], column_descriptions=descriptions,
        )
        assert rows[0].polarity == "high_is_good"

    def test_explicit_lineage_polarity_wins_over_descriptions(self):
        lineage = FeatureLineage(
            feature_name="nps_mean_90d",
            source_columns=["nps"],
            aggregation_kind="avg",
            window_days=90,
            polarity="high_is_bad",  # explicit override
        )
        descriptions = {
            "nps": ColumnDescriptionRow(
                table="account", column_name="nps", polarity="high_is_good",
            )
        }
        rows = build_feature_meta_rows(
            composite_name="cn1", lineages=[lineage], column_descriptions=descriptions,
        )
        assert rows[0].polarity == "high_is_bad"
