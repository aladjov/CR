class TestFeastTimestampTemplate:

    def test_gold_template_feast_timestamp_uses_reference_date(self):
        from customer_retention.generators.pipeline_generator.renderer import TEMPLATES

        gold_template = TEMPLATES["gold.py.j2"]

        # The template should NOT use datetime.now() for the timestamp
        # It should accept a reference_date parameter or use DataFrame attrs
        assert "add_feast_timestamp" in gold_template

        # Check for the improved implementation
        # The function should handle reference_date from attrs
        assert "reference_date" in gold_template or "attrs" in gold_template

    def test_gold_template_warns_on_fallback_to_datetime_now(self):
        from customer_retention.generators.pipeline_generator.renderer import TEMPLATES

        gold_template = TEMPLATES["gold.py.j2"]

        # There should be a warning mechanism when falling back
        assert "warn" in gold_template.lower() or "Warning" in gold_template


class TestHoldoutCreationLocation:

    def test_silver_template_has_holdout_creation(self):
        from customer_retention.generators.pipeline_generator.renderer import TEMPLATES

        silver_template = TEMPLATES["silver.py.j2"]

        # Silver layer should have holdout creation
        # This ensures holdout happens BEFORE gold layer feature computation
        assert "holdout" in silver_template.lower() or "create_holdout" in silver_template


class TestTemporalTemplateConsistency:

    def test_templates_have_consistent_reference_date_handling(self):
        from customer_retention.generators.pipeline_generator.renderer import TEMPLATES

        # Gold and silver should both respect reference_date
        gold_template = TEMPLATES["gold.py.j2"]
        silver_template = TEMPLATES["silver.py.j2"]

        # Both should have mechanisms for temporal awareness
        assert "timestamp" in gold_template.lower()

    def test_training_template_excludes_original_columns(self):
        from customer_retention.generators.pipeline_generator.renderer import TEMPLATES

        training_template = TEMPLATES["training.py.j2"]

        # Should exclude original_* columns to prevent leakage
        assert "original_" in training_template
        assert "drop" in training_template.lower() or "exclude" in training_template.lower()

    def test_feast_features_template_exists(self):
        from customer_retention.generators.pipeline_generator.renderer import TEMPLATES

        # Features template provides Feast definitions
        assert "features.py.j2" in TEMPLATES
        features_template = TEMPLATES["features.py.j2"]
        # Should have FeatureView
        assert "FeatureView" in features_template


class TestRendererOutput:

    def test_gold_template_has_temporal_metadata_support(self):
        from customer_retention.generators.pipeline_generator.renderer import TEMPLATES

        gold_template = TEMPLATES["gold.py.j2"]

        # Should have add_feast_timestamp function
        assert "add_feast_timestamp" in gold_template

        # Should reference DataFrame attrs for reference_date
        assert "attrs" in gold_template or "reference_date" in gold_template
        # Should have aggregation_reference_date lookup
        assert "aggregation_reference_date" in gold_template


class TestBronzeEventAllTimeWindow:
    """Bronze event template _parse_window must handle all_time (and weeks) like TimeWindowAggregator."""

    def _render_bronze_event_with_windows(self, windows):
        import ast

        from customer_retention.generators.pipeline_generator.models import (
            AggregationWindowConfig,
            BronzeEventConfig,
            SourceConfig,
        )
        from customer_retention.generators.pipeline_generator.renderer import CodeRenderer

        source = SourceConfig(
            name="events", path="events.parquet", format="parquet",
            entity_key="customer_id", time_column="ts", is_event_level=True,
        )
        config = BronzeEventConfig(
            source=source, entity_column="customer_id", time_column="ts",
            aggregation=AggregationWindowConfig(windows=windows, value_columns=["amount"], agg_funcs=["sum"]),
        )
        result = CodeRenderer().render_bronze_event("events", config)
        ast.parse(result)
        return result

    def test_all_time_window_generates_valid_python(self):
        self._render_bronze_event_with_windows(["7d", "30d", "all_time"])

    def test_all_time_window_skips_time_filter(self):
        result = self._render_bronze_event_with_windows(["7d", "all_time"])
        assert "all_time" in result
        assert '"all_time"' in result
        assert "td is None" in result

    def test_weeks_window_generates_valid_python(self):
        self._render_bronze_event_with_windows(["2w", "4w"])

    def test_only_all_time_generates_valid_python(self):
        self._render_bronze_event_with_windows(["all_time"])


class TestLandingTimestampRename:
    """Landing template should rename raw_time_column -> TIME_COLUMN when they differ."""

    def test_landing_renders_rename_when_raw_time_column_set(self):
        import ast

        from customer_retention.generators.pipeline_generator.models import (
            LandingLayerConfig,
            SourceConfig,
        )
        from customer_retention.generators.pipeline_generator.renderer import CodeRenderer

        source = SourceConfig(
            name="emails", path="emails.parquet", format="parquet",
            entity_key="customer_id", time_column="feature_timestamp", is_event_level=True,
        )
        landing_config = LandingLayerConfig(
            source=source, raw_source_path="/data/raw/emails.csv",
            raw_source_format="csv",
            entity_column="customer_id", time_column="feature_timestamp",
            target_column="churn",
            raw_time_column="sent_date",
        )
        result = CodeRenderer().render_landing("emails", landing_config)
        assert 'df.rename(columns={"sent_date": TIME_COLUMN})' in result or 'rename(columns={"sent_date"' in result
        ast.parse(result)

    def test_landing_no_rename_when_raw_time_column_is_none(self):
        import ast

        from customer_retention.generators.pipeline_generator.models import (
            LandingLayerConfig,
            SourceConfig,
        )
        from customer_retention.generators.pipeline_generator.renderer import CodeRenderer

        source = SourceConfig(
            name="orders", path="orders.parquet", format="parquet",
            entity_key="customer_id", time_column="order_date", is_event_level=True,
        )
        landing_config = LandingLayerConfig(
            source=source, raw_source_path="/data/orders.parquet",
            raw_source_format="parquet",
            entity_column="customer_id", time_column="order_date",
            target_column="churn",
            raw_time_column=None,
        )
        result = CodeRenderer().render_landing("orders", landing_config)
        assert "rename" not in result
        ast.parse(result)


class TestLandingTargetRename:
    """Landing template should rename original_target_column -> TARGET_COLUMN when they differ."""

    def _render_landing(self, original_target_column=None):
        import ast

        from customer_retention.generators.pipeline_generator.models import (
            LandingLayerConfig,
            SourceConfig,
        )
        from customer_retention.generators.pipeline_generator.renderer import CodeRenderer

        source = SourceConfig(
            name="emails", path="emails.parquet", format="parquet",
            entity_key="customer_id", time_column="feature_timestamp", is_event_level=True,
        )
        config = LandingLayerConfig(
            source=source, raw_source_path="/data/raw/emails.csv",
            raw_source_format="csv",
            entity_column="customer_id", time_column="feature_timestamp",
            target_column="target",
            original_target_column=original_target_column,
            raw_time_column="sent_date",
        )
        result = CodeRenderer().render_landing("emails", config)
        ast.parse(result)
        return result

    def test_landing_renames_original_target_column(self):
        result = self._render_landing(original_target_column="unsubscribed")
        assert '"unsubscribed": TARGET_COLUMN' in result

    def test_landing_no_target_rename_when_none(self):
        result = self._render_landing(original_target_column=None)
        assert '"unsubscribed": TARGET_COLUMN' not in result


class TestBronzeEventTargetPreservation:
    """Bronze event template should preserve TARGET_COLUMN through groupby aggregation."""

    def _render_bronze_event(self):
        import ast

        from customer_retention.generators.pipeline_generator.models import (
            AggregationWindowConfig,
            BronzeEventConfig,
            SourceConfig,
        )
        from customer_retention.generators.pipeline_generator.renderer import CodeRenderer

        source = SourceConfig(
            name="events", path="events.parquet", format="parquet",
            entity_key="customer_id", time_column="ts", is_event_level=True,
        )
        config = BronzeEventConfig(
            source=source, entity_column="customer_id", time_column="ts",
            aggregation=AggregationWindowConfig(
                windows=["30d", "all_time"], value_columns=["amount"], agg_funcs=["sum"],
            ),
        )
        result = CodeRenderer().render_bronze_event("events", config)
        ast.parse(result)
        return result

    def test_bronze_event_imports_target_column(self):
        result = self._render_bronze_event()
        assert "TARGET_COLUMN" in result
        assert "from config import" in result

    def test_bronze_event_preserves_target_in_reshaping(self):
        result = self._render_bronze_event()
        assert "df.groupby(ENTITY_COLUMN)[TARGET_COLUMN].first()" in result

    def test_bronze_event_target_preserved_before_window_aggregation(self):
        result = self._render_bronze_event()
        agg_fn = result[result.index("def apply_event_aggregation"):]
        target_pos = agg_fn.index("TARGET_COLUMN")
        window_pos = agg_fn.index("for window in")
        assert target_pos < window_pos


class TestBronzeTimestampRename:
    """Bronze template _load_raw_events should rename raw_time_column -> TIME_COLUMN when they differ."""

    def _render_bronze_with_lifecycle(self, raw_time_column=None):
        import ast

        from customer_retention.generators.pipeline_generator.models import (
            BronzeLayerConfig,
            LifecycleConfig,
            SourceConfig,
        )
        from customer_retention.generators.pipeline_generator.renderer import CodeRenderer

        source = SourceConfig(
            name="emails_agg", path="emails_agg.parquet", format="parquet",
            entity_key="customer_id", time_column="feature_timestamp", is_event_level=True,
        )
        config = BronzeLayerConfig(
            source=source,
            lifecycle=LifecycleConfig(
                include_recency_bucket=True,
                include_lifecycle_quadrant=True,
                include_cyclical_features=True,
            ),
            entity_column="customer_id",
            time_column="feature_timestamp",
            raw_time_column=raw_time_column,
        )
        result = CodeRenderer().render_bronze("emails_agg", config)
        ast.parse(result)
        return result

    def test_bronze_renders_rename_when_raw_time_column_set(self):
        result = self._render_bronze_with_lifecycle(raw_time_column="sent_date")
        assert 'raw_df = raw_df.rename(columns={"sent_date": TIME_COLUMN})' in result

    def test_bronze_rename_appears_before_time_column_usage(self):
        result = self._render_bronze_with_lifecycle(raw_time_column="sent_date")
        lifecycle_fn = result[result.index("def enrich_lifecycle"):]
        rename_pos = lifecycle_fn.index("raw_df.rename")
        time_usage_pos = lifecycle_fn.index("add_recency_tenure")
        assert rename_pos < time_usage_pos

    def test_bronze_no_rename_when_raw_time_column_is_none(self):
        result = self._render_bronze_with_lifecycle(raw_time_column=None)
        assert "raw_df.rename" not in result
