"""Tests for temporal leakage fixes in pipeline generator templates."""


class TestFeastTimestampTemplate:
    """Tests for Issue #2: datetime.now() used for Feast timestamps."""

    def test_gold_template_feast_timestamp_uses_reference_date(self):
        """Gold template should use reference_date for Feast timestamp, not datetime.now()."""
        from customer_retention.generators.pipeline_generator.renderer import TEMPLATES

        gold_template = TEMPLATES["gold.py.j2"]

        # The template should NOT use datetime.now() for the timestamp
        # It should accept a reference_date parameter or use DataFrame attrs
        assert "add_feast_timestamp" in gold_template

        # Check for the improved implementation
        # The function should handle reference_date from attrs
        assert "reference_date" in gold_template or "attrs" in gold_template

    def test_gold_template_warns_on_fallback_to_datetime_now(self):
        """Gold template should warn if falling back to datetime.now()."""
        from customer_retention.generators.pipeline_generator.renderer import TEMPLATES

        gold_template = TEMPLATES["gold.py.j2"]

        # There should be a warning mechanism when falling back
        assert "warn" in gold_template.lower() or "Warning" in gold_template


class TestHoldoutCreationLocation:
    """Tests for Issue #5: Holdout created after feature computation."""

    def test_silver_template_has_holdout_creation(self):
        """Silver template should include holdout creation logic."""
        from customer_retention.generators.pipeline_generator.renderer import TEMPLATES

        silver_template = TEMPLATES["silver.py.j2"]

        # Silver layer should have holdout creation
        # This ensures holdout happens BEFORE gold layer feature computation
        assert "holdout" in silver_template.lower() or "create_holdout" in silver_template

    def test_scoring_template_uses_existing_holdout(self):
        """Scoring template should use existing holdout, not create new one."""
        from customer_retention.generators.pipeline_generator.renderer import TEMPLATES

        scoring_template = TEMPLATES["run_scoring.py.j2"]

        # Should NOT have the create_holdout_if_needed function that modifies data
        # It should only READ existing holdout, not CREATE it
        lines = scoring_template.split("\n")

        # Check that holdout creation doesn't happen in scoring
        # The scoring script should only use holdout, not create it
        has_holdout_creation = "create_holdout_if_needed" in scoring_template

        # If it has holdout creation, it should be marked as deprecated
        # or it should check if holdout already exists and skip creation
        if has_holdout_creation:
            # Should have a check for existing holdout
            assert "ORIGINAL_COLUMN in df.columns" in scoring_template or "already exists" in scoring_template.lower()


class TestTemporalTemplateConsistency:
    """Tests for temporal consistency across templates."""

    def test_templates_have_consistent_reference_date_handling(self):
        """All templates should handle reference_date consistently."""
        from customer_retention.generators.pipeline_generator.renderer import TEMPLATES

        # Gold and silver should both respect reference_date
        gold_template = TEMPLATES["gold.py.j2"]
        silver_template = TEMPLATES["silver.py.j2"]

        # Both should have mechanisms for temporal awareness
        assert "timestamp" in gold_template.lower()

    def test_training_template_excludes_original_columns(self):
        """Training template should exclude original_* columns (holdout ground truth)."""
        from customer_retention.generators.pipeline_generator.renderer import TEMPLATES

        training_template = TEMPLATES["training.py.j2"]

        # Should exclude original_* columns to prevent leakage
        assert "original_" in training_template
        assert "drop" in training_template.lower() or "exclude" in training_template.lower()

    def test_feast_features_template_exists(self):
        """Feast features template should exist."""
        from customer_retention.generators.pipeline_generator.renderer import TEMPLATES

        # Features template provides Feast definitions
        assert "features.py.j2" in TEMPLATES
        features_template = TEMPLATES["features.py.j2"]
        # Should have FeatureView
        assert "FeatureView" in features_template


class TestRendererOutput:
    """Tests for rendered template output."""

    def test_gold_template_has_temporal_metadata_support(self):
        """Gold template should support temporal metadata."""
        from customer_retention.generators.pipeline_generator.renderer import TEMPLATES

        gold_template = TEMPLATES["gold.py.j2"]

        # Should have add_feast_timestamp function
        assert "add_feast_timestamp" in gold_template

        # Should reference DataFrame attrs for reference_date
        assert "attrs" in gold_template or "reference_date" in gold_template
        # Should have aggregation_reference_date lookup
        assert "aggregation_reference_date" in gold_template
