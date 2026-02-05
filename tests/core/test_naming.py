
import pytest

from customer_retention.core.naming import (
    Manifest,
    composite_name,
    dataset_name,
    script_name,
)


class TestCompositeName:
    def test_single_source(self):
        result = composite_name(["customer_emails"])
        prefix, hash_part = result.rsplit("__", 1)
        assert prefix == "cust_emai"
        assert len(hash_part) == 7

    def test_two_sources_sorted(self):
        result_ab = composite_name(["b_source", "a_source"])
        result_ba = composite_name(["a_source", "b_source"])
        assert result_ab == result_ba

    def test_deterministic(self):
        sources = ["customer_emails_aggregated", "customer_profiles"]
        assert composite_name(sources) == composite_name(sources)

    def test_different_sources_different_hash(self):
        cn1 = composite_name(["source_a"])
        cn2 = composite_name(["source_b"])
        assert cn1 != cn2

    def test_readable_prefix_multiple_sources(self):
        result = composite_name(["customer_emails_aggregated", "customer_profiles"])
        prefix, _ = result.rsplit("__", 1)
        assert prefix == "cust_emai_aggr_cust_prof"

    def test_empty_sources_raises(self):
        with pytest.raises(ValueError):
            composite_name([])

    def test_prefix_truncates_words_to_four_chars(self):
        result = composite_name(["very_long_source_name_here"])
        prefix, _ = result.rsplit("__", 1)
        assert prefix == "very_long_sour_name_here"


class TestDatasetName:
    def test_silver_dataset(self):
        cn = composite_name(["source_a", "source_b"])
        assert dataset_name("silver", cn) == f"silver_featureset_{cn}"

    def test_gold_dataset(self):
        cn = composite_name(["source_a"])
        assert dataset_name("gold", cn) == f"gold_features_{cn}"

    def test_bronze_dataset(self):
        assert dataset_name("bronze", "customer_profiles") == "customer_profiles"

    def test_landing_dataset(self):
        assert dataset_name("landing", "customer_emails") == "customer_emails"


class TestScriptName:
    def test_landing_script(self):
        assert script_name("landing", "customer_emails") == "landing_customer_emails"

    def test_bronze_event_script(self):
        assert script_name("bronze_event", "customer_emails") == "bronze_event_customer_emails"

    def test_bronze_entity_script(self):
        assert script_name("bronze_entity", "customer_emails_aggregated") == "bronze_entity_customer_emails_aggregated"

    def test_silver_script(self):
        cn = composite_name(["a", "b"])
        assert script_name("silver", cn) == f"silver_featureset_{cn}"

    def test_gold_script(self):
        cn = composite_name(["a", "b"])
        assert script_name("gold", cn) == f"gold_features_{cn}"


class TestManifest:
    def test_from_sources(self):
        sources = ["customer_emails", "customer_profiles"]
        m = Manifest.from_sources(sources, "customer_churn", recommendations_hash="abc123")
        assert m.pipeline_name == "customer_churn"
        assert m.composite_name == composite_name(sources)
        assert sorted(m.sources) == sorted(sources)
        assert m.recommendations_hash == "abc123"

    def test_dataset_names(self):
        m = Manifest.from_sources(["a", "b"], "test_pipeline")
        assert m.datasets["silver"] == f"silver_featureset_{m.composite_name}"
        assert m.datasets["gold"] == f"gold_features_{m.composite_name}"

    def test_script_names(self):
        m = Manifest.from_sources(["src_a"], "test_pipeline")
        assert m.scripts["landing"]["src_a"] == "landing_src_a"
        assert m.scripts["silver"] == f"silver_featureset_{m.composite_name}"
        assert m.scripts["gold"] == f"gold_features_{m.composite_name}"

    def test_feast_config(self):
        m = Manifest.from_sources(["a"], "test_pipeline")
        assert m.feast["feature_view"] == f"featureset_{m.composite_name}"
        assert m.feast["timestamp_column"] == "event_timestamp"

    def test_round_trip_json(self, tmp_path):
        m = Manifest.from_sources(["x", "y"], "pipe", recommendations_hash="h1")
        path = tmp_path / "manifest.json"
        m.save(path)
        loaded = Manifest.load(path)
        assert loaded.composite_name == m.composite_name
        assert loaded.pipeline_name == m.pipeline_name
        assert loaded.sources == m.sources
        assert loaded.datasets == m.datasets
        assert loaded.scripts == m.scripts
        assert loaded.feast == m.feast

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Manifest.load(tmp_path / "nope.json")

    def test_mlflow_experiment_name(self):
        m = Manifest.from_sources(["a"], "my_pipe")
        assert m.mlflow["experiment_name"] == "my_pipe"
