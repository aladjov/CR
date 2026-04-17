"""Tests for medallion architecture layered recommendations."""
from dataclasses import asdict
from pathlib import Path

from customer_retention.analysis.auto_explorer.layered_recommendations import (
    BronzeRecommendations,
    GoldRecommendations,
    LayeredRecommendation,
    RecommendationRegistry,
    SilverRecommendations,
)


class TestLayeredRecommendation:
    def test_creates_with_required_fields(self):
        rec = LayeredRecommendation(
            id="bronze_impute_age",
            layer="bronze",
            category="cleaning",
            action="impute",
            target_column="age",
            parameters={"strategy": "median"},
            rationale="5% missing values",
            source_notebook="03_quality"
        )
        assert rec.id == "bronze_impute_age"
        assert rec.layer == "bronze"
        assert rec.priority == 1

    def test_default_priority_and_dependencies(self):
        rec = LayeredRecommendation(
            id="test", layer="bronze", category="cleaning", action="drop",
            target_column="col", parameters={}, rationale="test", source_notebook="01"
        )
        assert rec.priority == 1
        assert rec.dependencies == []

    def test_serializes_to_dict(self):
        rec = LayeredRecommendation(
            id="bronze_cap_outliers_revenue",
            layer="bronze",
            category="outlier",
            action="cap",
            target_column="revenue",
            parameters={"method": "iqr", "factor": 1.5},
            rationale="8% outliers detected",
            source_notebook="03_quality",
            priority=2,
            dependencies=["bronze_impute_revenue"]
        )
        d = asdict(rec)
        assert d["id"] == "bronze_cap_outliers_revenue"
        assert d["parameters"]["method"] == "iqr"
        assert len(d["dependencies"]) == 1


class TestBronzeRecommendations:
    def test_creates_empty_container(self):
        bronze = BronzeRecommendations(source_file="customers.csv")
        assert bronze.source_file == "customers.csv"
        assert bronze.null_handling == []
        assert bronze.outlier_handling == []

    def test_adds_null_handling_recommendation(self):
        bronze = BronzeRecommendations(source_file="data.csv")
        rec = LayeredRecommendation(
            id="bronze_impute_age", layer="bronze", category="null",
            action="impute", target_column="age",
            parameters={"strategy": "median"},
            rationale="5% nulls", source_notebook="03"
        )
        bronze.null_handling.append(rec)
        assert len(bronze.null_handling) == 1

    def test_all_recommendations_property(self):
        bronze = BronzeRecommendations(source_file="data.csv")
        bronze.null_handling.append(LayeredRecommendation(
            id="r1", layer="bronze", category="null", action="impute",
            target_column="a", parameters={}, rationale="", source_notebook=""
        ))
        bronze.outlier_handling.append(LayeredRecommendation(
            id="r2", layer="bronze", category="outlier", action="cap",
            target_column="b", parameters={}, rationale="", source_notebook=""
        ))
        bronze.type_conversions.append(LayeredRecommendation(
            id="r3", layer="bronze", category="type", action="cast",
            target_column="c", parameters={}, rationale="", source_notebook=""
        ))
        all_recs = bronze.all_recommendations
        assert len(all_recs) == 3
        assert {r.id for r in all_recs} == {"r1", "r2", "r3"}


class TestSilverRecommendations:
    def test_creates_with_entity_column(self):
        silver = SilverRecommendations(entity_column="customer_id")
        assert silver.entity_column == "customer_id"
        assert silver.time_column is None
        assert silver.joins == []

    def test_creates_with_time_column_for_timeseries(self):
        silver = SilverRecommendations(
            entity_column="customer_id",
            time_column="event_date"
        )
        assert silver.time_column == "event_date"

    def test_all_recommendations_property(self):
        silver = SilverRecommendations(entity_column="customer_id")
        silver.aggregations.append(LayeredRecommendation(
            id="silver_agg_revenue", layer="silver", category="aggregation",
            action="sum", target_column="revenue",
            parameters={"windows": ["7d", "30d"]},
            rationale="Time window aggregation", source_notebook="04TS"
        ))
        assert len(silver.all_recommendations) == 1


class TestGoldRecommendations:
    def test_creates_with_target_column(self):
        gold = GoldRecommendations(target_column="churned")
        assert gold.target_column == "churned"
        assert gold.encoding == []
        assert gold.scaling == []

    def test_all_recommendations_property(self):
        gold = GoldRecommendations(target_column="churned")
        gold.encoding.append(LayeredRecommendation(
            id="gold_onehot_contract", layer="gold", category="encoding",
            action="one_hot", target_column="contract_type",
            parameters={"drop_first": True},
            rationale="Low cardinality", source_notebook="06"
        ))
        gold.scaling.append(LayeredRecommendation(
            id="gold_scale_revenue", layer="gold", category="scaling",
            action="standard", target_column="revenue",
            parameters={}, rationale="", source_notebook="06"
        ))
        assert len(gold.all_recommendations) == 2


class TestRecommendationRegistry:
    def test_creates_empty_registry(self):
        registry = RecommendationRegistry()
        assert registry.bronze is None
        assert registry.silver is None
        assert registry.gold is None

    def test_initializes_bronze_layer(self):
        registry = RecommendationRegistry()
        registry.init_bronze("customers.csv")
        assert registry.bronze is not None
        assert registry.bronze.source_file == "customers.csv"

    def test_initializes_silver_layer(self):
        registry = RecommendationRegistry()
        registry.init_silver("customer_id", time_column="event_date")
        assert registry.silver is not None
        assert registry.silver.entity_column == "customer_id"
        assert registry.silver.time_column == "event_date"

    def test_initializes_gold_layer(self):
        registry = RecommendationRegistry()
        registry.init_gold("churned")
        assert registry.gold is not None
        assert registry.gold.target_column == "churned"

    def test_add_recommendation_to_bronze(self):
        registry = RecommendationRegistry()
        registry.init_bronze("data.csv")
        registry.add_bronze_null(
            column="age", strategy="median",
            rationale="5% nulls", source_notebook="03"
        )
        assert len(registry.bronze.null_handling) == 1
        rec = registry.bronze.null_handling[0]
        assert rec.target_column == "age"
        assert rec.parameters["strategy"] == "median"

    def test_add_recommendation_to_silver(self):
        registry = RecommendationRegistry()
        registry.init_silver("customer_id", "event_date")
        registry.add_silver_aggregation(
            column="revenue", aggregation="sum", windows=["7d", "30d"],
            rationale="Revenue trend", source_notebook="04TS"
        )
        assert len(registry.silver.aggregations) == 1

    def test_add_recommendation_to_gold(self):
        registry = RecommendationRegistry()
        registry.init_gold("churned")
        registry.add_gold_encoding(
            column="contract", method="one_hot",
            rationale="3 categories", source_notebook="06"
        )
        assert len(registry.gold.encoding) == 1

    def test_get_all_recommendations(self):
        registry = RecommendationRegistry()
        registry.init_bronze("data.csv")
        registry.init_silver("customer_id")
        registry.init_gold("churned")
        registry.add_bronze_null("a", "median", "", "")
        registry.add_silver_aggregation("b", "sum", ["7d"], "", "")
        registry.add_gold_scaling("c", "standard", "", "")
        all_recs = registry.all_recommendations
        assert len(all_recs) == 3

    def test_get_recommendations_by_layer(self):
        registry = RecommendationRegistry()
        registry.init_bronze("data.csv")
        registry.init_gold("churned")
        registry.add_bronze_null("a", "median", "", "")
        registry.add_bronze_outlier("b", "cap", {"method": "iqr"}, "", "")
        registry.add_gold_encoding("c", "one_hot", "", "")
        bronze_recs = registry.get_by_layer("bronze")
        gold_recs = registry.get_by_layer("gold")
        assert len(bronze_recs) == 2
        assert len(gold_recs) == 1

    def test_serializes_to_dict(self):
        registry = RecommendationRegistry()
        registry.init_bronze("data.csv")
        registry.add_bronze_null("age", "median", "5% nulls", "03")
        d = registry.to_dict()
        assert "bronze" in d
        assert d["bronze"]["source_file"] == "data.csv"
        assert len(d["bronze"]["null_handling"]) == 1

    def test_loads_from_dict(self):
        data = {
            "bronze": {
                "source_file": "data.csv",
                "null_handling": [{
                    "id": "bronze_null_age",
                    "layer": "bronze",
                    "category": "null",
                    "action": "impute",
                    "target_column": "age",
                    "parameters": {"strategy": "median"},
                    "rationale": "5% nulls",
                    "source_notebook": "03",
                    "priority": 1,
                    "dependencies": []
                }],
                "outlier_handling": [],
                "type_conversions": [],
                "deduplication": [],
                "filtering": []
            }
        }
        registry = RecommendationRegistry.from_dict(data)
        assert registry.bronze is not None
        assert len(registry.bronze.null_handling) == 1
        assert registry.bronze.null_handling[0].target_column == "age"


class TestMultiSourceRegistry:
    def test_registers_multiple_bronze_sources(self):
        registry = RecommendationRegistry()
        registry.add_source("customers", "customers.csv")
        registry.add_source("events", "events.csv")
        assert "customers" in registry.sources
        assert "events" in registry.sources
        assert len(registry.sources) == 2

    def test_adds_recommendation_to_specific_source(self):
        registry = RecommendationRegistry()
        registry.add_source("customers", "customers.csv")
        registry.add_source("events", "events.csv")
        registry.add_bronze_null("age", "median", "5% nulls", "03", source="customers")
        registry.add_bronze_null("event_value", "zero", "3% nulls", "03", source="events")
        assert len(registry.sources["customers"].null_handling) == 1
        assert len(registry.sources["events"].null_handling) == 1

    def test_source_names_are_unique(self):
        registry = RecommendationRegistry()
        registry.add_source("customers", "customers.csv")
        registry.add_source("customers", "customers_v2.csv")
        assert registry.sources["customers"].source_file == "customers_v2.csv"

    def test_get_source_recommendations(self):
        registry = RecommendationRegistry()
        registry.add_source("customers", "customers.csv")
        registry.add_bronze_null("age", "median", "", "", source="customers")
        registry.add_bronze_outlier("revenue", "cap", {"method": "iqr"}, "", "", source="customers")
        recs = registry.get_source_recommendations("customers")
        assert len(recs) == 2

    def test_get_all_source_names(self):
        registry = RecommendationRegistry()
        registry.add_source("customers", "customers.csv")
        registry.add_source("events", "events.csv")
        registry.add_source("products", "products.csv")
        assert set(registry.source_names) == {"customers", "events", "products"}

    def test_backward_compatible_init_bronze(self):
        registry = RecommendationRegistry()
        registry.init_bronze("customers.csv")
        assert registry.bronze is not None
        assert "default" in registry.sources or registry.bronze is not None


class TestCategoricalAggregationRecommendations:
    """Tests for categorical aggregation recommendations in Silver layer."""

    def test_adds_mode_aggregation(self):
        registry = RecommendationRegistry()
        registry.init_silver("customer_id", "event_date")
        registry.add_silver_aggregation(
            column="channel", aggregation="mode", windows=["30d", "all_time"],
            rationale="Most frequent channel per customer", source_notebook="01d"
        )
        assert len(registry.silver.aggregations) == 1
        rec = registry.silver.aggregations[0]
        assert rec.parameters["aggregation"] == "mode"
        assert rec.target_column == "channel"

    def test_adds_nunique_aggregation(self):
        registry = RecommendationRegistry()
        registry.init_silver("customer_id", "event_date")
        registry.add_silver_aggregation(
            column="product_category", aggregation="nunique", windows=["7d", "30d"],
            rationale="Count distinct categories purchased", source_notebook="01d"
        )
        rec = registry.silver.aggregations[0]
        assert rec.parameters["aggregation"] == "nunique"

    def test_adds_mode_ratio_aggregation(self):
        registry = RecommendationRegistry()
        registry.init_silver("customer_id", "event_date")
        registry.add_silver_aggregation(
            column="channel", aggregation="mode_ratio", windows=["30d"],
            rationale="Consistency of channel usage", source_notebook="01d"
        )
        rec = registry.silver.aggregations[0]
        assert rec.parameters["aggregation"] == "mode_ratio"

    def test_adds_entropy_aggregation(self):
        registry = RecommendationRegistry()
        registry.init_silver("customer_id", "event_date")
        registry.add_silver_aggregation(
            column="product_category", aggregation="entropy", windows=["all_time"],
            rationale="Diversity of purchases", source_notebook="01d"
        )
        rec = registry.silver.aggregations[0]
        assert rec.parameters["aggregation"] == "entropy"

    def test_adds_value_counts_aggregation(self):
        registry = RecommendationRegistry()
        registry.init_silver("customer_id", "event_date")
        registry.add_silver_aggregation(
            column="channel", aggregation="value_counts", windows=["30d"],
            rationale="Individual counts per channel", source_notebook="01d"
        )
        rec = registry.silver.aggregations[0]
        assert rec.parameters["aggregation"] == "value_counts"

    def test_mixed_numeric_and_categorical_aggregations(self):
        registry = RecommendationRegistry()
        registry.init_silver("customer_id", "event_date")
        registry.add_silver_aggregation("amount", "sum", ["30d"], "Total spend", "01d")
        registry.add_silver_aggregation("amount", "mean", ["30d"], "Average spend", "01d")
        registry.add_silver_aggregation("channel", "mode", ["30d"], "Primary channel", "01d")
        registry.add_silver_aggregation("channel", "nunique", ["30d"], "Channel diversity", "01d")
        assert len(registry.silver.aggregations) == 4


class TestSilverJoinSpec:
    def test_adds_join_between_sources(self):
        registry = RecommendationRegistry()
        registry.add_source("customers", "customers.csv")
        registry.add_source("events", "events.csv")
        registry.init_silver("customer_id")
        registry.add_silver_join(
            left_source="customers",
            right_source="events",
            join_keys=["customer_id"],
            join_type="left",
            rationale="Enrich customers with event history"
        )
        assert len(registry.silver.joins) == 1
        join = registry.silver.joins[0]
        assert join.parameters["left_source"] == "customers"
        assert join.parameters["right_source"] == "events"

    def test_join_defines_merge_order(self):
        registry = RecommendationRegistry()
        registry.add_source("customers", "customers.csv")
        registry.add_source("events", "events.csv")
        registry.add_source("products", "products.csv")
        registry.init_silver("customer_id")
        registry.add_silver_join("customers", "events", ["customer_id"], "left", "Add events")
        registry.add_silver_join("_merged", "products", ["product_id"], "left", "Add products")
        assert len(registry.silver.joins) == 2


class TestMultiSourceSerialization:
    def test_serializes_multiple_sources(self):
        registry = RecommendationRegistry()
        registry.add_source("customers", "customers.csv")
        registry.add_source("events", "events.csv")
        registry.add_bronze_null("age", "median", "", "", source="customers")
        d = registry.to_dict()
        assert "sources" in d
        assert "customers" in d["sources"]
        assert "events" in d["sources"]

    def test_loads_multiple_sources_from_dict(self):
        data = {
            "sources": {
                "customers": {
                    "source_file": "customers.csv",
                    "null_handling": [],
                    "outlier_handling": [],
                    "type_conversions": [],
                    "deduplication": [],
                    "filtering": [],
                    "text_processing": []
                },
                "events": {
                    "source_file": "events.csv",
                    "null_handling": [],
                    "outlier_handling": [],
                    "type_conversions": [],
                    "deduplication": [],
                    "filtering": [],
                    "text_processing": []
                }
            }
        }
        registry = RecommendationRegistry.from_dict(data)
        assert len(registry.sources) == 2
        assert "customers" in registry.sources
        assert "events" in registry.sources


class TestBronzeFilteringRecommendations:
    def test_adds_filtering_to_bronze(self):
        registry = RecommendationRegistry()
        registry.init_bronze("data.csv")
        registry.add_bronze_filtering(
            column="rate", condition="value > 100", action="cap",
            rationale="Rate cannot exceed 100%", source_notebook="02"
        )
        assert len(registry.bronze.filtering) == 1
        rec = registry.bronze.filtering[0]
        assert rec.target_column == "rate"
        assert rec.parameters["condition"] == "value > 100"
        assert rec.action == "cap"

    def test_adds_filtering_to_source(self):
        registry = RecommendationRegistry()
        registry.add_source("events", "events.csv")
        registry.add_bronze_filtering(
            column="amount", condition="value < 0", action="filter",
            rationale="Negative amounts invalid", source_notebook="02", source="events"
        )
        assert len(registry.sources["events"].filtering) == 1

    def test_filtering_in_all_recommendations(self):
        registry = RecommendationRegistry()
        registry.init_bronze("data.csv")
        registry.add_bronze_filtering("rate", "value > 100", "cap", "", "")
        registry.add_bronze_null("age", "median", "", "")
        assert len(registry.all_recommendations) == 2


class TestBronzeModelingStrategyRecommendations:
    def test_adds_modeling_strategy_to_bronze(self):
        registry = RecommendationRegistry()
        registry.init_bronze("data.csv")
        registry.add_bronze_modeling_strategy(
            strategy="time_based_split", column="created_date",
            parameters={"test_ratio": 0.2, "by_column": "created_date"},
            rationale="Strong temporal trend detected (+500%)", source_notebook="02"
        )
        assert len(registry.bronze.modeling_strategy) == 1
        rec = registry.bronze.modeling_strategy[0]
        assert rec.action == "time_based_split"
        assert rec.parameters["test_ratio"] == 0.2

    def test_modeling_strategy_in_all_recommendations(self):
        registry = RecommendationRegistry()
        registry.init_bronze("data.csv")
        registry.add_bronze_modeling_strategy("time_based_split", "date", {}, "", "")
        registry.add_bronze_null("age", "median", "", "")
        assert len(registry.all_recommendations) == 2

    def test_serializes_modeling_strategy(self):
        registry = RecommendationRegistry()
        registry.init_bronze("data.csv")
        registry.add_bronze_modeling_strategy("time_based_split", "date", {"test_ratio": 0.2}, "trend", "02")
        d = registry.to_dict()
        assert len(d["bronze"]["modeling_strategy"]) == 1
        assert d["bronze"]["modeling_strategy"][0]["action"] == "time_based_split"

    def test_deserializes_modeling_strategy(self):
        data = {
            "bronze": {
                "source_file": "data.csv",
                "null_handling": [], "outlier_handling": [], "type_conversions": [],
                "deduplication": [], "filtering": [], "text_processing": [],
                "modeling_strategy": [{
                    "id": "bronze_modeling_date", "layer": "bronze", "category": "modeling",
                    "action": "time_based_split", "target_column": "date",
                    "parameters": {"test_ratio": 0.2}, "rationale": "trend",
                    "source_notebook": "02", "priority": 1, "dependencies": []
                }]
            }
        }
        registry = RecommendationRegistry.from_dict(data)
        assert len(registry.bronze.modeling_strategy) == 1


class TestSilverDerivedRecommendations:
    def test_adds_derived_column_to_silver(self):
        registry = RecommendationRegistry()
        registry.init_silver("customer_id")
        registry.add_silver_derived(
            column="days_since_signup", expression="(reference_date - signup_date).dt.days",
            feature_type="recency", rationale="Recency captures behavior patterns", source_notebook="02"
        )
        assert len(registry.silver.derived_columns) == 1
        rec = registry.silver.derived_columns[0]
        assert rec.target_column == "days_since_signup"
        assert rec.parameters["expression"] == "(reference_date - signup_date).dt.days"
        assert rec.parameters["feature_type"] == "recency"

    def test_adds_multiple_derived_columns(self):
        registry = RecommendationRegistry()
        registry.init_silver("customer_id")
        registry.add_silver_derived("tenure_years", "(ref - signup).days / 365", "tenure", "", "02")
        registry.add_silver_derived("signup_month_sin", "np.sin(2*pi*month/12)", "cyclical", "", "02")
        registry.add_silver_derived("is_weekend", "dayofweek >= 5", "extraction", "", "02")
        assert len(registry.silver.derived_columns) == 3

    def test_derived_in_all_recommendations(self):
        registry = RecommendationRegistry()
        registry.init_silver("customer_id")
        registry.add_silver_derived("days_since", "expr", "recency", "", "")
        registry.add_silver_aggregation("amount", "sum", ["7d"], "", "")
        assert len(registry.all_recommendations) == 2


class TestGoldTransformationRecommendations:
    def test_adds_transformation_to_gold(self):
        registry = RecommendationRegistry()
        registry.init_gold("churned")
        registry.add_gold_transformation(
            column="revenue", transform="log_transform",
            parameters={"base": "natural", "offset": 1},
            rationale="High skewness (3.5)", source_notebook="02"
        )
        assert len(registry.gold.transformations) == 1
        rec = registry.gold.transformations[0]
        assert rec.target_column == "revenue"
        assert rec.action == "log_transform"
        assert rec.parameters["base"] == "natural"

    def test_adds_cap_then_log_transformation(self):
        registry = RecommendationRegistry()
        registry.init_gold("churned")
        registry.add_gold_transformation(
            column="order_value", transform="cap_then_log",
            parameters={"cap_method": "iqr", "cap_multiplier": 1.5},
            rationale="High skewness with outliers", source_notebook="02"
        )
        rec = registry.gold.transformations[0]
        assert rec.action == "cap_then_log"
        assert rec.parameters["cap_method"] == "iqr"

    def test_adds_zero_inflation_transformation(self):
        registry = RecommendationRegistry()
        registry.init_gold("churned")
        registry.add_gold_transformation(
            column="click_rate", transform="zero_inflation_handling",
            parameters={"strategy": "separate_indicator", "transform_non_zero": "log"},
            rationale="50% zeros with high skewness", source_notebook="02"
        )
        rec = registry.gold.transformations[0]
        assert rec.action == "zero_inflation_handling"
        assert rec.parameters["strategy"] == "separate_indicator"

    def test_transformation_in_all_recommendations(self):
        registry = RecommendationRegistry()
        registry.init_gold("churned")
        registry.add_gold_transformation("revenue", "log_transform", {}, "", "")
        registry.add_gold_encoding("contract", "one_hot", "", "")
        registry.add_gold_scaling("tenure", "standard", "", "")
        assert len(registry.all_recommendations) == 3

    def test_serializes_transformations(self):
        registry = RecommendationRegistry()
        registry.init_gold("churned")
        registry.add_gold_transformation("revenue", "log_transform", {"offset": 1}, "skewed", "02")
        d = registry.to_dict()
        assert len(d["gold"]["transformations"]) == 1
        assert d["gold"]["transformations"][0]["action"] == "log_transform"


class TestBronzeDeduplicationRecommendations:
    def test_adds_deduplication_to_bronze(self):
        registry = RecommendationRegistry()
        registry.init_bronze("data.csv")
        registry.add_bronze_deduplication(
            key_column="customer_id", strategy="keep_first",
            rationale="12 duplicate keys detected", source_notebook="03"
        )
        assert len(registry.bronze.deduplication) == 1
        rec = registry.bronze.deduplication[0]
        assert rec.target_column == "customer_id"
        assert rec.action == "keep_first"

    def test_adds_deduplication_with_conflict_columns(self):
        registry = RecommendationRegistry()
        registry.init_bronze("data.csv")
        registry.add_bronze_deduplication(
            key_column="customer_id", strategy="keep_most_recent",
            rationale="Value conflicts in target column",
            source_notebook="03", conflict_columns=["retained", "revenue"]
        )
        rec = registry.bronze.deduplication[0]
        assert rec.parameters["conflict_columns"] == ["retained", "revenue"]
        assert rec.parameters["strategy"] == "keep_most_recent"

    def test_deduplication_in_all_recommendations(self):
        registry = RecommendationRegistry()
        registry.init_bronze("data.csv")
        registry.add_bronze_deduplication("customer_id", "keep_first", "", "")
        registry.add_bronze_null("age", "median", "", "")
        assert len(registry.all_recommendations) == 2


class TestBronzeConsistencyRecommendations:
    def test_adds_consistency_recommendation(self):
        registry = RecommendationRegistry()
        registry.init_bronze("data.csv")
        registry.add_bronze_consistency(
            column="city", issue_type="case_variants",
            action="normalize_lower", variants=["NYC", "nyc", "Nyc"],
            rationale="3 case variants detected", source_notebook="03"
        )
        assert len(registry.bronze.type_conversions) == 1
        rec = registry.bronze.type_conversions[0]
        assert rec.target_column == "city"
        assert rec.action == "normalize_lower"
        assert rec.parameters["variants"] == ["NYC", "nyc", "Nyc"]

    def test_adds_whitespace_consistency(self):
        registry = RecommendationRegistry()
        registry.init_bronze("data.csv")
        registry.add_bronze_consistency(
            column="name", issue_type="whitespace",
            action="strip", variants=[" John", "John ", " John "],
            rationale="Leading/trailing whitespace", source_notebook="03"
        )
        rec = registry.bronze.type_conversions[0]
        assert rec.action == "strip"


class TestBronzeImbalanceRecommendations:
    def test_adds_imbalance_strategy(self):
        registry = RecommendationRegistry()
        registry.init_bronze("data.csv")
        registry.add_bronze_imbalance_strategy(
            target_column="churned", imbalance_ratio=3.87,
            minority_class=0, strategy="stratified_sampling",
            rationale="3.87:1 imbalance ratio", source_notebook="03"
        )
        assert len(registry.bronze.modeling_strategy) == 1
        rec = registry.bronze.modeling_strategy[0]
        assert rec.action == "stratified_sampling"
        assert rec.parameters["imbalance_ratio"] == 3.87
        assert rec.parameters["minority_class"] == 0

    def test_adds_smote_recommendation(self):
        registry = RecommendationRegistry()
        registry.init_bronze("data.csv")
        registry.add_bronze_imbalance_strategy(
            target_column="churned", imbalance_ratio=10.5,
            minority_class=1, strategy="smote",
            rationale="Severe imbalance requires SMOTE", source_notebook="03"
        )
        rec = registry.bronze.modeling_strategy[0]
        assert rec.action == "smote"


class TestTextProcessingRecommendations:
    def test_adds_text_processing_to_bronze(self):
        registry = RecommendationRegistry()
        registry.init_bronze("data.csv")
        registry.add_bronze_text_processing(
            column="description",
            embedding_model="all-MiniLM-L6-v2",
            variance_threshold=0.95,
            n_components=5,
            rationale="TEXT column with ticket descriptions",
            source_notebook="02a"
        )
        assert len(registry.bronze.text_processing) == 1
        rec = registry.bronze.text_processing[0]
        assert rec.target_column == "description"
        assert rec.parameters["embedding_model"] == "all-MiniLM-L6-v2"
        assert rec.parameters["variance_threshold"] == 0.95
        assert rec.parameters["n_components"] == 5
        assert rec.parameters["approach"] == "pca"

    def test_adds_text_processing_to_source(self):
        registry = RecommendationRegistry()
        registry.add_source("tickets", "tickets.csv")
        registry.add_bronze_text_processing(
            column="ticket_text",
            embedding_model="all-MiniLM-L6-v2",
            variance_threshold=0.90,
            n_components=10,
            rationale="Support ticket text",
            source_notebook="01a_a",
            source="tickets"
        )
        assert len(registry.sources["tickets"].text_processing) == 1

    def test_text_processing_in_all_recommendations(self):
        registry = RecommendationRegistry()
        registry.init_bronze("data.csv")
        registry.add_bronze_null("age", "median", "", "")
        registry.add_bronze_text_processing("description", "all-MiniLM-L6-v2", 0.95, 5, "", "")
        all_recs = registry.all_recommendations
        assert len(all_recs) == 2

    def test_serializes_text_processing(self):
        registry = RecommendationRegistry()
        registry.init_bronze("data.csv")
        registry.add_bronze_text_processing("description", "all-MiniLM-L6-v2", 0.95, 5, "rationale", "02a")
        d = registry.to_dict()
        assert len(d["bronze"]["text_processing"]) == 1
        assert d["bronze"]["text_processing"][0]["parameters"]["approach"] == "pca"

    def test_deserializes_text_processing(self):
        data = {
            "bronze": {
                "source_file": "data.csv",
                "null_handling": [],
                "outlier_handling": [],
                "type_conversions": [],
                "deduplication": [],
                "filtering": [],
                "text_processing": [{
                    "id": "bronze_text_description",
                    "layer": "bronze",
                    "category": "text",
                    "action": "embed_reduce",
                    "target_column": "description",
                    "parameters": {
                        "embedding_model": "all-MiniLM-L6-v2",
                        "variance_threshold": 0.95,
                        "n_components": 5,
                        "approach": "pca"
                    },
                    "rationale": "TEXT column",
                    "source_notebook": "02a",
                    "priority": 1,
                    "dependencies": []
                }]
            }
        }
        registry = RecommendationRegistry.from_dict(data)
        assert len(registry.bronze.text_processing) == 1
        assert registry.bronze.text_processing[0].target_column == "description"


class TestGoldFeatureSelectionRecommendations:
    def test_adds_drop_multicollinear_to_gold(self):
        registry = RecommendationRegistry()
        registry.init_gold("churned")
        registry.add_gold_drop_multicollinear(
            column="feature_a", correlated_with="feature_b",
            correlation=0.85, rationale="High correlation detected",
            source_notebook="04"
        )
        assert len(registry.gold.feature_selection) == 1
        rec = registry.gold.feature_selection[0]
        assert rec.target_column == "feature_a"
        assert rec.action == "drop_multicollinear"
        assert rec.parameters["correlated_with"] == "feature_b"
        assert rec.parameters["correlation"] == 0.85

    def test_adds_drop_weak_to_gold(self):
        registry = RecommendationRegistry()
        registry.init_gold("churned")
        registry.add_gold_drop_weak(
            column="weak_feature", effect_size=0.05, correlation=0.02,
            rationale="Negligible predictive power", source_notebook="04"
        )
        assert len(registry.gold.feature_selection) == 1
        rec = registry.gold.feature_selection[0]
        assert rec.target_column == "weak_feature"
        assert rec.action == "drop_weak"
        assert rec.parameters["effect_size"] == 0.05
        assert rec.parameters["correlation"] == 0.02

    def test_drop_multicollinear_threads_slice_metadata_when_provided(self):
        """NB05 slice plumbing — slice_date/slice_strategy/slice_row_count are
        additive kwargs per docs/nb05_time_slice_relationship_plan.md §3.5."""
        registry = RecommendationRegistry()
        registry.init_gold("churned")
        registry.add_gold_drop_multicollinear(
            column="f_a", correlated_with="f_b", correlation=0.97,
            rationale="Pearson 0.97 at penultimate slice",
            source_notebook="05_relationship_analysis",
            slice_date="2026-03-01", slice_strategy="penultimate", slice_row_count=5000,
        )
        rec = registry.gold.feature_selection[-1]
        assert rec.parameters["slice_date"] == "2026-03-01"
        assert rec.parameters["slice_strategy"] == "penultimate"
        assert rec.parameters["slice_row_count"] == 5000

    def test_drop_multicollinear_slice_metadata_omitted_by_default(self):
        """Existing call sites that pass no slice kwargs must keep emitting
        a recommendation with NO slice_* keys in its parameters dict."""
        registry = RecommendationRegistry()
        registry.init_gold("churned")
        registry.add_gold_drop_multicollinear(
            column="f_a", correlated_with="f_b", correlation=0.85,
            rationale="High correlation", source_notebook="04",
        )
        rec = registry.gold.feature_selection[-1]
        assert "slice_date" not in rec.parameters
        assert "slice_strategy" not in rec.parameters
        assert "slice_row_count" not in rec.parameters

    def test_drop_weak_threads_slice_metadata_when_provided(self):
        registry = RecommendationRegistry()
        registry.init_gold("churned")
        registry.add_gold_drop_weak(
            column="weak_f", effect_size=0.05, correlation=0.02,
            rationale="Negligible at penultimate slice",
            source_notebook="05_relationship_analysis",
            slice_date="2026-03-01", slice_strategy="penultimate", slice_row_count=5000,
        )
        rec = registry.gold.feature_selection[-1]
        assert rec.parameters["slice_date"] == "2026-03-01"
        assert rec.parameters["slice_strategy"] == "penultimate"
        assert rec.parameters["slice_row_count"] == 5000

    def test_drop_weak_slice_metadata_omitted_for_variance_drops(self):
        """Variance-driven drops must NOT thread slice metadata (variance
        stays on full panel per §2); caller is responsible for withholding
        the kwargs. This test pins the no-op default behavior."""
        registry = RecommendationRegistry()
        registry.init_gold("churned")
        registry.add_gold_drop_weak(
            column="const_f", effect_size=0.0, correlation=0.0,
            rationale="zero variance", source_notebook="05_relationship_analysis",
        )
        rec = registry.gold.feature_selection[-1]
        assert "slice_date" not in rec.parameters
        assert "slice_strategy" not in rec.parameters
        assert "slice_row_count" not in rec.parameters

    def test_slice_metadata_roundtrips_via_to_dict_from_dict(self):
        """Step 9 — generator plumbing. slice_* keys in the opaque parameters
        dict must survive the to_dict / from_dict serialization loop so that
        pipeline generators reading the YAML recommendations file see them."""
        registry = RecommendationRegistry()
        registry.init_gold("churned")
        registry.add_gold_drop_multicollinear(
            column="f_a", correlated_with="f_b", correlation=0.97,
            rationale="0.97 at penultimate slice",
            source_notebook="05_relationship_analysis",
            slice_date="2026-03-01", slice_strategy="penultimate", slice_row_count=5000,
        )
        registry.add_gold_drop_weak(
            column="weak_f", effect_size=0.05, correlation=0.03,
            rationale="small effect at slice",
            source_notebook="05_relationship_analysis",
            slice_date="2026-03-01", slice_strategy="penultimate", slice_row_count=5000,
        )

        roundtripped = RecommendationRegistry.from_dict(registry.to_dict())
        recs = {r.target_column: r for r in roundtripped.gold.feature_selection}
        assert recs["f_a"].parameters["slice_date"] == "2026-03-01"
        assert recs["f_a"].parameters["slice_strategy"] == "penultimate"
        assert recs["f_a"].parameters["slice_row_count"] == 5000
        assert recs["weak_f"].parameters["slice_date"] == "2026-03-01"
        assert recs["weak_f"].parameters["slice_row_count"] == 5000

    def test_adds_prioritize_feature_to_gold(self):
        registry = RecommendationRegistry()
        registry.init_gold("churned")
        registry.add_gold_prioritize_feature(
            column="strong_feature", effect_size=2.5, correlation=0.72,
            rationale="Large effect size", source_notebook="04"
        )
        assert len(registry.gold.feature_selection) == 1
        rec = registry.gold.feature_selection[0]
        assert rec.target_column == "strong_feature"
        assert rec.action == "prioritize"
        assert rec.parameters["effect_size"] == 2.5

    def test_adds_drop_l1_zero_to_gold(self):
        registry = RecommendationRegistry()
        registry.init_gold("churned")
        registry.add_gold_drop_l1_zero(
            column="noise_feature", l1_coefficient=0.0,
            rationale="L1 zero coefficient", source_notebook="08"
        )
        assert len(registry.gold.feature_selection) == 1
        rec = registry.gold.feature_selection[0]
        assert rec.target_column == "noise_feature"
        assert rec.action == "drop_l1_zero"
        assert rec.parameters["l1_coefficient"] == 0.0

    def test_adds_drop_availability_to_gold(self):
        registry = RecommendationRegistry()
        registry.init_gold("churned")
        registry.add_gold_drop_availability(
            column="new_feature", issue_type="new_tracking", coverage_pct=42.5,
            rationale="Insufficient temporal coverage", source_notebook="08"
        )
        assert len(registry.gold.feature_selection) == 1
        rec = registry.gold.feature_selection[0]
        assert rec.target_column == "new_feature"
        assert rec.action == "drop_availability"
        assert rec.parameters["issue_type"] == "new_tracking"
        assert rec.parameters["coverage_pct"] == 42.5

    def test_adds_drop_zero_variance_to_gold(self):
        registry = RecommendationRegistry()
        registry.init_gold("churned")
        registry.add_gold_drop_zero_variance(
            column="constant_col", variance=0.0,
            rationale="Zero variance in training data", source_notebook="08"
        )
        assert len(registry.gold.feature_selection) == 1
        rec = registry.gold.feature_selection[0]
        assert rec.target_column == "constant_col"
        assert rec.action == "drop_zero_variance"
        assert rec.parameters["variance"] == 0.0

    def test_adds_drop_chi_squared_to_gold(self):
        registry = RecommendationRegistry()
        registry.init_gold("churned")
        registry.add_gold_drop_chi_squared(
            column="low_chi_feature", rationale="chi_squared below threshold",
            source_notebook="08"
        )
        assert len(registry.gold.feature_selection) == 1
        rec = registry.gold.feature_selection[0]
        assert rec.target_column == "low_chi_feature"
        assert rec.action == "drop_chi_squared"

    def test_adds_drop_gbdt_importance_to_gold(self):
        registry = RecommendationRegistry()
        registry.init_gold("churned")
        registry.add_gold_drop_gbdt_importance(
            column="low_gain_feature", importance=0.5,
            rationale="gbdt_importance below top-300", source_notebook="08"
        )
        assert len(registry.gold.feature_selection) == 1
        rec = registry.gold.feature_selection[0]
        assert rec.target_column == "low_gain_feature"
        assert rec.action == "drop_gbdt_importance"
        assert rec.parameters["importance"] == 0.5

    def test_adds_drop_rescue_consensus_to_gold(self):
        registry = RecommendationRegistry()
        registry.init_gold("churned")
        registry.add_gold_drop_rescue_consensus(
            column="account_events_7d_mean",
            chi_rank=742, chi_score=0.8,
            l1_coefficient=0.0, gbdt_total_gain=0.0,
            slice_date="2026-03-01", slice_strategy="penultimate", slice_row_count=5000,
            rationale="Dropped by all enabled selectors; chi-squared rank 742, "
                      "L1 coef 0, GBDT gain 0",
            source_notebook="08_baseline_experiments",
        )
        assert len(registry.gold.feature_selection) == 1
        rec = registry.gold.feature_selection[0]
        assert rec.target_column == "account_events_7d_mean"
        assert rec.action == "drop_rescue_consensus"
        params = rec.parameters
        assert params["chi_squared_rank"] == 742
        assert params["chi_squared_score"] == 0.8
        assert params["l1_coefficient"] == 0.0
        assert params["gbdt_total_gain"] == 0.0
        assert params["slice_date"] == "2026-03-01"
        assert params["slice_strategy"] == "penultimate"
        assert params["slice_row_count"] == 5000
        assert params["l1_considered"] is True
        assert params["gbdt_considered"] is True

    def test_feature_selection_in_all_recommendations(self):
        registry = RecommendationRegistry()
        registry.init_gold("churned")
        registry.add_gold_drop_multicollinear("a", "b", 0.9, "", "")
        registry.add_gold_drop_weak("c", 0.01, 0.01, "", "")
        registry.add_gold_encoding("category", "one_hot", "", "")
        assert len(registry.all_recommendations) == 3

    def test_serializes_new_drop_types_roundtrip(self):
        registry = RecommendationRegistry()
        registry.init_gold("churned")
        registry.add_gold_drop_availability("feat_a", "retired_tracking", 30.0, "retired", "08")
        registry.add_gold_drop_zero_variance("feat_b", 0.0, "constant", "08")
        d = registry.to_dict()
        restored = RecommendationRegistry.from_dict(d)
        actions = {r.target_column: r.action for r in restored.gold.feature_selection}
        assert actions == {"feat_a": "drop_availability", "feat_b": "drop_zero_variance"}
        assert restored.gold.feature_selection[0].parameters["issue_type"] == "retired_tracking"
        assert restored.gold.feature_selection[1].parameters["variance"] == 0.0

    def test_serializes_feature_selection(self):
        registry = RecommendationRegistry()
        registry.init_gold("churned")
        registry.add_gold_drop_multicollinear("a", "b", 0.85, "correlated", "04")
        d = registry.to_dict()
        assert len(d["gold"]["feature_selection"]) == 1
        assert d["gold"]["feature_selection"][0]["action"] == "drop_multicollinear"

    def test_deserializes_feature_selection(self):
        data = {
            "gold": {
                "target_column": "churned",
                "encoding": [],
                "scaling": [],
                "transformations": [],
                "feature_selection": [{
                    "id": "gold_feature_selection_a",
                    "layer": "gold",
                    "category": "feature_selection",
                    "action": "drop_multicollinear",
                    "target_column": "a",
                    "parameters": {"correlated_with": "b", "correlation": 0.85},
                    "rationale": "correlated",
                    "source_notebook": "04",
                    "priority": 1,
                    "dependencies": []
                }]
            }
        }
        registry = RecommendationRegistry.from_dict(data)
        assert len(registry.gold.feature_selection) == 1
        assert registry.gold.feature_selection[0].action == "drop_multicollinear"


class TestSilverInteractionRecommendations:
    def test_adds_ratio_feature_to_silver(self):
        registry = RecommendationRegistry()
        registry.init_silver("customer_id")
        registry.add_silver_ratio(
            column="open_to_click_ratio", numerator="open_rate", denominator="click_rate",
            rationale="Capture relative engagement", source_notebook="04"
        )
        assert len(registry.silver.derived_columns) == 1
        rec = registry.silver.derived_columns[0]
        assert rec.target_column == "open_to_click_ratio"
        assert rec.parameters["feature_type"] == "ratio"
        assert rec.parameters["numerator"] == "open_rate"
        assert rec.parameters["denominator"] == "click_rate"

    def test_adds_interaction_feature_to_silver(self):
        registry = RecommendationRegistry()
        registry.init_silver("customer_id")
        registry.add_silver_interaction(
            column="tenure_x_orders", features=["tenure", "order_count"],
            rationale="Capture interaction effect", source_notebook="04"
        )
        assert len(registry.silver.derived_columns) == 1
        rec = registry.silver.derived_columns[0]
        assert rec.target_column == "tenure_x_orders"
        assert rec.parameters["feature_type"] == "interaction"
        assert rec.parameters["features"] == ["tenure", "order_count"]

    def test_ratio_and_interaction_in_all_recommendations(self):
        registry = RecommendationRegistry()
        registry.init_silver("customer_id")
        registry.add_silver_ratio("ratio_1", "a", "b", "", "")
        registry.add_silver_interaction("interact_1", ["c", "d"], "", "")
        registry.add_silver_derived("derived_1", "expr", "recency", "", "")
        assert len(registry.all_recommendations) == 3


class TestSilverTemporalConfigRecommendations:
    def test_adds_temporal_config_to_silver(self):
        registry = RecommendationRegistry()
        registry.init_silver("customer_id", time_column="event_date")
        registry.add_silver_temporal_config(
            source_dataset="transactions",
            columns=["amount", "quantity"],
            lag_windows=4,
            lag_window_days=30,
            aggregations=["sum", "mean", "count"],
            feature_groups=["lagged_windows", "velocity", "recency"],
            rationale="Temporal features for transaction data",
            source_notebook="05"
        )
        assert len(registry.silver.aggregations) == 1
        rec = registry.silver.aggregations[0]
        assert rec.target_column == "transactions"
        assert rec.parameters["columns"] == ["amount", "quantity"]
        assert rec.parameters["lag_windows"] == 4
        assert rec.parameters["feature_groups"] == ["lagged_windows", "velocity", "recency"]

    def test_temporal_config_with_all_feature_groups(self):
        registry = RecommendationRegistry()
        registry.init_silver("customer_id")
        registry.add_silver_temporal_config(
            source_dataset="events",
            columns=["value"],
            lag_windows=3,
            lag_window_days=7,
            aggregations=["sum"],
            feature_groups=["lagged_windows", "velocity", "acceleration", "lifecycle",
                           "recency", "regularity", "cohort_comparison"],
            rationale="Full temporal analysis",
            source_notebook="05"
        )
        rec = registry.silver.aggregations[0]
        assert len(rec.parameters["feature_groups"]) == 7

    def test_temporal_config_in_all_recommendations(self):
        registry = RecommendationRegistry()
        registry.init_silver("customer_id")
        registry.add_silver_temporal_config("events", ["a"], 4, 30, ["sum"], ["lagged_windows"], "", "")
        registry.add_silver_derived("derived", "expr", "recency", "", "")
        assert len(registry.all_recommendations) == 2


class TestBronzeSegmentationRecommendations:
    def test_adds_segmentation_strategy_unified(self):
        registry = RecommendationRegistry()
        registry.init_bronze("data.csv")
        registry.add_bronze_segmentation_strategy(
            strategy="unified_model",
            confidence=0.85,
            n_segments=1,
            silhouette_score=-0.05,
            rationale="Weak clustering structure, unified model recommended",
            source_notebook="05"
        )
        assert len(registry.bronze.modeling_strategy) == 1
        rec = registry.bronze.modeling_strategy[0]
        assert rec.action == "unified_model"
        assert rec.parameters["confidence"] == 0.85
        assert rec.parameters["silhouette_score"] == -0.05

    def test_adds_segmentation_strategy_segmented(self):
        registry = RecommendationRegistry()
        registry.init_bronze("data.csv")
        registry.add_bronze_segmentation_strategy(
            strategy="segmented_model",
            confidence=0.92,
            n_segments=3,
            silhouette_score=0.45,
            rationale="Strong clustering structure supports segmented modeling",
            source_notebook="05"
        )
        rec = registry.bronze.modeling_strategy[0]
        assert rec.action == "segmented_model"
        assert rec.parameters["n_segments"] == 3

    def test_segmentation_strategy_in_all_recommendations(self):
        registry = RecommendationRegistry()
        registry.init_bronze("data.csv")
        registry.add_bronze_segmentation_strategy("unified_model", 0.8, 1, 0.1, "", "")
        registry.add_bronze_null("age", "median", "", "")
        assert len(registry.all_recommendations) == 2

    def test_serializes_segmentation_strategy(self):
        registry = RecommendationRegistry()
        registry.init_bronze("data.csv")
        registry.add_bronze_segmentation_strategy("unified_model", 0.85, 1, 0.1, "weak clusters", "05")
        d = registry.to_dict()
        assert len(d["bronze"]["modeling_strategy"]) == 1
        assert d["bronze"]["modeling_strategy"][0]["action"] == "unified_model"

    def test_deserializes_segmentation_strategy(self):
        data = {
            "bronze": {
                "source_file": "data.csv",
                "null_handling": [],
                "outlier_handling": [],
                "type_conversions": [],
                "deduplication": [],
                "filtering": [],
                "text_processing": [],
                "modeling_strategy": [{
                    "id": "bronze_segmentation_target",
                    "layer": "bronze",
                    "category": "segmentation",
                    "action": "segmented_model",
                    "target_column": "target",
                    "parameters": {"confidence": 0.9, "n_segments": 3, "silhouette_score": 0.5},
                    "rationale": "strong clusters",
                    "source_notebook": "05",
                    "priority": 1,
                    "dependencies": []
                }]
            }
        }
        registry = RecommendationRegistry.from_dict(data)
        assert len(registry.bronze.modeling_strategy) == 1
        assert registry.bronze.modeling_strategy[0].action == "segmented_model"


class TestBronzeFeatureCapacityRecommendations:
    def test_adds_feature_capacity_to_bronze(self):
        registry = RecommendationRegistry()
        registry.init_bronze("data.csv")
        registry.add_bronze_feature_capacity(
            epv=15.5,
            capacity_status="adequate",
            recommended_features=25,
            current_features=18,
            rationale="Sufficient data for current feature set",
            source_notebook="06"
        )
        assert len(registry.bronze.modeling_strategy) == 1
        rec = registry.bronze.modeling_strategy[0]
        assert rec.action == "feature_capacity"
        assert rec.parameters["epv"] == 15.5
        assert rec.parameters["capacity_status"] == "adequate"

    def test_feature_capacity_with_limited_status(self):
        registry = RecommendationRegistry()
        registry.init_bronze("data.csv")
        registry.add_bronze_feature_capacity(
            epv=7.2,
            capacity_status="limited",
            recommended_features=15,
            current_features=22,
            rationale="Consider reducing features or using regularization",
            source_notebook="06"
        )
        rec = registry.bronze.modeling_strategy[0]
        assert rec.parameters["capacity_status"] == "limited"
        assert rec.parameters["recommended_features"] == 15

    def test_feature_capacity_serialization(self):
        registry = RecommendationRegistry()
        registry.init_bronze("data.csv")
        registry.add_bronze_feature_capacity(12.0, "adequate", 20, 15, "good", "06")
        d = registry.to_dict()
        assert d["bronze"]["modeling_strategy"][0]["action"] == "feature_capacity"


class TestBronzeModelTypeRecommendations:
    def test_adds_model_type_to_bronze(self):
        registry = RecommendationRegistry()
        registry.init_bronze("data.csv")
        registry.add_bronze_model_type(
            model_type="tree_based",
            max_features_linear=10,
            max_features_regularized=25,
            max_features_tree=50,
            rationale="Sufficient data for tree-based models",
            source_notebook="06"
        )
        assert len(registry.bronze.modeling_strategy) == 1
        rec = registry.bronze.modeling_strategy[0]
        assert rec.action == "tree_based"
        assert rec.parameters["max_features_tree"] == 50

    def test_model_type_regularized(self):
        registry = RecommendationRegistry()
        registry.init_bronze("data.csv")
        registry.add_bronze_model_type(
            model_type="regularized_linear",
            max_features_linear=8,
            max_features_regularized=20,
            max_features_tree=40,
            rationale="Use L1/L2 regularization",
            source_notebook="06"
        )
        rec = registry.bronze.modeling_strategy[0]
        assert rec.action == "regularized_linear"

    def test_model_type_in_all_recommendations(self):
        registry = RecommendationRegistry()
        registry.init_bronze("data.csv")
        registry.add_bronze_model_type("tree_based", 10, 25, 50, "", "")
        registry.add_bronze_feature_capacity(15.0, "adequate", 20, 15, "", "")
        assert len(registry.all_recommendations) == 2

    def test_model_type_serialization(self):
        registry = RecommendationRegistry()
        registry.init_bronze("data.csv")
        registry.add_bronze_model_type("linear", 10, 20, 40, "simple model", "06")
        d = registry.to_dict()
        assert d["bronze"]["modeling_strategy"][0]["action"] == "linear"


class TestSilverDerivedFeaturesFromNotebook06:
    def test_adds_tenure_feature(self):
        registry = RecommendationRegistry()
        registry.init_silver("customer_id")
        registry.add_silver_derived(
            column="tenure_days",
            expression="reference_date - created_date",
            feature_type="recency",
            rationale="Customer tenure in days",
            source_notebook="06"
        )
        assert len(registry.silver.derived_columns) == 1
        rec = registry.silver.derived_columns[0]
        assert rec.target_column == "tenure_days"
        assert rec.parameters["feature_type"] == "recency"

    def test_adds_engagement_score(self):
        registry = RecommendationRegistry()
        registry.init_silver("customer_id")
        registry.add_silver_composite(
            column="email_engagement_score",
            columns=["open_rate", "click_rate"],
            rationale="Weighted email engagement",
            source_notebook="06"
        )
        rec = registry.silver.derived_columns[0]
        assert rec.target_column == "email_engagement_score"
        assert rec.action == "composite"
        assert rec.parameters["columns"] == ["open_rate", "click_rate"]

    def test_adds_service_adoption_score(self):
        registry = RecommendationRegistry()
        registry.init_silver("customer_id")
        registry.add_silver_composite(
            column="service_adoption_score",
            columns=["paperless", "refill", "doorstep"],
            rationale="Sum of service flags",
            source_notebook="06"
        )
        rec = registry.silver.derived_columns[0]
        assert rec.parameters["feature_type"] == "composite"
        assert rec.parameters["columns"] == ["paperless", "refill", "doorstep"]


class TestSilverCompositeRecommendations:
    def test_adds_composite_to_silver(self):
        registry = RecommendationRegistry()
        registry.init_silver("customer_id")
        registry.add_silver_composite(
            column="score", columns=["a", "b", "c"],
            rationale="mean score", source_notebook="06"
        )
        assert len(registry.silver.derived_columns) == 1
        rec = registry.silver.derived_columns[0]
        assert rec.target_column == "score"
        assert rec.action == "composite"
        assert rec.parameters["columns"] == ["a", "b", "c"]
        assert rec.parameters["feature_type"] == "composite"

    def test_composite_expression_reflects_columns(self):
        registry = RecommendationRegistry()
        registry.init_silver("customer_id")
        registry.add_silver_composite("s", ["x", "y"], "", "06")
        rec = registry.silver.derived_columns[0]
        assert rec.parameters["expression"] == "mean(x, y)"

    def test_composite_in_all_recommendations(self):
        registry = RecommendationRegistry()
        registry.init_silver("customer_id")
        registry.add_silver_composite("s", ["a", "b"], "", "06")
        registry.add_silver_ratio("r", "a", "b", "", "06")
        assert len(registry.all_recommendations) == 2

    def test_composite_round_trips_through_serialization(self, tmp_path):
        registry = RecommendationRegistry()
        registry.init_silver("customer_id")
        registry.add_silver_composite("score", ["a", "b"], "test", "06")
        path = str(tmp_path / "recs.yaml")
        registry.save(path)
        loaded = RecommendationRegistry.load(path)
        rec = loaded.silver.derived_columns[0]
        assert rec.action == "composite"
        assert rec.parameters["columns"] == ["a", "b"]


class TestRecommendationsHash:
    def test_empty_registry_returns_empty_hash(self):
        registry = RecommendationRegistry()
        hash_val = registry.compute_recommendations_hash()
        assert len(hash_val) == 8
        assert hash_val == registry.compute_recommendations_hash()

    def test_gold_recommendations_produce_hash(self):
        registry = RecommendationRegistry()
        registry.init_gold("churned")
        registry.add_gold_encoding("contract", "one_hot", "low cardinality", "06")
        registry.add_gold_scaling("revenue", "standard", "normalize", "06")
        hash_val = registry.compute_recommendations_hash()
        assert len(hash_val) == 8
        assert all(c in "0123456789abcdef" for c in hash_val)

    def test_same_recommendations_produce_same_hash(self):
        registry1 = RecommendationRegistry()
        registry1.init_gold("churned")
        registry1.add_gold_encoding("contract", "one_hot", "reason", "06")
        registry1.add_gold_scaling("revenue", "standard", "reason", "06")

        registry2 = RecommendationRegistry()
        registry2.init_gold("churned")
        registry2.add_gold_encoding("contract", "one_hot", "reason", "06")
        registry2.add_gold_scaling("revenue", "standard", "reason", "06")

        assert registry1.compute_recommendations_hash() == registry2.compute_recommendations_hash()

    def test_different_recommendations_produce_different_hash(self):
        registry1 = RecommendationRegistry()
        registry1.init_gold("churned")
        registry1.add_gold_encoding("contract", "one_hot", "", "06")

        registry2 = RecommendationRegistry()
        registry2.init_gold("churned")
        registry2.add_gold_encoding("contract", "label", "", "06")

        assert registry1.compute_recommendations_hash() != registry2.compute_recommendations_hash()

    def test_order_of_same_recommendations_produces_same_hash(self):
        registry1 = RecommendationRegistry()
        registry1.init_gold("churned")
        registry1.add_gold_encoding("a_column", "one_hot", "", "")
        registry1.add_gold_encoding("z_column", "label", "", "")

        registry2 = RecommendationRegistry()
        registry2.init_gold("churned")
        registry2.add_gold_encoding("z_column", "label", "", "")
        registry2.add_gold_encoding("a_column", "one_hot", "", "")

        assert registry1.compute_recommendations_hash() == registry2.compute_recommendations_hash()

    def test_hash_includes_transformations(self):
        registry = RecommendationRegistry()
        registry.init_gold("churned")
        registry.add_gold_transformation("revenue", "log_transform", {"offset": 1}, "", "")
        hash1 = registry.compute_recommendations_hash()

        registry.add_gold_transformation("amount", "sqrt", {}, "", "")
        hash2 = registry.compute_recommendations_hash()

        assert hash1 != hash2

    def test_hash_includes_feature_selection(self):
        registry = RecommendationRegistry()
        registry.init_gold("churned")
        registry.add_gold_drop_multicollinear("col_a", "col_b", 0.9, "", "")
        hash1 = registry.compute_recommendations_hash()

        registry.add_gold_drop_weak("weak_col", 0.01, 0.02, "", "")
        hash2 = registry.compute_recommendations_hash()

        assert hash1 != hash2

    def test_custom_hash_length(self):
        registry = RecommendationRegistry()
        registry.init_gold("churned")
        registry.add_gold_encoding("contract", "one_hot", "", "")

        hash_4 = registry.compute_recommendations_hash(length=4)
        hash_12 = registry.compute_recommendations_hash(length=12)

        assert len(hash_4) == 4
        assert len(hash_12) == 12
        assert hash_12.startswith(hash_4)

    def test_hash_excludes_non_gold_recommendations(self):
        registry1 = RecommendationRegistry()
        registry1.init_bronze("data.csv")
        registry1.init_gold("churned")
        registry1.add_bronze_null("age", "median", "", "")
        registry1.add_gold_encoding("contract", "one_hot", "", "")

        registry2 = RecommendationRegistry()
        registry2.init_gold("churned")
        registry2.add_gold_encoding("contract", "one_hot", "", "")

        assert registry1.compute_recommendations_hash() == registry2.compute_recommendations_hash()

    def test_hash_is_deterministic(self):
        registry = RecommendationRegistry()
        registry.init_gold("churned")
        registry.add_gold_encoding("contract", "one_hot", "low card", "06")
        registry.add_gold_scaling("revenue", "minmax", "bounded", "06")
        registry.add_gold_transformation("amount", "log_transform", {"base": "e"}, "skewed", "06")
        registry.add_gold_drop_multicollinear("col_a", "col_b", 0.95, "redundant", "04")

        hashes = [registry.compute_recommendations_hash() for _ in range(10)]
        assert len(set(hashes)) == 1


class TestRecommendationRegistrySaveLoad:
    def test_save_creates_yaml_file(self, tmp_path):
        registry = RecommendationRegistry()
        registry.init_bronze("data.csv")
        registry.add_bronze_null("age", "median", "5% nulls", "03")
        path = str(tmp_path / "recs.yaml")
        registry.save(path)
        assert Path(path).exists()

    def test_save_writes_valid_yaml(self, tmp_path):
        import yaml

        registry = RecommendationRegistry()
        registry.init_bronze("data.csv")
        registry.add_bronze_null("age", "median", "5% nulls", "03")
        path = str(tmp_path / "recs.yaml")
        registry.save(path)
        with open(path) as f:
            data = yaml.safe_load(f)
        assert "bronze" in data
        assert data["bronze"]["source_file"] == "data.csv"

    def test_load_roundtrips_with_save(self, tmp_path):
        registry = RecommendationRegistry()
        registry.init_bronze("data.csv")
        registry.init_silver("customer_id", time_column="event_date")
        registry.init_gold("churned")
        registry.add_bronze_null("age", "median", "5% nulls", "03")
        registry.add_silver_aggregation("revenue", "sum", ["7d"], "trend", "04")
        registry.add_gold_encoding("contract", "one_hot", "low card", "06")

        path = str(tmp_path / "recs.yaml")
        registry.save(path)
        loaded = RecommendationRegistry.load(path)

        assert len(loaded.all_recommendations) == len(registry.all_recommendations)
        assert loaded.bronze.source_file == "data.csv"
        assert loaded.silver.entity_column == "customer_id"
        assert loaded.gold.target_column == "churned"

    def test_load_restores_all_layers(self, tmp_path):
        registry = RecommendationRegistry()
        registry.init_bronze("data.csv")
        registry.init_silver("customer_id")
        registry.init_gold("churned")
        registry.add_bronze_null("a", "median", "", "")
        registry.add_silver_derived("b", "expr", "recency", "", "")
        registry.add_gold_scaling("c", "standard", "", "")

        path = str(tmp_path / "recs.yaml")
        registry.save(path)
        loaded = RecommendationRegistry.load(path)

        assert loaded.bronze is not None
        assert loaded.silver is not None
        assert loaded.gold is not None
        assert len(loaded.get_by_layer("bronze")) == 1
        assert len(loaded.get_by_layer("silver")) == 1
        assert len(loaded.get_by_layer("gold")) == 1

    def test_save_matches_manual_yaml_dump(self, tmp_path):
        import yaml

        registry = RecommendationRegistry()
        registry.init_bronze("data.csv")
        registry.add_bronze_null("age", "median", "5% nulls", "03")

        # Manual dump
        manual_path = tmp_path / "manual.yaml"
        with open(manual_path, "w") as f:
            yaml.dump(registry.to_dict(), f, default_flow_style=False, sort_keys=False)

        # save() method
        save_path = str(tmp_path / "save.yaml")
        registry.save(save_path)

        with open(manual_path) as f:
            manual_content = f.read()
        with open(save_path) as f:
            save_content = f.read()

        assert manual_content == save_content


class TestRecommendationRegistryDecimalSafe:
    def test_decimal_parameters_roundtrip(self, tmp_path):
        import decimal


        registry = RecommendationRegistry()
        registry.init_bronze("data.csv")
        registry.add_bronze_null("age", "median", "5% nulls", "03")
        # Inject Decimal values (as Spark aggregations would produce)
        rec = registry.bronze.null_handling[0]
        rec.parameters["threshold"] = decimal.Decimal("0.95")
        rec.parameters["ratio"] = decimal.Decimal("1.23456")

        path = str(tmp_path / "recs.yaml")
        registry.save(path)

        with open(path) as f:
            content = f.read()
        assert "!!python" not in content
        assert "decimal" not in content.lower()

        loaded = RecommendationRegistry.load(path)
        params = loaded.bronze.null_handling[0].parameters
        assert params["threshold"] == 0.95
        assert params["ratio"] == 1.23456


class TestRecommendationRegistryPandasTimestampSafe:
    def test_pandas_timestamp_minority_class_roundtrip(self, tmp_path):
        import pandas as pd

        registry = RecommendationRegistry()
        registry.init_bronze("data.csv")
        registry.add_bronze_imbalance_strategy(
            "target", 5.0, pd.Timestamp("2024-01-01"),
            "oversample", "severe imbalance", "01",
        )

        path = str(tmp_path / "recs.yaml")
        registry.save(path)

        with open(path) as f:
            content = f.read()
        assert "!!python" not in content

        loaded = RecommendationRegistry.load(path)
        params = loaded.bronze.modeling_strategy[0].parameters
        assert isinstance(params["minority_class"], str)
        assert "2024-01-01" in params["minority_class"]


class TestRecommendationRegistryMerge:

    def test_merge_empty_list(self):
        result = RecommendationRegistry.merge([])
        assert result.sources == {}
        assert result.bronze is None
        assert result.silver is None
        assert result.gold is None

    def test_merge_single_registry(self):
        reg = RecommendationRegistry()
        reg.add_source("customers", "customers.csv")
        reg.init_silver("customer_id")
        result = RecommendationRegistry.merge([reg])
        assert "customers" in result.sources
        assert result.silver is not None

    def test_merge_combines_sources(self):
        reg1 = RecommendationRegistry()
        reg1.add_source("customers", "customers.csv")
        reg2 = RecommendationRegistry()
        reg2.add_source("events", "events.csv")
        result = RecommendationRegistry.merge([reg1, reg2])
        assert "customers" in result.sources
        assert "events" in result.sources

    def test_merge_preserves_silver_gold(self):
        reg1 = RecommendationRegistry()
        reg1.init_silver("customer_id")
        reg1.add_silver_derived("tenure", "expr", "recency", "", "")
        reg2 = RecommendationRegistry()
        reg2.add_source("events", "events.csv")
        result = RecommendationRegistry.merge([reg1, reg2])
        assert result.silver is not None
        assert len(result.silver.derived_columns) == 1
        assert result.gold is None

    def test_merge_prefers_registry_with_gold(self):
        reg1 = RecommendationRegistry()
        reg1.init_silver("customer_id")
        reg1.add_silver_derived("t1", "expr", "recency", "", "")

        reg2 = RecommendationRegistry()
        reg2.init_silver("customer_id")
        reg2.init_gold("churned")
        reg2.add_gold_encoding("contract", "one_hot", "", "")
        reg2.add_silver_derived("t2", "expr", "recency", "", "")

        result = RecommendationRegistry.merge([reg1, reg2])
        assert result.gold is not None
        assert result.gold.target_column == "churned"
        assert len(result.gold.encoding) == 1
        assert result.silver is not None
        assert len(result.silver.derived_columns) == 1
        assert result.silver.derived_columns[0].target_column == "t2"

    def test_merge_combines_fit_artifacts(self):
        reg1 = RecommendationRegistry()
        reg1.fit_artifacts["rec_1"] = "artifact_a"
        reg2 = RecommendationRegistry()
        reg2.fit_artifacts["rec_2"] = "artifact_b"
        reg2.fit_artifacts["rec_1"] = "artifact_a_override"
        result = RecommendationRegistry.merge([reg1, reg2])
        assert result.fit_artifacts["rec_2"] == "artifact_b"
        assert result.fit_artifacts["rec_1"] == "artifact_a_override"


class TestFeatureSelectionConfig:
    def test_set_config_stores_thresholds_and_features(self):
        reg = RecommendationRegistry()
        reg.init_gold("target")
        reg.set_feature_selection_config(0.01, 0.95, ["feat_a", "feat_b", "feat_c"])
        cfg = reg.gold.feature_selection_config
        assert cfg.variance_threshold == 0.01
        assert cfg.correlation_threshold == 0.95
        assert cfg.analyzed_features == ["feat_a", "feat_b", "feat_c"]

    def test_config_features_sorted(self):
        reg = RecommendationRegistry()
        reg.init_gold("target")
        reg.set_feature_selection_config(0.01, 0.95, ["z_col", "a_col", "m_col"])
        assert reg.gold.feature_selection_config.analyzed_features == ["a_col", "m_col", "z_col"]

    def test_config_none_by_default(self):
        reg = RecommendationRegistry()
        reg.init_gold("target")
        assert reg.gold.feature_selection_config is None

    def test_config_noop_without_gold(self):
        reg = RecommendationRegistry()
        reg.set_feature_selection_config(0.01, 0.95, ["a"])
        assert reg.gold is None

    def test_config_roundtrips_through_save_load(self, tmp_path):
        reg = RecommendationRegistry()
        reg.init_gold("target")
        reg.set_feature_selection_config(0.01, 0.95, ["feat_a", "feat_b"])
        path = tmp_path / "recs.yaml"
        reg.save(path)
        loaded = RecommendationRegistry.load(path)
        cfg = loaded.gold.feature_selection_config
        assert cfg.variance_threshold == 0.01
        assert cfg.correlation_threshold == 0.95
        assert cfg.analyzed_features == ["feat_a", "feat_b"]

    def test_config_not_in_hash(self):
        reg1 = RecommendationRegistry()
        reg1.init_gold("target")
        reg2 = RecommendationRegistry()
        reg2.init_gold("target")
        reg2.set_feature_selection_config(0.01, 0.95, ["a", "b", "c"])
        assert reg1.compute_recommendations_hash() == reg2.compute_recommendations_hash()

    def test_config_preserved_through_merge(self):
        reg = RecommendationRegistry()
        reg.init_gold("target")
        reg.set_feature_selection_config(0.01, 0.95, ["a", "b"])
        merged = RecommendationRegistry.merge([reg])
        assert merged.gold.feature_selection_config.variance_threshold == 0.01

    def test_config_is_typed_dataclass(self):
        from customer_retention.analysis.auto_explorer.layered_recommendations import FeatureSelectionConfig
        reg = RecommendationRegistry()
        reg.init_gold("target")
        reg.set_feature_selection_config(0.01, 0.95, ["a"])
        assert isinstance(reg.gold.feature_selection_config, FeatureSelectionConfig)

    def test_config_roundtrip_produces_dataclass(self, tmp_path):
        from customer_retention.analysis.auto_explorer.layered_recommendations import FeatureSelectionConfig
        reg = RecommendationRegistry()
        reg.init_gold("target")
        reg.set_feature_selection_config(0.01, 0.95, ["a", "b"])
        path = tmp_path / "recs.yaml"
        reg.save(path)
        loaded = RecommendationRegistry.load(path)
        assert isinstance(loaded.gold.feature_selection_config, FeatureSelectionConfig)


class TestLandingRecommendations:
    def test_init_landing_creates_empty_block(self):
        reg = RecommendationRegistry()
        assert reg.landing is None
        reg.init_landing()
        assert reg.landing is not None
        assert reg.landing.filters == []
        assert reg.landing.lifecycle_enrichments == []

    def test_add_landing_filter_auto_inits(self):
        reg = RecommendationRegistry()
        reg.add_landing_filter(
            dataset="request",
            predicate="ACCOUNT_ID IS NOT NULL AND CREATED_DATE IS NOT NULL",
            rationale="34.2% of request rows have NULL ACCOUNT_ID",
            source_notebook="NB00 § 0.4.6",
        )
        assert len(reg.landing.filters) == 1
        rec = reg.landing.filters[0]
        assert rec.layer == "landing"
        assert rec.category == "landing_filtering"
        assert rec.action == "filter"
        assert rec.target_column == "request"
        assert rec.parameters["dataset"] == "request"
        assert rec.parameters["predicate"] == "ACCOUNT_ID IS NOT NULL AND CREATED_DATE IS NOT NULL"

    def test_landing_filter_id_does_not_collide_with_bronze_filter(self):
        reg = RecommendationRegistry()
        reg.init_bronze("requests.csv")
        reg.add_bronze_filtering(
            column="amount", condition="non_negative", action="drop",
            rationale="neg values invalid", source_notebook="NB03",
        )
        reg.add_landing_filter(
            dataset="request", predicate="amount >= 0",
            rationale="parity with bronze", source_notebook="NB00",
        )
        landing_id = reg.landing.filters[0].id
        bronze_id = reg.bronze.filtering[0].id
        assert landing_id != bronze_id
        assert landing_id.startswith("landing_landing_filtering")
        assert bronze_id.startswith("bronze_filtering")

    def test_add_landing_lifecycle_enrichment_serializes_config(self):
        from customer_retention.stages.lifecycle.config import LifecycleEnrichmentConfig
        cfg = LifecycleEnrichmentConfig(
            enriched_view_name="subscription_events",
            parent_entity_key="account_id",
            sub_entity_key="subscription_id",
            valid_from_column="start_date",
            valid_to_columns=("end_date",),
        )
        reg = RecommendationRegistry()
        reg.add_landing_lifecycle_enrichment(
            dataset="subscription", config=cfg,
            rationale="doubled stream", source_notebook="NB00",
        )
        rec = reg.landing.lifecycle_enrichments[0]
        assert rec.category == "lifecycle_enrichment"
        assert rec.action == "enrich"
        assert rec.target_column == "subscription"
        assert rec.parameters["dataset"] == "subscription"
        assert rec.parameters["config"]["enriched_view_name"] == "subscription_events"
        assert rec.parameters["config"]["valid_to_columns"] == ["end_date"]

    def test_landing_in_all_recommendations(self):
        reg = RecommendationRegistry()
        reg.add_landing_filter(
            dataset="request", predicate="x IS NOT NULL",
            rationale="r", source_notebook="nb",
        )
        assert any(r.layer == "landing" for r in reg.all_recommendations)

    def test_get_by_layer_landing(self):
        reg = RecommendationRegistry()
        reg.add_landing_filter(
            dataset="request", predicate="x > 0",
            rationale="r", source_notebook="nb",
        )
        recs = reg.get_by_layer("landing")
        assert len(recs) == 1
        assert recs[0].target_column == "request"

    def test_landing_roundtrips_through_to_dict_from_dict(self):
        reg = RecommendationRegistry()
        reg.add_landing_filter(
            dataset="request", predicate="ACCOUNT_ID IS NOT NULL",
            rationale="r", source_notebook="NB00",
        )
        data = reg.to_dict()
        assert "landing" in data
        assert len(data["landing"]["filters"]) == 1
        assert data["landing"]["filters"][0]["parameters"]["predicate"] == "ACCOUNT_ID IS NOT NULL"
        restored = RecommendationRegistry.from_dict(data)
        assert restored.landing is not None
        assert len(restored.landing.filters) == 1
        assert restored.landing.filters[0].parameters["predicate"] == "ACCOUNT_ID IS NOT NULL"

    def test_landing_roundtrips_through_save_load(self, tmp_path):
        reg = RecommendationRegistry()
        reg.add_landing_filter(
            dataset="account", predicate="STATUS = 'active'",
            rationale="r", source_notebook="NB00",
        )
        path = tmp_path / "recs.yaml"
        reg.save(path)
        loaded = RecommendationRegistry.load(path)
        assert loaded.landing is not None
        assert loaded.landing.filters[0].target_column == "account"

    def test_from_dict_tolerates_missing_landing_key(self):
        data = {
            "bronze": {
                "source_file": "x.csv",
                "null_handling": [],
                "outlier_handling": [],
                "type_conversions": [],
                "deduplication": [],
                "filtering": [],
                "text_processing": [],
                "modeling_strategy": [],
            },
        }
        restored = RecommendationRegistry.from_dict(data)
        assert restored.landing is None

    def test_merge_preserves_landing_from_primary(self):
        primary = RecommendationRegistry()
        primary.init_gold("target")
        primary.add_landing_filter(
            dataset="request", predicate="x IS NOT NULL",
            rationale="r", source_notebook="nb",
        )
        secondary = RecommendationRegistry()
        merged = RecommendationRegistry.merge([secondary, primary])
        assert merged.landing is not None
        assert len(merged.landing.filters) == 1
        assert merged.landing.filters[0].target_column == "request"
