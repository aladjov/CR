import pytest


def _try_import():
    try:
        from customer_retention.stages.temporal.temporal_merger import (
            classify_silver_columns,
        )
    except ImportError:
        pytest.skip("classify_silver_columns not yet implemented (pre-fix state)")
    return classify_silver_columns


def _oracle(
    silver_columns,
    bronze_columns_by_dataset,
    base_source,
    separator="__",
    spine_columns=("entity_id", "as_of_date"),
):
    origin = {}
    merged_so_far = set(spine_columns) | set(bronze_columns_by_dataset[base_source])
    rename_history = {}
    sequence = [base_source] + [n for n in bronze_columns_by_dataset if n != base_source]
    for name in sequence:
        cols = set(bronze_columns_by_dataset[name])
        if name == base_source:
            for c in cols:
                origin[c] = name
            continue
        conflicts = cols & merged_so_far
        nonconflicts = cols - merged_so_far
        for c in conflicts:
            renamed = f"{name}{separator}{c}"
            rename_history[renamed] = name
            merged_so_far.add(renamed)
        for c in nonconflicts:
            origin[c] = name
            merged_so_far.add(c)
    for c in silver_columns:
        if c in spine_columns:
            origin[c] = "spine"
        elif c in rename_history:
            origin[c] = rename_history[c]
    return {c: origin[c] for c in silver_columns if c in origin}


class TestNoConflicts:
    def test_each_dataset_contributes_unprefixed_when_names_unique(self):
        fn = _try_import()
        bronze = {
            "account": ["ACCOUNT_ID", "REVENUE_MARKET_SEGMENT"],
            "contract": ["ACCOUNT_ID", "event_count_30d", "event_type_start_count_90d"],
            "subscription": ["ACCOUNT_ID", "NET_PRICE_sum_30d"],
        }
        silver = [
            "entity_id", "as_of_date",
            "REVENUE_MARKET_SEGMENT",
            "event_count_30d", "event_type_start_count_90d",
            "NET_PRICE_sum_30d",
        ]
        origin = fn(silver, bronze, base_source="account")
        assert origin["REVENUE_MARKET_SEGMENT"] == "account"
        assert origin["event_count_30d"] == "contract"
        assert origin["event_type_start_count_90d"] == "contract"
        assert origin["NET_PRICE_sum_30d"] == "subscription"
        assert origin["entity_id"] == "spine"
        assert origin["as_of_date"] == "spine"


class TestConflicts:
    def test_collision_with_base_is_prefixed(self):
        fn = _try_import()
        bronze = {
            "account": ["ACCOUNT_ID", "CREATED_DATE"],
            "contract": ["ACCOUNT_ID", "CREATED_DATE", "event_count_30d"],
        }
        silver = ["entity_id", "as_of_date", "CREATED_DATE",
                  "contract__CREATED_DATE", "event_count_30d"]
        origin = fn(silver, bronze, base_source="account")
        assert origin["CREATED_DATE"] == "account"
        assert origin["contract__CREATED_DATE"] == "contract"
        assert origin["event_count_30d"] == "contract"

    def test_collision_between_two_event_sources_prefixes_the_later_one(self):
        fn = _try_import()
        bronze = {
            "account": ["ACCOUNT_ID"],
            "contract": ["ACCOUNT_ID", "event_count_30d"],
            "subscription": ["ACCOUNT_ID", "event_count_30d"],
        }
        silver = ["entity_id", "as_of_date",
                  "event_count_30d", "subscription__event_count_30d"]
        origin = fn(silver, bronze, base_source="account")
        assert origin["event_count_30d"] == "contract"
        assert origin["subscription__event_count_30d"] == "subscription"


class TestSpineColumns:
    def test_join_keys_map_to_spine(self):
        fn = _try_import()
        bronze = {"account": ["ACCOUNT_ID"], "contract": ["ACCOUNT_ID", "event_count_30d"]}
        silver = ["entity_id", "as_of_date", "event_count_30d"]
        origin = fn(silver, bronze, base_source="account")
        assert origin["entity_id"] == "spine"
        assert origin["as_of_date"] == "spine"


class TestAmbiguousMapping:
    def test_raises_when_unprefixed_column_not_in_any_bronze(self):
        fn = _try_import()
        bronze = {"account": ["ACCOUNT_ID"], "contract": ["ACCOUNT_ID", "event_count_30d"]}
        silver = ["entity_id", "as_of_date", "mystery_feature"]
        with pytest.raises(ValueError, match="mystery_feature"):
            fn(silver, bronze, base_source="account")

    def test_raises_on_unknown_base_source(self):
        fn = _try_import()
        bronze = {"account": ["ACCOUNT_ID"], "contract": ["ACCOUNT_ID"]}
        silver = ["entity_id", "as_of_date"]
        with pytest.raises(KeyError, match="missing"):
            fn(silver, bronze, base_source="missing")


class TestCustomSeparator:
    def test_respects_configured_conflict_separator(self):
        fn = _try_import()
        bronze = {
            "account": ["ACCOUNT_ID", "STATUS"],
            "contract": ["ACCOUNT_ID", "STATUS"],
        }
        silver = ["entity_id", "as_of_date", "STATUS", "contract--STATUS"]
        origin = fn(silver, bronze, base_source="account", conflict_separator="--")
        assert origin["STATUS"] == "account"
        assert origin["contract--STATUS"] == "contract"


class TestContributionCoverage:
    def test_every_non_spine_dataset_contributes_at_least_one_column(self):
        fn = _try_import()
        bronze = {
            "account": ["ACCOUNT_ID", "REVENUE_MARKET_SEGMENT"],
            "contract": ["ACCOUNT_ID", "event_count_30d"],
            "case": ["ACCOUNT_ID", "case__only_feature"],
        }
        silver = ["entity_id", "as_of_date",
                  "REVENUE_MARKET_SEGMENT", "event_count_30d", "case__only_feature"]
        origin = fn(silver, bronze, base_source="account")
        contributions = {}
        for col, src in origin.items():
            if src == "spine":
                continue
            contributions[src] = contributions.get(src, 0) + 1
        for name in bronze:
            assert contributions.get(name, 0) >= 1, f"{name} contributed 0 columns"


class TestRunRegression:
    def test_reproduces_observed_run_shape_all_unprefixed(self):
        fn = _try_import()
        bronze = {
            "account": [
                "ACCOUNT_ID", "CREATED_DATE", "LAST_MODIFIED_DATE",
                "REVENUE_MARKET_SEGMENT", "ANNUAL_REVENUE_RANGE",
                "INDUSTRY_SEGMENT", "churned",
            ],
            "contract": [
                "ACCOUNT_ID", "event_count_30d", "event_count_90d",
                "event_count_365d", "event_count_all_time",
                "CLOSED_DATE_hour_mean_30d", "CLOSED_DATE_dow_sum_30d",
            ],
            "subscription": [
                "ACCOUNT_ID", "NET_PRICE_sum_30d", "NET_PRICE_mean_90d",
            ],
            "case": [
                "ACCOUNT_ID", "case_event_count_30d",
            ],
        }
        silver = [c for cols in bronze.values() for c in cols if c != "ACCOUNT_ID"]
        silver = ["entity_id", "as_of_date"] + sorted(set(silver))
        origin = fn(silver, bronze, base_source="account")
        expected = _oracle(silver, bronze, base_source="account")
        assert origin == expected
        non_spine = [c for c in silver if origin[c] != "spine"]
        prefixed = [c for c in non_spine if "__" in c]
        unprefixed = [c for c in non_spine if "__" not in c]
        assert len(unprefixed) > 0
        assert all("__" not in c for c in unprefixed)
        assert all("__" in c for c in prefixed)
        contributions = {}
        for c in non_spine:
            contributions[origin[c]] = contributions.get(origin[c], 0) + 1
        for name in bronze:
            assert contributions.get(name, 0) >= 1, f"{name} contributed 0 columns"
