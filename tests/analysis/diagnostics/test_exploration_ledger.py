from pathlib import Path

import pytest

from customer_retention.analysis.auto_explorer.findings import ExplorationFindings
from customer_retention.analysis.auto_explorer.layered_recommendations import (
    GoldRecommendations,
    LayeredRecommendation,
    RecommendationRegistry,
)
from customer_retention.analysis.diagnostics.exploration_ledger import (
    attribute_feature_to_source,
    build_column_drop_register,
    build_dataset_column_ledger,
    build_nb05_drop_ledger,
    build_nb08_per_source_survival,
    build_nb08_top30_attribution,
    build_recommendation_audit,
    build_recommendation_summary_by_action,
    write_diagnostic_yaml,
)


def _rec(action: str, target: str) -> LayeredRecommendation:
    return LayeredRecommendation(
        id=f"{action}-{target}",
        layer="gold",
        category="transformation",
        action=action,
        target_column=target,
        parameters={},
        rationale="",
        source_notebook="nb04",
    )


@pytest.fixture
def simple_source_columns():
    return {
        "case": ["CASE_TYPE", "RESOLUTION_TARGET_DATE_TIME", "CASE_ID"],
        "subscription": ["NET_PRICE", "STATUS"],
        "implementation_project": ["ACTIVATED_DATE"],
    }


class TestAttributeFeatureToSource:
    def test_raw_column_direct_match(self, simple_source_columns):
        source, base, derivation = attribute_feature_to_source(
            "CASE_TYPE", simple_source_columns
        )
        assert source == "case"
        assert base == "CASE_TYPE"
        assert derivation == "raw"

    def test_aggregated_feature(self, simple_source_columns):
        source, base, derivation = attribute_feature_to_source(
            "NET_PRICE_sum_180d", simple_source_columns
        )
        assert source == "subscription"
        assert base == "NET_PRICE"
        assert derivation == "sum_180d"

    def test_is_zero_flag(self, simple_source_columns):
        source, base, derivation = attribute_feature_to_source(
            "NET_PRICE_count_30d_is_zero", simple_source_columns
        )
        assert source == "subscription"
        assert base == "NET_PRICE"
        assert derivation == "count_30d_is_zero"

    def test_lag_prefix(self, simple_source_columns):
        source, base, derivation = attribute_feature_to_source(
            "lag2_NET_PRICE_sum_180d", simple_source_columns
        )
        assert source == "subscription"
        assert base == "NET_PRICE"

    def test_unknown_feature_returns_unknown(self, simple_source_columns):
        source, base, derivation = attribute_feature_to_source(
            "totally_made_up_feature", simple_source_columns
        )
        assert source == "unknown"
        assert base == "totally_made_up_feature"
        assert derivation == "raw"

    def test_longest_prefix_wins(self):
        cols = {"case": ["CASE", "CASE_TYPE"]}
        source, base, derivation = attribute_feature_to_source(
            "CASE_TYPE_sum_30d", cols
        )
        assert base == "CASE_TYPE"


class TestBuildRecommendationAudit:
    def test_audit_marks_gated_opt_in_zero_inflation(self, simple_source_columns):
        registry = RecommendationRegistry()
        registry.gold = GoldRecommendations(target_column="target")
        registry.gold.transformations = [
            _rec("zero_inflation_handling", "CASE_TYPE_count_30d"),
            _rec("zero_inflation_handling", "NET_PRICE_sum_30d"),
            _rec("log_transform", "NET_PRICE_mean_30d"),
        ]
        opt_in = {"subscription": ["NET_PRICE"]}
        df = build_recommendation_audit(
            registry,
            opt_in=opt_in,
            excluded_leaking={},
            source_columns=simple_source_columns,
        )
        rows = df.to_dict(orient="records")
        by_col = {r["base_column"] + "_" + r["action"]: r["status"] for r in rows}
        assert by_col["CASE_TYPE_zero_inflation_handling"] == "gated_opt_in"
        assert by_col["NET_PRICE_zero_inflation_handling"] == "applied"
        assert by_col["NET_PRICE_log_transform"] == "applied"

    def test_audit_marks_gated_excluded_leaking(self, simple_source_columns):
        registry = RecommendationRegistry()
        registry.gold = GoldRecommendations(target_column="target")
        registry.gold.transformations = [_rec("log_transform", "ACTIVATED_DATE_days")]
        df = build_recommendation_audit(
            registry,
            opt_in={},
            excluded_leaking={"implementation_project": ["ACTIVATED_DATE"]},
            source_columns=simple_source_columns,
        )
        rows = df.to_dict(orient="records")
        assert rows[0]["status"] == "gated_excluded_leaking"

    def test_empty_gold_returns_empty_df(self, simple_source_columns):
        registry = RecommendationRegistry()
        df = build_recommendation_audit(
            registry, opt_in={}, excluded_leaking={}, source_columns=simple_source_columns
        )
        assert len(df) == 0

    def test_summary_counts_by_action(self, simple_source_columns):
        registry = RecommendationRegistry()
        registry.gold = GoldRecommendations(target_column="target")
        registry.gold.transformations = [
            _rec("zero_inflation_handling", "A"),
            _rec("zero_inflation_handling", "B"),
            _rec("log_transform", "C"),
        ]
        audit = build_recommendation_audit(
            registry, opt_in={}, excluded_leaking={}, source_columns=simple_source_columns
        )
        summary = build_recommendation_summary_by_action(audit)
        rows = {r["action"]: r for r in summary.to_dict(orient="records")}
        assert rows["zero_inflation_handling"]["total"] == 2
        assert rows["zero_inflation_handling"]["gated_opt_in"] == 2
        assert rows["zero_inflation_handling"]["applied"] == 0
        assert rows["log_transform"]["total"] == 1
        assert rows["log_transform"]["applied"] == 1


class TestColumnLedger:
    def test_ledger_counts_raw_and_kept(self):
        findings_dict = {
            "case": ExplorationFindings(source_path="/f/case.csv", source_format="csv"),
            "subscription": ExplorationFindings(source_path="/f/subscription.csv", source_format="csv"),
        }
        ledger = build_dataset_column_ledger(
            findings_dict,
            raw_columns_by_dataset={"case": ["A", "B", "C", "D"], "subscription": ["X", "Y"]},
            drop_columns_by_dataset={"case": ["A", "B"], "subscription": []},
            auto_drop_text_by_dataset={"case": ["C"], "subscription": []},
        )
        by_name = {r["dataset"]: r for r in ledger.to_dict(orient="records")}
        assert by_name["case"]["raw_cols"] == 4
        assert by_name["case"]["dropped_DROP_COLUMNS"] == 2
        assert by_name["case"]["dropped_AUTO_TEXT"] == 1
        assert by_name["case"]["survived_to_bronze"] == 1
        assert by_name["subscription"]["survived_to_bronze"] == 2

    def test_column_drop_register_rows(self):
        register = build_column_drop_register(
            drop_columns_by_dataset={"case": ["A"]},
            auto_drop_text_by_dataset={"case": ["B"]},
            audit_scores={"case": {"A": 0.1, "B": 0.9}},
        )
        rows = register.to_dict(orient="records")
        by_col = {r["column"]: r for r in rows}
        assert by_col["A"]["reason"] == "DROP_COLUMNS"
        assert by_col["A"]["audit_score"] == pytest.approx(0.1)
        assert by_col["B"]["reason"] == "AUTO_DROP_TEXT"


class TestNb05DropLedger:
    def test_bucket_counts_per_source(self, simple_source_columns):
        profile = {
            "kept": ["NET_PRICE_sum_30d", "CASE_TYPE"],
            "drop_zero_variance": ["STATUS_count_30d"],
            "drop_weak": ["CASE_ID_count_30d"],
            "drop_multicollinear": ["ACTIVATED_DATE_days"],
            "drop_excluded_leaking": [],
        }
        df = build_nb05_drop_ledger(profile, simple_source_columns)
        by_ds = {r["dataset"]: r for r in df.to_dict(orient="records")}
        assert by_ds["subscription"]["survives"] == 1
        assert by_ds["subscription"]["drop_zero_var"] == 1
        assert by_ds["case"]["survives"] == 1
        assert by_ds["case"]["drop_weak"] == 1
        assert by_ds["implementation_project"]["drop_multicollinear"] == 1

    def test_empty_profile_returns_empty_df(self, simple_source_columns):
        df = build_nb05_drop_ledger({}, simple_source_columns)
        assert len(df) == 0
        assert "dataset" in df.columns
        assert "survives" in df.columns


class TestNb08PerSourceSurvival:
    def test_counts_per_stage_per_source(self, simple_source_columns):
        stages = {
            "post_load_gold": ["NET_PRICE_sum_30d", "CASE_TYPE", "ACTIVATED_DATE_days"],
            "post_zero_var": ["NET_PRICE_sum_30d", "CASE_TYPE"],
            "post_chi_sq": ["NET_PRICE_sum_30d"],
        }
        df = build_nb08_per_source_survival(stages, simple_source_columns)
        by_ds = {r["dataset"]: r for r in df.to_dict(orient="records")}
        assert by_ds["subscription"]["post_load_gold"] == 1
        assert by_ds["subscription"]["post_chi_sq"] == 1
        assert by_ds["case"]["post_chi_sq"] == 0

    def test_empty_stages_returns_empty_df(self, simple_source_columns):
        df = build_nb08_per_source_survival({}, simple_source_columns)
        assert len(df) == 0


class TestNb08Top30Attribution:
    def test_ranks_by_importance(self, simple_source_columns):
        scores = {
            "gbdt": {
                "NET_PRICE_sum_30d": 0.8,
                "CASE_TYPE": 0.6,
                "ACTIVATED_DATE_days": 0.3,
            },
        }
        df = build_nb08_top30_attribution(scores, simple_source_columns, top_n=2)
        assert list(df["feature"]) == ["NET_PRICE_sum_30d", "CASE_TYPE"]
        assert df["source_dataset"].tolist() == ["subscription", "case"]
        assert df["gbdt_importance"].tolist() == [0.8, 0.6]

    def test_empty_scores_returns_empty_df(self, simple_source_columns):
        df = build_nb08_top30_attribution({}, simple_source_columns)
        assert len(df) == 0

    def test_unknown_feature_attributed_to_unknown_source(self, simple_source_columns):
        df = build_nb08_top30_attribution(
            {"gbdt": {"completely_made_up_feature_name": 0.5}},
            simple_source_columns,
        )
        assert df.iloc[0]["source_dataset"] == "unknown"


class TestYamlWriter:
    def test_round_trip(self, tmp_path: Path):
        from customer_retention.core.compat import native_pd

        df = native_pd.DataFrame([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}])
        path = tmp_path / "diag_x.yaml"
        write_diagnostic_yaml(df, path)
        assert path.exists()
        import yaml

        loaded = yaml.safe_load(path.read_text())
        assert loaded == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]

    def test_empty_df_writes_empty_list(self, tmp_path: Path):
        from customer_retention.core.compat import native_pd

        df = native_pd.DataFrame()
        path = tmp_path / "diag_x.yaml"
        write_diagnostic_yaml(df, path)
        import yaml

        assert yaml.safe_load(path.read_text()) == []
