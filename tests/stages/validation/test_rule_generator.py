"""Tests for automatic validation rule generation."""

from customer_retention.analysis.auto_explorer.findings import ColumnFinding, ExplorationFindings
from customer_retention.core.config.column_config import ColumnType
from customer_retention.stages.validation.rule_generator import RuleGenerator


def make_column(name: str, col_type: ColumnType, **metrics) -> ColumnFinding:
    return ColumnFinding(
        name=name,
        inferred_type=col_type,
        confidence=0.9,
        evidence=["test"],
        type_metrics=metrics.get("type_metrics", {}),
        universal_metrics=metrics.get("universal_metrics", {})
    )


class TestBinaryRules:
    def test_binary_column_gets_binary_rule(self):
        col = make_column("is_active", ColumnType.BINARY)
        rules = RuleGenerator.for_column(col)

        assert "is_active" in rules
        assert rules["is_active"]["type"] == "binary"
        assert rules["is_active"]["valid_values"] == [0, 1]

    def test_target_binary_gets_binary_rule(self):
        col = make_column("churn", ColumnType.TARGET,
                          type_metrics={"distinct_count": 2})
        rules = RuleGenerator.for_column(col)

        assert rules["churn"]["type"] == "binary"


class TestPercentageRules:
    def test_rate_in_name_gets_percentage_rule(self):
        col = make_column("click_rate", ColumnType.NUMERIC_CONTINUOUS,
                          type_metrics={"min": 0, "max": 100})
        rules = RuleGenerator.for_column(col)

        assert rules["click_rate"]["type"] == "percentage"
        assert rules["click_rate"]["min"] == 0
        assert rules["click_rate"]["max"] == 100

    def test_pct_in_name_gets_percentage_rule(self):
        col = make_column("open_pct", ColumnType.NUMERIC_CONTINUOUS,
                          type_metrics={"min": 0, "max": 95})
        rules = RuleGenerator.for_column(col)

        assert rules["open_pct"]["type"] == "percentage"

    def test_percent_in_name_gets_percentage_rule(self):
        col = make_column("discount_percent", ColumnType.NUMERIC_CONTINUOUS)
        rules = RuleGenerator.for_column(col)

        assert rules["discount_percent"]["type"] == "percentage"

    def test_ratio_columns_0_to_1_get_percentage_rule(self):
        col = make_column("conversion_ratio", ColumnType.NUMERIC_CONTINUOUS,
                          type_metrics={"min": 0, "max": 0.85})
        rules = RuleGenerator.for_column(col)

        assert rules["conversion_ratio"]["type"] == "percentage"
        assert rules["conversion_ratio"]["max"] == 1


class TestNonNegativeRules:
    def test_count_column_gets_non_negative(self):
        col = make_column("order_count", ColumnType.NUMERIC_DISCRETE,
                          type_metrics={"min": 0, "max": 150})
        rules = RuleGenerator.for_column(col)

        assert rules["order_count"]["type"] == "non_negative"

    def test_amount_column_gets_non_negative(self):
        col = make_column("total_amount", ColumnType.NUMERIC_CONTINUOUS,
                          type_metrics={"min": 0, "max": 5000})
        rules = RuleGenerator.for_column(col)

        assert rules["total_amount"]["type"] == "non_negative"

    def test_column_with_negative_min_no_rule(self):
        col = make_column("balance", ColumnType.NUMERIC_CONTINUOUS,
                          type_metrics={"min": -500, "max": 5000})
        rules = RuleGenerator.for_column(col)

        assert rules == {}


class TestSkippedColumns:
    def test_identifier_skipped(self):
        col = make_column("customer_id", ColumnType.IDENTIFIER)
        rules = RuleGenerator.for_column(col)
        assert rules == {}

    def test_datetime_skipped(self):
        col = make_column("created_at", ColumnType.DATETIME)
        rules = RuleGenerator.for_column(col)
        assert rules == {}

    def test_text_skipped(self):
        col = make_column("notes", ColumnType.TEXT)
        rules = RuleGenerator.for_column(col)
        assert rules == {}

    def test_categorical_skipped(self):
        col = make_column("category", ColumnType.CATEGORICAL_NOMINAL)
        rules = RuleGenerator.for_column(col)
        assert rules == {}


class TestFromFindings:
    def test_generates_rules_for_all_applicable_columns(self):
        findings = ExplorationFindings(
            source_path="test.csv",
            source_format="csv",
            columns={
                "id": make_column("id", ColumnType.IDENTIFIER),
                "is_active": make_column("is_active", ColumnType.BINARY),
                "click_rate": make_column("click_rate", ColumnType.NUMERIC_CONTINUOUS,
                                          type_metrics={"min": 0, "max": 100}),
                "order_count": make_column("order_count", ColumnType.NUMERIC_DISCRETE,
                                           type_metrics={"min": 0, "max": 50}),
            }
        )

        rules = RuleGenerator.from_findings(findings)

        assert "id" not in rules
        assert "is_active" in rules
        assert "click_rate" in rules
        assert "order_count" in rules
        assert len(rules) == 3
