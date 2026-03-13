from customer_retention.analysis.auto_explorer.findings import (
    ColumnClassification,
    ColumnFinding,
    ExplorationFindings,
    classify_columns,
)
from customer_retention.core.config.column_config import ColumnType


def _make_findings(col_types: dict[str, ColumnType]) -> ExplorationFindings:
    return ExplorationFindings(
        source_path="/tmp/test",
        source_format="csv",
        columns={
            name: ColumnFinding(name=name, inferred_type=ct, confidence=1.0, evidence=[])
            for name, ct in col_types.items()
        },
    )


class TestClassifyColumns:

    def test_all_types_classified(self):
        findings = _make_findings({
            "age": ColumnType.NUMERIC_CONTINUOUS,
            "count": ColumnType.NUMERIC_DISCRETE,
            "city": ColumnType.CATEGORICAL_NOMINAL,
            "rank": ColumnType.CATEGORICAL_ORDINAL,
            "month": ColumnType.CATEGORICAL_CYCLICAL,
            "created_at": ColumnType.DATETIME,
            "event_ts": ColumnType.FEATURE_TIMESTAMP,
            "label_ts": ColumnType.LABEL_TIMESTAMP,
            "is_active": ColumnType.BINARY,
            "bio": ColumnType.TEXT,
            "user_id": ColumnType.IDENTIFIER,
            "churn": ColumnType.TARGET,
        })
        cc = classify_columns(findings)
        assert cc.numeric == ["age", "count"]
        assert cc.categorical == ["city", "rank", "month"]
        assert cc.datetime == ["created_at", "event_ts", "label_ts"]
        assert cc.binary == ["is_active"]
        assert cc.text == ["bio"]
        assert cc.identifier == ["user_id"]
        assert cc.target == "churn"

    def test_empty_findings(self):
        cc = classify_columns(_make_findings({}))
        assert cc == ColumnClassification()

    def test_exclude_filter(self):
        findings = _make_findings({
            "age": ColumnType.NUMERIC_CONTINUOUS,
            "event_timestamp": ColumnType.FEATURE_TIMESTAMP,
            "as_of_date": ColumnType.DATETIME,
        })
        cc = classify_columns(findings, exclude={"event_timestamp", "as_of_date"})
        assert cc.numeric == ["age"]
        assert cc.datetime == []

    def test_unknown_type_not_classified(self):
        findings = _make_findings({"mystery": ColumnType.UNKNOWN})
        cc = classify_columns(findings)
        assert cc.numeric == []
        assert cc.categorical == []
        assert cc.target is None
