"""Tests for the ``sps_table_descriptions.md`` bootstrap parser."""
from __future__ import annotations

from pathlib import Path

from customer_retention.stages.causal.interpretation.markdown_bootstrap import (
    parse_table_descriptions_md,
)


def _write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "fixture.md"
    path.write_text(content)
    return path


class TestParseTableDescriptionsMd:
    def test_paren_format_single_table(self, tmp_path):
        md = (
            "prod.salesforce.account:\n"
            "\n"
            "ACCOUNT_ID (string): Unique identifier for the account.\n"
            "NPS_SCORE (int): 0-10 loyalty score.\n"
        )
        rows = parse_table_descriptions_md(_write(tmp_path, md))
        assert [(r.table, r.column_name, r.business_definition) for r in rows] == [
            ("account", "ACCOUNT_ID", "Unique identifier for the account."),
            ("account", "NPS_SCORE", "0-10 loyalty score."),
        ]
        assert all(r.catalog == "prod" for r in rows)
        assert all(r.schema == "salesforce" for r in rows)
        assert all(r.source == "imported_from_md" for r in rows)

    def test_dash_format_single_table(self, tmp_path):
        md = (
            "prod.salesforce.subscription:\n"
            "\n"
            "SUBSCRIPTION_ID: string — A unique identifier for each subscription.\n"
            "NET_PRICE: decimal(14,2) — The net price of the subscription.\n"
        )
        rows = parse_table_descriptions_md(_write(tmp_path, md))
        assert [(r.column_name, r.business_definition) for r in rows] == [
            ("SUBSCRIPTION_ID", "A unique identifier for each subscription."),
            ("NET_PRICE", "The net price of the subscription."),
        ]

    def test_fqn_in_prose_line(self, tmp_path):
        md = (
            "The table prod.reporting.orders contains aggregated data.\n"
            "\n"
            "orderId (string): The order identifier.\n"
        )
        rows = parse_table_descriptions_md(_write(tmp_path, md))
        assert len(rows) == 1
        assert rows[0].catalog == "prod"
        assert rows[0].schema == "reporting"
        assert rows[0].table == "orders"
        assert rows[0].column_name == "orderId"

    def test_multiple_tables_in_sequence(self, tmp_path):
        md = (
            "prod.s.a:\n"
            "\n"
            "X (string): desc of x.\n"
            "\n"
            "prod.s.b:\n"
            "\n"
            "Y (string): desc of y.\n"
        )
        rows = parse_table_descriptions_md(_write(tmp_path, md))
        assert [(r.table, r.column_name) for r in rows] == [("a", "X"), ("b", "Y")]

    def test_ignores_lines_before_first_table_header(self, tmp_path):
        md = (
            "SPS Tables Description\n"
            "some intro prose with no fqn.\n"
            "\n"
            "prod.s.a:\n"
            "\n"
            "X (string): desc.\n"
        )
        rows = parse_table_descriptions_md(_write(tmp_path, md))
        assert len(rows) == 1
        assert rows[0].column_name == "X"

    def test_ignores_dataset_dict_tail(self, tmp_path):
        md = (
            "prod.s.a:\n"
            "\n"
            "X (string): desc.\n"
            "\n"
            "datasets = {\n"
            '#Salesforce Account:\n'
            '"account": "prod.salesforce.account",\n'
            "}\n"
        )
        rows = parse_table_descriptions_md(_write(tmp_path, md))
        assert [r.column_name for r in rows] == ["X"]

    def test_skips_malformed_lines(self, tmp_path):
        md = (
            "prod.s.a:\n"
            "\n"
            "this line has no pattern at all\n"
            "X (string): valid line.\n"
            "   \n"
            "Y: string — also valid.\n"
        )
        rows = parse_table_descriptions_md(_write(tmp_path, md))
        assert [r.column_name for r in rows] == ["X", "Y"]

    def test_auto_business_name_title_cases_snake(self, tmp_path):
        md = "prod.s.a:\n\nACCOUNT_ID (string): x.\n"
        rows = parse_table_descriptions_md(_write(tmp_path, md))
        assert rows[0].business_name == "Account Id"

    def test_auto_business_name_leaves_camelcase_unchanged(self, tmp_path):
        md = "prod.s.a:\n\ndc4SenderId (string): x.\n"
        rows = parse_table_descriptions_md(_write(tmp_path, md))
        assert rows[0].business_name == "dc4SenderId"

    def test_unit_polarity_pii_stay_null(self, tmp_path):
        md = "prod.s.a:\n\nX (string): desc.\n"
        rows = parse_table_descriptions_md(_write(tmp_path, md))
        r = rows[0]
        assert r.unit is None
        assert r.polarity is None
        assert r.pii_class is None
        assert r.value_examples is None
        assert r.last_verified_at is None

    def test_parses_real_sps_file(self):
        path = Path(__file__).resolve().parents[3] / "docs" / "sps_table_descriptions.md"
        if not path.exists():
            return
        rows = parse_table_descriptions_md(path)
        assert len(rows) > 100
        tables = {r.table for r in rows}
        assert "account" in tables
        assert "subscription" in tables
        for r in rows:
            assert r.column_name
            assert r.table
            assert r.source == "imported_from_md"
