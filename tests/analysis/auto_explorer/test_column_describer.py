from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from customer_retention.analysis.auto_explorer.column_describer import (
    _PREFERRED_ENDPOINTS,
    _build_prompt,
    _parse_llm_response,
    _parse_table_fqn,
    describe_datasets,
    fetch_uc_column_comments,
    generate_column_descriptions,
    resolve_endpoint,
    validate_descriptions,
)

# ---------------------------------------------------------------------------
# _parse_table_fqn
# ---------------------------------------------------------------------------

class TestParseTableFqn:
    def test_valid_three_part_name(self):
        assert _parse_table_fqn("cat.sch.tbl") == ("cat", "sch", "tbl")

    def test_real_world_name(self):
        cat, sch, tbl = _parse_table_fqn(
            "prod_corp_snowflake_provisioning_shared.salesforce.account"
        )
        assert cat == "prod_corp_snowflake_provisioning_shared"
        assert sch == "salesforce"
        assert tbl == "account"

    def test_too_few_parts_raises(self):
        with pytest.raises(ValueError, match="Expected 'catalog.schema.table'"):
            _parse_table_fqn("schema.table")

    def test_too_many_parts_raises(self):
        with pytest.raises(ValueError, match="Expected 'catalog.schema.table'"):
            _parse_table_fqn("a.b.c.d")

    def test_single_name_raises(self):
        with pytest.raises(ValueError):
            _parse_table_fqn("table_only")


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_contains_table_name(self):
        prompt = _build_prompt("cat.sch.tbl", ["COL_A", "COL_B"])
        assert "cat.sch.tbl" in prompt

    def test_contains_all_columns(self):
        cols = ["ALPHA", "BETA", "GAMMA"]
        prompt = _build_prompt("c.s.t", cols)
        for col in cols:
            assert f"- {col}" in prompt

    def test_format_instruction_present(self):
        prompt = _build_prompt("c.s.t", ["X"])
        assert "COLUMN_NAME: description text here" in prompt


# ---------------------------------------------------------------------------
# _parse_llm_response
# ---------------------------------------------------------------------------

class TestParseLlmResponse:
    def test_parses_well_formed_lines(self):
        text = (
            "COL_A: Unique identifier for the record.\n"
            "COL_B: Date the record was created.\n"
        )
        result = _parse_llm_response(text, ["COL_A", "COL_B"])
        assert result == {
            "COL_A": "Unique identifier for the record.",
            "COL_B": "Date the record was created.",
        }

    def test_ignores_unknown_columns(self):
        text = "UNKNOWN_COL: Some description.\n"
        result = _parse_llm_response(text, ["COL_A"])
        assert result == {}

    def test_ignores_blank_lines(self):
        text = "\n\nCOL_A: desc\n\n"
        result = _parse_llm_response(text, ["COL_A"])
        assert result == {"COL_A": "desc"}

    def test_ignores_preamble_text(self):
        text = (
            "Here are the descriptions:\n"
            "COL_A: The primary key.\n"
        )
        result = _parse_llm_response(text, ["COL_A"])
        assert result == {"COL_A": "The primary key."}

    def test_strips_whitespace(self):
        text = "  COL_A :  some desc  \n"
        result = _parse_llm_response(text, ["COL_A"])
        assert result == {"COL_A": "some desc"}

    def test_empty_description_skipped(self):
        text = "COL_A: \n"
        result = _parse_llm_response(text, ["COL_A"])
        assert result == {}

    def test_colon_in_description_preserved(self):
        text = "COL_A: Format: YYYY-MM-DD\n"
        result = _parse_llm_response(text, ["COL_A"])
        assert result == {"COL_A": "Format: YYYY-MM-DD"}

    def test_column_name_with_numbers(self):
        text = "COL_123: A numbered column.\n"
        result = _parse_llm_response(text, ["COL_123"])
        assert result == {"COL_123": "A numbered column."}

    def test_empty_response(self):
        assert _parse_llm_response("", ["COL_A"]) == {}

    def test_all_garbage(self):
        text = "I cannot generate descriptions because...\nPlease try again.\n"
        assert _parse_llm_response(text, ["COL_A"]) == {}


# ---------------------------------------------------------------------------
# validate_descriptions
# ---------------------------------------------------------------------------

class TestValidateDescriptions:
    def test_all_described(self):
        cols = ["A", "B", "C"]
        descs = {"A": "d1", "B": "d2", "C": "d3"}
        assert validate_descriptions(cols, descs) == []

    def test_some_missing(self):
        cols = ["A", "B", "C"]
        descs = {"A": "d1"}
        assert validate_descriptions(cols, descs) == ["B", "C"]

    def test_none_described(self):
        cols = ["A", "B"]
        assert validate_descriptions(cols, {}) == ["A", "B"]

    def test_empty_columns(self):
        assert validate_descriptions([], {"A": "d1"}) == []

    def test_preserves_column_order(self):
        cols = ["Z", "A", "M"]
        missing = validate_descriptions(cols, {})
        assert missing == ["Z", "A", "M"]


# ---------------------------------------------------------------------------
# resolve_endpoint
# ---------------------------------------------------------------------------

def _make_endpoint(name, task="llm/v1/chat", ready="READY"):
    return {"name": name, "task": task, "state": {"ready": ready}}


class TestResolveEndpoint:
    def test_explicit_endpoint_returned_as_is(self):
        assert resolve_endpoint("my-custom-ep") == "my-custom-ep"

    def test_none_returns_preferred_endpoint(self):
        eps = [
            _make_endpoint("databricks-dbrx-instruct"),
            _make_endpoint(_PREFERRED_ENDPOINTS[0]),
        ]
        mock_client = MagicMock()
        mock_client.list_endpoints.return_value = eps
        with patch("mlflow.deployments.get_deploy_client", return_value=mock_client):
            result = resolve_endpoint(None)
        assert result == _PREFERRED_ENDPOINTS[0]

    def test_falls_back_to_any_databricks_chat(self):
        eps = [_make_endpoint("databricks-some-new-model")]
        mock_client = MagicMock()
        mock_client.list_endpoints.return_value = eps
        with patch("mlflow.deployments.get_deploy_client", return_value=mock_client):
            result = resolve_endpoint(None)
        assert result == "databricks-some-new-model"

    def test_skips_non_ready_endpoints(self):
        eps = [
            _make_endpoint(_PREFERRED_ENDPOINTS[0], ready="NOT_READY"),
            _make_endpoint("databricks-fallback-model"),
        ]
        mock_client = MagicMock()
        mock_client.list_endpoints.return_value = eps
        with patch("mlflow.deployments.get_deploy_client", return_value=mock_client):
            result = resolve_endpoint(None)
        assert result == "databricks-fallback-model"

    def test_skips_embedding_endpoints(self):
        eps = [
            _make_endpoint("databricks-bge-large-en", task="llm/v1/embeddings"),
            _make_endpoint("databricks-chat-model"),
        ]
        mock_client = MagicMock()
        mock_client.list_endpoints.return_value = eps
        with patch("mlflow.deployments.get_deploy_client", return_value=mock_client):
            result = resolve_endpoint(None)
        assert result == "databricks-chat-model"

    def test_raises_when_no_endpoints(self):
        mock_client = MagicMock()
        mock_client.list_endpoints.return_value = []
        with patch("mlflow.deployments.get_deploy_client", return_value=mock_client):
            with pytest.raises(RuntimeError, match="No Foundation Model"):
                resolve_endpoint(None)

    def test_raises_when_list_fails(self):
        with patch(
            "mlflow.deployments.get_deploy_client",
            side_effect=RuntimeError("no creds"),
        ):
            with pytest.raises(RuntimeError, match="Cannot auto-detect"):
                resolve_endpoint(None)

    def test_ignores_custom_non_databricks_endpoints(self):
        eps = [_make_endpoint("my-custom-model")]  # no databricks- prefix
        mock_client = MagicMock()
        mock_client.list_endpoints.return_value = eps
        with patch("mlflow.deployments.get_deploy_client", return_value=mock_client):
            with pytest.raises(RuntimeError, match="No Foundation Model"):
                resolve_endpoint(None)


# ---------------------------------------------------------------------------
# fetch_uc_column_comments — Databricks guard
# ---------------------------------------------------------------------------

class TestFetchUcColumnComments:
    @patch(
        "customer_retention.analysis.auto_explorer.column_describer.is_databricks",
        return_value=False,
    )
    def test_returns_empty_when_not_databricks(self, _mock):
        assert fetch_uc_column_comments("cat.sch.tbl") == {}

    @patch(
        "customer_retention.analysis.auto_explorer.column_describer.is_databricks",
        return_value=True,
    )
    def test_returns_comments_on_databricks(self, _mock_db):
        mock_spark = MagicMock()
        row1 = {"column_name": "COL_A", "comment": "Primary key"}
        row2 = {"column_name": "COL_B", "comment": "Created date"}
        mock_spark.sql.return_value.collect.return_value = [row1, row2]

        result = fetch_uc_column_comments("cat.sch.tbl", spark=mock_spark)
        assert result == {"COL_A": "Primary key", "COL_B": "Created date"}
        mock_spark.sql.assert_called_once()

    @patch(
        "customer_retention.analysis.auto_explorer.column_describer.is_databricks",
        return_value=True,
    )
    def test_returns_empty_on_sql_error(self, _mock_db):
        mock_spark = MagicMock()
        mock_spark.sql.side_effect = RuntimeError("table not found")
        assert fetch_uc_column_comments("cat.sch.tbl", spark=mock_spark) == {}


# ---------------------------------------------------------------------------
# generate_column_descriptions — Databricks guard + LLM call
# ---------------------------------------------------------------------------

class TestGenerateColumnDescriptions:
    @patch(
        "customer_retention.analysis.auto_explorer.column_describer.is_databricks",
        return_value=False,
    )
    def test_returns_empty_when_not_databricks(self, _mock):
        result = generate_column_descriptions("c.s.t", ["A", "B"])
        assert result == {}

    @patch(
        "customer_retention.analysis.auto_explorer.column_describer.is_databricks",
        return_value=True,
    )
    def test_skips_llm_when_all_columns_have_comments(self, _mock_db):
        existing = {"A": "desc A", "B": "desc B"}
        result = generate_column_descriptions(
            "c.s.t", ["A", "B"], existing_comments=existing
        )
        assert result == existing

    @patch(
        "customer_retention.analysis.auto_explorer.column_describer.is_databricks",
        return_value=True,
    )
    @patch(
        "customer_retention.analysis.auto_explorer.column_describer.resolve_endpoint",
        return_value="databricks-mock-model",
    )
    def test_calls_llm_for_missing_columns(self, _mock_resolve, _mock_db):
        mock_client = MagicMock()
        mock_client.predict.return_value = {
            "choices": [
                {"message": {"content": "COL_B: The second column.\n"}}
            ]
        }
        with patch("mlflow.deployments.get_deploy_client", return_value=mock_client):
            result = generate_column_descriptions(
                "c.s.t",
                ["COL_A", "COL_B"],
                existing_comments={"COL_A": "Already documented"},
            )
        assert result == {
            "COL_A": "Already documented",
            "COL_B": "The second column.",
        }
        # Only COL_B should be in the prompt
        prompt_sent = mock_client.predict.call_args[1]["inputs"]["messages"][0]["content"]
        assert "COL_B" in prompt_sent
        assert "COL_A" not in prompt_sent

    @patch(
        "customer_retention.analysis.auto_explorer.column_describer.is_databricks",
        return_value=True,
    )
    @patch(
        "customer_retention.analysis.auto_explorer.column_describer.resolve_endpoint",
        return_value="databricks-mock-model",
    )
    def test_returns_existing_on_llm_failure(self, _mock_resolve, _mock_db):
        with patch(
            "mlflow.deployments.get_deploy_client",
            side_effect=RuntimeError("endpoint down"),
        ):
            result = generate_column_descriptions(
                "c.s.t",
                ["A", "B"],
                existing_comments={"A": "known"},
            )
        assert result == {"A": "known"}

    @patch(
        "customer_retention.analysis.auto_explorer.column_describer.is_databricks",
        return_value=True,
    )
    def test_uses_explicit_endpoint(self, _mock_db):
        mock_client = MagicMock()
        mock_client.predict.return_value = {
            "choices": [{"message": {"content": "A: desc\n"}}]
        }
        with patch("mlflow.deployments.get_deploy_client", return_value=mock_client):
            generate_column_descriptions(
                "c.s.t", ["A"], endpoint="my-custom-endpoint"
            )
        mock_client.predict.assert_called_once()
        assert mock_client.predict.call_args[1]["endpoint"] == "my-custom-endpoint"


# ---------------------------------------------------------------------------
# describe_datasets — end-to-end orchestration
# ---------------------------------------------------------------------------

class TestDescribeDatasets:
    @patch(
        "customer_retention.analysis.auto_explorer.column_describer.is_databricks",
        return_value=False,
    )
    def test_skips_everything_when_not_databricks(self, _mock):
        datasets = {"accounts": "cat.sch.accounts"}
        frames = {"accounts": pd.DataFrame({"A": [1]})}
        assert describe_datasets(datasets, frames) == {}

    @patch(
        "customer_retention.analysis.auto_explorer.column_describer.fetch_uc_column_comments",
    )
    @patch(
        "customer_retention.analysis.auto_explorer.column_describer.generate_column_descriptions",
    )
    @patch(
        "customer_retention.analysis.auto_explorer.column_describer.is_databricks",
        return_value=True,
    )
    def test_describes_file_path_datasets_via_llm(self, _mock_db, mock_gen, mock_uc):
        mock_gen.return_value = {"X": "A simple column."}
        datasets = {"local": "../data/local.csv"}
        frames = {"local": pd.DataFrame({"X": [1]})}

        result = describe_datasets(datasets, frames)

        assert result == {"local": {"X": "A simple column."}}
        mock_uc.assert_not_called()  # no UC lookup for CSV sources
        assert mock_gen.call_args[0][0] == "file:local"

    @patch(
        "customer_retention.analysis.auto_explorer.column_describer.fetch_uc_column_comments",
        return_value={},
    )
    @patch(
        "customer_retention.analysis.auto_explorer.column_describer.generate_column_descriptions",
    )
    @patch(
        "customer_retention.analysis.auto_explorer.column_describer.is_databricks",
        return_value=True,
    )
    def test_orchestrates_uc_and_llm(self, _mock_db, mock_gen, mock_uc):
        mock_gen.return_value = {"COL_A": "desc A"}
        datasets = {"acct": "cat.sch.acct"}
        frames = {"acct": pd.DataFrame({"COL_A": [1]})}

        result = describe_datasets(datasets, frames)
        assert result == {"acct": {"COL_A": "desc A"}}
        mock_uc.assert_called_once_with("cat.sch.acct")
        mock_gen.assert_called_once()

    @patch(
        "customer_retention.analysis.auto_explorer.column_describer.is_databricks",
        return_value=True,
    )
    def test_skips_missing_frames(self, _mock_db):
        datasets = {"acct": "cat.sch.acct"}
        frames = {}  # no matching frame loaded
        result = describe_datasets(datasets, frames)
        assert result == {}
