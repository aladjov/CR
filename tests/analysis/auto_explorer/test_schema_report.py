from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from customer_retention.analysis.auto_explorer.schema_report import (
    build_schema_report,
    generate_schema_report,
    normalize_dtype_label,
)


class TestNormalizeDtypeLabel:
    def test_pandas_object_maps_to_string(self):
        assert normalize_dtype_label("object") == "string"

    def test_pandas_int64_maps_to_bigint(self):
        assert normalize_dtype_label("int64") == "bigint"

    def test_pandas_int64_nullable_maps_to_bigint(self):
        assert normalize_dtype_label("Int64") == "bigint"

    def test_pandas_int32_maps_to_int(self):
        assert normalize_dtype_label("int32") == "int"

    def test_pandas_float64_maps_to_double(self):
        assert normalize_dtype_label("float64") == "double"

    def test_pandas_float32_maps_to_float(self):
        assert normalize_dtype_label("float32") == "float"

    def test_pandas_datetime_maps_to_timestamp(self):
        assert normalize_dtype_label("datetime64[ns]") == "timestamp"

    def test_pandas_datetime_tz_maps_to_timestamp(self):
        assert normalize_dtype_label("datetime64[ns, UTC]") == "timestamp"

    def test_pandas_bool_maps_to_boolean(self):
        assert normalize_dtype_label("bool") == "boolean"

    def test_pandas_nullable_boolean(self):
        assert normalize_dtype_label("boolean") == "boolean"

    def test_pandas_category_maps_to_string(self):
        assert normalize_dtype_label("category") == "string"

    def test_spark_string_passthrough(self):
        assert normalize_dtype_label("string") == "string"

    def test_spark_bigint_passthrough(self):
        assert normalize_dtype_label("bigint") == "bigint"

    def test_spark_timestamp_ntz(self):
        assert normalize_dtype_label("timestamp_ntz") == "timestamp"

    def test_spark_timestamp(self):
        assert normalize_dtype_label("timestamp") == "timestamp"

    def test_spark_date(self):
        assert normalize_dtype_label("date") == "date"

    def test_spark_decimal_passthrough(self):
        assert normalize_dtype_label("decimal(18,2)") == "decimal(18,2)"

    def test_spark_decimal_no_scale(self):
        assert normalize_dtype_label("decimal(10,0)") == "decimal(10,0)"

    def test_spark_double_passthrough(self):
        assert normalize_dtype_label("double") == "double"

    def test_spark_int(self):
        assert normalize_dtype_label("int") == "int"

    def test_spark_long_maps_to_bigint(self):
        assert normalize_dtype_label("long") == "bigint"

    def test_spark_short_maps_to_smallint(self):
        assert normalize_dtype_label("short") == "smallint"

    def test_spark_tinyint(self):
        assert normalize_dtype_label("tinyint") == "tinyint"

    def test_unknown_type_passthrough(self):
        assert normalize_dtype_label("binary") == "binary"

    def test_pandas_int8_maps_to_tinyint(self):
        assert normalize_dtype_label("int8") == "tinyint"

    def test_pandas_int16_maps_to_smallint(self):
        assert normalize_dtype_label("int16") == "smallint"


class TestGenerateSchemaReport:
    def test_single_dataset_basic(self):
        df = pd.DataFrame({"ID": [1, 2], "NAME": ["a", "b"]})
        result = generate_schema_report(
            {"customers": df}, {"customers": "catalog.schema.customers"},
        )
        assert "catalog.schema.customers:" in result
        assert "ID (bigint)" in result
        assert "NAME (string)" in result

    def test_multiple_datasets(self):
        df1 = pd.DataFrame({"A": [1]})
        df2 = pd.DataFrame({"B": ["x"], "C": [1.5]})
        result = generate_schema_report(
            {"t1": df1, "t2": df2},
            {"t1": "db.schema.t1", "t2": "db.schema.t2"},
        )
        assert "db.schema.t1:" in result
        assert "db.schema.t2:" in result
        assert "A (bigint)" in result
        assert "B (string)" in result
        assert "C (double)" in result

    def test_dataset_order_matches_input(self):
        frames = {"zz": pd.DataFrame({"X": [1]}), "aa": pd.DataFrame({"Y": [2]})}
        paths = {"zz": "db.zz", "aa": "db.aa"}
        result = generate_schema_report(frames, paths)
        zz_pos = result.index("db.zz:")
        aa_pos = result.index("db.aa:")
        assert zz_pos < aa_pos

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = generate_schema_report({"empty": df}, {"empty": "db.empty"})
        assert "db.empty:" in result

    def test_file_path_used_when_no_table_path(self):
        df = pd.DataFrame({"COL": [1]})
        result = generate_schema_report(
            {"local": df}, {"local": "/data/local.csv"},
        )
        assert "/data/local.csv" in result

    def test_title_present(self):
        df = pd.DataFrame({"A": [1]})
        result = generate_schema_report({"t": df}, {"t": "db.t"})
        assert result.startswith("Dataset Schema Report")

    def test_mixed_dtypes(self):
        df = pd.DataFrame({
            "ID": pd.array([1, 2], dtype="int64"),
            "AMOUNT": pd.array([1.5, 2.5], dtype="float64"),
            "NAME": pd.array(["a", "b"], dtype="object"),
            "CREATED": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "FLAG": pd.array([True, False], dtype="bool"),
        })
        result = generate_schema_report({"mixed": df}, {"mixed": "db.mixed"})
        assert "ID (bigint)" in result
        assert "AMOUNT (double)" in result
        assert "NAME (string)" in result
        assert "CREATED (timestamp)" in result
        assert "FLAG (boolean)" in result

    def test_column_order_preserved(self):
        df = pd.DataFrame({"ZEBRA": [1], "ALPHA": [2], "MIDDLE": [3]})
        result = generate_schema_report({"t": df}, {"t": "db.t"})
        z_pos = result.index("ZEBRA")
        a_pos = result.index("ALPHA")
        m_pos = result.index("MIDDLE")
        assert z_pos < a_pos < m_pos

    def test_missing_frame_skipped(self):
        df = pd.DataFrame({"A": [1]})
        result = generate_schema_report(
            {"present": df}, {"present": "db.p", "missing": "db.m"},
        )
        assert "db.p:" in result
        assert "db.m" not in result

    def test_no_description_by_default(self):
        df = pd.DataFrame({"COL": [1]})
        result = generate_schema_report({"t": df}, {"t": "db.t"})
        lines = [line for line in result.splitlines() if "COL" in line]
        assert len(lines) == 1
        assert lines[0].strip() == "COL (bigint)"

    def test_descriptions_included_when_provided(self):
        df = pd.DataFrame({"ACCOUNT_ID": ["x"]})
        descs = {"ACCOUNT_ID": "Unique identifier for the account."}
        result = generate_schema_report(
            {"t": df}, {"t": "db.t"}, column_descriptions={"t": descs},
        )
        assert "ACCOUNT_ID (string): Unique identifier for the account." in result

    def test_partial_descriptions(self):
        df = pd.DataFrame({"A": [1], "B": ["x"]})
        descs = {"A": "First column"}
        result = generate_schema_report(
            {"t": df}, {"t": "db.t"}, column_descriptions={"t": descs},
        )
        assert "A (bigint): First column" in result
        assert "B (string)" in result
        assert "B (string):" not in result


# ---------------------------------------------------------------------------
# build_schema_report
# ---------------------------------------------------------------------------


_MOCK_RESOLVE = patch(
    "customer_retention.analysis.auto_explorer.column_describer.resolve_endpoint",
    return_value="databricks-mock-model",
)


class TestBuildSchemaReport:
    def test_returns_result_object(self):
        df = pd.DataFrame({"A": [1], "B": ["x"]})
        result = build_schema_report({"t": df}, {"t": "db.t"})
        assert "Dataset Schema Report" in result.text
        assert result.column_count == 2
        assert result.dataset_count == 1
        assert result.warnings == []
        assert result.llm_endpoint_used is None

    def test_llm_disabled_by_default(self):
        df = pd.DataFrame({"A": [1]})
        result = build_schema_report({"t": df}, {"t": "cat.sch.tbl"})
        assert "A (bigint)" in result.text
        assert result.warnings == []
        assert result.llm_endpoint_used is None

    @patch(
        "customer_retention.analysis.auto_explorer.schema_report.is_databricks",
        return_value=False,
    )
    def test_llm_skipped_when_not_databricks(self, _mock):
        df = pd.DataFrame({"A": [1]})
        result = build_schema_report(
            {"t": df}, {"t": "cat.sch.tbl"}, llm_descriptions=True
        )
        assert any("skipped" in w for w in result.warnings)
        assert "A (bigint)" in result.text
        assert result.llm_endpoint_used is None

    @_MOCK_RESOLVE
    @patch(
        "customer_retention.analysis.auto_explorer.schema_report.is_databricks",
        return_value=True,
    )
    @patch(
        "customer_retention.analysis.auto_explorer.column_describer.describe_datasets",
    )
    def test_llm_descriptions_merged_into_report(
        self, mock_describe, _mock_db, _mock_resolve
    ):
        mock_describe.return_value = {"t": {"A": "The primary key."}}
        df = pd.DataFrame({"A": [1], "B": ["x"]})
        result = build_schema_report(
            {"t": df}, {"t": "cat.sch.tbl"}, llm_descriptions=True
        )
        assert "A (bigint): The primary key." in result.text
        assert "B (string)" in result.text
        assert result.llm_endpoint_used == "databricks-mock-model"

    @_MOCK_RESOLVE
    @patch(
        "customer_retention.analysis.auto_explorer.schema_report.is_databricks",
        return_value=True,
    )
    @patch(
        "customer_retention.analysis.auto_explorer.column_describer.describe_datasets",
    )
    def test_warns_on_missing_descriptions(
        self, mock_describe, _mock_db, _mock_resolve
    ):
        mock_describe.return_value = {"t": {"A": "desc A"}}
        df = pd.DataFrame({"A": [1], "B": ["x"], "C": [1.0]})
        result = build_schema_report(
            {"t": df}, {"t": "cat.sch.tbl"}, llm_descriptions=True
        )
        assert len(result.warnings) == 1
        assert "2/3" in result.warnings[0]
        assert "B" in result.warnings[0]
        assert "C" in result.warnings[0]

    @_MOCK_RESOLVE
    @patch(
        "customer_retention.analysis.auto_explorer.schema_report.is_databricks",
        return_value=True,
    )
    @patch(
        "customer_retention.analysis.auto_explorer.column_describer.describe_datasets",
    )
    def test_no_warning_when_all_described(
        self, mock_describe, _mock_db, _mock_resolve
    ):
        mock_describe.return_value = {"t": {"A": "d1", "B": "d2"}}
        df = pd.DataFrame({"A": [1], "B": ["x"]})
        result = build_schema_report(
            {"t": df}, {"t": "cat.sch.tbl"}, llm_descriptions=True
        )
        assert result.warnings == []

    @patch(
        "customer_retention.analysis.auto_explorer.schema_report.is_databricks",
        return_value=True,
    )
    @patch(
        "customer_retention.analysis.auto_explorer.column_describer.resolve_endpoint",
        side_effect=RuntimeError(
            "No Foundation Model chat endpoints available on this workspace."
        ),
    )
    def test_endpoint_resolution_failure_becomes_warning(
        self, _mock_resolve, _mock_db
    ):
        df = pd.DataFrame({"A": [1]})
        result = build_schema_report(
            {"t": df}, {"t": "cat.sch.tbl"}, llm_descriptions=True
        )
        assert len(result.warnings) == 1
        assert "No Foundation Model" in result.warnings[0]
        assert "LLM_ENDPOINT" in result.warnings[0]
        assert result.llm_endpoint_used is None
        # Report still generated, just without descriptions.
        assert "A (bigint)" in result.text
