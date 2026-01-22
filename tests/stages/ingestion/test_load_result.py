import pytest
from customer_retention.stages.ingestion import LoadResult


class TestLoadResult:
    def test_create_successful_result(self):
        result = LoadResult(
            success=True,
            row_count=100,
            column_count=10,
            duration_seconds=1.5,
            source_name="test_source"
        )
        assert result.success is True
        assert result.row_count == 100
        assert result.column_count == 10
        assert result.duration_seconds == 1.5
        assert result.source_name == "test_source"

    def test_create_failed_result(self):
        result = LoadResult(
            success=False,
            row_count=0,
            column_count=0,
            duration_seconds=0.5,
            source_name="test_source",
            errors=["File not found"]
        )
        assert result.success is False
        assert len(result.errors) == 1

    def test_default_warnings_and_errors(self):
        result = LoadResult(
            success=True,
            row_count=100,
            column_count=10,
            duration_seconds=1.5,
            source_name="test_source"
        )
        assert result.warnings == []
        assert result.errors == []

    def test_has_warnings_false(self):
        result = LoadResult(
            success=True,
            row_count=100,
            column_count=10,
            duration_seconds=1.5,
            source_name="test_source"
        )
        assert result.has_warnings() is False

    def test_has_warnings_true(self):
        result = LoadResult(
            success=True,
            row_count=100,
            column_count=10,
            duration_seconds=1.5,
            source_name="test_source",
            warnings=["Some warning"]
        )
        assert result.has_warnings() is True

    def test_has_errors_false(self):
        result = LoadResult(
            success=True,
            row_count=100,
            column_count=10,
            duration_seconds=1.5,
            source_name="test_source"
        )
        assert result.has_errors() is False

    def test_has_errors_true(self):
        result = LoadResult(
            success=False,
            row_count=0,
            column_count=0,
            duration_seconds=0.5,
            source_name="test_source",
            errors=["Some error"]
        )
        assert result.has_errors() is True

    def test_add_warning(self):
        result = LoadResult(
            success=True,
            row_count=100,
            column_count=10,
            duration_seconds=1.5,
            source_name="test_source"
        )
        result.add_warning("Warning message")
        assert len(result.warnings) == 1
        assert result.warnings[0] == "Warning message"

    def test_add_multiple_warnings(self):
        result = LoadResult(
            success=True,
            row_count=100,
            column_count=10,
            duration_seconds=1.5,
            source_name="test_source"
        )
        result.add_warning("Warning 1")
        result.add_warning("Warning 2")
        assert len(result.warnings) == 2

    def test_add_error(self):
        result = LoadResult(
            success=False,
            row_count=0,
            column_count=0,
            duration_seconds=0.5,
            source_name="test_source"
        )
        result.add_error("Error message")
        assert len(result.errors) == 1
        assert result.errors[0] == "Error message"

    def test_add_multiple_errors(self):
        result = LoadResult(
            success=False,
            row_count=0,
            column_count=0,
            duration_seconds=0.5,
            source_name="test_source"
        )
        result.add_error("Error 1")
        result.add_error("Error 2")
        assert len(result.errors) == 2

    def test_get_summary_success(self):
        result = LoadResult(
            success=True,
            row_count=100,
            column_count=10,
            duration_seconds=1.5,
            source_name="test_source"
        )
        summary = result.get_summary()
        assert "SUCCESS" in summary
        assert "test_source" in summary
        assert "100 rows" in summary
        assert "10 columns" in summary
        assert "1.50s" in summary

    def test_get_summary_failed(self):
        result = LoadResult(
            success=False,
            row_count=0,
            column_count=0,
            duration_seconds=0.5,
            source_name="test_source"
        )
        summary = result.get_summary()
        assert "FAILED" in summary
        assert "test_source" in summary

    def test_schema_info(self):
        result = LoadResult(
            success=True,
            row_count=100,
            column_count=2,
            duration_seconds=1.5,
            source_name="test_source",
            schema_info={"id": "int64", "name": "object"}
        )
        assert len(result.schema_info) == 2
        assert result.schema_info["id"] == "int64"
        assert result.schema_info["name"] == "object"

    def test_json_serialization(self):
        result = LoadResult(
            success=True,
            row_count=100,
            column_count=10,
            duration_seconds=1.5,
            source_name="test_source"
        )
        json_data = result.model_dump()
        assert json_data["success"] is True
        assert json_data["row_count"] == 100
