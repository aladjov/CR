import pandas as pd
import pytest
from pydantic import ValidationError

from customer_retention.core.config import ColumnConfig, ColumnType
from customer_retention.core.config.column_config import (
    NON_FEATURE_COLUMN_TYPES,
    NON_FEATURE_DTYPES,
    select_model_ready_columns,
)


class TestColumnType:
    def test_all_column_types_exist(self):
        assert ColumnType.IDENTIFIER == "identifier"
        assert ColumnType.TARGET == "target"
        assert ColumnType.FEATURE_TIMESTAMP == "feature_timestamp"
        assert ColumnType.LABEL_TIMESTAMP == "label_timestamp"
        assert ColumnType.NUMERIC_CONTINUOUS == "numeric_continuous"
        assert ColumnType.NUMERIC_DISCRETE == "numeric_discrete"
        assert ColumnType.CATEGORICAL_NOMINAL == "categorical_nominal"
        assert ColumnType.CATEGORICAL_ORDINAL == "categorical_ordinal"
        assert ColumnType.CATEGORICAL_CYCLICAL == "categorical_cyclical"
        assert ColumnType.DATETIME == "datetime"
        assert ColumnType.BINARY == "binary"
        assert ColumnType.TEXT == "text"
        assert ColumnType.UNKNOWN == "unknown"

    def test_column_type_count(self):
        assert len(ColumnType) == 13


class TestColumnConfig:
    def test_create_minimal_column_config(self):
        col = ColumnConfig(name="custid", column_type=ColumnType.IDENTIFIER)
        assert col.name == "custid"
        assert col.column_type == ColumnType.IDENTIFIER
        assert col.nullable is True

    def test_create_with_all_attributes(self):
        col = ColumnConfig(
            name="age",
            column_type=ColumnType.NUMERIC_CONTINUOUS,
            nullable=False,
            encoding_strategy="standard",
            scaling_strategy="minmax",
            missing_strategy="mean",
            min_value=0,
            max_value=120,
            description="Customer age",
            business_name="Age",
            is_feature=True,
            exclude_from_model=False
        )
        assert col.name == "age"
        assert col.min_value == 0
        assert col.max_value == 120
        assert col.description == "Customer age"

    def test_cyclical_column_requires_cyclical_max(self):
        with pytest.raises(ValidationError, match="cyclical_max required"):
            ColumnConfig(name="favday", column_type=ColumnType.CATEGORICAL_CYCLICAL)

    def test_cyclical_column_with_cyclical_max(self):
        col = ColumnConfig(
            name="favday",
            column_type=ColumnType.CATEGORICAL_CYCLICAL,
            cyclical_max=7
        )
        assert col.cyclical_max == 7

    def test_ordinal_column_requires_ordinal_order(self):
        with pytest.raises(ValidationError, match="ordinal_order required"):
            ColumnConfig(name="satisfaction", column_type=ColumnType.CATEGORICAL_ORDINAL)

    def test_ordinal_column_with_ordinal_order(self):
        col = ColumnConfig(
            name="satisfaction",
            column_type=ColumnType.CATEGORICAL_ORDINAL,
            ordinal_order=["low", "medium", "high"]
        )
        assert col.ordinal_order == ["low", "medium", "high"]

    def test_should_be_used_as_feature_identifier(self):
        col = ColumnConfig(name="custid", column_type=ColumnType.IDENTIFIER)
        assert col.should_be_used_as_feature() is False

    def test_should_be_used_as_feature_target(self):
        col = ColumnConfig(name="retained", column_type=ColumnType.TARGET)
        assert col.should_be_used_as_feature() is False

    def test_should_be_used_as_feature_feature_timestamp(self):
        col = ColumnConfig(name="feature_ts", column_type=ColumnType.FEATURE_TIMESTAMP)
        assert col.should_be_used_as_feature() is False

    def test_should_be_used_as_feature_label_timestamp(self):
        col = ColumnConfig(name="label_ts", column_type=ColumnType.LABEL_TIMESTAMP)
        assert col.should_be_used_as_feature() is False

    def test_should_be_used_as_feature_numeric(self):
        col = ColumnConfig(name="age", column_type=ColumnType.NUMERIC_CONTINUOUS)
        assert col.should_be_used_as_feature() is True

    def test_should_be_used_as_feature_excluded(self):
        col = ColumnConfig(
            name="age",
            column_type=ColumnType.NUMERIC_CONTINUOUS,
            exclude_from_model=True
        )
        assert col.should_be_used_as_feature() is False

    def test_should_be_used_as_feature_explicit_true(self):
        col = ColumnConfig(
            name="custid",
            column_type=ColumnType.IDENTIFIER,
            is_feature=True
        )
        assert col.should_be_used_as_feature() is True

    def test_should_be_used_as_feature_explicit_false(self):
        col = ColumnConfig(
            name="age",
            column_type=ColumnType.NUMERIC_CONTINUOUS,
            is_feature=False
        )
        assert col.should_be_used_as_feature() is False

    def test_is_categorical_nominal(self):
        col = ColumnConfig(name="city", column_type=ColumnType.CATEGORICAL_NOMINAL)
        assert col.is_categorical() is True

    def test_is_categorical_ordinal(self):
        col = ColumnConfig(
            name="satisfaction",
            column_type=ColumnType.CATEGORICAL_ORDINAL,
            ordinal_order=["low", "high"]
        )
        assert col.is_categorical() is True

    def test_is_categorical_cyclical(self):
        col = ColumnConfig(
            name="favday",
            column_type=ColumnType.CATEGORICAL_CYCLICAL,
            cyclical_max=7
        )
        assert col.is_categorical() is True

    def test_is_categorical_binary(self):
        col = ColumnConfig(name="paperless", column_type=ColumnType.BINARY)
        assert col.is_categorical() is True

    def test_is_categorical_numeric(self):
        col = ColumnConfig(name="age", column_type=ColumnType.NUMERIC_CONTINUOUS)
        assert col.is_categorical() is False

    def test_is_numeric_continuous(self):
        col = ColumnConfig(name="age", column_type=ColumnType.NUMERIC_CONTINUOUS)
        assert col.is_numeric() is True

    def test_is_numeric_discrete(self):
        col = ColumnConfig(name="esent", column_type=ColumnType.NUMERIC_DISCRETE)
        assert col.is_numeric() is True

    def test_is_numeric_categorical(self):
        col = ColumnConfig(name="city", column_type=ColumnType.CATEGORICAL_NOMINAL)
        assert col.is_numeric() is False

    def test_is_temporal_datetime(self):
        col = ColumnConfig(name="created", column_type=ColumnType.DATETIME)
        assert col.is_temporal() is True

    def test_is_temporal_numeric(self):
        col = ColumnConfig(name="age", column_type=ColumnType.NUMERIC_CONTINUOUS)
        assert col.is_temporal() is False

    def test_nullable_default(self):
        col = ColumnConfig(name="age", column_type=ColumnType.NUMERIC_CONTINUOUS)
        assert col.nullable is True

    def test_nullable_explicit_false(self):
        col = ColumnConfig(
            name="retained",
            column_type=ColumnType.TARGET,
            nullable=False
        )
        assert col.nullable is False

    def test_json_serialization(self):
        col = ColumnConfig(name="age", column_type=ColumnType.NUMERIC_CONTINUOUS)
        json_data = col.model_dump()
        assert json_data["name"] == "age"
        assert json_data["column_type"] == "numeric_continuous"

    def test_json_deserialization(self):
        data = {"name": "age", "column_type": "numeric_continuous"}
        col = ColumnConfig(**data)
        assert col.name == "age"
        assert col.column_type == ColumnType.NUMERIC_CONTINUOUS

    def test_validation_rules(self):
        col = ColumnConfig(
            name="age",
            column_type=ColumnType.NUMERIC_CONTINUOUS,
            min_value=0,
            max_value=120,
            allowed_values=["adult"],
            regex_pattern=r"\d+"
        )
        assert col.min_value == 0
        assert col.max_value == 120
        assert col.allowed_values == ["adult"]
        assert col.regex_pattern == r"\d+"

    def test_should_be_used_as_feature_datetime_false_by_default(self):
        col = ColumnConfig(name="created_at", column_type=ColumnType.DATETIME)
        assert col.should_be_used_as_feature() is False

    def test_should_be_used_as_feature_datetime_explicit_override(self):
        col = ColumnConfig(name="created_at", column_type=ColumnType.DATETIME, is_feature=True)
        assert col.should_be_used_as_feature() is True

    def test_should_be_used_as_feature_text_false_by_default(self):
        col = ColumnConfig(name="notes", column_type=ColumnType.TEXT)
        assert col.should_be_used_as_feature() is False

    def test_should_be_used_as_feature_text_explicit_override(self):
        col = ColumnConfig(name="notes", column_type=ColumnType.TEXT, is_feature=True)
        assert col.should_be_used_as_feature() is True


class TestNonFeatureColumnTypes:
    def test_includes_identifier(self):
        assert ColumnType.IDENTIFIER in NON_FEATURE_COLUMN_TYPES

    def test_includes_target(self):
        assert ColumnType.TARGET in NON_FEATURE_COLUMN_TYPES

    def test_includes_feature_timestamp(self):
        assert ColumnType.FEATURE_TIMESTAMP in NON_FEATURE_COLUMN_TYPES

    def test_includes_label_timestamp(self):
        assert ColumnType.LABEL_TIMESTAMP in NON_FEATURE_COLUMN_TYPES

    def test_includes_datetime(self):
        assert ColumnType.DATETIME in NON_FEATURE_COLUMN_TYPES

    def test_includes_text(self):
        assert ColumnType.TEXT in NON_FEATURE_COLUMN_TYPES

    def test_excludes_numeric_continuous(self):
        assert ColumnType.NUMERIC_CONTINUOUS not in NON_FEATURE_COLUMN_TYPES

    def test_excludes_categorical_nominal(self):
        assert ColumnType.CATEGORICAL_NOMINAL not in NON_FEATURE_COLUMN_TYPES

    def test_excludes_binary(self):
        assert ColumnType.BINARY not in NON_FEATURE_COLUMN_TYPES


class TestNonFeatureDtypes:
    def test_includes_datetime64(self):
        assert "datetime64" in NON_FEATURE_DTYPES

    def test_includes_datetimetz(self):
        assert "datetimetz" in NON_FEATURE_DTYPES

    def test_includes_timedelta64(self):
        assert "timedelta64" in NON_FEATURE_DTYPES

    def test_is_frozenset(self):
        assert isinstance(NON_FEATURE_DTYPES, frozenset)


class TestSelectModelReadyColumns:
    def test_excludes_datetime64(self):
        df = pd.DataFrame({"dt": pd.date_range("2024-01-01", periods=3), "val": [1.0, 2.0, 3.0]})
        result = select_model_ready_columns(df)
        assert "dt" not in result.columns
        assert "val" in result.columns

    def test_excludes_datetimetz(self):
        df = pd.DataFrame({"dt": pd.date_range("2024-01-01", periods=3, tz="UTC"), "val": [1, 2, 3]})
        result = select_model_ready_columns(df)
        assert "dt" not in result.columns
        assert "val" in result.columns

    def test_excludes_timedelta(self):
        df = pd.DataFrame({"td": pd.to_timedelta([1, 2, 3], unit="D"), "val": [1.0, 2.0, 3.0]})
        result = select_model_ready_columns(df)
        assert "td" not in result.columns
        assert "val" in result.columns

    def test_preserves_numeric_bool_object(self):
        df = pd.DataFrame({
            "num": [1.0, 2.0], "integer": [1, 2],
            "flag": [True, False], "cat": ["a", "b"],
        })
        result = select_model_ready_columns(df)
        assert list(result.columns) == ["num", "integer", "flag", "cat"]

    def test_preserves_nullable_int(self):
        df = pd.DataFrame({"val": pd.array([1, 2, None], dtype="Int64")})
        result = select_model_ready_columns(df)
        assert "val" in result.columns

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = select_model_ready_columns(df)
        assert len(result.columns) == 0
