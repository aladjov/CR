import pytest
from pydantic import ValidationError

from customer_retention.core.config import ColumnConfig, ColumnType


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
