import pytest
import pandas as pd
import numpy as np
from customer_retention.core.config import ColumnType
from customer_retention.stages.cleaning import (
    MissingValueHandler, ImputationStrategy, ImputationResult
)


class TestImputationStrategies:
    @pytest.fixture
    def numeric_series_with_nulls(self):
        return pd.Series([1.0, 2.0, None, 4.0, 5.0, None, 7.0, 8.0, 9.0, 10.0])

    @pytest.fixture
    def categorical_series_with_nulls(self):
        return pd.Series(["A", "B", None, "A", "B", "A", None, "C", "A", "B"])

    def test_mean_imputation(self, numeric_series_with_nulls):
        handler = MissingValueHandler(strategy=ImputationStrategy.MEAN)
        result = handler.fit_transform(numeric_series_with_nulls)

        assert result.series.isna().sum() == 0
        expected_mean = numeric_series_with_nulls.dropna().mean()
        assert result.series[2] == pytest.approx(expected_mean)
        assert result.strategy_used == ImputationStrategy.MEAN
        assert result.values_imputed == 2

    def test_median_imputation(self, numeric_series_with_nulls):
        handler = MissingValueHandler(strategy=ImputationStrategy.MEDIAN)
        result = handler.fit_transform(numeric_series_with_nulls)

        assert result.series.isna().sum() == 0
        expected_median = numeric_series_with_nulls.dropna().median()
        assert result.series[2] == pytest.approx(expected_median)
        assert result.strategy_used == ImputationStrategy.MEDIAN
        assert result.values_imputed == 2

    def test_mode_imputation(self, categorical_series_with_nulls):
        handler = MissingValueHandler(strategy=ImputationStrategy.MODE)
        result = handler.fit_transform(categorical_series_with_nulls)

        assert result.series.isna().sum() == 0
        assert result.series[2] == "A"
        assert result.strategy_used == ImputationStrategy.MODE
        assert result.values_imputed == 2

    def test_constant_imputation_numeric(self, numeric_series_with_nulls):
        handler = MissingValueHandler(strategy=ImputationStrategy.CONSTANT, fill_value=0.0)
        result = handler.fit_transform(numeric_series_with_nulls)

        assert result.series.isna().sum() == 0
        assert result.series[2] == 0.0
        assert result.fill_value == 0.0

    def test_constant_imputation_categorical(self, categorical_series_with_nulls):
        handler = MissingValueHandler(strategy=ImputationStrategy.CONSTANT, fill_value="Unknown")
        result = handler.fit_transform(categorical_series_with_nulls)

        assert result.series.isna().sum() == 0
        assert result.series[2] == "Unknown"

    def test_drop_row_returns_mask(self, numeric_series_with_nulls):
        handler = MissingValueHandler(strategy=ImputationStrategy.DROP_ROW)
        result = handler.fit_transform(numeric_series_with_nulls)

        assert result.rows_dropped == 2
        assert len(result.drop_mask) == 10
        assert result.drop_mask[2] is True
        assert result.drop_mask[5] is True


class TestMissingIndicator:
    def test_creates_missing_indicator_when_configured(self):
        series = pd.Series([1.0, None, 3.0, None, 5.0])
        handler = MissingValueHandler(strategy=ImputationStrategy.MEAN, add_indicator=True)
        result = handler.fit_transform(series)

        assert result.indicator_column is not None
        assert result.indicator_column.sum() == 2
        assert result.indicator_column[1] == 1
        assert result.indicator_column[3] == 1
        assert result.indicator_column[0] == 0


class TestStrategySelectionByColumnType:
    def test_default_strategy_for_numeric_continuous(self):
        handler = MissingValueHandler.from_column_type(ColumnType.NUMERIC_CONTINUOUS)
        assert handler.strategy == ImputationStrategy.MEDIAN

    def test_default_strategy_for_numeric_discrete(self):
        handler = MissingValueHandler.from_column_type(ColumnType.NUMERIC_DISCRETE)
        assert handler.strategy == ImputationStrategy.MODE

    def test_default_strategy_for_categorical_nominal(self):
        handler = MissingValueHandler.from_column_type(ColumnType.CATEGORICAL_NOMINAL)
        assert handler.strategy == ImputationStrategy.MODE

    def test_default_strategy_for_binary(self):
        handler = MissingValueHandler.from_column_type(ColumnType.BINARY)
        assert handler.strategy == ImputationStrategy.MODE

    def test_default_strategy_for_datetime(self):
        handler = MissingValueHandler.from_column_type(ColumnType.DATETIME)
        assert handler.strategy == ImputationStrategy.DROP_ROW

    def test_default_strategy_for_identifier_raises_error(self):
        handler = MissingValueHandler.from_column_type(ColumnType.IDENTIFIER)
        series = pd.Series(["ID001", None, "ID003"])
        with pytest.raises(ValueError, match="should not have missing"):
            handler.fit_transform(series)

    def test_default_strategy_for_target_is_drop_row(self):
        handler = MissingValueHandler.from_column_type(ColumnType.TARGET)
        assert handler.strategy == ImputationStrategy.DROP_ROW

    def test_default_strategy_for_categorical_ordinal(self):
        handler = MissingValueHandler.from_column_type(ColumnType.CATEGORICAL_ORDINAL)
        assert handler.strategy == ImputationStrategy.MODE

    def test_default_strategy_for_categorical_cyclical(self):
        handler = MissingValueHandler.from_column_type(ColumnType.CATEGORICAL_CYCLICAL)
        assert handler.strategy == ImputationStrategy.MODE

    def test_default_strategy_for_text_is_constant(self):
        handler = MissingValueHandler.from_column_type(ColumnType.TEXT)
        assert handler.strategy == ImputationStrategy.CONSTANT
        assert handler.fill_value == ""


class TestAdvancedImputation:
    @pytest.fixture
    def time_series_with_nulls(self):
        return pd.Series([1.0, 2.0, None, None, 5.0, 6.0, None, 8.0])

    def test_forward_fill_imputation(self, time_series_with_nulls):
        handler = MissingValueHandler(strategy=ImputationStrategy.FORWARD_FILL)
        result = handler.fit_transform(time_series_with_nulls)

        assert result.series[2] == 2.0
        assert result.series[3] == 2.0
        assert result.series[6] == 6.0

    def test_backward_fill_imputation(self, time_series_with_nulls):
        handler = MissingValueHandler(strategy=ImputationStrategy.BACKWARD_FILL)
        result = handler.fit_transform(time_series_with_nulls)

        assert result.series[2] == 5.0
        assert result.series[3] == 5.0
        assert result.series[6] == 8.0

    def test_interpolate_imputation(self, time_series_with_nulls):
        handler = MissingValueHandler(strategy=ImputationStrategy.INTERPOLATE)
        result = handler.fit_transform(time_series_with_nulls)

        assert result.series[2] == pytest.approx(3.0)
        assert result.series[3] == pytest.approx(4.0)
        assert result.series[6] == pytest.approx(7.0)


class TestKNNImputation:
    @pytest.mark.skipif(
        not hasattr(__import__('importlib').util.find_spec('sklearn'), '__name__') if __import__('importlib').util.find_spec('sklearn') else True,
        reason="sklearn not available"
    )
    def test_knn_imputation_numeric(self):
        pytest.importorskip("sklearn")
        df = pd.DataFrame({
            "A": [1.0, 2.0, None, 4.0, 5.0],
            "B": [1.1, 2.1, 3.1, 4.1, 5.1],
        })
        handler = MissingValueHandler(strategy=ImputationStrategy.KNN, knn_neighbors=2)
        result = handler.fit_transform(df["A"], reference_df=df)

        assert result.series.isna().sum() == 0
        assert 2.0 <= result.series[2] <= 4.0


class TestFitTransformSeparation:
    def test_fit_then_transform_on_new_data(self):
        train_series = pd.Series([1.0, 2.0, None, 4.0, 5.0, 6.0])
        test_series = pd.Series([None, 3.0, None, 7.0])

        handler = MissingValueHandler(strategy=ImputationStrategy.MEAN)
        handler.fit(train_series)
        result = handler.transform(test_series)

        expected_fill = train_series.dropna().mean()
        assert result.series[0] == pytest.approx(expected_fill)
        assert result.series[2] == pytest.approx(expected_fill)

    def test_transform_without_fit_raises_error(self):
        handler = MissingValueHandler(strategy=ImputationStrategy.MEAN)
        with pytest.raises(ValueError, match="not fitted"):
            handler.transform(pd.Series([1.0, None, 3.0]))


class TestEdgeCases:
    def test_no_missing_values_returns_unchanged(self):
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        handler = MissingValueHandler(strategy=ImputationStrategy.MEAN)
        result = handler.fit_transform(series)

        assert result.values_imputed == 0
        pd.testing.assert_series_equal(result.series, series)

    def test_all_missing_values(self):
        series = pd.Series([None, None, None])
        handler = MissingValueHandler(strategy=ImputationStrategy.MEAN)

        with pytest.raises(ValueError, match="all values are missing"):
            handler.fit_transform(series)

    def test_mode_with_multiple_modes_uses_first(self):
        series = pd.Series(["A", "A", None, "B", "B"])
        handler = MissingValueHandler(strategy=ImputationStrategy.MODE)
        result = handler.fit_transform(series)

        assert result.series[2] in ["A", "B"]
