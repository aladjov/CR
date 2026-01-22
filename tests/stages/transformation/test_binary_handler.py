import pandas as pd
import pytest

from customer_retention.stages.transformation import BinaryHandler


class TestBinaryStandardization:
    def test_standardize_0_1(self):
        series = pd.Series([0, 1, 0, 1, 0])
        handler = BinaryHandler()
        result = handler.fit_transform(series)

        assert set(result.series.unique()) == {0, 1}

    def test_standardize_true_false(self):
        series = pd.Series([True, False, True, False])
        handler = BinaryHandler()
        result = handler.fit_transform(series)

        assert set(result.series.unique()) == {0, 1}
        assert result.series[series].iloc[0] == 1

    def test_standardize_yes_no(self):
        series = pd.Series(["Yes", "No", "Yes", "No", "Yes"])
        handler = BinaryHandler()
        result = handler.fit_transform(series)

        assert set(result.series.unique()) == {0, 1}
        assert result.series[series == "Yes"].iloc[0] == 1

    def test_standardize_y_n(self):
        series = pd.Series(["Y", "N", "Y", "N"])
        handler = BinaryHandler()
        result = handler.fit_transform(series)

        assert set(result.series.unique()) == {0, 1}

    def test_standardize_1_2_shift(self):
        series = pd.Series([1, 2, 1, 2, 1])
        handler = BinaryHandler()
        result = handler.fit_transform(series)

        assert set(result.series.unique()) == {0, 1}


class TestPositiveClassMapping:
    def test_explicit_positive_class(self):
        series = pd.Series(["Active", "Inactive", "Active"])
        handler = BinaryHandler(positive_class="Active")
        result = handler.fit_transform(series)

        assert result.series[series == "Active"].iloc[0] == 1
        assert result.series[series == "Inactive"].iloc[0] == 0

    def test_inferred_positive_class(self):
        series = pd.Series([1, 0, 1, 0])
        handler = BinaryHandler()
        result = handler.fit_transform(series)

        assert result.positive_class is not None


class TestResultOutput:
    def test_result_contains_mapping(self):
        series = pd.Series(["Yes", "No", "Yes"])
        handler = BinaryHandler()
        result = handler.fit_transform(series)

        assert result.mapping is not None
        assert "Yes" in result.mapping or "No" in result.mapping

    def test_result_contains_original_values(self):
        series = pd.Series(["Active", "Inactive"])
        handler = BinaryHandler()
        result = handler.fit_transform(series)

        assert result.original_values is not None
        assert "Active" in result.original_values
        assert "Inactive" in result.original_values


class TestFitTransformSeparation:
    def test_fit_stores_mapping(self):
        series = pd.Series(["Yes", "No", "Yes"])
        handler = BinaryHandler()
        handler.fit(series)

        assert handler._mapping is not None

    def test_transform_uses_fitted_mapping(self):
        train = pd.Series(["Yes", "No", "Yes"])
        test = pd.Series(["No", "Yes", "No"])

        handler = BinaryHandler()
        handler.fit(train)
        result = handler.transform(test)

        assert result.series[0] == 0
        assert result.series[1] == 1

    def test_transform_without_fit_raises_error(self):
        handler = BinaryHandler()
        with pytest.raises(ValueError, match="not fitted"):
            handler.transform(pd.Series(["Yes", "No"]))


class TestEdgeCases:
    def test_handles_nulls(self):
        series = pd.Series([1, 0, None, 1, None])
        handler = BinaryHandler()
        result = handler.fit_transform(series)

        assert pd.isna(result.series[2])
        assert pd.isna(result.series[4])
        assert result.series[0] == 1
        assert result.series[1] == 0

    def test_all_same_value(self):
        series = pd.Series([1, 1, 1, 1])
        handler = BinaryHandler()
        result = handler.fit_transform(series)

        assert (result.series == 1).all()

    def test_mixed_case_yes_no(self):
        series = pd.Series(["YES", "no", "Yes", "NO"])
        handler = BinaryHandler()
        result = handler.fit_transform(series)

        assert set(result.series.unique()) == {0, 1}
