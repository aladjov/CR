import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.transformation import CategoricalEncoder, EncodingStrategy


class TestOneHotEncoding:
    @pytest.fixture
    def categorical_series(self):
        return pd.Series(["A", "B", "C", "A", "B", "A"])

    def test_one_hot_creates_columns(self, categorical_series):
        encoder = CategoricalEncoder(strategy=EncodingStrategy.ONE_HOT)
        result = encoder.fit_transform(categorical_series)

        assert len(result.columns_created) == 2  # n-1 with drop_first=True

    def test_one_hot_drop_first(self, categorical_series):
        encoder = CategoricalEncoder(strategy=EncodingStrategy.ONE_HOT, drop_first=True)
        result = encoder.fit_transform(categorical_series)

        unique_values = categorical_series.nunique()
        assert len(result.columns_created) == unique_values - 1

    def test_one_hot_no_drop_first(self, categorical_series):
        encoder = CategoricalEncoder(strategy=EncodingStrategy.ONE_HOT, drop_first=False)
        result = encoder.fit_transform(categorical_series)

        unique_values = categorical_series.nunique()
        assert len(result.columns_created) == unique_values

    def test_one_hot_values_are_binary(self, categorical_series):
        encoder = CategoricalEncoder(strategy=EncodingStrategy.ONE_HOT, drop_first=False)
        result = encoder.fit_transform(categorical_series)

        for col in result.df.columns:
            assert set(result.df[col].unique()).issubset({0, 1})

    def test_one_hot_handles_unknown_categories(self):
        train = pd.Series(["A", "B", "C"])
        test = pd.Series(["A", "B", "D"])

        encoder = CategoricalEncoder(strategy=EncodingStrategy.ONE_HOT, handle_unknown="ignore")
        encoder.fit(train)
        result = encoder.transform(test)

        assert result.df.shape[0] == 3


class TestLabelEncoding:
    def test_label_encoding(self):
        series = pd.Series(["cat", "dog", "bird", "cat", "dog"])
        encoder = CategoricalEncoder(strategy=EncodingStrategy.LABEL)
        result = encoder.fit_transform(series)

        assert result.series is not None
        assert result.series.dtype in [np.int64, np.int32, int]
        assert len(result.mapping) == 3

    def test_label_encoding_mapping_preserved(self):
        train = pd.Series(["A", "B", "C"])
        test = pd.Series(["B", "A", "C"])

        encoder = CategoricalEncoder(strategy=EncodingStrategy.LABEL)
        encoder.fit(train)

        train_result = encoder.transform(train)
        test_result = encoder.transform(test)

        assert train_result.series[train == "A"].iloc[0] == test_result.series[test == "A"].iloc[0]


class TestOrdinalEncoding:
    def test_ordinal_encoding_respects_order(self):
        series = pd.Series(["Low", "Medium", "High", "Low", "High"])
        categories = ["Low", "Medium", "High"]

        encoder = CategoricalEncoder(strategy=EncodingStrategy.ORDINAL, categories=categories)
        result = encoder.fit_transform(series)

        assert result.series[series == "Low"].iloc[0] == 0
        assert result.series[series == "Medium"].iloc[0] == 1
        assert result.series[series == "High"].iloc[0] == 2

    def test_ordinal_encoding_unknown_raises_by_default(self):
        series = pd.Series(["Low", "Medium", "High"])
        categories = ["Low", "Medium", "High"]

        encoder = CategoricalEncoder(strategy=EncodingStrategy.ORDINAL, categories=categories)
        encoder.fit(series)

        with pytest.raises(ValueError, match="unknown"):
            encoder.transform(pd.Series(["Low", "Very High"]))


class TestCyclicalEncoding:
    def test_cyclical_encoding_creates_two_columns(self):
        series = pd.Series([0, 1, 2, 3, 4, 5, 6])
        encoder = CategoricalEncoder(strategy=EncodingStrategy.CYCLICAL, period=7)
        result = encoder.fit_transform(series)

        assert len(result.columns_created) == 2
        assert "_sin" in result.columns_created[0]
        assert "_cos" in result.columns_created[1]

    def test_cyclical_encoding_sin_cos_range(self):
        series = pd.Series([0, 1, 2, 3, 4, 5, 6])
        encoder = CategoricalEncoder(strategy=EncodingStrategy.CYCLICAL, period=7)
        result = encoder.fit_transform(series)

        sin_col = [c for c in result.df.columns if "_sin" in c][0]
        cos_col = [c for c in result.df.columns if "_cos" in c][0]

        assert result.df[sin_col].min() >= -1
        assert result.df[sin_col].max() <= 1
        assert result.df[cos_col].min() >= -1
        assert result.df[cos_col].max() <= 1

    def test_cyclical_encoding_day_of_week(self):
        series = pd.Series([0, 1, 2, 3, 4, 5, 6])  # Monday to Sunday
        encoder = CategoricalEncoder(strategy=EncodingStrategy.CYCLICAL, period=7)
        result = encoder.fit_transform(series)

        sin_col = [c for c in result.df.columns if "_sin" in c][0]
        cos_col = [c for c in result.df.columns if "_cos" in c][0]

        assert result.df[cos_col].iloc[0] == pytest.approx(1.0, abs=0.01)


class TestFrequencyEncoding:
    def test_frequency_encoding(self):
        series = pd.Series(["A", "A", "A", "B", "B", "C"])
        encoder = CategoricalEncoder(strategy=EncodingStrategy.FREQUENCY)
        result = encoder.fit_transform(series)

        assert result.series[series == "A"].iloc[0] == pytest.approx(3/6)
        assert result.series[series == "B"].iloc[0] == pytest.approx(2/6)
        assert result.series[series == "C"].iloc[0] == pytest.approx(1/6)


class TestTargetEncoding:
    def test_target_encoding_basic(self):
        series = pd.Series(["A", "A", "A", "B", "B", "C"])
        target = pd.Series([1, 1, 0, 0, 0, 1])

        encoder = CategoricalEncoder(strategy=EncodingStrategy.TARGET)
        result = encoder.fit_transform(series, target=target)

        a_mean = target[series == "A"].mean()
        assert result.series[series == "A"].iloc[0] == pytest.approx(a_mean, abs=0.1)

    def test_target_encoding_with_smoothing(self):
        series = pd.Series(["A", "A", "A", "B", "B", "C"])
        target = pd.Series([1, 1, 0, 0, 0, 1])

        encoder = CategoricalEncoder(strategy=EncodingStrategy.TARGET, smoothing=1.0)
        result = encoder.fit_transform(series, target=target)

        assert result.series is not None


class TestFitTransformSeparation:
    def test_fit_stores_mapping(self):
        series = pd.Series(["A", "B", "C"])
        encoder = CategoricalEncoder(strategy=EncodingStrategy.LABEL)
        encoder.fit(series)

        assert encoder._mapping is not None
        assert len(encoder._mapping) == 3

    def test_transform_without_fit_raises_error(self):
        encoder = CategoricalEncoder(strategy=EncodingStrategy.LABEL)
        with pytest.raises(ValueError, match="not fitted"):
            encoder.transform(pd.Series(["A", "B"]))


class TestResultOutput:
    def test_result_contains_mapping(self):
        series = pd.Series(["A", "B", "C"])
        encoder = CategoricalEncoder(strategy=EncodingStrategy.LABEL)
        result = encoder.fit_transform(series)

        assert result.mapping is not None
        assert len(result.mapping) == 3

    def test_result_contains_columns_created_for_one_hot(self):
        series = pd.Series(["A", "B", "C"])
        encoder = CategoricalEncoder(strategy=EncodingStrategy.ONE_HOT)
        result = encoder.fit_transform(series)

        assert result.columns_created is not None
        assert len(result.columns_created) > 0


class TestEdgeCases:
    def test_handles_nulls(self):
        series = pd.Series(["A", "B", None, "A", None])
        encoder = CategoricalEncoder(strategy=EncodingStrategy.LABEL)
        result = encoder.fit_transform(series)

        assert pd.isna(result.series[2])
        assert pd.isna(result.series[4])

    def test_single_category(self):
        series = pd.Series(["A", "A", "A"])
        encoder = CategoricalEncoder(strategy=EncodingStrategy.ONE_HOT, drop_first=True)
        result = encoder.fit_transform(series)

        assert len(result.columns_created) == 0

    def test_rare_category_handling(self):
        series = pd.Series(["A"] * 100 + ["B"] * 50 + ["C", "D", "E"])
        encoder = CategoricalEncoder(strategy=EncodingStrategy.ONE_HOT, min_frequency=5)
        result = encoder.fit_transform(series)

        assert any("other" in col.lower() or len(result.columns_created) < 5 for col in result.columns_created)
