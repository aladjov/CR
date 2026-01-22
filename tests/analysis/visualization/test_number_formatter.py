import pytest

from customer_retention.analysis.visualization.number_formatter import NumberFormatter


class TestNumberFormatterCompact:
    @pytest.fixture
    def formatter(self):
        return NumberFormatter()

    def test_small_numbers_unchanged(self, formatter):
        assert formatter.compact(0) == "0.00"
        assert formatter.compact(1) == "1.00"
        assert formatter.compact(999) == "999.00"

    def test_thousands(self, formatter):
        assert formatter.compact(1000) == "1.00K"
        assert formatter.compact(1500) == "1.50K"
        assert formatter.compact(12345) == "12.35K"
        assert formatter.compact(999999) == "1000.00K"

    def test_millions(self, formatter):
        assert formatter.compact(1000000) == "1.00M"
        assert formatter.compact(2500000) == "2.50M"
        assert formatter.compact(12345678) == "12.35M"

    def test_billions(self, formatter):
        assert formatter.compact(1000000000) == "1.00B"
        assert formatter.compact(7890000000) == "7.89B"

    def test_trillions(self, formatter):
        assert formatter.compact(1000000000000) == "1.00T"
        assert formatter.compact(2500000000000) == "2.50T"

    def test_negative_numbers(self, formatter):
        assert formatter.compact(-1000) == "-1.00K"
        assert formatter.compact(-2500000) == "-2.50M"
        assert formatter.compact(-500) == "-500.00"

    def test_custom_precision(self, formatter):
        assert formatter.compact(1234, precision=0) == "1K"
        assert formatter.compact(1234, precision=1) == "1.2K"
        assert formatter.compact(1234, precision=3) == "1.234K"

    def test_decimals_preserved_for_small(self, formatter):
        assert formatter.compact(0.5) == "0.50"
        assert formatter.compact(12.345) == "12.35"
        assert formatter.compact(99.999) == "100.00"


class TestNumberFormatterPercentage:
    @pytest.fixture
    def formatter(self):
        return NumberFormatter()

    def test_positive_percentage(self, formatter):
        assert formatter.percentage(25.5) == "+25.50%"
        assert formatter.percentage(100) == "+100.00%"

    def test_negative_percentage(self, formatter):
        assert formatter.percentage(-15.3) == "-15.30%"

    def test_zero_percentage(self, formatter):
        assert formatter.percentage(0) == "+0.00%"

    def test_custom_precision(self, formatter):
        assert formatter.percentage(25.555, precision=1) == "+25.6%"
        assert formatter.percentage(25.555, precision=0) == "+26%"

    def test_without_sign(self, formatter):
        assert formatter.percentage(25.5, show_sign=False) == "25.50%"
        assert formatter.percentage(-15.3, show_sign=False) == "-15.30%"


class TestNumberFormatterRate:
    @pytest.fixture
    def formatter(self):
        return NumberFormatter()

    def test_rate_with_suffix(self, formatter):
        assert formatter.rate(2.5, "/mo") == "+2.50/mo"
        assert formatter.rate(-1.2, "/day") == "-1.20/day"

    def test_rate_precision(self, formatter):
        assert formatter.rate(2.555, "/mo", precision=1) == "+2.6/mo"


class TestNumberFormatterPlotlyFormat:
    @pytest.fixture
    def formatter(self):
        return NumberFormatter()

    def test_plotly_format_for_indicator(self, formatter):
        fmt = formatter.plotly_format(precision=2, show_sign=True)
        assert fmt == "+.2f"

    def test_plotly_format_without_sign(self, formatter):
        fmt = formatter.plotly_format(precision=2, show_sign=False)
        assert fmt == ".2f"

    def test_plotly_format_custom_precision(self, formatter):
        assert formatter.plotly_format(precision=0, show_sign=True) == "+.0f"
        assert formatter.plotly_format(precision=3, show_sign=False) == ".3f"
