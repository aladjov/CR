import pandas as pd
import pytest

from customer_retention.core.config import ColumnType
from customer_retention.stages.profiling import ColumnProfiler, ProfilerFactory


@pytest.fixture
def sample_text_series():
    """Sample text data."""
    return pd.Series([
        "This is a normal text string",
        "Another example with some words",
        "Short text",
        "This is a much longer text string with many more words to analyze",
        "Text with numbers 123 and symbols !@#",
        "email@example.com should be detected",
        "Phone: 555-123-4567",
        None,
        "",
        "Final text entry"
    ])


@pytest.fixture
def pii_text_series():
    """Text data with PII."""
    return pd.Series([
        "Contact me at john.doe@example.com",
        "My phone is 555-123-4567",
        "SSN: 123-45-6789",
        "Credit card: 4532-1234-5678-9010",
        "Regular text without PII",
        "Another email: jane@company.org"
    ])


class TestTextProfiler:
    def test_profiler_exists(self):
        profiler = ProfilerFactory.get_profiler(ColumnType.TEXT)
        assert profiler is not None
        assert isinstance(profiler, ColumnProfiler)

    def test_profile_returns_text_metrics(self, sample_text_series):
        profiler = ProfilerFactory.get_profiler(ColumnType.TEXT)
        result = profiler.profile(sample_text_series)

        assert "text_metrics" in result
        assert result["text_metrics"] is not None

    def test_text_length_metrics(self, sample_text_series):
        profiler = ProfilerFactory.get_profiler(ColumnType.TEXT)
        result = profiler.profile(sample_text_series)

        text_metrics = result["text_metrics"]
        assert hasattr(text_metrics, "length_min")
        assert hasattr(text_metrics, "length_max")
        assert hasattr(text_metrics, "length_mean")
        assert hasattr(text_metrics, "length_median")

        assert text_metrics.length_min >= 0
        assert text_metrics.length_max > 0
        assert text_metrics.length_mean > 0

    def test_empty_text_detection(self, sample_text_series):
        profiler = ProfilerFactory.get_profiler(ColumnType.TEXT)
        result = profiler.profile(sample_text_series)

        text_metrics = result["text_metrics"]
        assert hasattr(text_metrics, "empty_count")
        assert hasattr(text_metrics, "empty_percentage")
        assert text_metrics.empty_count >= 1  # We have one empty string
        assert text_metrics.empty_percentage > 0

    def test_word_count_metrics(self, sample_text_series):
        profiler = ProfilerFactory.get_profiler(ColumnType.TEXT)
        result = profiler.profile(sample_text_series)

        text_metrics = result["text_metrics"]
        assert hasattr(text_metrics, "word_count_mean")
        assert text_metrics.word_count_mean > 0

    def test_contains_digits_percentage(self, sample_text_series):
        profiler = ProfilerFactory.get_profiler(ColumnType.TEXT)
        result = profiler.profile(sample_text_series)

        text_metrics = result["text_metrics"]
        assert hasattr(text_metrics, "contains_digits_pct")
        assert text_metrics.contains_digits_pct >= 0
        assert text_metrics.contains_digits_pct <= 100

    def test_contains_special_chars_percentage(self, sample_text_series):
        profiler = ProfilerFactory.get_profiler(ColumnType.TEXT)
        result = profiler.profile(sample_text_series)

        text_metrics = result["text_metrics"]
        assert hasattr(text_metrics, "contains_special_pct")
        assert text_metrics.contains_special_pct >= 0
        assert text_metrics.contains_special_pct <= 100


class TestPIIDetection:
    def test_pii_detected_flag(self, pii_text_series):
        profiler = ProfilerFactory.get_profiler(ColumnType.TEXT)
        result = profiler.profile(pii_text_series)

        text_metrics = result["text_metrics"]
        assert hasattr(text_metrics, "pii_detected")
        assert text_metrics.pii_detected is True

    def test_pii_types_detected(self, pii_text_series):
        profiler = ProfilerFactory.get_profiler(ColumnType.TEXT)
        result = profiler.profile(pii_text_series)

        text_metrics = result["text_metrics"]
        assert hasattr(text_metrics, "pii_types")
        assert isinstance(text_metrics.pii_types, list)
        assert len(text_metrics.pii_types) > 0

    def test_email_detection(self):
        series = pd.Series(["Contact: john.doe@example.com", "No PII here"])
        profiler = ProfilerFactory.get_profiler(ColumnType.TEXT)
        result = profiler.profile(series)

        text_metrics = result["text_metrics"]
        assert text_metrics.pii_detected is True
        assert "email" in text_metrics.pii_types

    def test_phone_detection(self):
        series = pd.Series(["Call me at 555-123-4567", "No PII here"])
        profiler = ProfilerFactory.get_profiler(ColumnType.TEXT)
        result = profiler.profile(series)

        text_metrics = result["text_metrics"]
        assert text_metrics.pii_detected is True
        assert "phone" in text_metrics.pii_types

    def test_ssn_detection(self):
        series = pd.Series(["SSN: 123-45-6789", "Regular text"])
        profiler = ProfilerFactory.get_profiler(ColumnType.TEXT)
        result = profiler.profile(series)

        text_metrics = result["text_metrics"]
        assert text_metrics.pii_detected is True
        assert "ssn" in text_metrics.pii_types

    def test_credit_card_detection(self):
        series = pd.Series(["Card: 4532-1234-5678-9010", "Regular text"])
        profiler = ProfilerFactory.get_profiler(ColumnType.TEXT)
        result = profiler.profile(series)

        text_metrics = result["text_metrics"]
        assert text_metrics.pii_detected is True
        assert "credit_card" in text_metrics.pii_types

    def test_no_pii_in_clean_text(self, sample_text_series):
        # Remove the entries with email and phone
        clean_series = pd.Series([
            "This is a normal text string",
            "Another example with some words",
            "Short text"
        ])
        profiler = ProfilerFactory.get_profiler(ColumnType.TEXT)
        result = profiler.profile(clean_series)

        text_metrics = result["text_metrics"]
        assert text_metrics.pii_detected is False
        assert len(text_metrics.pii_types) == 0


class TestTextQualityChecks:
    def test_pii_check_critical(self, pii_text_series):
        """Test that PII detection triggers CRITICAL quality check."""
        from customer_retention.stages.profiling.quality_checks import PIIDetectedCheck

        check = PIIDetectedCheck()
        profiler = ProfilerFactory.get_profiler(ColumnType.TEXT)
        result = profiler.profile(pii_text_series)

        check_result = check.run("test_col", result["text_metrics"])

        assert check_result is not None
        assert check_result.passed is False
        assert check_result.severity.value == "critical"

    def test_empty_text_check(self):
        """Test excessive empty text check."""
        from customer_retention.stages.profiling.quality_checks import EmptyTextCheck

        # >50% empty
        series = pd.Series(["text"] * 40 + [""] * 60)
        profiler = ProfilerFactory.get_profiler(ColumnType.TEXT)
        result = profiler.profile(series)

        check = EmptyTextCheck()
        check_result = check.run("test_col", result["text_metrics"])

        assert check_result is not None
        assert check_result.passed is False

    def test_short_text_check(self):
        """Test very short text average check."""
        from customer_retention.stages.profiling.quality_checks import ShortTextCheck

        # Very short texts (< 10 chars avg)
        series = pd.Series(["a", "bb", "ccc", "dd", "e"])
        profiler = ProfilerFactory.get_profiler(ColumnType.TEXT)
        result = profiler.profile(series)

        check = ShortTextCheck()
        check_result = check.run("test_col", result["text_metrics"])

        assert check_result is not None
        assert check_result.passed is False
