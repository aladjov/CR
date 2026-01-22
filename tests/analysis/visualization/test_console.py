"""Tests for console output utilities."""

import pytest
from unittest.mock import patch, MagicMock
from customer_retention.analysis.visualization import console


class TestBar:
    def test_zero_percent(self):
        result = console._bar(0, width=10)
        assert result == "░░░░░░░░░░"

    def test_hundred_percent_no_empty(self):
        result = console._bar(100, width=10)
        assert result == "██████████"
        assert "░" not in result

    def test_fifty_percent(self):
        result = console._bar(50, width=10)
        assert result == "█████░░░░░"

    def test_clamps_negative(self):
        result = console._bar(-10, width=10)
        assert "░░░░░░░░░░" in result

    def test_clamps_over_hundred(self):
        result = console._bar(150, width=10)
        assert result == "██████████"


class TestMarkdownOutput:
    @patch.object(console, '_add')
    def test_header_uppercase(self, mock_add):
        console.header("test header")
        mock_add.assert_called_once()
        assert "TEST HEADER" in mock_add.call_args[0][0]
        assert "####" in mock_add.call_args[0][0]

    @patch.object(console, '_add')
    def test_subheader_bold(self, mock_add):
        console.subheader("section")
        assert "**section**" in mock_add.call_args[0][0]

    @patch.object(console, '_add')
    def test_success_has_ok(self, mock_add):
        console.success("done")
        assert "[OK]" in mock_add.call_args[0][0]

    @patch.object(console, '_add')
    def test_warning_has_exclamation(self, mock_add):
        console.warning("caution")
        assert "[!]" in mock_add.call_args[0][0]

    @patch.object(console, '_add')
    def test_error_has_x(self, mock_add):
        console.error("failed")
        assert "[X]" in mock_add.call_args[0][0]

    @patch.object(console, '_add')
    def test_info_italic(self, mock_add):
        console.info("hint")
        assert "(i)" in mock_add.call_args[0][0]
        assert "*" in mock_add.call_args[0][0]

    @patch.object(console, '_add')
    def test_metric_value_bold(self, mock_add):
        console.metric("Count", 42)
        assert "Count:" in mock_add.call_args[0][0]
        assert "**42**" in mock_add.call_args[0][0]


class TestScore:
    @patch.object(console, '_add')
    def test_excellent_rating(self, mock_add):
        console.score(95)
        assert "Excellent" in mock_add.call_args[0][0]

    @patch.object(console, '_add')
    def test_good_rating(self, mock_add):
        console.score(75)
        assert "Good" in mock_add.call_args[0][0]

    @patch.object(console, '_add')
    def test_fair_rating(self, mock_add):
        console.score(55)
        assert "Fair" in mock_add.call_args[0][0]

    @patch.object(console, '_add')
    def test_poor_rating(self, mock_add):
        console.score(30)
        assert "Poor" in mock_add.call_args[0][0]

    @patch.object(console, '_add')
    def test_bar_in_code_block(self, mock_add):
        console.score(50)
        assert "`" in mock_add.call_args[0][0]
        assert "█" in mock_add.call_args[0][0]


class TestProgress:
    @patch.object(console, '_add')
    def test_hundred_percent_full_bar(self, mock_add):
        console.progress("Test", 100)
        assert "░" not in mock_add.call_args[0][0]

    @patch.object(console, '_add')
    def test_value_bold(self, mock_add):
        console.progress("Test", 75.5)
        assert "**75.5%**" in mock_add.call_args[0][0]


class TestCheck:
    @patch.object(console, '_add')
    def test_passed_shows_ok(self, mock_add):
        console.check("validation", True)
        assert "[OK]" in mock_add.call_args[0][0]

    @patch.object(console, '_add')
    def test_failed_shows_x(self, mock_add):
        console.check("validation", False)
        assert "[X]" in mock_add.call_args[0][0]

    @patch.object(console, '_add')
    def test_detail_included(self, mock_add):
        console.check("check", False, "3 errors")
        assert "3 errors" in mock_add.call_args[0][0]


class TestOverview:
    @patch.object(console, '_add')
    def test_all_values_bold(self, mock_add):
        console.overview(1000, 10, 5.5, 99.0, "churn")
        calls = [c[0][0] for c in mock_add.call_args_list]
        all_text = "\n".join(calls)
        assert "**1,000**" in all_text
        assert "**10**" in all_text
        assert "**5.5 MB**" in all_text
        assert "**99.0%**" in all_text
        assert "**churn**" in all_text

    @patch.object(console, '_add')
    def test_target_optional(self, mock_add):
        console.overview(100, 5, 1.0, 80.0)
        calls = [c[0][0] for c in mock_add.call_args_list]
        all_text = "\n".join(calls)
        assert "Target" not in all_text


class TestLists:
    @patch.object(console, '_add')
    def test_bullets(self, mock_add):
        console.bullets(["one", "two"])
        calls = [c[0][0] for c in mock_add.call_args_list]
        assert "- one" in calls[0]
        assert "- two" in calls[1]

    @patch.object(console, '_add')
    def test_numbers(self, mock_add):
        console.numbers(["first", "second"])
        calls = [c[0][0] for c in mock_add.call_args_list]
        assert "1. first" in calls[0]
        assert "2. second" in calls[1]


class TestSectionBatching:
    def test_start_section_disables_auto_flush(self):
        console._auto_flush = True
        console.start_section()
        assert console._auto_flush is False
        console._auto_flush = True

    def test_end_section_enables_auto_flush(self):
        console._auto_flush = False
        console._buffer = []
        console.end_section()
        assert console._auto_flush is True
