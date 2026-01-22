import pytest
import os
from unittest.mock import patch, MagicMock

from customer_retention.analysis.visualization.display import detect_environment, DisplayManager


class TestDetectEnvironment:
    def test_terminal_environment(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("customer_retention.analysis.visualization.display.get_ipython", side_effect=NameError):
                env = detect_environment()
                assert env == "terminal"

    def test_databricks_environment(self):
        with patch.dict(os.environ, {"DATABRICKS_RUNTIME_VERSION": "10.4"}, clear=False):
            env = detect_environment()
            assert env == "databricks"

    def test_jupyter_environment(self):
        with patch.dict(os.environ, {}, clear=True):
            mock_shell = MagicMock()
            mock_shell.__class__.__name__ = "ZMQInteractiveShell"
            with patch("customer_retention.analysis.visualization.display.get_ipython", return_value=mock_shell):
                env = detect_environment()
                assert env == "jupyter"


class TestDisplayManager:
    def test_detect_environment_static(self):
        result = DisplayManager.detect_environment()
        assert result in ["databricks", "jupyter", "ipython", "terminal"]

    def test_format_number_thousands(self):
        assert DisplayManager.format_number(1000) == "1,000"
        assert DisplayManager.format_number(1234567) == "1,234,567"

    def test_format_number_decimal(self):
        assert DisplayManager.format_number(1234.567, decimals=2) == "1,234.57"

    def test_format_percentage(self):
        assert DisplayManager.format_percentage(0.756) == "75.6%"
        assert DisplayManager.format_percentage(0.5) == "50.0%"

    def test_format_percentage_custom_decimals(self):
        assert DisplayManager.format_percentage(0.75678, decimals=2) == "75.68%"

    def test_get_completeness_color_high(self):
        assert DisplayManager.get_completeness_color(95) == "#2ca02c"  # green
        assert DisplayManager.get_completeness_color(100) == "#2ca02c"

    def test_get_completeness_color_medium(self):
        assert DisplayManager.get_completeness_color(80) == "#ff7f0e"  # orange
        assert DisplayManager.get_completeness_color(90) == "#ff7f0e"

    def test_get_completeness_color_low(self):
        assert DisplayManager.get_completeness_color(50) == "#d62728"  # red
        assert DisplayManager.get_completeness_color(0) == "#d62728"

    def test_create_summary_html(self):
        html = DisplayManager.create_summary_html(
            source_path="test.csv",
            row_count=1000,
            column_count=10,
            completeness_pct=85.5,
            memory_bytes=1024 * 1024  # 1 MB
        )
        assert "test.csv" in html
        assert "1,000" in html
        assert "10" in html
        assert "85" in html or "86" in html
        assert "MB" in html

    def test_create_summary_html_small_memory(self):
        html = DisplayManager.create_summary_html(
            source_path="test.csv",
            row_count=100,
            column_count=5,
            completeness_pct=45.0,
            memory_bytes=512  # 512 bytes
        )
        assert "test.csv" in html
        assert "512 B" in html
