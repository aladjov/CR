import os
from unittest.mock import MagicMock, patch

from customer_retention.analysis.visualization.display import (
    DisplayManager,
    detect_environment,
    display_figure,
    display_summary,
    display_table,
)


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

    def test_format_memory_gb(self):
        result = DisplayManager.format_memory(2 * 1024 ** 3)
        assert "GB" in result
        assert "2.0" in result

    def test_format_memory_kb(self):
        result = DisplayManager.format_memory(5 * 1024)
        assert "KB" in result
        assert "5.0" in result


class TestDetectEnvironmentIPython:
    def test_ipython_terminal_environment(self):
        with patch.dict(os.environ, {}, clear=True):
            mock_shell = MagicMock()
            mock_shell.__class__.__name__ = "TerminalInteractiveShell"
            with patch(
                "customer_retention.analysis.visualization.display.get_ipython",
                return_value=mock_shell,
            ):
                env = detect_environment()
                assert env == "ipython"

    def test_shell_none_returns_terminal(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "customer_retention.analysis.visualization.display.get_ipython",
                return_value=None,
            ):
                env = detect_environment()
                assert env == "terminal"

    def test_import_error_returns_terminal(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "customer_retention.analysis.visualization.display.get_ipython",
                side_effect=ImportError,
            ):
                env = detect_environment()
                assert env == "terminal"

    def test_attribute_error_returns_terminal(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "customer_retention.analysis.visualization.display.get_ipython",
                side_effect=AttributeError,
            ):
                env = detect_environment()
                assert env == "terminal"


class TestDisplayFigure:
    def test_plotly_figure_in_notebook_env(self):
        fig = MagicMock()
        fig.update_layout = MagicMock()
        fig.show = MagicMock()

        with patch(
            "customer_retention.analysis.visualization.display.detect_environment",
            return_value="jupyter",
        ):
            display_figure(fig, title="Test", width=800, height=600)
            fig.update_layout.assert_any_call(title="Test")
            fig.update_layout.assert_any_call(width=800)
            fig.update_layout.assert_any_call(height=600)
            fig.show.assert_called_once_with(config={"responsive": False})

    def test_plotly_figure_in_terminal_env(self):
        fig = MagicMock()
        fig.update_layout = MagicMock()
        fig.write_html = MagicMock()

        with patch(
            "customer_retention.analysis.visualization.display.detect_environment",
            return_value="terminal",
        ):
            with patch("tempfile.NamedTemporaryFile") as mock_tmp:
                mock_file = MagicMock()
                mock_file.__enter__ = MagicMock(return_value=mock_file)
                mock_file.__exit__ = MagicMock(return_value=False)
                mock_file.name = "/tmp/test.html"
                mock_tmp.return_value = mock_file
                with patch("webbrowser.open") as mock_open:
                    display_figure(fig)
                    fig.write_html.assert_called_once()
                    mock_open.assert_called_once()

    def test_plotly_figure_no_title_no_dims(self):
        fig = MagicMock()
        fig.update_layout = MagicMock()
        fig.show = MagicMock()

        with patch(
            "customer_retention.analysis.visualization.display.detect_environment",
            return_value="jupyter",
        ):
            display_figure(fig)
            fig.show.assert_called_once()

    def test_matplotlib_figure(self):
        fig = MagicMock(spec=[])  # No update_layout
        fig.savefig = MagicMock()
        fig.suptitle = MagicMock()

        with patch(
            "customer_retention.analysis.visualization.display.detect_environment",
            return_value="terminal",
        ):
            with patch("matplotlib.pyplot.show") as mock_show:
                display_figure(fig, title="My Plot")
                fig.suptitle.assert_called_once_with("My Plot")
                mock_show.assert_called_once()

    def test_matplotlib_figure_no_title(self):
        fig = MagicMock(spec=[])
        fig.savefig = MagicMock()

        with patch(
            "customer_retention.analysis.visualization.display.detect_environment",
            return_value="terminal",
        ):
            with patch("matplotlib.pyplot.show") as mock_show:
                display_figure(fig)
                mock_show.assert_called_once()

    def test_plotly_figure_in_databricks(self):
        fig = MagicMock()
        fig.update_layout = MagicMock()
        fig.show = MagicMock()

        with patch(
            "customer_retention.analysis.visualization.display.detect_environment",
            return_value="databricks",
        ):
            display_figure(fig, title="Test")
            fig.show.assert_called_once_with(config={"responsive": False})

    def test_plotly_figure_in_ipython(self):
        fig = MagicMock()
        fig.update_layout = MagicMock()
        fig.show = MagicMock()

        with patch(
            "customer_retention.analysis.visualization.display.detect_environment",
            return_value="ipython",
        ):
            display_figure(fig)
            fig.show.assert_called_once_with(config={"responsive": False})


class TestDisplayTable:
    def test_terminal_with_title_and_to_string(self, capsys):
        df = MagicMock()
        df.to_string = MagicMock(return_value="col1  col2\n1     2")
        df.head = MagicMock(return_value=df)

        with patch(
            "customer_retention.analysis.visualization.display.detect_environment",
            return_value="terminal",
        ):
            display_table(df, title="My Table")
            df.head.assert_called_once_with(50)
            df.to_string.assert_called_once()

        captured = capsys.readouterr()
        assert "My Table" in captured.out

    def test_terminal_without_title(self):
        df = MagicMock()
        df.to_string = MagicMock(return_value="col1\n1")
        df.head = MagicMock(return_value=df)

        with patch(
            "customer_retention.analysis.visualization.display.detect_environment",
            return_value="terminal",
        ):
            display_table(df)
            df.head.assert_called_once_with(50)
            df.to_string.assert_called_once()

    def test_terminal_no_to_string(self, capsys):
        df = MagicMock(spec=[])  # No to_string

        with patch(
            "customer_retention.analysis.visualization.display.detect_environment",
            return_value="terminal",
        ):
            display_table(df, title="Title")

        captured = capsys.readouterr()
        assert "Title" in captured.out

    def test_jupyter_with_to_html(self):
        df = MagicMock()
        df.to_html = MagicMock(return_value="<table></table>")
        df.head = MagicMock(return_value=df)

        mock_display = MagicMock()
        mock_html_cls = MagicMock()

        with patch(
            "customer_retention.analysis.visualization.display.detect_environment",
            return_value="jupyter",
        ):
            with patch.dict("sys.modules", {"IPython": MagicMock(), "IPython.display": MagicMock()}):
                with patch(
                    "customer_retention.analysis.visualization.display.detect_environment",
                    return_value="jupyter",
                ):
                    # Simulate the try block in jupyter env
                    display_table(df, title="Test Table")

    def test_jupyter_without_to_html(self):
        df = MagicMock(spec=[])  # No to_html, no to_string

        with patch(
            "customer_retention.analysis.visualization.display.detect_environment",
            return_value="jupyter",
        ):
            display_table(df, title="Test")

    def test_jupyter_no_title(self):
        df = MagicMock()
        df.to_html = MagicMock(return_value="<table></table>")
        df.head = MagicMock(return_value=df)

        with patch(
            "customer_retention.analysis.visualization.display.detect_environment",
            return_value="jupyter",
        ):
            display_table(df)


class TestDisplaySummary:
    def _make_findings(self, null_pct=10.0, memory_mb=5.0, row_count=100, column_count=5):
        col_mock = MagicMock()
        col_mock.universal_metrics = {"null_percentage": null_pct}
        columns_dict = {"col1": col_mock}

        findings = MagicMock()
        findings.columns.values.return_value = list(columns_dict.values())
        findings.source_path = "data.csv"
        findings.row_count = row_count
        findings.column_count = column_count
        findings.memory_usage_mb = memory_mb
        return findings

    def test_display_summary_terminal(self, capsys):
        findings = self._make_findings()
        charts = MagicMock()

        with patch(
            "customer_retention.analysis.visualization.display.detect_environment",
            return_value="terminal",
        ):
            display_summary(findings, charts)

        captured = capsys.readouterr()
        assert "data.csv" in captured.out
        assert "100" in captured.out

    def test_display_summary_jupyter(self):
        findings = self._make_findings()
        charts = MagicMock()

        with patch(
            "customer_retention.analysis.visualization.display.detect_environment",
            return_value="jupyter",
        ):
            with patch(
                "customer_retention.analysis.visualization.display.DisplayManager.create_summary_html",
                return_value="<html>summary</html>",
            ) as mock_html:
                display_summary(findings, charts)
                mock_html.assert_called_once()
                call_kwargs = mock_html.call_args
                assert call_kwargs[0][0] == "data.csv"  # source_path
                assert call_kwargs[0][1] == 100  # row_count

    def test_display_summary_no_memory(self, capsys):
        findings = self._make_findings(memory_mb=0, row_count=100, column_count=10)
        charts = MagicMock()

        with patch(
            "customer_retention.analysis.visualization.display.detect_environment",
            return_value="terminal",
        ):
            display_summary(findings, charts)

        captured = capsys.readouterr()
        assert "data.csv" in captured.out

    def test_display_summary_no_columns(self, capsys):
        findings = MagicMock()
        findings.columns.values.return_value = []
        findings.source_path = "data.csv"
        findings.row_count = 0
        findings.column_count = 0
        findings.memory_usage_mb = 0
        charts = MagicMock()

        with patch(
            "customer_retention.analysis.visualization.display.detect_environment",
            return_value="terminal",
        ):
            display_summary(findings, charts)

        captured = capsys.readouterr()
        assert "data.csv" in captured.out

    def test_display_summary_databricks(self):
        findings = self._make_findings()
        charts = MagicMock()

        with patch(
            "customer_retention.analysis.visualization.display.detect_environment",
            return_value="databricks",
        ):
            with patch(
                "customer_retention.analysis.visualization.display.DisplayManager.create_summary_html",
                return_value="<html>summary</html>",
            ) as mock_html:
                display_summary(findings, charts)
                mock_html.assert_called_once()

    def test_display_summary_jupyter_import_error(self):
        findings = self._make_findings()
        charts = MagicMock()

        with patch(
            "customer_retention.analysis.visualization.display.detect_environment",
            return_value="jupyter",
        ):
            with patch(
                "customer_retention.analysis.visualization.display.DisplayManager.create_summary_html",
                return_value="<html>summary</html>",
            ):
                # Force ImportError on the IPython.display import inside display_summary
                import builtins
                original_import = builtins.__import__

                def mock_import(name, *args, **kwargs):
                    if name == "IPython.display":
                        raise ImportError("no IPython")
                    return original_import(name, *args, **kwargs)

                with patch("builtins.__import__", side_effect=mock_import):
                    display_summary(findings, charts)


class TestDisplayTableImportError:
    def test_jupyter_import_error_with_to_string(self):
        df = MagicMock()
        df.to_string = MagicMock(return_value="table text")
        df.head = MagicMock(return_value=df)

        with patch(
            "customer_retention.analysis.visualization.display.detect_environment",
            return_value="jupyter",
        ):
            import builtins
            original_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "IPython.display":
                    raise ImportError("no IPython")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                display_table(df, title="Test")

    def test_jupyter_import_error_without_to_string(self):
        df = MagicMock(spec=[])  # No to_string

        with patch(
            "customer_retention.analysis.visualization.display.detect_environment",
            return_value="jupyter",
        ):
            import builtins
            original_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "IPython.display":
                    raise ImportError("no IPython")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                display_table(df)
