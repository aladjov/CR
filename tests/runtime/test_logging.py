from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from customer_retention.runtime import cr
from customer_retention.runtime import logging as cr_logging


@pytest.fixture(autouse=True)
def _reset_notebook_cache():
    cr_logging._reset_cache_for_tests()
    yield
    cr_logging._reset_cache_for_tests()


class TestInNotebookDetection:
    def test_no_ipython_returns_false(self, monkeypatch):
        monkeypatch.setattr(cr_logging, "_detect_notebook", lambda: False)
        cr_logging._reset_cache_for_tests()
        assert cr.in_notebook() is False

    def test_result_is_cached(self, monkeypatch):
        call_count = [0]
        def counter():
            call_count[0] += 1
            return False
        monkeypatch.setattr(cr_logging, "_detect_notebook", counter)
        cr_logging._reset_cache_for_tests()
        cr.in_notebook()
        cr.in_notebook()
        cr.in_notebook()
        assert call_count[0] == 1


class TestLogNonNotebook:
    def test_log_uses_stdlib_logger(self, caplog, monkeypatch):
        monkeypatch.setattr(cr_logging, "_detect_notebook", lambda: False)
        cr_logging._reset_cache_for_tests()
        with caplog.at_level(logging.INFO, logger="customer_retention.runtime"):
            cr.log("hello world", step="demo")
        assert any("hello world" in rec.message for rec in caplog.records)

    def test_log_table_on_df_with_show_calls_show(self, monkeypatch):
        monkeypatch.setattr(cr_logging, "_detect_notebook", lambda: False)
        cr_logging._reset_cache_for_tests()
        df = MagicMock()
        df.show = MagicMock()
        cr.log_table(df, message="demo", max_rows=5)
        df.show.assert_called_once_with(5)

    def test_log_table_on_pandas_like_df_falls_back_to_head(self, monkeypatch, caplog):
        monkeypatch.setattr(cr_logging, "_detect_notebook", lambda: False)
        cr_logging._reset_cache_for_tests()
        df = MagicMock(spec=["head"])
        df.head = MagicMock(return_value="head_result")
        with caplog.at_level(logging.INFO, logger="customer_retention.runtime"):
            cr.log_table(df, max_rows=3)
        df.head.assert_called_once_with(3)


class TestLogNotebook:
    def test_log_displays_markdown(self, monkeypatch):
        monkeypatch.setattr(cr_logging, "_detect_notebook", lambda: True)
        cr_logging._reset_cache_for_tests()

        markdown_instances = []
        displayed = []

        class FakeMarkdown:
            def __init__(self, text):
                self.text = text
                markdown_instances.append(self)

        def fake_display(obj):
            displayed.append(obj)

        fake_ipython_display = MagicMock()
        fake_ipython_display.Markdown = FakeMarkdown
        fake_ipython_display.display = fake_display
        monkeypatch.setitem(__import__("sys").modules, "IPython.display", fake_ipython_display)

        cr.log("# Header")
        assert len(markdown_instances) == 1
        assert markdown_instances[0].text == "# Header"

    def test_log_table_displays_df(self, monkeypatch):
        monkeypatch.setattr(cr_logging, "_detect_notebook", lambda: True)
        cr_logging._reset_cache_for_tests()

        displayed = []

        class FakeMarkdown:
            def __init__(self, text):
                self.text = text

        fake_ipython_display = MagicMock()
        fake_ipython_display.Markdown = FakeMarkdown
        fake_ipython_display.display = lambda obj: displayed.append(obj)
        monkeypatch.setitem(__import__("sys").modules, "IPython.display", fake_ipython_display)

        df = object()
        cr.log_table(df, message="caption")
        assert any(isinstance(obj, FakeMarkdown) for obj in displayed)
        assert df in displayed
