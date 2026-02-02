"""Tests for the PlotlyToImagePreprocessor."""
from unittest.mock import patch

import pytest
from nbformat.notebooknode import NotebookNode

from customer_retention.analysis.plotly_preprocessor import PlotlyToImagePreprocessor


@pytest.fixture
def preprocessor():
    return PlotlyToImagePreprocessor()


class TestPlotlyToImagePreprocessor:
    def test_skips_non_code_cells(self, preprocessor):
        cell = NotebookNode({
            "cell_type": "markdown",
            "source": "# title",
            "metadata": {},
        })
        result_cell, _ = preprocessor.preprocess_cell(cell, {}, 0)
        assert result_cell.cell_type == "markdown"

    def test_skips_cells_without_outputs(self, preprocessor):
        cell = NotebookNode({
            "cell_type": "code",
            "source": "x = 1",
            "metadata": {},
            "outputs": [],
            "execution_count": 1,
        })
        result_cell, _ = preprocessor.preprocess_cell(cell, {}, 0)
        assert result_cell.outputs == []

    def test_skips_non_plotly_outputs(self, preprocessor):
        cell = NotebookNode({
            "cell_type": "code",
            "source": "print('hi')",
            "metadata": {},
            "outputs": [
                NotebookNode({
                    "output_type": "stream",
                    "name": "stdout",
                    "text": "hi\n",
                }),
            ],
            "execution_count": 1,
        })
        result_cell, _ = preprocessor.preprocess_cell(cell, {}, 0)
        assert len(result_cell.outputs) == 1
        assert result_cell.outputs[0].output_type == "stream"

    def test_skips_non_display_data(self, preprocessor):
        cell = NotebookNode({
            "cell_type": "code",
            "source": "1 + 1",
            "metadata": {},
            "outputs": [
                NotebookNode({
                    "output_type": "execute_result",
                    "data": {"text/plain": "2"},
                    "metadata": {},
                    "execution_count": 1,
                }),
            ],
            "execution_count": 1,
        })
        result_cell, _ = preprocessor.preprocess_cell(cell, {}, 0)
        assert result_cell.outputs[0].output_type == "execute_result"

    def test_noop_when_kaleido_missing(self):
        pp = PlotlyToImagePreprocessor()
        with patch.dict("sys.modules", {"kaleido": None}):
            pp._kaleido_available = None
            with patch("builtins.__import__", side_effect=ImportError("no kaleido")):
                pp._kaleido_available = None
                # Force re-check
                pp._kaleido_available = False
                nb = NotebookNode({
                    "cells": [],
                    "metadata": {},
                    "nbformat": 4,
                    "nbformat_minor": 5,
                })
                result_nb, _ = pp.preprocess(nb, {})
                assert result_nb is nb

    def test_noop_when_plotly_missing(self):
        pp = PlotlyToImagePreprocessor()
        pp._plotly_available = False
        nb = NotebookNode({
            "cells": [],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        })
        result_nb, _ = pp.preprocess(nb, {})
        assert result_nb is nb

    def test_extract_plotly_from_html_returns_none_for_non_plotly(self, preprocessor):
        result = preprocessor._extract_plotly_from_html("<div>hello</div>")
        assert result is None
