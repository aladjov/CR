"""Tests for the Jupyter post-save hook."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from customer_retention.analysis.jupyter_save_hook import post_save_export


def _create_minimal_notebook(path: Path) -> None:
    nb = {
        "cells": [{"cell_type": "code", "source": "1+1", "metadata": {},
                    "outputs": [], "execution_count": None}],
        "metadata": {"kernelspec": {"display_name": "Python 3",
                                     "language": "python", "name": "python3"},
                      "language_info": {"name": "python", "version": "3.10.0"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb))


class TestPostSaveExport:
    def test_exports_exploration_notebook(self, tmp_path):
        nb_dir = tmp_path / "exploration_notebooks"
        nb_dir.mkdir()
        nb_path = nb_dir / "01_data_discovery.ipynb"
        _create_minimal_notebook(nb_path)
        model = {"type": "notebook"}
        cm = MagicMock()

        with patch(
            "customer_retention.analysis.jupyter_save_hook.export_notebook_html"
        ) as mock_export:
            post_save_export(model, str(nb_path), cm)
            mock_export.assert_called_once()
            assert mock_export.call_args[0][0] == nb_path

    def test_skips_non_notebook_types(self, tmp_path):
        model = {"type": "file"}
        cm = MagicMock()

        with patch(
            "customer_retention.analysis.jupyter_save_hook.export_notebook_html"
        ) as mock_export:
            post_save_export(model, str(tmp_path / "data.csv"), cm)
            mock_export.assert_not_called()

    def test_skips_notebooks_outside_exploration_dir(self, tmp_path):
        nb_path = tmp_path / "random_dir" / "test.ipynb"
        nb_path.parent.mkdir()
        _create_minimal_notebook(nb_path)
        model = {"type": "notebook"}
        cm = MagicMock()

        with patch(
            "customer_retention.analysis.jupyter_save_hook.export_notebook_html"
        ) as mock_export:
            post_save_export(model, str(nb_path), cm)
            mock_export.assert_not_called()

    def test_handles_export_error_gracefully(self, tmp_path):
        nb_dir = tmp_path / "exploration_notebooks"
        nb_dir.mkdir()
        nb_path = nb_dir / "test.ipynb"
        _create_minimal_notebook(nb_path)
        model = {"type": "notebook"}
        cm = MagicMock()

        with patch(
            "customer_retention.analysis.jupyter_save_hook.export_notebook_html",
            side_effect=RuntimeError("export failed"),
        ):
            post_save_export(model, str(nb_path), cm)

    def test_uses_experiments_docs_as_output_dir(self, tmp_path):
        nb_dir = tmp_path / "exploration_notebooks"
        nb_dir.mkdir()
        nb_path = nb_dir / "test.ipynb"
        _create_minimal_notebook(nb_path)
        model = {"type": "notebook"}
        cm = MagicMock()
        experiments_dir = tmp_path / "experiments"

        with patch(
            "customer_retention.analysis.jupyter_save_hook.export_notebook_html"
        ) as mock_export, patch(
            "customer_retention.analysis.jupyter_save_hook.get_experiments_dir",
            return_value=experiments_dir,
        ):
            post_save_export(model, str(nb_path), cm)
            assert mock_export.call_args[0][1] == experiments_dir / "docs"
