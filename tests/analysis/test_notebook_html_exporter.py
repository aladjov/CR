"""Tests for notebook HTML exporter."""
import json
from pathlib import Path
from unittest.mock import patch

from customer_retention.analysis.notebook_html_exporter import (
    _cleanup_processed,
    _preprocess_plotly,
    check_exported_html,
    display_html_documentation,
    export_notebook_html,
)


def _create_minimal_notebook(path: Path) -> None:
    """Create a minimal valid .ipynb file."""
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "source": "print('hello')",
                "metadata": {},
                "outputs": [],
                "execution_count": None,
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb))


class TestExportNotebookHtml:
    def test_exports_notebook_to_html(self, tmp_path):
        nb_path = tmp_path / "test_notebook.ipynb"
        _create_minimal_notebook(nb_path)
        output_dir = tmp_path / "docs"

        result = export_notebook_html(nb_path, output_dir)

        assert result is not None
        assert result.exists()
        assert result.suffix == ".html"
        assert result.name == "test_notebook.html"

    def test_returns_none_when_nbconvert_missing(self, tmp_path):
        nb_path = tmp_path / "test.ipynb"
        _create_minimal_notebook(nb_path)
        output_dir = tmp_path / "docs"

        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = export_notebook_html(nb_path, output_dir)

        assert result is None

    def test_returns_none_when_notebook_missing(self, tmp_path):
        nb_path = tmp_path / "nonexistent.ipynb"
        output_dir = tmp_path / "docs"

        result = export_notebook_html(nb_path, output_dir)

        assert result is None

    def test_output_dir_created_if_missing(self, tmp_path):
        nb_path = tmp_path / "test.ipynb"
        _create_minimal_notebook(nb_path)
        output_dir = tmp_path / "nested" / "docs"

        assert not output_dir.exists()
        export_notebook_html(nb_path, output_dir)
        assert output_dir.exists()

    def test_uses_custom_template_when_available(self, tmp_path):
        nb_path = tmp_path / "test.ipynb"
        _create_minimal_notebook(nb_path)
        output_dir = tmp_path / "docs"

        with patch("subprocess.run") as mock_run, \
             patch("customer_retention.analysis.notebook_html_exporter.TEMPLATE_DIR") as mock_tpl:
            mock_tpl.exists.return_value = True
            mock_tpl.__str__ = lambda self: "/templates/tutorial_html"
            mock_run.return_value.returncode = 0

            export_notebook_html(nb_path, output_dir)

            call_args = mock_run.call_args[0][0]
            assert "--template" in call_args
            template_idx = call_args.index("--template")
            assert call_args[template_idx + 1] == "/templates/tutorial_html"


class TestPreprocessPlotly:
    def test_returns_original_on_import_error(self, tmp_path):
        nb_path = tmp_path / "test.ipynb"
        _create_minimal_notebook(nb_path)
        output_dir = tmp_path / "docs"
        with patch.dict("sys.modules", {"customer_retention.analysis.plotly_preprocessor": None}):
            result = _preprocess_plotly(nb_path, output_dir)
        assert result == nb_path

    def test_skips_round_trip_when_deps_unavailable(self, tmp_path):
        nb_path = tmp_path / "test.ipynb"
        _create_minimal_notebook(nb_path)
        output_dir = tmp_path / "docs"
        with patch(
            "customer_retention.analysis.plotly_preprocessor.PlotlyToImagePreprocessor"
        ) as MockPP:
            MockPP.return_value.kaleido_available = False
            MockPP.return_value.plotly_available = True
            result = _preprocess_plotly(nb_path, output_dir)
        assert result == nb_path
        assert not (output_dir / "_processed").exists()

    def test_preprocesses_when_deps_available(self, tmp_path):
        import nbformat
        nb_path = tmp_path / "test.ipynb"
        _create_minimal_notebook(nb_path)
        output_dir = tmp_path / "docs"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(nb_path) as f:
            nb = nbformat.read(f, as_version=4)
        with patch(
            "customer_retention.analysis.plotly_preprocessor.PlotlyToImagePreprocessor"
        ) as MockPP:
            MockPP.return_value.kaleido_available = True
            MockPP.return_value.plotly_available = True
            MockPP.return_value.preprocess.return_value = (nb, {})
            result = _preprocess_plotly(nb_path, output_dir)
        assert result.exists()
        assert result.parent.name == "_processed"

    def test_fallback_on_processing_error(self, tmp_path):
        nb_path = tmp_path / "test.ipynb"
        nb_path.write_text("not valid json notebook")
        output_dir = tmp_path / "docs"
        result = _preprocess_plotly(nb_path, output_dir)
        assert result == nb_path


class TestCleanupProcessed:
    def test_removes_processed_file(self, tmp_path):
        original = tmp_path / "original.ipynb"
        original.touch()
        processed_dir = tmp_path / "_processed"
        processed_dir.mkdir()
        processed = processed_dir / "original.ipynb"
        processed.touch()

        _cleanup_processed(processed, original)

        assert not processed.exists()
        assert not processed_dir.exists()

    def test_noop_when_same_path(self, tmp_path):
        original = tmp_path / "test.ipynb"
        original.touch()

        _cleanup_processed(original, original)

        assert original.exists()

    def test_noop_when_processed_missing(self, tmp_path):
        original = tmp_path / "original.ipynb"
        original.touch()
        processed = tmp_path / "_processed" / "original.ipynb"

        _cleanup_processed(processed, original)


class TestCheckExportedHtml:
    def test_returns_found_and_missing(self, tmp_path):
        nb_dir = tmp_path / "notebooks"
        nb_dir.mkdir()
        for name in ["01_alpha", "02_beta", "03_gamma"]:
            _create_minimal_notebook(nb_dir / f"{name}.ipynb")

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "01_alpha.html").write_text("<html></html>")
        (docs_dir / "03_gamma.html").write_text("<html></html>")

        found, missing = check_exported_html(docs_dir, nb_dir)

        assert [p.name for p in found] == ["01_alpha.html", "03_gamma.html"]
        assert missing == ["02_beta"]

    def test_all_exported(self, tmp_path):
        nb_dir = tmp_path / "notebooks"
        nb_dir.mkdir()
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        for name in ["01_a", "02_b"]:
            _create_minimal_notebook(nb_dir / f"{name}.ipynb")
            (docs_dir / f"{name}.html").write_text("<html></html>")

        found, missing = check_exported_html(docs_dir, nb_dir)

        assert len(found) == 2
        assert missing == []

    def test_none_exported(self, tmp_path):
        nb_dir = tmp_path / "notebooks"
        nb_dir.mkdir()
        for name in ["01_a", "02_b"]:
            _create_minimal_notebook(nb_dir / f"{name}.ipynb")
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()

        found, missing = check_exported_html(docs_dir, nb_dir)

        assert found == []
        assert missing == ["01_a", "02_b"]

    def test_empty_docs_dir(self, tmp_path):
        nb_dir = tmp_path / "notebooks"
        nb_dir.mkdir()
        _create_minimal_notebook(nb_dir / "01_a.ipynb")
        docs_dir = tmp_path / "docs_missing"

        found, missing = check_exported_html(docs_dir, nb_dir)

        assert found == []
        assert missing == ["01_a"]


class TestDisplayHtmlDocumentation:
    def test_displays_each_html_file(self, tmp_path):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "01_a.html").write_text("<h1>A</h1>")
        (docs_dir / "02_b.html").write_text("<h1>B</h1>")

        with patch("IPython.display.display") as mock_display:
            display_html_documentation(docs_dir)

        # 2 files × 2 calls each (heading + iframe)
        assert mock_display.call_count == 4

    def test_displays_in_sorted_order(self, tmp_path):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "02_beta.html").write_text("<h1>B</h1>")
        (docs_dir / "01_alpha.html").write_text("<h1>A</h1>")

        rendered_stems: list[str] = []

        def capture_display(html_obj):
            text = html_obj.data
            if "<h2>" in text:
                stem = text.replace("<h2>", "").replace("</h2>", "")
                rendered_stems.append(stem)

        with patch("IPython.display.display", side_effect=capture_display):
            display_html_documentation(docs_dir)

        assert rendered_stems == ["01_alpha", "02_beta"]

    def test_noop_on_empty_dir(self, tmp_path):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()

        with patch("IPython.display.display") as mock_display:
            display_html_documentation(docs_dir)

        mock_display.assert_not_called()
