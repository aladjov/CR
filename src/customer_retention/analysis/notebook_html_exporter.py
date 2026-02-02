"""Export a notebook as self-contained HTML for documentation snapshots."""
import html
import subprocess
import sys
from pathlib import Path
from typing import Optional

TEMPLATE_DIR = Path(__file__).parents[2] / ".." / "scripts" / "templates" / "tutorial_html"


def _preprocess_plotly(notebook_path: Path, output_dir: Path) -> Path:
    try:
        import nbformat

        from customer_retention.analysis.plotly_preprocessor import PlotlyToImagePreprocessor
    except ImportError:
        return notebook_path

    preprocessor = PlotlyToImagePreprocessor()
    if not preprocessor.kaleido_available or not preprocessor.plotly_available:
        return notebook_path

    try:
        with open(notebook_path, "r", encoding="utf-8") as fh:
            nb = nbformat.read(fh, as_version=4)
        nb, _ = preprocessor.preprocess(nb, {})
        processed_dir = output_dir / "_processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        processed_path = processed_dir / notebook_path.name
        with open(processed_path, "w", encoding="utf-8") as fh:
            nbformat.write(nb, fh)
        return processed_path
    except Exception:
        return notebook_path


def _cleanup_processed(processed_path: Path, original_path: Path) -> None:
    """Remove the temporary processed notebook if it differs from the original."""
    if processed_path != original_path and processed_path.exists():
        try:
            processed_path.unlink()
            parent = processed_path.parent
            if parent.name == "_processed" and not any(parent.iterdir()):
                parent.rmdir()
        except OSError:
            pass


def export_notebook_html(notebook_path: Path, output_dir: Path) -> Optional[Path]:
    """Export *notebook_path* to a self-contained HTML file in *output_dir*.

    Returns the output path on success, ``None`` on failure (missing
    ``nbconvert``, file not found, conversion error).  No exceptions are
    raised so callers can treat this as best-effort documentation.
    """
    if not notebook_path.exists():
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = notebook_path.stem + ".html"

    processed_path = _preprocess_plotly(notebook_path, output_dir)

    cmd = [
        sys.executable, "-m", "nbconvert",
        "--to", "html",
        "--output", output_name,
        "--output-dir", str(output_dir),
    ]

    if TEMPLATE_DIR.exists():
        cmd.extend(["--template", str(TEMPLATE_DIR)])

    cmd.append(str(processed_path))

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        _cleanup_processed(processed_path, notebook_path)
        return None

    _cleanup_processed(processed_path, notebook_path)

    result = output_dir / output_name
    return result if result.exists() else None


def check_exported_html(
    docs_dir: Path, notebook_dir: Path
) -> tuple[list[Path], list[str]]:
    """Check which notebook HTML exports exist and which are missing.

    Returns ``(found_paths, missing_stems)`` where *found_paths* are existing
    HTML files that correspond to notebooks and *missing_stems* are notebook
    stems with no matching HTML.
    """
    expected_stems = sorted(p.stem for p in notebook_dir.glob("*.ipynb"))

    if not docs_dir.exists():
        return [], expected_stems

    html_by_stem = {p.stem: p for p in docs_dir.glob("*.html")}

    found: list[Path] = []
    missing: list[str] = []
    for stem in expected_stems:
        if stem in html_by_stem:
            found.append(html_by_stem[stem])
        else:
            missing.append(stem)

    return sorted(found), sorted(missing)


def display_html_documentation(docs_dir: Path) -> None:
    """Render every HTML file in *docs_dir* inline inside a Jupyter notebook.

    Each file is wrapped in an ``<iframe srcdoc="...">`` for CSS isolation.
    """
    from IPython.display import HTML, display

    if not docs_dir.exists():
        return

    html_files = sorted(docs_dir.glob("*.html"))
    for path in html_files:
        content = path.read_text(encoding="utf-8")
        escaped = html.escape(content)
        display(HTML(f"<h2>{html.escape(path.stem)}</h2>"))
        display(
            HTML(
                f'<iframe srcdoc="{escaped}" '
                f'style="width:100%;height:600px;border:1px solid #ccc;" '
                f"sandbox></iframe>"
            )
        )
