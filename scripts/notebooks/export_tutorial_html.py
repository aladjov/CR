#!/usr/bin/env python3
"""
Export Jupyter notebooks to self-contained tutorial HTML.

This script converts notebooks to HTML with:
- Code cells collapsed in details/summary sections
- Markdown and outputs readily visible
- Self-contained HTML (embedded CSS/JS)
- Optional table of contents generation

Usage:
    python scripts/notebooks/export_tutorial_html.py exploration_notebooks/*.ipynb --output docs/tutorial/
    python scripts/notebooks/export_tutorial_html.py --run exploration_notebooks/ --output docs/tutorial/
"""
import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

SCRIPT_DIR = Path(__file__).parent
TEMPLATE_DIR = SCRIPT_DIR.parent / "templates" / "tutorial_html"


@dataclass
class ExportConfig:
    output_dir: Path
    template_dir: Path = TEMPLATE_DIR
    embed_images: bool = True
    execute_notebooks: bool = False
    kernel_name: str = "python3"
    timeout: int = 600
    allow_errors: bool = True
    convert_plotly: bool = True  # Convert Plotly figures to static images
    filter_warnings: bool = False  # Keep warnings visible by default


class TutorialExporter:
    def __init__(self, config: ExportConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self._plotly_preprocessor = None

    def _get_plotly_preprocessor(self):
        if self._plotly_preprocessor is None:
            try:
                # Import from same directory
                import sys
                sys.path.insert(0, str(SCRIPT_DIR))
                from plotly_image_preprocessor import PlotlyToImagePreprocessor
                self._plotly_preprocessor = PlotlyToImagePreprocessor()
            except ImportError as e:
                print(f"  Note: Plotly conversion not available ({e})")
                self._plotly_preprocessor = False
        return self._plotly_preprocessor

    def _get_warning_filter(self):
        try:
            import sys
            sys.path.insert(0, str(SCRIPT_DIR))
            from plotly_image_preprocessor import WarningFilterPreprocessor
            return WarningFilterPreprocessor()
        except ImportError:
            return None

    def _preprocess_plotly(self, notebook_path: Path) -> Path:
        """Convert Plotly figures to static images and optionally filter warnings."""
        try:
            import nbformat
            with open(notebook_path, "r", encoding="utf-8") as f:
                nb = nbformat.read(f, as_version=4)

            # Apply warning filter only if configured
            if self.config.filter_warnings:
                warning_filter = self._get_warning_filter()
                if warning_filter:
                    nb, _ = warning_filter.preprocess(nb, {})

            # Apply Plotly conversion if enabled
            if self.config.convert_plotly:
                preprocessor = self._get_plotly_preprocessor()
                if preprocessor:
                    nb, _ = preprocessor.preprocess(nb, {})

            # Save to temp location
            processed_dir = self.config.output_dir / "_processed"
            processed_dir.mkdir(parents=True, exist_ok=True)
            processed_path = processed_dir / notebook_path.name

            with open(processed_path, "w", encoding="utf-8") as f:
                nbformat.write(nb, f)

            return processed_path
        except Exception as e:
            print(f"  Warning: Could not preprocess notebook: {e}")
            return notebook_path

    def export_notebook(self, notebook_path: Path) -> Optional[Path]:
        output_name = notebook_path.stem + ".html"
        output_path = self.config.output_dir / output_name

        # Preprocess to convert Plotly figures
        processed_path = self._preprocess_plotly(notebook_path)

        cmd = self._build_nbconvert_command(processed_path, output_path)
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"  Exported: {output_path.name}")
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"  ERROR exporting {notebook_path.name}: {e.stderr}")
            return None

    def _build_nbconvert_command(self, notebook_path: Path, output_path: Path) -> List[str]:
        cmd = [
            sys.executable, "-m", "nbconvert",
            "--to", "html",
            "--template", str(self.config.template_dir),
            "--output", output_path.name,
            "--output-dir", str(output_path.parent),
        ]
        if self.config.embed_images:
            cmd.append("--HTMLExporter.embed_images=True")
        if self.config.execute_notebooks:
            cmd.extend([
                "--execute",
                f"--ExecutePreprocessor.kernel_name={self.config.kernel_name}",
                f"--ExecutePreprocessor.timeout={self.config.timeout}",
            ])
            if self.config.allow_errors:
                cmd.append("--allow-errors")
        cmd.append(str(notebook_path))
        return cmd

    def export_directory(self, notebooks_dir: Path, pattern: str = "*.ipynb") -> List[Path]:
        notebooks = sorted(notebooks_dir.glob(pattern))
        if not notebooks:
            print(f"No notebooks found in {notebooks_dir}")
            return []
        print(f"Found {len(notebooks)} notebooks in {notebooks_dir}")
        exported = []
        for nb_path in notebooks:
            result = self.export_notebook(nb_path)
            if result:
                exported.append(result)
        return exported

    def create_index_page(self, exported_files: List[Path], title: str = "Tutorial") -> Path:
        index_path = self.config.output_dir / "index.html"
        notebooks_list = "\n".join(
            f'        <li><a href="{f.name}">{self._format_title(f.stem)}</a></li>'
            for f in exported_files
        )
        html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 0 auto;
            padding: 40px 20px;
            color: #24292e;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 60px 40px;
            margin: -40px -20px 40px -20px;
            text-align: center;
            border-radius: 0 0 8px 8px;
        }}
        .header h1 {{ margin: 0; font-size: 2.5em; }}
        .header p {{ opacity: 0.9; margin: 15px 0 0 0; font-size: 1.2em; }}
        .section {{
            background: white;
            border: 1px solid #e1e4e8;
            border-radius: 8px;
            padding: 30px;
            margin-bottom: 30px;
        }}
        .section h2 {{
            margin-top: 0;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }}
        .notebook-list {{
            list-style: none;
            padding: 0;
        }}
        .notebook-list li {{
            padding: 15px 20px;
            border: 1px solid #e1e4e8;
            border-radius: 6px;
            margin-bottom: 10px;
            transition: all 0.2s;
        }}
        .notebook-list li:hover {{
            border-color: #667eea;
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.15);
        }}
        .notebook-list a {{
            color: #0366d6;
            text-decoration: none;
            font-weight: 500;
            font-size: 1.1em;
        }}
        .notebook-list a:hover {{ text-decoration: underline; }}
        .footer {{
            text-align: center;
            color: #586069;
            font-size: 0.9em;
            margin-top: 40px;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            background: #f1f8ff;
            color: #0366d6;
            border-radius: 20px;
            font-size: 0.85em;
            margin-left: 10px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <p>Customer Retention ML Framework - Sample Run</p>
    </div>

    <div class="section">
        <h2>Notebooks</h2>
        <p>Click on a notebook to view the tutorial. Code cells are collapsed by default - click "Show/Hide Code" to expand.</p>
        <ul class="notebook-list">
{notebooks_list}
        </ul>
    </div>

    <div class="section">
        <h2>About This Tutorial</h2>
        <p>This tutorial demonstrates the Customer Retention ML Framework using a sample Kaggle dataset.
        Each notebook walks through a stage of the ML pipeline, from data exploration to model deployment.</p>
        <p><strong>Features demonstrated:</strong></p>
        <ul>
            <li>Automatic data profiling and type detection</li>
            <li>Data quality assessment and cleaning</li>
            <li>Feature engineering and selection</li>
            <li>Model training with MLflow tracking</li>
            <li>Business-aligned model deployment</li>
        </ul>
    </div>

    <div class="footer">
        <p>Generated with Customer Retention ML Framework</p>
    </div>
</body>
</html>'''
        index_path.write_text(html_content)
        print(f"\nCreated index page: {index_path}")
        return index_path

    def _format_title(self, stem: str) -> str:
        title = stem.replace("_", " ").replace("-", " ")
        parts = title.split()
        if parts and parts[0].isdigit():
            parts[0] = f"{parts[0]}."
        return " ".join(parts).title()


def run_notebooks_with_papermill(
    notebooks_dir: Path,
    output_dir: Path,
    data_path: Optional[str] = None,
) -> List[Path]:
    import papermill as pm
    executed_dir = output_dir / "executed"
    executed_dir.mkdir(parents=True, exist_ok=True)
    notebooks = sorted(notebooks_dir.glob("*.ipynb"))
    executed_paths = []
    parameters = {}
    if data_path:
        parameters["DATA_PATH"] = data_path
    print(f"\nExecuting {len(notebooks)} notebooks...")
    for nb_path in notebooks:
        output_path = executed_dir / nb_path.name
        print(f"  Running: {nb_path.name}")
        try:
            pm.execute_notebook(
                str(nb_path),
                str(output_path),
                parameters=parameters,
                kernel_name="python3",
            )
            executed_paths.append(output_path)
            print(f"    Done: {output_path.name}")
        except Exception as e:
            print(f"    ERROR: {e}")
            if output_path.exists():
                executed_paths.append(output_path)
    return executed_paths


def main():
    parser = argparse.ArgumentParser(
        description="Export Jupyter notebooks to tutorial HTML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Export existing notebooks
    python scripts/notebooks/export_tutorial_html.py exploration_notebooks/*.ipynb -o docs/tutorial/

    # Run notebooks first, then export
    python scripts/notebooks/export_tutorial_html.py --run exploration_notebooks/ -o docs/tutorial/

    # Run with specific data path
    python scripts/notebooks/export_tutorial_html.py --run exploration_notebooks/ \\
        --data-path /data/kaggle_churn.csv -o docs/tutorial/
"""
    )
    parser.add_argument(
        "notebooks",
        nargs="*",
        help="Notebook files or directory to export"
    )
    parser.add_argument(
        "-o", "--output",
        default="./tutorial_output",
        help="Output directory for HTML files (default: ./tutorial_output)"
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute notebooks before exporting (requires papermill)"
    )
    parser.add_argument(
        "--data-path",
        help="Data path parameter to pass to notebooks when executing"
    )
    parser.add_argument(
        "--title",
        default="Customer Retention Tutorial",
        help="Title for the index page"
    )
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="Don't create an index.html page"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute notebooks during conversion (alternative to --run)"
    )
    parser.add_argument(
        "--no-plotly-convert",
        action="store_true",
        help="Don't convert Plotly figures to static images"
    )
    parser.add_argument(
        "--filter-warnings",
        action="store_true",
        help="Filter out warning messages from notebook outputs"
    )
    args = parser.parse_args()
    if not args.notebooks:
        parser.print_help()
        sys.exit(1)
    output_dir = Path(args.output)
    config = ExportConfig(
        output_dir=output_dir,
        execute_notebooks=args.execute,
        convert_plotly=not args.no_plotly_convert,
        filter_warnings=args.filter_warnings,
    )
    exporter = TutorialExporter(config)
    all_exported = []
    for nb_arg in args.notebooks:
        nb_path = Path(nb_arg)
        if nb_path.is_dir():
            if args.run:
                executed = run_notebooks_with_papermill(
                    nb_path, output_dir, args.data_path
                )
                for exec_nb in executed:
                    result = exporter.export_notebook(exec_nb)
                    if result:
                        all_exported.append(result)
            else:
                exported = exporter.export_directory(nb_path)
                all_exported.extend(exported)
        elif nb_path.exists() and nb_path.suffix == ".ipynb":
            if args.run:
                executed_dir = output_dir / "executed"
                executed_dir.mkdir(parents=True, exist_ok=True)
                import papermill as pm
                output_nb = executed_dir / nb_path.name
                print(f"Executing: {nb_path.name}")
                try:
                    params = {"DATA_PATH": args.data_path} if args.data_path else {}
                    pm.execute_notebook(str(nb_path), str(output_nb), parameters=params)
                    result = exporter.export_notebook(output_nb)
                    if result:
                        all_exported.append(result)
                except Exception as e:
                    print(f"  ERROR: {e}")
            else:
                result = exporter.export_notebook(nb_path)
                if result:
                    all_exported.append(result)
        else:
            print(f"Warning: {nb_arg} not found or not a notebook")
    if all_exported and not args.no_index:
        exporter.create_index_page(all_exported, args.title)
    print(f"\nExported {len(all_exported)} notebooks to {output_dir}")


if __name__ == "__main__":
    main()
