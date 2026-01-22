"""
NBConvert preprocessors for tutorial HTML export.

Includes:
- PlotlyToImagePreprocessor: Convert Plotly figures to static PNG images
- WarningFilterPreprocessor: Remove warning/stderr outputs from cells
"""
import base64
import json
import re
from typing import Tuple

from nbconvert.preprocessors import Preprocessor


class WarningFilterPreprocessor(Preprocessor):
    """Remove warning and stderr outputs from notebook cells."""

    # Patterns to filter out
    WARNING_PATTERNS = [
        r'FutureWarning:',
        r'UserWarning:',
        r'DeprecationWarning:',
        r'RuntimeWarning:',
        r'warnings\.warn\(',
        r'/.*\.py:\d+:.*Warning',
        r'^\s*/.*site-packages/.*\.py:\d+',
    ]

    def preprocess_cell(self, cell, resources, index):
        if cell.cell_type != "code":
            return cell, resources

        outputs = getattr(cell, "outputs", None)
        if not outputs:
            return cell, resources

        filtered_outputs = []
        for output in outputs:
            if self._should_keep_output(output):
                filtered_outputs.append(output)

        cell.outputs = filtered_outputs
        return cell, resources

    def _should_keep_output(self, output):
        """Check if output should be kept (not a warning)."""
        output_type = getattr(output, "output_type", None) or output.get("output_type")

        # Keep non-stream outputs (display_data, execute_result, etc.)
        if output_type != "stream":
            return True

        # Check stream name - stderr often contains warnings
        name = getattr(output, "name", None) or output.get("name", "")
        text = getattr(output, "text", None) or output.get("text", "")

        if isinstance(text, list):
            text = "".join(text)

        # Check if text matches any warning pattern
        for pattern in self.WARNING_PATTERNS:
            if re.search(pattern, text):
                return False

        return True


class PlotlyToImagePreprocessor(Preprocessor):
    """Convert Plotly figures to static PNG images in notebook outputs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._kaleido_available = None
        self._plotly_available = None

    @property
    def kaleido_available(self):
        if self._kaleido_available is None:
            try:
                import kaleido
                self._kaleido_available = True
            except ImportError:
                self._kaleido_available = False
                self.log.warning("kaleido not available - Plotly figures will not be converted to images")
        return self._kaleido_available

    @property
    def plotly_available(self):
        if self._plotly_available is None:
            try:
                import plotly
                self._plotly_available = True
            except ImportError:
                self._plotly_available = False
                self.log.warning("plotly not available - cannot convert figures")
        return self._plotly_available

    def preprocess(self, nb, resources):
        if not self.kaleido_available or not self.plotly_available:
            return nb, resources
        return super().preprocess(nb, resources)

    def preprocess_cell(self, cell, resources, index):
        if cell.cell_type != "code":
            return cell, resources

        outputs = getattr(cell, "outputs", None)
        if not outputs:
            return cell, resources

        new_outputs = []
        converted_count = 0
        for output in outputs:
            converted = self._convert_plotly_output(output)
            new_outputs.append(converted)
            if converted is not output:
                converted_count += 1

        cell.outputs = new_outputs
        if converted_count > 0:
            self.log.info(f"Converted {converted_count} Plotly figures in cell {index}")
        return cell, resources

    def _convert_plotly_output(self, output):
        from nbformat.notebooknode import NotebookNode

        # Get output_type safely
        output_type = getattr(output, "output_type", None) or output.get("output_type")
        if output_type != "display_data":
            return output

        # Get data safely
        data = getattr(output, "data", None) or output.get("data", {})

        # Check for Plotly JSON
        plotly_json = None
        if "application/vnd.plotly.v1+json" in data:
            plotly_json = data["application/vnd.plotly.v1+json"]
        elif "text/html" in data:
            html = data.get("text/html", "")
            if isinstance(html, list):
                html = "".join(html)
            plotly_json = self._extract_plotly_from_html(html)

        if plotly_json is None:
            return output

        # Convert to PNG
        try:
            png_bytes = self._plotly_to_png(plotly_json)
            if png_bytes:
                png_b64 = base64.b64encode(png_bytes).decode("utf-8")
                # Return as NotebookNode to maintain format
                return NotebookNode({
                    "output_type": "display_data",
                    "data": {"image/png": png_b64},
                    "metadata": {}
                })
        except Exception as e:
            self.log.warning(f"Failed to convert Plotly figure: {e}")

        return output

    def _extract_plotly_from_html(self, html: str):
        """Extract Plotly JSON from HTML widget output."""
        import re

        # Look for Plotly.newPlot or Plotly.react calls
        patterns = [
            r'Plotly\.(?:newPlot|react)\s*\(\s*["\'][\w-]+["\']\s*,\s*(\[.*?\])\s*,\s*(\{.*?\})',
            r'var\s+data\s*=\s*(\[.*?\]);',
            r'"data"\s*:\s*(\[.*?\])',
        ]

        for pattern in patterns:
            match = re.search(pattern, html, re.DOTALL)
            if match:
                try:
                    data_str = match.group(1)
                    data = json.loads(data_str)
                    # Try to get layout too
                    layout = {}
                    if len(match.groups()) > 1:
                        try:
                            layout = json.loads(match.group(2))
                        except:
                            pass
                    return {"data": data, "layout": layout}
                except json.JSONDecodeError:
                    continue
        return None

    def _plotly_to_png(self, fig_dict: dict, width: int = 1200, height: int = 600) -> bytes:
        """Convert Plotly figure dict to PNG bytes."""
        import plotly.graph_objects as go
        import plotly.io as pio

        if isinstance(fig_dict, dict):
            fig = go.Figure(fig_dict)
        else:
            fig = fig_dict

        # Get dimensions from original layout if available
        orig_layout = fig_dict.get("layout", {}) if isinstance(fig_dict, dict) else {}
        orig_width = orig_layout.get("width")
        orig_height = orig_layout.get("height")

        # Respect original dimensions, use wider default for better readability
        if orig_height:
            height = orig_height
        if orig_width:
            width = max(orig_width, 1200)  # Use at least 1200 for readability

        # Update layout for better static rendering
        fig.update_layout(
            width=width,
            height=height,
            margin=dict(l=50, r=50, t=50, b=50),
        )

        # Use scale=1.0 to preserve original font sizes
        png_bytes = pio.to_image(fig, format="png", scale=1.0)
        return png_bytes


def convert_notebook_plotly_to_images(notebook_path: str, output_path: str = None):
    """
    Convert all Plotly figures in a notebook to static images.

    Args:
        notebook_path: Path to input notebook
        output_path: Path for output notebook (default: overwrite input)
    """
    import nbformat

    if output_path is None:
        output_path = notebook_path

    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    preprocessor = PlotlyToImagePreprocessor()
    nb, _ = preprocessor.preprocess(nb, {})

    with open(output_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"Converted Plotly figures in {notebook_path}")
    if output_path != notebook_path:
        print(f"Saved to {output_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python plotly_image_preprocessor.py <notebook.ipynb> [output.ipynb]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    convert_notebook_plotly_to_images(input_path, output_path)
