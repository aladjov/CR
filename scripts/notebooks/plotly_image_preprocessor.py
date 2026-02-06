"""
NBConvert preprocessors for tutorial HTML export.

Includes:
- PlotlyToImagePreprocessor: Convert Plotly figures to static PNG images (imported from src/)
- WarningFilterPreprocessor: Remove warning/stderr outputs from cells
"""

import re

from nbconvert.preprocessors import Preprocessor

try:
    from customer_retention.analysis.plotly_preprocessor import PlotlyToImagePreprocessor  # noqa: F401
except ImportError:
    PlotlyToImagePreprocessor = None


class WarningFilterPreprocessor(Preprocessor):
    """Remove warning and stderr outputs from notebook cells."""

    # Patterns to filter out
    WARNING_PATTERNS = [
        r"FutureWarning:",
        r"UserWarning:",
        r"DeprecationWarning:",
        r"RuntimeWarning:",
        r"warnings\.warn\(",
        r"/.*\.py:\d+:.*Warning",
        r"^\s*/.*site-packages/.*\.py:\d+",
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

        text = getattr(output, "text", None) or output.get("text", "")

        if isinstance(text, list):
            text = "".join(text)

        # Check if text matches any warning pattern
        for pattern in self.WARNING_PATTERNS:
            if re.search(pattern, text):
                return False

        return True


def convert_notebook_plotly_to_images(notebook_path: str, output_path: str = None):
    if PlotlyToImagePreprocessor is None:
        raise ImportError("customer_retention.analysis.plotly_preprocessor not available")
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
