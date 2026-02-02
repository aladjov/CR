"""NBConvert preprocessor that converts Plotly figures to static PNG images."""
import base64
import json
import re

from nbconvert.preprocessors import Preprocessor


class PlotlyToImagePreprocessor(Preprocessor):
    """Convert Plotly figures to static PNG images in notebook outputs.

    Requires ``plotly`` and ``kaleido`` to be installed.  When either is
    missing the preprocessor is a no-op so callers can always apply it
    without guarding imports.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._kaleido_available = None
        self._plotly_available = None

    @property
    def kaleido_available(self):
        if self._kaleido_available is None:
            try:
                import kaleido  # noqa: F401
                self._kaleido_available = True
            except ImportError:
                self._kaleido_available = False
                self.log.warning("kaleido not available - Plotly figures will not be converted to images")
        return self._kaleido_available

    @property
    def plotly_available(self):
        if self._plotly_available is None:
            try:
                import plotly  # noqa: F401
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

        output_type = getattr(output, "output_type", None) or output.get("output_type")
        if output_type != "display_data":
            return output

        data = getattr(output, "data", None) or output.get("data", {})

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

        try:
            png_bytes = self._plotly_to_png(plotly_json)
            if png_bytes:
                png_b64 = base64.b64encode(png_bytes).decode("utf-8")
                return NotebookNode({
                    "output_type": "display_data",
                    "data": {"image/png": png_b64},
                    "metadata": {}
                })
        except Exception as e:
            self.log.warning(f"Failed to convert Plotly figure: {e}")

        return output

    def _extract_plotly_from_html(self, html: str):
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
                    layout = {}
                    if len(match.groups()) > 1:
                        try:
                            layout = json.loads(match.group(2))
                        except (json.JSONDecodeError, IndexError):
                            pass
                    return {"data": data, "layout": layout}
                except json.JSONDecodeError:
                    continue
        return None

    def _plotly_to_png(self, fig_dict: dict, width: int = 1200, height: int = 600) -> bytes:
        import plotly.graph_objects as go
        import plotly.io as pio

        if isinstance(fig_dict, dict):
            fig = go.Figure(fig_dict)
        else:
            fig = fig_dict

        orig_layout = fig_dict.get("layout", {}) if isinstance(fig_dict, dict) else {}
        orig_width = orig_layout.get("width")
        orig_height = orig_layout.get("height")

        if orig_height:
            height = orig_height
        if orig_width:
            width = max(orig_width, 1200)

        fig.update_layout(
            width=width,
            height=height,
            margin=dict(l=50, r=50, t=50, b=50),
        )

        png_bytes = pio.to_image(fig, format="png", scale=1.0)
        return png_bytes
