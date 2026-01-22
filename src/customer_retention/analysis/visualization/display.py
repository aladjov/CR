import os
from typing import Any, Optional


def get_ipython():
    from IPython import get_ipython as _get_ipython
    return _get_ipython()


def detect_environment() -> str:
    if "DATABRICKS_RUNTIME_VERSION" in os.environ:
        return "databricks"
    try:
        shell = get_ipython()
        if shell is not None:
            shell_name = shell.__class__.__name__
            if "ZMQInteractiveShell" in shell_name:
                return "jupyter"
            elif "TerminalInteractiveShell" in shell_name:
                return "ipython"
    except (ImportError, NameError, AttributeError):
        pass
    return "terminal"


class DisplayManager:
    @staticmethod
    def detect_environment() -> str:
        return detect_environment()

    @staticmethod
    def format_number(value: float, decimals: int = 0) -> str:
        if decimals > 0:
            return f"{value:,.{decimals}f}"
        return f"{int(value):,}"

    @staticmethod
    def format_percentage(value: float, decimals: int = 1) -> str:
        return f"{value * 100:.{decimals}f}%"

    @staticmethod
    def get_completeness_color(pct: float) -> str:
        if pct >= 95:
            return "#2ca02c"  # green
        if pct >= 80:
            return "#ff7f0e"  # orange
        return "#d62728"  # red

    @staticmethod
    def format_memory(bytes_size: int) -> str:
        if bytes_size >= 1024 ** 3:
            return f"{bytes_size / (1024 ** 3):.1f} GB"
        if bytes_size >= 1024 ** 2:
            return f"{bytes_size / (1024 ** 2):.1f} MB"
        if bytes_size >= 1024:
            return f"{bytes_size / 1024:.1f} KB"
        return f"{bytes_size} B"

    @staticmethod
    def create_summary_html(source_path: str, row_count: int, column_count: int,
                            completeness_pct: float, memory_bytes: int) -> str:
        completeness_color = DisplayManager.get_completeness_color(completeness_pct)
        memory_str = DisplayManager.format_memory(memory_bytes)
        return f"""
        <div style="font-family: sans-serif; padding: 20px; color: #333;">
            <h2 style="color: #222;">Data Exploration: {source_path}</h2>
            <div style="display: flex; gap: 20px; margin-bottom: 20px;">
                <div style="background: #f0f0f0; padding: 15px; border-radius: 8px;">
                    <h4 style="margin: 0 0 8px 0; color: #555;">Rows</h4>
                    <span style="font-size: 24px; font-weight: bold; color: #222;">{row_count:,}</span>
                </div>
                <div style="background: #f0f0f0; padding: 15px; border-radius: 8px;">
                    <h4 style="margin: 0 0 8px 0; color: #555;">Columns</h4>
                    <span style="font-size: 24px; font-weight: bold; color: #222;">{column_count}</span>
                </div>
                <div style="background: #f0f0f0; padding: 15px; border-radius: 8px;">
                    <h4 style="margin: 0 0 8px 0; color: #555;">Completeness</h4>
                    <span style="font-size: 24px; font-weight: bold; color: {completeness_color};">{completeness_pct:.1f}%</span>
                </div>
                <div style="background: #f0f0f0; padding: 15px; border-radius: 8px;">
                    <h4 style="margin: 0 0 8px 0; color: #555;">Memory</h4>
                    <span style="font-size: 24px; font-weight: bold; color: #222;">{memory_str}</span>
                </div>
            </div>
        </div>
        """


def display_figure(fig: Any, title: Optional[str] = None, width: Optional[int] = None, height: Optional[int] = None):
    env = detect_environment()
    if hasattr(fig, "update_layout"):
        if title:
            fig.update_layout(title=title)
        if width:
            fig.update_layout(width=width)
        if height:
            fig.update_layout(height=height)
        if env in ["databricks", "jupyter", "ipython"]:
            # Disable responsive mode to respect explicit width/height
            fig.show(config={"responsive": False})
        else:
            import tempfile
            import webbrowser
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
                fig.write_html(f.name, config={"responsive": False})
                webbrowser.open(f"file://{f.name}")
    elif hasattr(fig, "savefig"):
        import matplotlib.pyplot as plt
        if title:
            fig.suptitle(title)
        plt.show()


def display_table(df: Any, max_rows: int = 50, title: Optional[str] = None):
    """Display a pandas DataFrame in the appropriate format for the current environment."""
    env = detect_environment()
    if env in ["databricks", "jupyter", "ipython"]:
        try:
            from IPython.display import display, HTML
            if title:
                display(HTML(f"<h4>{title}</h4>"))
            if hasattr(df, "to_html"):
                html = df.head(max_rows).to_html(classes="table table-striped", index=False)
                display(HTML(html))
            else:
                display(df)
        except ImportError:
            print(df.head(max_rows).to_string() if hasattr(df, "to_string") else str(df))
    else:
        if title:
            print(f"\n{title}")
            print("-" * len(title))
        if hasattr(df, "to_string"):
            print(df.head(max_rows).to_string())
        else:
            print(df)


def display_summary(findings: Any, charts: Any):
    env = detect_environment()

    # Calculate completeness (average of non-null percentages across columns)
    completeness_scores = []
    for col in findings.columns.values():
        null_pct = col.universal_metrics.get("null_percentage", 0)
        completeness_scores.append(100 - null_pct)
    completeness_pct = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 100.0

    # Get memory usage (stored in MB, convert to bytes for display formatting)
    memory_mb = getattr(findings, 'memory_usage_mb', 0)
    if memory_mb > 0:
        memory_bytes = int(memory_mb * 1024 * 1024)
    else:
        # Rough estimate: row_count * column_count * 8 bytes average per cell
        memory_bytes = findings.row_count * findings.column_count * 8

    html = DisplayManager.create_summary_html(
        findings.source_path,
        findings.row_count,
        findings.column_count,
        completeness_pct,
        memory_bytes
    )
    if env in ["databricks", "jupyter", "ipython"]:
        try:
            from IPython.display import display, HTML
            display(HTML(html))
        except ImportError:
            print(html)
    else:
        print(html)
