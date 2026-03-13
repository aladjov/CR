"""Reusable ipywidgets-based paginated column viewer for notebook exploration."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from customer_retention.analysis.visualization.attention_scorer import AttentionScore
from customer_retention.analysis.visualization.display import (
    detect_environment,
    display_table,
)
from customer_retention.core.compat import native_pd

try:
    import ipywidgets as widgets

    HAS_WIDGETS = True
except ImportError:
    HAS_WIDGETS = False


def has_interactive_widgets() -> bool:
    """Return True if ipywidgets is available and we're in a notebook environment."""
    if not HAS_WIDGETS:
        return False
    return detect_environment() in ("jupyter", "databricks")


@dataclass
class ColumnEntry:
    """A single column to be displayed in the paginator."""

    name: str
    column_type: str
    attention_score: AttentionScore
    analysis_data: Any = None
    recommendation_data: Any = None
    extra: dict = field(default_factory=dict)


class ColumnPaginator:
    """Paginated column viewer with filtering and sorting.

    Parameters
    ----------
    entries : list of ColumnEntry
        All columns to display.
    render_callback : callable(entry, output_area) -> None
        Renders charts for one column into the output area.
    summary_callback : callable(entry) -> dict
        Returns one row for the summary table.
    page_size : int
        Number of columns per page.
    title : str
        Title shown above the paginator.
    fallback_top_n : int
        When widgets are unavailable, render this many top-scored columns.
    """

    def __init__(
        self,
        entries: List[ColumnEntry],
        render_callback: Callable,
        summary_callback: Callable,
        page_size: int = 10,
        title: str = "Column Analysis",
        fallback_top_n: int = 20,
    ):
        self._entries = sorted(entries, key=lambda e: e.attention_score.score, reverse=True)
        self._render_callback = render_callback
        self._summary_callback = summary_callback
        self._page_size = page_size
        self._title = title
        self._fallback_top_n = fallback_top_n

        # Filter/sort state
        self._search_text = ""
        self._sort_key = "score"
        self._min_score = 0
        self._current_page = 0
        self._filtered_entries: List[ColumnEntry] = list(self._entries)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show(self):
        """Display the paginator — widgets if available, fallback otherwise."""
        # Always show summary table first
        self._show_summary_table()

        if has_interactive_widgets():
            self._show_widget()
        else:
            self._show_fallback()

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------

    def _show_summary_table(self):
        rows = [self._summary_callback(entry) for entry in self._entries]
        if rows:
            df = native_pd.DataFrame(rows)
            display_table(df, title=f"{self._title} Summary ({len(rows)} columns)")

    # ------------------------------------------------------------------
    # Widget-based paginator
    # ------------------------------------------------------------------

    def _show_widget(self):
        from IPython.display import display as ipy_display

        search_box = widgets.Text(
            value="",
            placeholder="Search column name...",
            description="Search:",
            layout=widgets.Layout(width="300px"),
        )
        sort_dropdown = widgets.Dropdown(
            options=[("Score (high first)", "score"), ("Name (A-Z)", "name"), ("Type", "type")],
            value="score",
            description="Sort:",
            layout=widgets.Layout(width="250px"),
        )
        min_score_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=100,
            step=5,
            description="Min score:",
            layout=widgets.Layout(width="300px"),
        )

        prev_btn = widgets.Button(description="<< Prev", layout=widgets.Layout(width="100px"))
        next_btn = widgets.Button(description="Next >>", layout=widgets.Layout(width="100px"))
        page_label = widgets.HTML(value="")

        output_area = widgets.Output()

        def _update_page_label():
            total = len(self._filtered_entries)
            total_pages = max(1, math.ceil(total / self._page_size))
            page_label.value = (
                f"<b>Page {self._current_page + 1} of {total_pages}</b> "
                f"({total} columns)"
            )

        def _apply_filters_and_render(_=None):
            self._search_text = search_box.value.strip().lower()
            self._sort_key = sort_dropdown.value
            self._min_score = min_score_slider.value
            self._apply_filters()
            self._current_page = 0
            _update_page_label()
            self._render_page(self._current_page, output_area)

        def _on_prev(_=None):
            if self._current_page > 0:
                self._current_page -= 1
                _update_page_label()
                self._render_page(self._current_page, output_area)

        def _on_next(_=None):
            total_pages = max(1, math.ceil(len(self._filtered_entries) / self._page_size))
            if self._current_page < total_pages - 1:
                self._current_page += 1
                _update_page_label()
                self._render_page(self._current_page, output_area)

        search_box.observe(lambda change: _apply_filters_and_render(), names="value")
        sort_dropdown.observe(lambda change: _apply_filters_and_render(), names="value")
        min_score_slider.observe(lambda change: _apply_filters_and_render(), names="value")
        prev_btn.on_click(_on_prev)
        next_btn.on_click(_on_next)

        controls = widgets.HBox([search_box, sort_dropdown, min_score_slider])
        nav = widgets.HBox([prev_btn, page_label, next_btn])
        container = widgets.VBox([
            widgets.HTML(f"<h3>{self._title}</h3>"),
            controls,
            nav,
            output_area,
        ])

        # Initial render
        self._apply_filters()
        _update_page_label()
        self._render_page(0, output_area)

        ipy_display(container)

    # ------------------------------------------------------------------
    # Fallback (no widgets)
    # ------------------------------------------------------------------

    def _show_fallback(self):
        top = self._entries[: self._fallback_top_n]
        print(f"\n{self._title}: showing top {len(top)} of {len(self._entries)} columns by attention score\n")
        for entry in top:
            self._render_callback(entry, None)

    # ------------------------------------------------------------------
    # Filtering and sorting
    # ------------------------------------------------------------------

    def _apply_filters(self):
        filtered = self._entries
        if self._min_score > 0:
            filtered = [e for e in filtered if e.attention_score.score >= self._min_score]
        if self._search_text:
            filtered = [e for e in filtered if self._search_text in e.name.lower()]

        if self._sort_key == "score":
            filtered.sort(key=lambda e: e.attention_score.score, reverse=True)
        elif self._sort_key == "name":
            filtered.sort(key=lambda e: e.name.lower())
        elif self._sort_key == "type":
            filtered.sort(key=lambda e: (e.column_type, e.name.lower()))

        self._filtered_entries = filtered

    # ------------------------------------------------------------------
    # Page rendering
    # ------------------------------------------------------------------

    def _render_page(self, page: int, output_area: Optional[Any] = None):
        start = page * self._page_size
        end = start + self._page_size
        page_entries = self._filtered_entries[start:end]

        if output_area is not None:
            output_area.clear_output(wait=True)
            with output_area:
                for entry in page_entries:
                    self._render_callback(entry, output_area)
        else:
            for entry in page_entries:
                self._render_callback(entry, None)
