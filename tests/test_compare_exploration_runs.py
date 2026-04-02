"""Tests for scripts/experiments/compare_exploration_runs.py — dual-format drift report."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "experiments"))

from compare_exploration_runs import (
    CellMatch,
    CellMatchStatus,
    DriftResult,
    HtmlDriftRenderer,
    ManifestCell,
    MarkdownDriftRenderer,
    NotebookDrift,
    ParsedManifest,
    RunMetadata,
    compute_drift_result,
    match_cells,
    parse_manifest,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manifest(notebooks: dict[str, list[tuple[str, str, str, str]]], section_map: str = "") -> str:
    """Build a valid manifest string.

    notebooks: {nb_name: [(cell_name, cell_id, section, content), ...]}
    """
    if not section_map:
        lines = ["<!-- Section Map -->"]
        for nb, cells in notebooks.items():
            out_count = sum(1 for _, _, _, c in cells if c.strip())
            lines.append(f"<!-- {nb} ({len(cells)} cells, {out_count} with output) -->")
        section_map = "\n".join(lines)

    parts = [section_map]
    for nb, cells in notebooks.items():
        cell_blocks = []
        for name, cid, section, content in cells:
            sec_part = f" (section: {section})" if section else ""
            cell_blocks.append(f"## code: {name} (id: {cid}){sec_part}\n\n{content}")
        parts.append(f"# {nb}\n\n" + "\n\n".join(cell_blocks))
    return "\n\n".join(parts) + "\n"


def _cell(name="test", cell_id="aabb1122", section="", content="output") -> ManifestCell:
    return ManifestCell(name=name, cell_id=cell_id, section=section, content=content)


def _minimal_drift(notebook_drifts=None) -> DriftResult:
    old_m = ParsedManifest(notebooks={}, section_map_raw="")
    new_m = ParsedManifest(notebooks={}, section_map_raw="")
    return DriftResult(
        old_meta=RunMetadata(sha="aaa1111", version="1.0.0", date="Jan 01, 2026"),
        new_meta=RunMetadata(sha="bbb2222", version="2.0.0", date="Feb 01, 2026"),
        old_manifest=old_m, new_manifest=new_m,
        notebook_drifts=notebook_drifts or [],
        snapshot_grid_old=None, snapshot_grid_new=None,
        project_context_old=None, project_context_new=None,
        feature_profile_old=None, feature_profile_new=None,
        old_timings={}, new_timings={}, metric_tables=[],
    )


# ---------------------------------------------------------------------------
# TestParseManifest
# ---------------------------------------------------------------------------

class TestParseManifest:
    def test_extracts_cell_id_and_section_from_new_format(self):
        text = _make_manifest({"00_start_here": [
            ("core_imports", "a355da3c", "0_1_project_metadata", "**Project Setup**"),
        ]})
        m = parse_manifest(text)
        cell = m.notebooks["00_start_here"][0]
        assert cell.cell_id == "a355da3c"
        assert cell.section == "0_1_project_metadata"
        assert cell.name == "core_imports"
        assert "Project Setup" in cell.content

    def test_handles_old_format_without_section(self):
        text = _make_manifest({"01_data": [
            ("cell_cell-1", "cell-1", "", "some output"),
        ]})
        m = parse_manifest(text)
        cell = m.notebooks["01_data"][0]
        assert cell.cell_id == "cell-1"
        assert cell.section == ""

    def test_preserves_cell_order(self):
        text = _make_manifest({"nb": [
            ("alpha", "id1", "", "a"), ("beta", "id2", "", "b"), ("gamma", "id3", "", "c"),
        ]})
        m = parse_manifest(text)
        names = [c.name for c in m.notebooks["nb"]]
        assert names == ["alpha", "beta", "gamma"]

    def test_multiple_notebooks(self):
        text = _make_manifest({
            "00_start": [("init", "aa11", "", "out1")],
            "01_discover": [("load", "bb22", "", "out2")],
        })
        m = parse_manifest(text)
        assert set(m.notebooks) == {"00_start", "01_discover"}

    def test_section_map_raw_preserved(self):
        text = _make_manifest({"nb": [("c", "id1", "", "x")]})
        m = parse_manifest(text)
        assert "<!-- nb" in m.section_map_raw
        assert "Section Map" in m.section_map_raw

    def test_empty_manifest(self):
        m = parse_manifest("<!-- Section Map -->\n")
        assert m.notebooks == {}


# ---------------------------------------------------------------------------
# TestMatchCells
# ---------------------------------------------------------------------------

class TestMatchCells:
    def test_matches_by_cell_id(self):
        old = [_cell(name="old_name", cell_id="aabb1122", content="same")]
        new = [_cell(name="old_name", cell_id="aabb1122", content="same")]
        matches = match_cells(old, new)
        assert len(matches) == 1
        assert matches[0].status == CellMatchStatus.UNCHANGED

    def test_matches_by_name_when_ids_differ(self):
        old = [_cell(name="load_data", cell_id="cell-1", content="same")]
        new = [_cell(name="load_data", cell_id="ff001122", content="same")]
        matches = match_cells(old, new)
        assert len(matches) == 1
        assert matches[0].status == CellMatchStatus.UNCHANGED

    def test_renamed_detected_when_id_matches_name_differs(self):
        old = [_cell(name="cell_abc", cell_id="aabb1122", content="same")]
        new = [_cell(name="core_imports", cell_id="aabb1122", content="same")]
        matches = match_cells(old, new)
        assert len(matches) == 1
        assert matches[0].status == CellMatchStatus.RENAMED

    def test_unmatched_old_marked_removed(self):
        old = [_cell(name="gone", cell_id="dead0000", content="x")]
        new = []
        matches = match_cells(old, new)
        assert len(matches) == 1
        assert matches[0].status == CellMatchStatus.REMOVED

    def test_unmatched_new_marked_added(self):
        old = []
        new = [_cell(name="brand_new", cell_id="new00001", content="y")]
        matches = match_cells(old, new)
        assert len(matches) == 1
        assert matches[0].status == CellMatchStatus.ADDED

    def test_identical_content_unchanged(self):
        old = [_cell(name="x", cell_id="id1", content="hello world")]
        new = [_cell(name="x", cell_id="id1", content="hello world")]
        matches = match_cells(old, new)
        assert matches[0].status == CellMatchStatus.UNCHANGED
        assert matches[0].diff_text == ""

    def test_modified_includes_diff_text(self):
        old = [_cell(name="x", cell_id="id1", content="line one\nline two")]
        new = [_cell(name="x", cell_id="id1", content="line one\nline THREE")]
        matches = match_cells(old, new)
        assert matches[0].status == CellMatchStatus.MODIFIED
        assert matches[0].diff_text != ""
        assert "-line two" in matches[0].diff_text
        assert "+line THREE" in matches[0].diff_text

    def test_skips_jupyter_default_ids_for_id_matching(self):
        old = [_cell(name="cell_cell-1", cell_id="cell-1", content="x")]
        new = [_cell(name="core_imports", cell_id="a355da3c", content="y")]
        matches = match_cells(old, new)
        assert len(matches) == 1
        assert matches[0].status == CellMatchStatus.MODIFIED
        assert matches[0].old_cell.name == "cell_cell-1"
        assert matches[0].new_cell.name == "core_imports"

    def test_positional_pairing_when_no_id_or_name_match(self):
        old = [_cell(name="cell_aaa", cell_id="cell-1", content="setup"), _cell(name="cell_bbb", cell_id="cell-2", content="run")]
        new = [_cell(name="init", cell_id="ff11", content="setup v2"), _cell(name="execute", cell_id="ff22", content="run v2")]
        matches = match_cells(old, new)
        assert len(matches) == 2
        assert all(m.status == CellMatchStatus.MODIFIED for m in matches)

    def test_section_aware_positional_pairing(self):
        old = [
            _cell(name="a", cell_id="cell-1", section="sec_1", content="alpha"),
            _cell(name="b", cell_id="cell-2", section="sec_2", content="beta"),
        ]
        new = [
            _cell(name="x", cell_id="ff01", section="sec_2", content="beta_new"),
            _cell(name="y", cell_id="ff02", section="sec_1", content="alpha_new"),
        ]
        matches = match_cells(old, new)
        sec1_match = [m for m in matches if m.old_cell and m.old_cell.section == "sec_1"][0]
        assert sec1_match.new_cell.section == "sec_1"
        sec2_match = [m for m in matches if m.old_cell and m.old_cell.section == "sec_2"][0]
        assert sec2_match.new_cell.section == "sec_2"


# ---------------------------------------------------------------------------
# TestMarkdownDriftRenderer
# ---------------------------------------------------------------------------

class TestMarkdownDriftRenderer:
    def test_render_title_format(self):
        drift = _minimal_drift()
        md = MarkdownDriftRenderer(drift).render()
        assert "# Drift Report: aaa1111 -> bbb2222" in md

    def test_render_summary_shows_both_labels(self):
        drift = _minimal_drift()
        md = MarkdownDriftRenderer(drift).render()
        assert "v1.0.0" in md
        assert "v2.0.0" in md

    def test_render_section_map_as_md_table(self):
        old_m = ParsedManifest(
            notebooks={}, section_map_raw="<!-- nb1 (10 cells, 5 with output) -->"
        )
        new_m = ParsedManifest(
            notebooks={}, section_map_raw="<!-- nb1 (12 cells, 6 with output) -->"
        )
        drift = _minimal_drift()
        drift.old_manifest = old_m
        drift.new_manifest = new_m
        md = MarkdownDriftRenderer(drift).render()
        assert "| nb1 |" in md
        assert "| Notebook |" in md

    def test_render_cell_diff_uses_fenced_code(self):
        matches = [CellMatch(
            old_cell=_cell(name="x", content="old"), new_cell=_cell(name="x", content="new"),
            status=CellMatchStatus.MODIFIED, diff_text="-old\n+new",
        )]
        drift = _minimal_drift([NotebookDrift("nb", matches)])
        md = MarkdownDriftRenderer(drift).render()
        assert "```diff" in md

    def test_render_full_produces_valid_markdown(self):
        drift = _minimal_drift()
        md = MarkdownDriftRenderer(drift).render()
        assert md.startswith("# ")
        assert "## Summary" in md


# ---------------------------------------------------------------------------
# TestHtmlDriftRenderer
# ---------------------------------------------------------------------------

class TestHtmlDriftRenderer:
    def test_wraps_in_html_document(self):
        drift = _minimal_drift()
        html = HtmlDriftRenderer(drift).render()
        assert "<!DOCTYPE html>" in html
        assert "</html>" in html

    def test_color_coded_diff_spans(self):
        matches = [CellMatch(
            old_cell=_cell(name="x", content="old line"), new_cell=_cell(name="x", content="new line"),
            status=CellMatchStatus.MODIFIED, diff_text="-old line\n+new line",
        )]
        drift = _minimal_drift([NotebookDrift("nb", matches)])
        html = HtmlDriftRenderer(drift).render()
        assert "diff-removed" in html or "background" in html

    def test_details_summary_per_notebook(self):
        matches = [CellMatch(
            old_cell=None, new_cell=_cell(name="x", content="new"),
            status=CellMatchStatus.ADDED, diff_text="",
        )]
        drift = _minimal_drift([NotebookDrift("00_start_here", matches)])
        html = HtmlDriftRenderer(drift).render()
        assert "<details" in html
        assert "<summary" in html
        assert "00_start_here" in html

    def test_embedded_style_tag(self):
        drift = _minimal_drift()
        html = HtmlDriftRenderer(drift).render()
        assert "<style>" in html

    def test_scrollable_containers(self):
        matches = [CellMatch(
            old_cell=_cell(name="x", content="old"), new_cell=_cell(name="x", content="new"),
            status=CellMatchStatus.MODIFIED, diff_text="-old\n+new",
        )]
        drift = _minimal_drift([NotebookDrift("nb", matches)])
        html = HtmlDriftRenderer(drift).render()
        assert "overflow" in html


# ---------------------------------------------------------------------------
# TestYamlDiffRows
# ---------------------------------------------------------------------------

class TestYamlDiffRows:
    def test_detects_changed_values(self):
        renderer = MarkdownDriftRenderer(_minimal_drift())
        rows = renderer.yaml_diff_rows({"a": 1}, {"a": 2}, ["a"])
        assert rows[0] == ("a", "1", "2", True)

    def test_missing_key_shows_na(self):
        renderer = MarkdownDriftRenderer(_minimal_drift())
        rows = renderer.yaml_diff_rows({"a": 1}, {}, ["a", "b"])
        assert rows[0] == ("a", "1", "N/A", True)
        assert rows[1] == ("b", "N/A", "N/A", False)

    def test_none_dicts_handled(self):
        renderer = MarkdownDriftRenderer(_minimal_drift())
        rows = renderer.yaml_diff_rows(None, None, ["x"])
        assert rows[0] == ("x", "N/A", "N/A", False)


# ---------------------------------------------------------------------------
# TestCli
# ---------------------------------------------------------------------------

class TestCli:
    def test_format_both_produces_two_files(self, tmp_path):
        old_manifest = tmp_path / "old" / "cell_outputs.md"
        new_manifest = tmp_path / "new" / "cell_outputs.md"
        old_manifest.parent.mkdir()
        new_manifest.parent.mkdir()
        text = _make_manifest({"nb": [("c1", "id1", "", "output")]})
        old_manifest.write_text(text)
        new_manifest.write_text(text)
        for d in [old_manifest.parent, new_manifest.parent]:
            (d / "git_sha.txt").write_text("abcdef1234567")
            (d / "version.txt").write_text("1.0.0")
            (d / "commit_date.txt").write_text("Jan 01, 2026")

        out = tmp_path / "drift_report.md"
        drift = compute_drift_result(old_manifest, new_manifest, old_manifest.parent, new_manifest.parent)
        MarkdownDriftRenderer(drift).write(out.with_suffix(".md"))
        HtmlDriftRenderer(drift).write(out.with_suffix(".html"))
        assert out.with_suffix(".md").exists()
        assert out.with_suffix(".html").exists()

    def test_format_md_content(self, tmp_path):
        old_manifest = tmp_path / "old" / "cell_outputs.md"
        new_manifest = tmp_path / "new" / "cell_outputs.md"
        old_manifest.parent.mkdir()
        new_manifest.parent.mkdir()
        text = _make_manifest({"nb": [("c1", "id1", "", "output")]})
        old_manifest.write_text(text)
        new_manifest.write_text(text)
        for d in [old_manifest.parent, new_manifest.parent]:
            (d / "git_sha.txt").write_text("abcdef1234567")
            (d / "version.txt").write_text("1.0.0")
            (d / "commit_date.txt").write_text("Jan 01, 2026")

        drift = compute_drift_result(old_manifest, new_manifest, old_manifest.parent, new_manifest.parent)
        md = MarkdownDriftRenderer(drift).render()
        assert "# Drift Report" in md

    def test_format_html_content(self, tmp_path):
        old_manifest = tmp_path / "old" / "cell_outputs.md"
        new_manifest = tmp_path / "new" / "cell_outputs.md"
        old_manifest.parent.mkdir()
        new_manifest.parent.mkdir()
        text = _make_manifest({"nb": [("c1", "id1", "", "output")]})
        old_manifest.write_text(text)
        new_manifest.write_text(text)
        for d in [old_manifest.parent, new_manifest.parent]:
            (d / "git_sha.txt").write_text("abcdef1234567")
            (d / "version.txt").write_text("1.0.0")
            (d / "commit_date.txt").write_text("Jan 01, 2026")

        drift = compute_drift_result(old_manifest, new_manifest, old_manifest.parent, new_manifest.parent)
        html = HtmlDriftRenderer(drift).render()
        assert "<!DOCTYPE html>" in html
