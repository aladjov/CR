import re
import shutil
from pathlib import Path

import nbformat
import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"
DUMMY_NB = FIXTURE_DIR / "99_dummy_migration.ipynb"


def _migrate(nb_path, dry_run=False):
    from scripts.notebooks.migrate_notebook_cell_ids import (
        DUMMY_SPLIT_MAP,
        DUMMY_TAG_ONLY_MAP,
        migrate_notebook,
    )
    return migrate_notebook(
        nb_path,
        dry_run=dry_run,
        split_map=DUMMY_SPLIT_MAP,
        tag_only_map=DUMMY_TAG_ONLY_MAP,
    )


def _read_nb(path):
    return nbformat.read(str(path), as_version=4)


def _cell_source(cell):
    return cell.source if isinstance(cell.source, str) else "".join(cell.source)


def _cell_first_line(cell):
    return _cell_source(cell).split("\n")[0]


@pytest.fixture
def work_nb(tmp_path):
    dest = tmp_path / "99_dummy_migration.ipynb"
    shutil.copy2(DUMMY_NB, dest)
    return dest


class TestMigrationRoundTrip:

    def test_round_trip_preserves_cell_count(self, work_nb):
        original = _read_nb(work_nb)
        original_count = len(original.cells)
        orig_returned, final_returned = _migrate(work_nb)
        migrated = _read_nb(work_nb)
        assert orig_returned == original_count
        assert final_returned == original_count + 1
        assert len(migrated.cells) == original_count + 1

    def test_round_trip_preserves_markdown_exactly(self, work_nb):
        original = _read_nb(work_nb)
        md_before = [_cell_source(c) for c in original.cells if c.cell_type == "markdown"]
        _migrate(work_nb)
        migrated = _read_nb(work_nb)
        md_after = [_cell_source(c) for c in migrated.cells if c.cell_type == "markdown"]
        assert md_before == md_after

    def test_round_trip_preserves_outputs(self, work_nb):
        original = _read_nb(work_nb)
        cells_with_outputs_before = sum(
            1 for c in original.cells if c.cell_type == "code" and c.outputs
        )
        _migrate(work_nb)
        migrated = _read_nb(work_nb)
        cells_with_outputs_after = sum(
            1 for c in migrated.cells if c.cell_type == "code" and c.outputs
        )
        assert cells_with_outputs_after >= cells_with_outputs_before

    def test_round_trip_preserves_newlines_in_source(self, work_nb):
        _migrate(work_nb)
        migrated = _read_nb(work_nb)
        for cell in migrated.cells:
            if cell.cell_type != "code":
                continue
            source = _cell_source(cell)
            if not source.strip():
                continue
            lines = source.split("\n")
            assert lines[-1] == "" or source.endswith("\n"), (
                f"Cell {cell.id} source does not end with newline"
            )

    def test_split_boundary_correct(self, work_nb):
        _migrate(work_nb)
        migrated = _read_nb(work_nb)
        config_cell = None
        code_cell = None
        for i, c in enumerate(migrated.cells):
            first = _cell_first_line(c)
            if first == "# @cr:config" and "SAMPLE_FRACTION" in _cell_source(c):
                config_cell = c
                if i + 1 < len(migrated.cells):
                    code_cell = migrated.cells[i + 1]
                break
        assert config_cell is not None, "No config cell found after split"
        assert "from pathlib" not in _cell_source(config_cell)
        assert code_cell is not None
        assert "from pathlib" in _cell_source(code_cell) or "data_dir" in _cell_source(code_cell)

    def test_split_preserves_empty_lines(self, work_nb):
        _migrate(work_nb)
        migrated = _read_nb(work_nb)
        all_source = "".join(_cell_source(c) for c in migrated.cells if c.cell_type == "code")
        assert "PROJECT_NAME" in all_source
        assert "data_dir" in all_source

    def test_magic_comment_is_valid_python(self, work_nb):
        _migrate(work_nb)
        migrated = _read_nb(work_nb)
        for cell in migrated.cells:
            if cell.cell_type != "code":
                continue
            source = _cell_source(cell)
            if not source.strip():
                continue
            try:
                compile(source, f"<cell:{cell.id}>", "exec")
            except SyntaxError as e:
                pytest.fail(f"Cell {cell.id} syntax error: {e}\nSource:\n{source}")

    def test_notebook_validates_with_nbformat(self, work_nb):
        _migrate(work_nb)
        migrated = _read_nb(work_nb)
        nbformat.validate(migrated)

    def test_idempotent(self, work_nb):
        _migrate(work_nb)
        first = _read_nb(work_nb)
        first_sources = [_cell_source(c) for c in first.cells]
        first_ids = [c.id for c in first.cells]
        _migrate(work_nb)
        second = _read_nb(work_nb)
        second_sources = [_cell_source(c) for c in second.cells]
        second_ids = [c.id for c in second.cells]
        assert first_ids == second_ids
        assert first_sources == second_sources

    def test_dry_run_no_modification(self, work_nb):
        original_bytes = work_nb.read_bytes()
        _migrate(work_nb, dry_run=True)
        assert work_nb.read_bytes() == original_bytes


class TestMigrationCellIds:

    def test_all_ids_follow_pattern(self, work_nb):
        _migrate(work_nb)
        migrated = _read_nb(work_nb)
        pattern = re.compile(r"^nb\w+-\d{3}$")
        for cell in migrated.cells:
            assert pattern.match(cell.id), f"Cell ID {cell.id!r} does not match pattern"

    def test_no_duplicate_ids(self, work_nb):
        _migrate(work_nb)
        migrated = _read_nb(work_nb)
        ids = [c.id for c in migrated.cells]
        assert len(ids) == len(set(ids))

    def test_config_tag_present(self, work_nb):
        _migrate(work_nb)
        migrated = _read_nb(work_nb)
        tagged_config = [
            c for c in migrated.cells
            if c.cell_type == "code" and _cell_first_line(c) == "# @cr:config"
        ]
        assert len(tagged_config) >= 2

    def test_user_code_tag_present(self, work_nb):
        _migrate(work_nb)
        migrated = _read_nb(work_nb)
        tagged_user = [
            c for c in migrated.cells
            if c.cell_type == "code" and _cell_first_line(c) == "# @cr:user_code"
        ]
        assert len(tagged_user) >= 1
