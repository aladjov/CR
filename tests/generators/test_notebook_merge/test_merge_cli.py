import nbformat
import pytest

from customer_retention.generators.notebook_merge.cli import (
    compute_all_merges,
    merge_notebook_pair,
)


def _write_nb(path, cells):
    nb = nbformat.v4.new_notebook()
    for cid, tag, body in cells:
        cell = nbformat.v4.new_code_cell(source=f"# @cr:{tag}\n{body}")
        cell.id = cid
        nb.cells.append(cell)
    nbformat.write(nb, str(path))


class TestMergeNotebookPair:

    def test_basic_pair_merge(self, tmp_path):
        base_dir = tmp_path / "base"
        base_dir.mkdir()
        _write_nb(base_dir / "00_start.ipynb", [("c1", "config", "X = 1\n")])
        theirs = tmp_path / "theirs.ipynb"
        _write_nb(theirs, [("c1", "config", "X = 42\n")])
        ours = tmp_path / "ours.ipynb"
        _write_nb(ours, [("c1", "config", "X = 1\n")])
        output = tmp_path / "merged.ipynb"

        report = merge_notebook_pair(
            theirs, ours, output, base_path=base_dir / "00_start.ipynb",
        )
        assert not report.has_conflicts
        merged = nbformat.read(str(output), as_version=4)
        assert "X = 42" in "".join(merged.cells[0].source)

    def test_pair_merge_conflict(self, tmp_path):
        base_dir = tmp_path / "base"
        base_dir.mkdir()
        _write_nb(base_dir / "00.ipynb", [("c1", "config", "X = 1\n")])
        theirs = tmp_path / "theirs.ipynb"
        _write_nb(theirs, [("c1", "config", "X = 42\n")])
        ours = tmp_path / "ours.ipynb"
        _write_nb(ours, [("c1", "config", "X = 99\n")])
        output = tmp_path / "merged.ipynb"

        report = merge_notebook_pair(
            theirs, ours, output, base_path=base_dir / "00.ipynb",
        )
        assert report.has_conflicts

    def test_dry_run_no_write(self, tmp_path):
        base_dir = tmp_path / "base"
        base_dir.mkdir()
        _write_nb(base_dir / "nb.ipynb", [("c1", "config", "X = 1\n")])
        theirs = tmp_path / "theirs.ipynb"
        _write_nb(theirs, [("c1", "config", "X = 42\n")])
        ours = tmp_path / "ours.ipynb"
        _write_nb(ours, [("c1", "config", "X = 1\n")])
        output = tmp_path / "merged.ipynb"

        merge_notebook_pair(
            theirs, ours, output, base_path=base_dir / "nb.ipynb", dry_run=True,
        )
        assert not output.exists()


class TestComputeAllMerges:

    def test_directory_merge(self, tmp_path):
        base_dir = tmp_path / "base"
        theirs_dir = tmp_path / "theirs"
        ours_dir = tmp_path / "ours"
        for d in (base_dir, theirs_dir, ours_dir):
            d.mkdir()

        _write_nb(base_dir / "00_start.ipynb", [("c1", "config", "X = 1\n")])
        _write_nb(base_dir / "01_discover.ipynb", [("c2", "config", "Y = 2\n")])
        _write_nb(theirs_dir / "00_start.ipynb", [("c1", "config", "X = 42\n")])
        _write_nb(theirs_dir / "01_discover.ipynb", [("c2", "config", "Y = 2\n")])
        _write_nb(ours_dir / "00_start.ipynb", [("c1", "config", "X = 1\n")])
        _write_nb(ours_dir / "01_discover.ipynb", [("c2", "config", "Y = 99\n")])

        pending = compute_all_merges(theirs_dir, ours_dir, base_dir)
        assert "00_start.ipynb" in pending
        assert "01_discover.ipynb" in pending

    def test_notebook_only_in_theirs(self, tmp_path):
        base_dir = tmp_path / "base"
        theirs_dir = tmp_path / "theirs"
        ours_dir = tmp_path / "ours"
        for d in (base_dir, theirs_dir, ours_dir):
            d.mkdir()

        _write_nb(base_dir / "00.ipynb", [("c1", "config", "X = 1\n")])
        _write_nb(theirs_dir / "00.ipynb", [("c1", "config", "X = 1\n")])
        _write_nb(theirs_dir / "extra.ipynb", [("c9", "config", "Z = 3\n")])
        _write_nb(ours_dir / "00.ipynb", [("c1", "config", "X = 1\n")])

        pending = compute_all_merges(theirs_dir, ours_dir, base_dir)
        assert "extra.ipynb" in pending

    def test_notebook_only_in_ours(self, tmp_path):
        base_dir = tmp_path / "base"
        theirs_dir = tmp_path / "theirs"
        ours_dir = tmp_path / "ours"
        for d in (base_dir, theirs_dir, ours_dir):
            d.mkdir()

        _write_nb(base_dir / "00.ipynb", [("c1", "config", "X = 1\n")])
        _write_nb(theirs_dir / "00.ipynb", [("c1", "config", "X = 1\n")])
        _write_nb(ours_dir / "00.ipynb", [("c1", "config", "X = 1\n")])
        _write_nb(ours_dir / "extra.ipynb", [("c9", "user_code", "custom()\n")])

        pending = compute_all_merges(theirs_dir, ours_dir, base_dir)
        assert "extra.ipynb" in pending

    def test_single_notebook_filter(self, tmp_path):
        base_dir = tmp_path / "base"
        theirs_dir = tmp_path / "theirs"
        ours_dir = tmp_path / "ours"
        for d in (base_dir, theirs_dir, ours_dir):
            d.mkdir()

        _write_nb(base_dir / "00.ipynb", [("c1", "config", "X = 1\n")])
        _write_nb(base_dir / "01.ipynb", [("c2", "config", "Y = 2\n")])
        _write_nb(theirs_dir / "00.ipynb", [("c1", "config", "X = 42\n")])
        _write_nb(theirs_dir / "01.ipynb", [("c2", "config", "Y = 2\n")])
        _write_nb(ours_dir / "00.ipynb", [("c1", "config", "X = 1\n")])
        _write_nb(ours_dir / "01.ipynb", [("c2", "config", "Y = 2\n")])

        pending = compute_all_merges(theirs_dir, ours_dir, base_dir, notebook="00.ipynb")
        assert "00.ipynb" in pending
        assert "01.ipynb" not in pending


class TestApplyMerges:

    def test_apply_writes_output(self, tmp_path):
        from customer_retention.generators.notebook_merge.cli import apply_pending_merges

        base_dir = tmp_path / "base"
        theirs_dir = tmp_path / "theirs"
        ours_dir = tmp_path / "ours"
        output_dir = tmp_path / "output"
        for d in (base_dir, theirs_dir, ours_dir):
            d.mkdir()

        _write_nb(base_dir / "00.ipynb", [("c1", "config", "X = 1\n")])
        _write_nb(theirs_dir / "00.ipynb", [("c1", "config", "X = 42\n")])
        _write_nb(ours_dir / "00.ipynb", [("c1", "config", "X = 1\n")])

        pending = compute_all_merges(theirs_dir, ours_dir, base_dir)
        apply_pending_merges(pending, output_dir)
        assert (output_dir / "00.ipynb").exists()
        nb = nbformat.read(str(output_dir / "00.ipynb"), as_version=4)
        assert "X = 42" in "".join(nb.cells[0].source)


class TestMainCLI:

    def test_dir_mode(self, tmp_path, capsys):
        from customer_retention.generators.notebook_merge.cli import main

        base_dir = tmp_path / "base"
        theirs_dir = tmp_path / "theirs"
        ours_dir = tmp_path / "ours"
        output_dir = tmp_path / "output"
        for d in (base_dir, theirs_dir, ours_dir):
            d.mkdir()

        _write_nb(base_dir / "00.ipynb", [("c1", "config", "X = 1\n")])
        _write_nb(theirs_dir / "00.ipynb", [("c1", "config", "X = 42\n")])
        _write_nb(ours_dir / "00.ipynb", [("c1", "config", "X = 1\n")])

        main([
            "dir", str(theirs_dir), str(ours_dir),
            "-o", str(output_dir),
            "--base-dir", str(base_dir),
        ])
        assert (output_dir / "00.ipynb").exists()

    def test_pair_mode(self, tmp_path, capsys):
        from customer_retention.generators.notebook_merge.cli import main

        base_dir = tmp_path / "base"
        base_dir.mkdir()
        _write_nb(base_dir / "nb.ipynb", [("c1", "config", "X = 1\n")])
        theirs = tmp_path / "theirs.ipynb"
        _write_nb(theirs, [("c1", "config", "X = 42\n")])
        ours = tmp_path / "ours.ipynb"
        _write_nb(ours, [("c1", "config", "X = 1\n")])
        output = tmp_path / "merged.ipynb"

        main([
            "pair", str(theirs), str(ours),
            "-o", str(output),
            "--base", str(base_dir / "nb.ipynb"),
        ])
        assert output.exists()

    def test_fail_on_conflict_exit_code(self, tmp_path):
        from customer_retention.generators.notebook_merge.cli import main

        base_dir = tmp_path / "base"
        theirs_dir = tmp_path / "theirs"
        ours_dir = tmp_path / "ours"
        output_dir = tmp_path / "output"
        for d in (base_dir, theirs_dir, ours_dir):
            d.mkdir()

        _write_nb(base_dir / "00.ipynb", [("c1", "config", "X = 1\n")])
        _write_nb(theirs_dir / "00.ipynb", [("c1", "config", "X = 42\n")])
        _write_nb(ours_dir / "00.ipynb", [("c1", "config", "X = 99\n")])

        with pytest.raises(SystemExit) as exc_info:
            main([
                "dir", str(theirs_dir), str(ours_dir),
                "-o", str(output_dir),
                "--base-dir", str(base_dir),
                "--fail-on-conflict",
            ])
        assert exc_info.value.code == 1
