from unittest.mock import patch

import nbformat
import pytest
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from customer_retention.generators.notebook_sync.cli import main, sync_directory, sync_notebook


def _write_nb(path, cells):
    nb = new_notebook()
    nb.cells = cells
    nbformat.write(nb, str(path))


def _code_cell(cell_id, source):
    c = new_code_cell(source=source)
    c.id = cell_id
    return c


def _md_cell(cell_id, source):
    c = new_markdown_cell(source=source)
    c.id = cell_id
    return c


def _read_nb(path):
    return nbformat.read(str(path), as_version=4)


class TestSyncNotebook:

    def test_code_cell_updated(self, tmp_path):
        repo = tmp_path / "repo" / "nb.ipynb"
        user = tmp_path / "user" / "nb.ipynb"
        repo.parent.mkdir()
        user.parent.mkdir()
        _write_nb(repo, [_code_cell("c1", "# @cr:code\nimport os\nimport sys\n")])
        _write_nb(user, [_code_cell("c1", "# @cr:code\nimport os\n")])
        report = sync_notebook(repo, user, backup=False)
        assert report.has_changes
        merged = _read_nb(user)
        assert "import sys" in merged.cells[0].source

    def test_config_cell_preserved(self, tmp_path):
        repo = tmp_path / "repo" / "nb.ipynb"
        user = tmp_path / "user" / "nb.ipynb"
        repo.parent.mkdir()
        user.parent.mkdir()
        _write_nb(repo, [_code_cell("c1", "# @cr:config\nX = 1\n")])
        _write_nb(user, [_code_cell("c1", "# @cr:config\nX = 42\n")])
        report = sync_notebook(repo, user, backup=False)
        merged = _read_nb(user)
        assert "X = 42" in merged.cells[0].source

    def test_dry_run_no_write(self, tmp_path):
        repo = tmp_path / "repo" / "nb.ipynb"
        user = tmp_path / "user" / "nb.ipynb"
        repo.parent.mkdir()
        user.parent.mkdir()
        _write_nb(repo, [_code_cell("c1", "# @cr:code\nimport os\nimport sys\n")])
        _write_nb(user, [_code_cell("c1", "# @cr:code\nimport os\n")])
        original_bytes = user.read_bytes()
        report = sync_notebook(repo, user, dry_run=True)
        assert report.has_changes
        assert user.read_bytes() == original_bytes

    def test_backup_created(self, tmp_path):
        repo = tmp_path / "repo" / "nb.ipynb"
        user = tmp_path / "user" / "nb.ipynb"
        repo.parent.mkdir()
        user.parent.mkdir()
        _write_nb(repo, [_code_cell("c1", "# @cr:code\nnew_code()\n")])
        _write_nb(user, [_code_cell("c1", "# @cr:code\nold_code()\n")])
        sync_notebook(repo, user, backup=True)
        assert user.with_suffix(".ipynb.bak").exists()

    def test_no_backup_when_no_changes(self, tmp_path):
        repo = tmp_path / "repo" / "nb.ipynb"
        user = tmp_path / "user" / "nb.ipynb"
        repo.parent.mkdir()
        user.parent.mkdir()
        _write_nb(repo, [_code_cell("c1", "# @cr:code\nimport os\n")])
        _write_nb(user, [_code_cell("c1", "# @cr:code\nimport os\n")])
        sync_notebook(repo, user, backup=True)
        assert not user.with_suffix(".ipynb.bak").exists()


class TestSyncDirectory:

    def test_syncs_multiple_notebooks(self, tmp_path):
        repo = tmp_path / "repo"
        user = tmp_path / "user"
        repo.mkdir()
        user.mkdir()
        for name in ["01_a.ipynb", "02_b.ipynb"]:
            _write_nb(repo / name, [_code_cell("c1", "# @cr:code\nnew()\n")])
            _write_nb(user / name, [_code_cell("c1", "# @cr:code\nold()\n")])
        results = sync_directory(repo, user, backup=False)
        assert len(results) == 2
        assert all(r.has_changes for r in results.values())

    def test_single_notebook_filter(self, tmp_path):
        repo = tmp_path / "repo"
        user = tmp_path / "user"
        repo.mkdir()
        user.mkdir()
        for name in ["01_a.ipynb", "02_b.ipynb"]:
            _write_nb(repo / name, [_code_cell("c1", "# @cr:code\nnew()\n")])
            _write_nb(user / name, [_code_cell("c1", "# @cr:code\nold()\n")])
        results = sync_directory(repo, user, notebook="01_a.ipynb", backup=False)
        assert len(results) == 1
        assert "01_a.ipynb" in results

    def test_skips_missing_user_notebook(self, tmp_path, capsys):
        repo = tmp_path / "repo"
        user = tmp_path / "user"
        repo.mkdir()
        user.mkdir()
        _write_nb(repo / "01_a.ipynb", [_code_cell("c1", "x = 1\n")])
        results = sync_directory(repo, user, backup=False)
        assert len(results) == 0
        assert "not found in user dir" in capsys.readouterr().out

    def test_user_only_notebook_never_touched(self, tmp_path):
        repo = tmp_path / "repo"
        user = tmp_path / "user"
        repo.mkdir()
        user.mkdir()
        _write_nb(repo / "01_a.ipynb", [_code_cell("c1", "# @cr:code\nnew()\n")])
        _write_nb(user / "01_a.ipynb", [_code_cell("c1", "# @cr:code\nold()\n")])
        dbx_init_source = (
            "# @cr:config\n"
            "from customer_retention.integrations.databricks_init import databricks_init\n"
            "\n"
            "result = databricks_init(\n"
            '    catalog="prod_catalog",\n'
            '    schema="churn_model",\n'
            '    workspace_path="Users/me@company.com/cr",\n'
            '    model_name="customer_retention",\n'
            ")\n"
        )
        _write_nb(user / "00_databricks_setup.ipynb", [_code_cell("dbx1", dbx_init_source)])
        original_dbx = (user / "00_databricks_setup.ipynb").read_bytes()
        results = sync_directory(repo, user, backup=False)
        assert "00_databricks_setup.ipynb" not in results
        assert (user / "00_databricks_setup.ipynb").read_bytes() == original_dbx


class TestMainCli:

    def test_main_runs_sync(self, tmp_path):
        repo = tmp_path / "repo"
        user = tmp_path / "user"
        repo.mkdir()
        user.mkdir()
        _write_nb(repo / "01_a.ipynb", [_code_cell("c1", "# @cr:code\nnew()\n")])
        _write_nb(user / "01_a.ipynb", [_code_cell("c1", "# @cr:code\nold()\n")])
        main(["--repo-dir", str(repo), "--user-dir", str(user), "--no-backup"])
        merged = _read_nb(user / "01_a.ipynb")
        assert "new()" in merged.cells[0].source

    def test_main_dry_run(self, tmp_path):
        repo = tmp_path / "repo"
        user = tmp_path / "user"
        repo.mkdir()
        user.mkdir()
        _write_nb(repo / "01_a.ipynb", [_code_cell("c1", "# @cr:code\nnew()\n")])
        _write_nb(user / "01_a.ipynb", [_code_cell("c1", "# @cr:code\nold()\n")])
        original = (user / "01_a.ipynb").read_bytes()
        main(["--repo-dir", str(repo), "--user-dir", str(user), "--dry-run"])
        assert (user / "01_a.ipynb").read_bytes() == original

    def test_main_missing_repo_dir(self, tmp_path):
        with pytest.raises(SystemExit):
            main(["--repo-dir", str(tmp_path / "nonexistent"), "--user-dir", str(tmp_path)])


def _setup_removal_scenario(tmp_path):
    repo = tmp_path / "repo"
    user = tmp_path / "user"
    repo.mkdir()
    user.mkdir()
    _write_nb(repo / "01_a.ipynb", [_code_cell("c1", "# @cr:code\nimport os\n")])
    _write_nb(user / "01_a.ipynb", [
        _code_cell("c1", "# @cr:code\nimport os\n"),
        _code_cell("c2", "print('orphan')\n"),
    ])
    return repo, user


def _setup_no_removal_scenario(tmp_path):
    repo = tmp_path / "repo"
    user = tmp_path / "user"
    repo.mkdir()
    user.mkdir()
    _write_nb(repo / "01_a.ipynb", [_code_cell("c1", "# @cr:code\nnew()\n")])
    _write_nb(user / "01_a.ipynb", [_code_cell("c1", "# @cr:code\nold()\n")])
    return repo, user


class TestConfirmationFlow:

    def test_no_removals_writes_immediately(self, tmp_path):
        repo, user = _setup_no_removal_scenario(tmp_path)
        main(["--repo-dir", str(repo), "--user-dir", str(user), "--no-backup"])
        merged = _read_nb(user / "01_a.ipynb")
        assert "new()" in merged.cells[0].source

    def test_removals_without_force_prompts(self, tmp_path, capsys):
        repo, user = _setup_removal_scenario(tmp_path)
        original = (user / "01_a.ipynb").read_bytes()
        with patch("builtins.input", return_value="n"):
            main(["--repo-dir", str(repo), "--user-dir", str(user), "--no-backup"])
        assert (user / "01_a.ipynb").read_bytes() == original
        output = capsys.readouterr().out
        assert "REMOVED" in output or "removed" in output.lower()

    def test_removals_user_confirms_writes(self, tmp_path):
        repo, user = _setup_removal_scenario(tmp_path)
        with patch("builtins.input", return_value="y"):
            main(["--repo-dir", str(repo), "--user-dir", str(user), "--no-backup"])
        merged = _read_nb(user / "01_a.ipynb")
        assert len(merged.cells) == 1

    def test_removals_user_declines_no_writes(self, tmp_path, capsys):
        repo, user = _setup_removal_scenario(tmp_path)
        original = (user / "01_a.ipynb").read_bytes()
        with patch("builtins.input", return_value="n"):
            main(["--repo-dir", str(repo), "--user-dir", str(user), "--no-backup"])
        assert (user / "01_a.ipynb").read_bytes() == original
        output = capsys.readouterr().out
        assert "# @cr:" in output or "tag" in output.lower()

    def test_force_skips_prompt_even_with_removals(self, tmp_path):
        repo, user = _setup_removal_scenario(tmp_path)
        main(["--repo-dir", str(repo), "--user-dir", str(user), "--no-backup", "--force"])
        merged = _read_nb(user / "01_a.ipynb")
        assert len(merged.cells) == 1

    def test_dry_run_with_removals_shows_removals_no_prompt_no_writes(self, tmp_path, capsys):
        repo, user = _setup_removal_scenario(tmp_path)
        original = (user / "01_a.ipynb").read_bytes()
        main(["--repo-dir", str(repo), "--user-dir", str(user), "--dry-run"])
        assert (user / "01_a.ipynb").read_bytes() == original
        output = capsys.readouterr().out
        assert "removed" in output.lower() or "REMOVED" in output

    def test_removal_details_show_source_preview(self, tmp_path, capsys):
        repo, user = _setup_removal_scenario(tmp_path)
        with patch("builtins.input", return_value="n"):
            main(["--repo-dir", str(repo), "--user-dir", str(user), "--no-backup"])
        output = capsys.readouterr().out
        assert "print('orphan')" in output

    def test_removal_details_show_cell_id(self, tmp_path, capsys):
        repo, user = _setup_removal_scenario(tmp_path)
        with patch("builtins.input", return_value="n"):
            main(["--repo-dir", str(repo), "--user-dir", str(user), "--no-backup"])
        output = capsys.readouterr().out
        assert "c2" in output

    def test_single_notebook_mode_also_prompts(self, tmp_path, capsys):
        repo, user = _setup_removal_scenario(tmp_path)
        original = (user / "01_a.ipynb").read_bytes()
        with patch("builtins.input", return_value="n"):
            main([
                "--repo-dir", str(repo), "--user-dir", str(user),
                "--notebook", "01_a.ipynb", "--no-backup",
            ])
        assert (user / "01_a.ipynb").read_bytes() == original

    def test_multiple_notebooks_with_removals(self, tmp_path):
        repo = tmp_path / "repo"
        user = tmp_path / "user"
        repo.mkdir()
        user.mkdir()
        _write_nb(repo / "01_a.ipynb", [_code_cell("c1", "# @cr:code\nimport os\n")])
        _write_nb(user / "01_a.ipynb", [
            _code_cell("c1", "# @cr:code\nimport os\n"),
            _code_cell("c2", "print('orphan1')\n"),
        ])
        _write_nb(repo / "02_b.ipynb", [_code_cell("c1", "# @cr:code\nimport sys\n")])
        _write_nb(user / "02_b.ipynb", [
            _code_cell("c1", "# @cr:code\nimport sys\n"),
            _code_cell("c3", "print('orphan2')\n"),
        ])
        with patch("builtins.input", return_value="y"):
            main(["--repo-dir", str(repo), "--user-dir", str(user), "--no-backup"])
        assert len(_read_nb(user / "01_a.ipynb").cells) == 1
        assert len(_read_nb(user / "02_b.ipynb").cells) == 1

    def test_force_flag_accepted_by_argparse(self, tmp_path):
        repo, user = _setup_no_removal_scenario(tmp_path)
        main(["--repo-dir", str(repo), "--user-dir", str(user), "--no-backup", "--force"])
        merged = _read_nb(user / "01_a.ipynb")
        assert "new()" in merged.cells[0].source

    def test_backup_created_on_confirmed_write(self, tmp_path):
        repo, user = _setup_removal_scenario(tmp_path)
        with patch("builtins.input", return_value="y"):
            main(["--repo-dir", str(repo), "--user-dir", str(user)])
        assert (user / "01_a.ipynb.bak").exists()


class TestCausalRegeneration:
    """Regen-before-sync: avoid silently pushing stale causal ipynb artifacts."""

    def _setup_causal_layout(self, tmp_path):
        """Build project structure: exploration_notebooks/, causal_notebooks/,
        scripts/notebooks/build_causal_notebooks.py."""
        repo = tmp_path / "exploration_notebooks"
        repo.mkdir()
        _write_nb(repo / "01.ipynb", [_code_cell("c1", "pass\n")])
        user = tmp_path / "exploration_user"
        user.mkdir()
        _write_nb(user / "01.ipynb", [_code_cell("c1", "pass\n")])

        causal_repo = tmp_path / "causal_notebooks"
        causal_repo.mkdir()
        _write_nb(causal_repo / "c01.ipynb", [_code_cell("c1", "pass\n")])
        causal_user = tmp_path / "causal_user"
        causal_user.mkdir()
        _write_nb(causal_user / "c01.ipynb", [_code_cell("c1", "pass\n")])

        scripts = tmp_path / "scripts" / "notebooks"
        scripts.mkdir(parents=True)
        build = scripts / "build_causal_notebooks.py"
        # Sentinel script — writes a marker file so the test can verify the
        # build ran before the sync did.
        build.write_text(
            "import pathlib\n"
            f"pathlib.Path({str(tmp_path)!r}).joinpath('.build_ran').touch()\n"
            "print('regenerated causal notebooks')\n"
        )
        return repo, user, causal_repo, causal_user, build

    def test_regenerates_before_syncing_by_default(self, tmp_path, capsys):
        repo, user, causal_repo, causal_user, _ = self._setup_causal_layout(tmp_path)
        main([
            "--repo-dir", str(repo),
            "--user-dir", str(user),
            "--causal-repo-dir", str(causal_repo),
            "--causal-user-dir", str(causal_user),
            "--no-backup",
        ])
        marker = tmp_path / ".build_ran"
        assert marker.exists(), "build_causal_notebooks.py should have executed"
        out = capsys.readouterr().out
        assert "regenerating from build_causal_notebooks.py" in out
        assert "regenerated causal notebooks" in out

    def test_no_regenerate_flag_skips_build(self, tmp_path):
        repo, user, causal_repo, causal_user, _ = self._setup_causal_layout(tmp_path)
        main([
            "--repo-dir", str(repo),
            "--user-dir", str(user),
            "--causal-repo-dir", str(causal_repo),
            "--causal-user-dir", str(causal_user),
            "--no-backup",
            "--no-regenerate-causal",
        ])
        assert not (tmp_path / ".build_ran").exists()

    def test_dry_run_skips_actual_build(self, tmp_path, capsys):
        repo, user, causal_repo, causal_user, _ = self._setup_causal_layout(tmp_path)
        main([
            "--repo-dir", str(repo),
            "--user-dir", str(user),
            "--causal-repo-dir", str(causal_repo),
            "--causal-user-dir", str(causal_user),
            "--dry-run",
        ])
        # Dry-run should not actually invoke the build
        assert not (tmp_path / ".build_ran").exists()
        out = capsys.readouterr().out
        assert "would regenerate" in out
        assert "build_causal_notebooks.py" in out

    def test_missing_build_script_degrades_gracefully(self, tmp_path, capsys):
        repo, user, causal_repo, causal_user, build = self._setup_causal_layout(tmp_path)
        build.unlink()  # remove the build script
        # Should complete without raising
        main([
            "--repo-dir", str(repo),
            "--user-dir", str(user),
            "--causal-repo-dir", str(causal_repo),
            "--causal-user-dir", str(causal_user),
            "--no-backup",
        ])
        out = capsys.readouterr().out
        assert "build script not found" in out

    def test_build_failure_aborts_sync(self, tmp_path):
        repo, user, causal_repo, causal_user, build = self._setup_causal_layout(tmp_path)
        build.write_text("import sys\nsys.exit(7)\n")
        with pytest.raises(SystemExit):
            main([
                "--repo-dir", str(repo),
                "--user-dir", str(user),
                "--causal-repo-dir", str(causal_repo),
                "--causal-user-dir", str(causal_user),
                "--no-backup",
            ])
