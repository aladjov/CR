from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

from customer_retention.core.compat.remote_path import (
    RemotePath,
    _is_scheme_path,
    _RemoteStat,
    _RemoteTextReader,
    _RemoteTextWriter,
    _translate_remote_error,
    make_path,
)


class _FileInfo:
    def __init__(self, name: str, size: int = 0, modification_time: int = 0, is_dir: bool = False):
        self.name = name + ("/" if is_dir else "")
        self.size = size
        self.modificationTime = modification_time
        self._is_dir = is_dir

    def isDir(self) -> bool:  # noqa: N802 — mimics Databricks FileInfo API
        return self._is_dir


def _make_dbutils(files=None, head_content="", ls_raises=None):
    dbu = MagicMock()

    def _ls(path):
        if ls_raises and path in ls_raises:
            raise Exception(f"Path not found: {path}")
        if files and path in files:
            return files[path]
        raise Exception(f"Path not found: {path}")

    dbu.fs.ls = MagicMock(side_effect=_ls)
    dbu.fs.head = MagicMock(return_value=head_content)
    dbu.fs.put = MagicMock()
    dbu.fs.mkdirs = MagicMock()
    dbu.fs.rm = MagicMock()
    return dbu


@pytest.fixture()
def mock_dbutils():
    dbu = _make_dbutils()
    with patch("customer_retention.core.compat.remote_path._get_dbutils", return_value=dbu):
        yield dbu


class TestRemotePathProperties:
    def test_name(self):
        p = RemotePath("/Volumes/catalog/schema/data/file.csv")
        assert p.name == "file.csv"

    def test_stem(self):
        p = RemotePath("/Volumes/catalog/schema/data/file.csv")
        assert p.stem == "file"

    def test_suffix(self):
        p = RemotePath("/Volumes/catalog/schema/data/file.csv")
        assert p.suffix == ".csv"

    def test_parts(self):
        p = RemotePath("/Volumes/catalog/schema")
        assert p.parts == ("/", "Volumes", "catalog", "schema")

    def test_parent(self):
        p = RemotePath("/Volumes/catalog/schema/data")
        parent = p.parent
        assert isinstance(parent, RemotePath)
        assert str(parent) == "/Volumes/catalog/schema"

    def test_parents(self):
        p = RemotePath("/a/b/c")
        parents = p.parents
        assert all(isinstance(pp, RemotePath) for pp in parents)
        assert str(parents[0]) == "/a/b"
        assert str(parents[1]) == "/a"

    def test_str(self):
        p = RemotePath("/Volumes/catalog/schema")
        assert str(p) == "/Volumes/catalog/schema"

    def test_fspath(self):
        import os

        p = RemotePath("/Volumes/catalog/schema")
        assert os.fspath(p) == "/Volumes/catalog/schema"

    def test_repr(self):
        p = RemotePath("/a/b")
        assert repr(p) == "RemotePath('/a/b')"


class TestRemotePathOperators:
    def test_truediv_str(self):
        p = RemotePath("/a/b")
        result = p / "c"
        assert isinstance(result, RemotePath)
        assert str(result) == "/a/b/c"

    def test_truediv_chained(self):
        p = RemotePath("/base")
        result = p / "sub" / "file.txt"
        assert isinstance(result, RemotePath)
        assert str(result) == "/base/sub/file.txt"

    def test_eq_remote_paths(self):
        assert RemotePath("/a/b") == RemotePath("/a/b")
        assert RemotePath("/a/b") != RemotePath("/a/c")

    def test_eq_str(self):
        assert RemotePath("/a/b") == "/a/b"

    def test_hash(self):
        s = {RemotePath("/a"), RemotePath("/a"), RemotePath("/b")}
        assert len(s) == 2

    def test_lt(self):
        assert RemotePath("/a") < RemotePath("/b")

    def test_with_name(self):
        p = RemotePath("/a/b/file.csv")
        renamed = p.with_name("other.yaml")
        assert isinstance(renamed, RemotePath)
        assert str(renamed) == "/a/b/other.yaml"

    def test_with_suffix(self):
        p = RemotePath("/a/b/file.csv")
        changed = p.with_suffix(".yaml")
        assert isinstance(changed, RemotePath)
        assert str(changed) == "/a/b/file.yaml"

    def test_joinpath(self):
        p = RemotePath("/a")
        result = p.joinpath("b", "c")
        assert isinstance(result, RemotePath)
        assert str(result) == "/a/b/c"

    def test_relative_to(self):
        p = RemotePath("/a/b/c")
        rel = p.relative_to("/a")
        assert isinstance(rel, PurePosixPath)
        assert str(rel) == "b/c"

    def test_as_posix(self):
        p = RemotePath("/a/b/c")
        assert p.as_posix() == "/a/b/c"


class TestRemotePathResolve:
    def test_resolve_returns_self(self):
        p = RemotePath("/Volumes/catalog/schema/data")
        resolved = p.resolve()
        assert resolved is p

    def test_resolve_returns_remote_path(self):
        p = RemotePath("/data/findings/file.yaml")
        resolved = p.resolve()
        assert isinstance(resolved, RemotePath)
        assert str(resolved) == "/data/findings/file.yaml"

    def test_resolve_in_set_membership(self):
        p1 = RemotePath("/a/b")
        p2 = RemotePath("/a/b")
        loaded = {p1.resolve()}
        assert p2.resolve() in loaded

    def test_resolve_after_glob(self, mock_dbutils):
        mock_dbutils.fs.ls.side_effect = None
        mock_dbutils.fs.ls.return_value = [
            _FileInfo("ds_findings.yaml"),
        ]
        p = RemotePath("/findings")
        candidates = p.glob("*_findings.yaml")
        resolved = candidates[0].resolve()
        assert isinstance(resolved, RemotePath)
        assert str(resolved) == "/findings/ds_findings.yaml"


class TestRemotePathFromRemotePath:
    def test_construct_from_remote_path(self):
        p1 = RemotePath("/a/b")
        p2 = RemotePath(p1)
        assert str(p2) == "/a/b"
        assert isinstance(p2, RemotePath)

    def test_construct_from_pure_posix(self):
        pp = PurePosixPath("/x/y")
        p = RemotePath(pp)
        assert str(p) == "/x/y"


class TestRemotePathExists:
    def test_exists_true(self, mock_dbutils):
        mock_dbutils.fs.ls.side_effect = None
        mock_dbutils.fs.ls.return_value = [_FileInfo("file.csv", size=100)]
        p = RemotePath("/data/file.csv")
        assert p.exists() is True

    def test_exists_false(self, mock_dbutils):
        mock_dbutils.fs.ls.side_effect = None
        mock_dbutils.fs.ls.return_value = [_FileInfo("other.csv")]
        p = RemotePath("/data/file.csv")
        assert p.exists() is False

    def test_exists_parent_not_found(self, mock_dbutils):
        mock_dbutils.fs.ls.side_effect = Exception("not found")
        p = RemotePath("/data/file.csv")
        assert p.exists() is False


class TestRemotePathIsDir:
    def test_is_dir_true(self, mock_dbutils):
        mock_dbutils.fs.ls.side_effect = None
        mock_dbutils.fs.ls.return_value = [_FileInfo("sub/", is_dir=True)]
        p = RemotePath("/data")
        assert p.is_dir() is True

    def test_is_dir_false(self, mock_dbutils):
        mock_dbutils.fs.ls.side_effect = Exception("not a dir")
        p = RemotePath("/data/file.csv")
        assert p.is_dir() is False


class TestRemotePathMkdir:
    def test_mkdir(self, mock_dbutils):
        p = RemotePath("/data/new_dir")
        p.mkdir(parents=True, exist_ok=True)
        mock_dbutils.fs.mkdirs.assert_called_once_with("/data/new_dir")


class TestRemotePathReadWrite:
    def test_read_text(self, mock_dbutils):
        mock_dbutils.fs.head.return_value = "hello world"
        p = RemotePath("/data/file.txt")
        assert p.read_text() == "hello world"
        mock_dbutils.fs.head.assert_called_once_with("/data/file.txt", 1_048_576)

    def test_write_text(self, mock_dbutils):
        p = RemotePath("/data/file.txt")
        p.write_text("some content")
        mock_dbutils.fs.put.assert_called_once_with("/data/file.txt", "some content", overwrite=True)


class TestRemotePathOpen:
    def test_open_read(self, mock_dbutils):
        mock_dbutils.fs.head.return_value = "line1\nline2\n"
        p = RemotePath("/data/file.txt")
        with p.open("r") as f:
            content = f.read()
        assert content == "line1\nline2\n"

    def test_open_write(self, mock_dbutils):
        p = RemotePath("/data/file.txt")
        with p.open("w") as f:
            f.write("written content")
        mock_dbutils.fs.put.assert_called_once_with("/data/file.txt", "written content", overwrite=True)

    def test_open_write_yaml(self, mock_dbutils):
        p = RemotePath("/data/config.yaml")
        data = {"key": "value", "list": [1, 2, 3]}
        with p.open("w") as f:
            yaml.dump(data, f, default_flow_style=False)
        call_args = mock_dbutils.fs.put.call_args
        written = call_args[0][1]
        loaded = yaml.safe_load(written)
        assert loaded == data

    def test_open_read_json(self, mock_dbutils):
        data = {"name": "test", "value": 42}
        mock_dbutils.fs.head.return_value = json.dumps(data)
        p = RemotePath("/data/meta.json")
        with p.open("r") as f:
            loaded = json.load(f)
        assert loaded == data


class TestRemotePathGlob:
    def test_simple_glob(self, mock_dbutils):
        mock_dbutils.fs.ls.side_effect = None
        mock_dbutils.fs.ls.return_value = [
            _FileInfo("a_findings.yaml"),
            _FileInfo("b_findings.yaml"),
            _FileInfo("readme.md"),
        ]
        p = RemotePath("/data")
        results = p.glob("*_findings.yaml")
        assert len(results) == 2
        assert all(isinstance(r, RemotePath) for r in results)
        assert str(results[0]) == "/data/a_findings.yaml"
        assert str(results[1]) == "/data/b_findings.yaml"

    def test_glob_no_matches(self, mock_dbutils):
        mock_dbutils.fs.ls.side_effect = None
        mock_dbutils.fs.ls.return_value = [_FileInfo("file.txt")]
        p = RemotePath("/data")
        results = p.glob("*.yaml")
        assert results == []

    def test_glob_dir_not_found(self, mock_dbutils):
        mock_dbutils.fs.ls.side_effect = Exception("not found")
        p = RemotePath("/nonexistent")
        results = p.glob("*.yaml")
        assert results == []


class TestRemotePathIterdir:
    def test_iterdir(self, mock_dbutils):
        mock_dbutils.fs.ls.side_effect = None
        mock_dbutils.fs.ls.return_value = [
            _FileInfo("a/", is_dir=True),
            _FileInfo("b.txt"),
        ]
        p = RemotePath("/data")
        entries = list(p.iterdir())
        assert len(entries) == 2
        assert all(isinstance(e, RemotePath) for e in entries)
        assert str(entries[0]) == "/data/a"
        assert str(entries[1]) == "/data/b.txt"

    def test_iterdir_empty(self, mock_dbutils):
        mock_dbutils.fs.ls.side_effect = Exception("not found")
        p = RemotePath("/empty")
        assert list(p.iterdir()) == []


class TestRemotePathStat:
    def test_stat(self, mock_dbutils):
        mock_dbutils.fs.ls.side_effect = None
        mock_dbutils.fs.ls.return_value = [
            _FileInfo("file.txt", size=1024, modification_time=1700000000000),
        ]
        p = RemotePath("/data/file.txt")
        st = p.stat()
        assert isinstance(st, _RemoteStat)
        assert st.st_size == 1024
        assert st.st_mtime == 1700000000.0

    def test_stat_not_found(self, mock_dbutils):
        mock_dbutils.fs.ls.side_effect = None
        mock_dbutils.fs.ls.return_value = [_FileInfo("other.txt")]
        p = RemotePath("/data/file.txt")
        with pytest.raises(FileNotFoundError):
            p.stat()


class TestRemotePathUnlink:
    def test_unlink(self, mock_dbutils):
        p = RemotePath("/data/file.txt")
        p.unlink()
        mock_dbutils.fs.rm.assert_called_once_with("/data/file.txt", recurse=False)

    def test_unlink_missing_ok(self, mock_dbutils):
        mock_dbutils.fs.rm.side_effect = Exception("not found")
        p = RemotePath("/data/file.txt")
        p.unlink(missing_ok=True)

    def test_unlink_missing_raises(self, mock_dbutils):
        mock_dbutils.fs.rm.side_effect = Exception("not found")
        p = RemotePath("/data/file.txt")
        with pytest.raises(Exception, match="not found"):
            p.unlink(missing_ok=False)


class TestRemotePathRmdir:
    def test_rmdir(self, mock_dbutils):
        p = RemotePath("/data/empty_dir")
        p.rmdir()
        mock_dbutils.fs.rm.assert_called_once_with("/data/empty_dir", recurse=False)

    def test_rmdir_raises_on_failure(self, mock_dbutils):
        mock_dbutils.fs.rm.side_effect = Exception("not empty")
        p = RemotePath("/data/nonempty_dir")
        with pytest.raises(OSError):
            p.rmdir()


class TestRemoteTextReader:
    def test_read(self):
        reader = _RemoteTextReader("hello\nworld\n")
        with reader as r:
            assert r.read() == "hello\nworld\n"

    def test_readline(self):
        reader = _RemoteTextReader("line1\nline2\n")
        with reader as r:
            assert r.readline() == "line1\n"
            assert r.readline() == "line2\n"

    def test_readlines(self):
        reader = _RemoteTextReader("a\nb\nc\n")
        with reader as r:
            assert r.readlines() == ["a\n", "b\n", "c\n"]

    def test_iter(self):
        reader = _RemoteTextReader("x\ny\n")
        with reader as r:
            lines = list(r)
        assert lines == ["x\n", "y\n"]


class TestRemoteTextWriter:
    def test_write_and_flush(self, mock_dbutils):
        p = RemotePath("/out/file.txt")
        writer = _RemoteTextWriter(p)
        with writer as w:
            w.write("first ")
            w.write("second")
        mock_dbutils.fs.put.assert_called_once_with("/out/file.txt", "first second", overwrite=True)


class TestMakePath:
    def test_local_mode(self):
        with (
            patch("customer_retention.core.compat.detection.is_remote_spark", return_value=False),
            patch("customer_retention.core.compat.detection.is_databricks", return_value=False),
        ):
            result = make_path("/some/local/path")
            assert isinstance(result, Path)
            assert str(result) == "/some/local/path"

    def test_remote_mode(self):
        with (
            patch("customer_retention.core.compat.detection.is_remote_spark", return_value=True),
            patch("customer_retention.core.compat.detection.is_databricks", return_value=False),
        ):
            result = make_path("/Volumes/catalog/schema/experiments")
            assert isinstance(result, RemotePath)
            assert str(result) == "/Volumes/catalog/schema/experiments"

    def test_databricks_fuse_path_returns_path(self):
        with (
            patch("customer_retention.core.compat.detection.is_remote_spark", return_value=False),
            patch("customer_retention.core.compat.detection.is_databricks", return_value=True),
        ):
            result = make_path("/Volumes/catalog/schema/experiments")
            assert isinstance(result, Path)
            assert str(result) == "/Volumes/catalog/schema/experiments"

    def test_databricks_scheme_path_returns_remote(self):
        with (
            patch("customer_retention.core.compat.detection.is_remote_spark", return_value=False),
            patch("customer_retention.core.compat.detection.is_databricks", return_value=True),
        ):
            result = make_path("dbfs:/FileStore/experiments")
            assert isinstance(result, RemotePath)
            assert str(result) == "dbfs:/FileStore/experiments"

    def test_databricks_workspace_fuse_path(self):
        with (
            patch("customer_retention.core.compat.detection.is_remote_spark", return_value=False),
            patch("customer_retention.core.compat.detection.is_databricks", return_value=True),
        ):
            result = make_path("/Workspace/Users/me/experiments")
            assert isinstance(result, Path)

    def test_databricks_s3_scheme_path(self):
        with (
            patch("customer_retention.core.compat.detection.is_remote_spark", return_value=False),
            patch("customer_retention.core.compat.detection.is_databricks", return_value=True),
        ):
            result = make_path("s3://bucket/experiments")
            assert isinstance(result, RemotePath)

    def test_already_remote_path(self):
        rp = RemotePath("/a/b")
        result = make_path(rp)
        assert result is rp

    def test_already_path_local_mode(self):
        with (
            patch("customer_retention.core.compat.detection.is_remote_spark", return_value=False),
            patch("customer_retention.core.compat.detection.is_databricks", return_value=False),
        ):
            p = Path("/some/path")
            result = make_path(p)
            assert result is p

    def test_force_remote_true(self):
        result = make_path("/a/b", force_remote=True)
        assert isinstance(result, RemotePath)

    def test_force_remote_false(self):
        result = make_path("/a/b", force_remote=False)
        assert isinstance(result, Path)


class TestMakePathPropagation:
    def test_truediv_propagates_type(self):
        rp = RemotePath("/base")
        child = rp / "sub" / "file.txt"
        assert isinstance(child, RemotePath)
        assert str(child) == "/base/sub/file.txt"

    def test_parent_propagates_type(self):
        rp = RemotePath("/a/b/c")
        parent = rp.parent
        assert isinstance(parent, RemotePath)
        assert str(parent) == "/a/b"


class TestRemotePathWithRunNamespace:
    def test_namespace_paths_propagate_remote(self, mock_dbutils):
        from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace

        root = RemotePath("/Volumes/catalog/schema/experiments")
        ns = RunNamespace(root=root, run_id="test-run-001")

        assert isinstance(ns.run_dir, RemotePath)
        assert str(ns.run_dir) == "/Volumes/catalog/schema/experiments/runs/test-run-001"

        assert isinstance(ns.datasets_dir, RemotePath)
        assert isinstance(ns.merged_dir, RemotePath)
        assert isinstance(ns.session_dir, RemotePath)

        findings_path = ns.multi_dataset_findings_path
        assert isinstance(findings_path, RemotePath)
        assert str(findings_path).endswith("multi_dataset_findings.yaml")

        dataset_findings = ns.dataset_findings_dir("my_data")
        assert isinstance(dataset_findings, RemotePath)
        assert "datasets/my_data/findings" in str(dataset_findings)

    def test_namespace_setup_calls_mkdirs(self, mock_dbutils):
        from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace

        root = RemotePath("/Volumes/cat/sch/exp")
        ns = RunNamespace(root=root, run_id="r1")
        ns.setup()
        assert mock_dbutils.fs.mkdirs.call_count >= 7

    def test_namespace_list_datasets(self, mock_dbutils):
        from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace

        mock_dbutils.fs.ls.side_effect = None
        mock_dbutils.fs.ls.return_value = [
            _FileInfo("dataset_a/", is_dir=True),
            _FileInfo("dataset_b/", is_dir=True),
        ]
        root = RemotePath("/Volumes/cat/sch/exp")
        ns = RunNamespace(root=root, run_id="r1")
        datasets = ns.list_datasets()
        assert datasets == ["dataset_a", "dataset_b"]

    def test_namespace_discover_findings(self, mock_dbutils):
        from customer_retention.analysis.auto_explorer.run_namespace import RunNamespace

        def ls_side_effect(path):
            if path.endswith("datasets"):
                return [_FileInfo("ds1/", is_dir=True)]
            if path.endswith("ds1"):
                return [_FileInfo("findings/", is_dir=True)]
            if path.endswith("findings"):
                return [
                    _FileInfo("ds1_findings.yaml"),
                    _FileInfo("ds1_aggregated_findings.yaml"),
                ]
            raise Exception("not found")

        mock_dbutils.fs.ls.side_effect = ls_side_effect
        root = RemotePath("/exp")
        ns = RunNamespace(root=root, run_id="r1")
        findings = ns.discover_all_findings(prefer_aggregated=True)
        assert len(findings) == 1
        assert "aggregated" in str(findings[0])


class TestExperimentsIntegration:
    def test_get_experiments_dir_remote(self):
        with (
            patch.dict("os.environ", {"CR_EXPERIMENTS_DIR": "/Volumes/cat/sch/experiments"}),
            patch("customer_retention.core.compat.detection.is_remote_spark", return_value=True),
            patch("customer_retention.core.compat.detection.is_databricks", return_value=False),
        ):
            from customer_retention.core.config.experiments import get_experiments_dir

            result = get_experiments_dir()
            assert isinstance(result, RemotePath)
            assert str(result) == "/Volumes/cat/sch/experiments"

    def test_get_experiments_dir_local(self):
        with (
            patch.dict("os.environ", {"CR_EXPERIMENTS_DIR": "/local/experiments"}, clear=False),
            patch("customer_retention.core.compat.detection.is_remote_spark", return_value=False),
            patch("customer_retention.core.compat.detection.is_databricks", return_value=False),
        ):
            from customer_retention.core.config.experiments import get_experiments_dir

            result = get_experiments_dir()
            assert isinstance(result, Path)


class TestIsSchemaPath:
    def test_dbfs_scheme(self):
        assert _is_scheme_path("dbfs:/FileStore/data") is True

    def test_s3_scheme(self):
        assert _is_scheme_path("s3://bucket/key") is True

    def test_abfss_scheme(self):
        assert _is_scheme_path("abfss://container@storage.dfs.core.windows.net/path") is True

    def test_absolute_posix_path(self):
        assert _is_scheme_path("/Volumes/catalog/schema/vol") is False

    def test_workspace_path(self):
        assert _is_scheme_path("/Workspace/Users/me/project") is False

    def test_relative_path(self):
        assert _is_scheme_path("experiments/data") is False

    def test_empty_string(self):
        assert _is_scheme_path("") is False


class TestMakePathDatabricksFuse:
    def test_databricks_fuse_setup_experiments(self, tmp_path):
        with (
            patch("customer_retention.core.compat.detection.is_remote_spark", return_value=False),
            patch("customer_retention.core.compat.detection.is_databricks", return_value=True),
        ):
            base = make_path(str(tmp_path / "experiments"))
            assert isinstance(base, Path)
            sub = base / "data" / "bronze"
            sub.mkdir(parents=True, exist_ok=True)
            assert sub.exists()

    def test_remote_spark_always_returns_remote(self):
        with (
            patch("customer_retention.core.compat.detection.is_remote_spark", return_value=True),
            patch("customer_retention.core.compat.detection.is_databricks", return_value=False),
        ):
            result = make_path("/Volumes/catalog/schema/experiments")
            assert isinstance(result, RemotePath)

    def test_cr_spark_remote_env_returns_remote(self):
        with (
            patch.dict("os.environ", {"CR_SPARK_REMOTE": "1"}),
            patch("customer_retention.core.compat.detection.is_remote_spark", return_value=False),
            patch("customer_retention.core.compat.detection.is_databricks", return_value=False),
        ):
            result = make_path("/Volumes/catalog/schema/experiments")
            assert isinstance(result, RemotePath)


class TestTranslateRemoteError:
    def test_file_not_found_exception(self):
        exc = Exception("java.io.FileNotFoundException: /data/missing.txt")
        with pytest.raises(FileNotFoundError, match="No such file"):
            _translate_remote_error(exc, "/data/missing.txt")

    def test_generic_exception_becomes_os_error(self):
        exc = Exception("Some Py4J gateway error")
        with pytest.raises(OSError, match="Remote filesystem error"):
            _translate_remote_error(exc, "/data/file.txt")

    def test_preserves_original_as_cause(self):
        original = Exception("java.io.FileNotFoundException: gone")
        with pytest.raises(FileNotFoundError) as exc_info:
            _translate_remote_error(original, "/gone")
        assert exc_info.value.__cause__ is original


class TestRemotePathExceptionTranslation:
    def test_read_text_file_not_found_raises_file_not_found_error(self, mock_dbutils):
        mock_dbutils.fs.head.side_effect = Exception(
            "java.io.FileNotFoundException: /data/missing.txt"
        )
        p = RemotePath("/data/missing.txt")
        with pytest.raises(FileNotFoundError):
            p.read_text()

    def test_read_text_generic_error_raises_os_error(self, mock_dbutils):
        mock_dbutils.fs.head.side_effect = Exception("Py4J gateway error")
        p = RemotePath("/data/file.txt")
        with pytest.raises(OSError):
            p.read_text()

    def test_write_text_error_raises_os_error(self, mock_dbutils):
        mock_dbutils.fs.put.side_effect = Exception("Permission denied")
        p = RemotePath("/data/file.txt")
        with pytest.raises(OSError):
            p.write_text("content")

    def test_write_text_file_not_found_raises_file_not_found_error(self, mock_dbutils):
        mock_dbutils.fs.put.side_effect = Exception(
            "java.io.FileNotFoundException: /readonly/file.txt"
        )
        p = RemotePath("/readonly/file.txt")
        with pytest.raises(FileNotFoundError):
            p.write_text("content")

    def test_stat_parent_not_found_raises_file_not_found_error(self, mock_dbutils):
        mock_dbutils.fs.ls.side_effect = Exception(
            "java.io.FileNotFoundException: /nonexistent/"
        )
        p = RemotePath("/nonexistent/file.txt")
        with pytest.raises(FileNotFoundError):
            p.stat()

    def test_stat_generic_error_raises_os_error(self, mock_dbutils):
        mock_dbutils.fs.ls.side_effect = Exception("Py4J connection lost")
        p = RemotePath("/data/file.txt")
        with pytest.raises(OSError):
            p.stat()
