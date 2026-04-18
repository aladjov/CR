from __future__ import annotations

import fnmatch
import io
import os
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Optional, Union


class _RemoteStat:
    def __init__(self, st_size: int = 0, st_mtime: float = 0.0) -> None:
        self.st_size = st_size
        self.st_mtime = st_mtime


class _RemoteTextReader:
    def __init__(self, content: str) -> None:
        self._buf = io.StringIO(content)

    def read(self, size: int = -1) -> str:
        return self._buf.read(size)

    def readline(self, size: int = -1) -> str:
        return self._buf.readline(size)

    def readlines(self) -> list[str]:
        return self._buf.readlines()

    def __iter__(self) -> Iterator[str]:
        return iter(self._buf)

    def __enter__(self) -> _RemoteTextReader:
        return self

    def __exit__(self, *args: Any) -> None:
        self._buf.close()


class _RemoteTextWriter:
    def __init__(self, path: RemotePath) -> None:
        self._path = path
        self._buf = io.StringIO()

    def write(self, data: str) -> int:
        return self._buf.write(data)

    def __enter__(self) -> _RemoteTextWriter:
        return self

    def __exit__(self, *args: Any) -> None:
        content = self._buf.getvalue()
        self._buf.close()
        dbu = _get_dbutils()
        dbu.fs.put(str(self._path), content, overwrite=True)


def _get_dbutils() -> Any:
    from customer_retention.core.compat.detection import get_dbutils

    dbu = get_dbutils()
    if dbu is None and os.environ.get("CR_SPARK_REMOTE"):
        from customer_retention.core.compat.detection import connect_remote_spark

        connect_remote_spark()
        dbu = get_dbutils()
    if dbu is None:
        raise RuntimeError("dbutils is not available — cannot perform remote filesystem operations")
    return dbu


def _translate_remote_error(exc: Exception, path_str: str) -> None:
    msg = str(exc)
    if "FileNotFoundException" in msg or "java.io.FileNotFoundException" in msg:
        raise FileNotFoundError(f"No such file: {path_str}") from exc
    raise OSError(f"Remote filesystem error for {path_str}: {msg}") from exc


class RemotePath:
    __slots__ = ("_pure",)

    def __init__(self, path: Union[str, PurePosixPath, RemotePath]) -> None:
        if isinstance(path, RemotePath):
            self._pure = path._pure
        elif isinstance(path, PurePosixPath):
            self._pure = path
        else:
            self._pure = PurePosixPath(str(path))

    @property
    def name(self) -> str:
        return self._pure.name

    @property
    def stem(self) -> str:
        return self._pure.stem

    @property
    def suffix(self) -> str:
        return self._pure.suffix

    @property
    def parts(self) -> tuple[str, ...]:
        return self._pure.parts

    @property
    def parent(self) -> RemotePath:
        return RemotePath(self._pure.parent)

    @property
    def parents(self) -> tuple[RemotePath, ...]:
        return tuple(RemotePath(p) for p in self._pure.parents)

    def __truediv__(self, other: Union[str, os.PathLike[str]]) -> RemotePath:
        return RemotePath(self._pure / str(other))

    def __rtruediv__(self, other: Union[str, os.PathLike[str]]) -> RemotePath:
        return RemotePath(PurePosixPath(str(other)) / self._pure)

    def __str__(self) -> str:
        return str(self._pure)

    def __repr__(self) -> str:
        return f"RemotePath({str(self._pure)!r})"

    def __fspath__(self) -> str:
        return str(self._pure)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, RemotePath):
            return self._pure == other._pure
        if isinstance(other, (str, PurePosixPath, Path)):
            return str(self._pure) == str(other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._pure)

    def __lt__(self, other: object) -> bool:
        if isinstance(other, RemotePath):
            return self._pure < other._pure
        return NotImplemented

    def exists(self) -> bool:
        dbu = _get_dbutils()
        parent_str = str(self._pure.parent)
        try:
            entries = dbu.fs.ls(parent_str)
            target = self.name
            for entry in entries:
                entry_name = entry.name.rstrip("/")
                if entry_name == target:
                    return True
            return False
        except Exception:
            if str(self._pure) == "/" or str(self._pure) == ".":
                return True
            return False

    def is_dir(self) -> bool:
        dbu = _get_dbutils()
        try:
            dbu.fs.ls(str(self._pure))
            return True
        except Exception:
            return False

    def is_file(self) -> bool:
        return self.exists() and not self.is_dir()

    def mkdir(self, parents: bool = False, exist_ok: bool = False) -> None:
        dbu = _get_dbutils()
        dbu.fs.mkdirs(str(self._pure))

    def read_text(self, encoding: str = "utf-8") -> str:
        dbu = _get_dbutils()
        try:
            return dbu.fs.head(str(self._pure), 1_048_576)
        except Exception as exc:
            _translate_remote_error(exc, str(self._pure))

    def write_text(self, content: str, encoding: str = "utf-8") -> None:
        dbu = _get_dbutils()
        try:
            dbu.fs.put(str(self._pure), content, overwrite=True)
        except Exception as exc:
            _translate_remote_error(exc, str(self._pure))

    def open(self, mode: str = "r", encoding: str = "utf-8") -> Union[_RemoteTextReader, _RemoteTextWriter]:
        if "w" in mode:
            return _RemoteTextWriter(self)
        return _RemoteTextReader(self.read_text(encoding))

    def glob(self, pattern: str) -> list[RemotePath]:
        if "**" in pattern:
            return self._recursive_glob(pattern)
        return self._simple_glob(pattern)

    def _simple_glob(self, pattern: str) -> list[RemotePath]:
        dbu = _get_dbutils()
        try:
            entries = dbu.fs.ls(str(self._pure))
        except Exception:
            return []
        results = []
        for entry in entries:
            entry_name = entry.name.rstrip("/")
            if fnmatch.fnmatch(entry_name, pattern):
                results.append(self / entry_name)
        return sorted(results, key=lambda p: p.name)

    def _recursive_glob(self, pattern: str) -> list[RemotePath]:
        parts = PurePosixPath(pattern).parts
        return self._walk_glob(parts)

    def _walk_glob(self, parts: tuple[str, ...]) -> list[RemotePath]:
        if not parts:
            return [self]
        dbu = _get_dbutils()
        try:
            entries = dbu.fs.ls(str(self._pure))
        except Exception:
            return []
        results = []
        if parts[0] == "**":
            remaining = parts[1:]
            if remaining:
                for entry in entries:
                    entry_name = entry.name.rstrip("/")
                    child = self / entry_name
                    if fnmatch.fnmatch(entry_name, remaining[0]):
                        results.extend(child._walk_glob(remaining[1:]))
                    if entry.isDir():
                        results.extend(child._walk_glob(parts))
            else:
                results.append(self)
                for entry in entries:
                    entry_name = entry.name.rstrip("/")
                    child = self / entry_name
                    results.append(child)
                    if entry.isDir():
                        results.extend(child._walk_glob(parts))
        else:
            for entry in entries:
                entry_name = entry.name.rstrip("/")
                if fnmatch.fnmatch(entry_name, parts[0]):
                    child = self / entry_name
                    results.extend(child._walk_glob(parts[1:]))
        return results

    def iterdir(self) -> Iterator[RemotePath]:
        dbu = _get_dbutils()
        try:
            entries = dbu.fs.ls(str(self._pure))
        except Exception:
            return
        for entry in entries:
            entry_name = entry.name.rstrip("/")
            yield self / entry_name

    def unlink(self, missing_ok: bool = False) -> None:
        dbu = _get_dbutils()
        try:
            dbu.fs.rm(str(self._pure), recurse=False)
        except Exception:
            if not missing_ok:
                raise

    def rmdir(self) -> None:
        dbu = _get_dbutils()
        try:
            dbu.fs.rm(str(self._pure), recurse=False)
        except Exception:
            raise OSError(f"Cannot remove directory: {self._pure}") from None

    def stat(self) -> _RemoteStat:
        dbu = _get_dbutils()
        parent_str = str(self._pure.parent)
        try:
            entries = dbu.fs.ls(parent_str)
        except Exception as exc:
            _translate_remote_error(exc, str(self._pure))
        target = self.name
        for entry in entries:
            entry_name = entry.name.rstrip("/")
            if entry_name == target:
                mtime = getattr(entry, "modificationTime", 0) / 1000.0
                return _RemoteStat(st_size=entry.size, st_mtime=mtime)
        raise FileNotFoundError(f"No such file: {self._pure}")

    def relative_to(self, other: Union[str, Path, RemotePath]) -> PurePosixPath:
        return self._pure.relative_to(str(other))

    def with_name(self, name: str) -> RemotePath:
        return RemotePath(self._pure.with_name(name))

    def with_suffix(self, suffix: str) -> RemotePath:
        return RemotePath(self._pure.with_suffix(suffix))

    def joinpath(self, *args: Union[str, os.PathLike[str]]) -> RemotePath:
        result = self._pure
        for arg in args:
            result = result / str(arg)
        return RemotePath(result)

    def resolve(self, strict: bool = False) -> RemotePath:
        return self

    def is_absolute(self) -> bool:
        return self._pure.is_absolute()

    def as_posix(self) -> str:
        return str(self._pure)


def _is_scheme_path(path_str: str) -> bool:
    """Whether *path_str* uses a URI scheme (``dbfs:/``, ``s3://``, ``abfss://``)."""
    first_segment = path_str.split("/", 1)[0]
    return ":" in first_segment


def is_unity_volume_path(path: Union[str, Path, "RemotePath"]) -> bool:
    return str(path).startswith("/Volumes/")


def atomic_write_text(path: Union[Path, "RemotePath"], content: str) -> None:
    """Replace *path*'s contents with *content* in a single atomic operation.

    Concurrent per-dataset Databricks ``for_each_task`` jobs write the same
    shared YAML files (snapshot grid, project context). On Unity Catalog
    Volumes FUSE mounts, standard ``open('w') → write → close`` can produce
    byte-level interleaves where the shorter writer truncates the tail of
    the longer one, corrupting the file. Routing Volumes writes through
    ``dbutils.fs.put`` bypasses FUSE and commits one atomic PUT at the UC
    service — last-writer-wins, never interleaved.
    """
    if isinstance(path, RemotePath):
        path.write_text(content)
        return
    if is_unity_volume_path(path):
        dbu = _maybe_get_dbutils()
        if dbu is not None:
            dbu.fs.put(str(path), content, overwrite=True)
            return
        path.write_text(content)
        return
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(content)
    os.replace(str(tmp), str(path))


def _maybe_get_dbutils() -> Any:
    try:
        return _get_dbutils()
    except RuntimeError:
        return None


def make_path(path: Union[str, Path, RemotePath], force_remote: Optional[bool] = None) -> Union[Path, RemotePath]:
    if isinstance(path, RemotePath):
        return path
    from customer_retention.core.compat.detection import is_databricks, is_remote_spark

    path_str = str(path)
    if force_remote is not None:
        remote = force_remote
    elif is_remote_spark() or bool(os.environ.get("CR_SPARK_REMOTE")):
        remote = True
    elif is_databricks():
        # On Databricks runtime, FUSE-mounted paths (absolute POSIX) work as
        # regular Path — only URI-scheme paths (dbfs:/, s3:/) need dbutils.fs.
        remote = _is_scheme_path(path_str)
    else:
        remote = False
    if remote:
        return RemotePath(path_str)
    return Path(path_str) if not isinstance(path, Path) else path
