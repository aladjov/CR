"""Permanent per-cell Spark job counter for exploration notebooks.

A sentinel-delimited block is added once to each notebook's init cell and
stays permanently (like ``@cr:`` cell IDs).  It auto-activates only during
batch execution (``CR_BATCH_EXECUTION=1``), registers IPython hooks to
count Spark jobs per ``@cr:``-tagged cell, and writes a JSONL sidecar
consumed by ``cell_profiling.merge_sidecar_metrics``.

The same code runs identically on local and Databricks — the metric
(Spark job delta) is directly comparable across environments.
"""

from __future__ import annotations

import json
from pathlib import Path

_SENTINEL_START = "# --- cr:profiler ---"
_SENTINEL_END = "# --- /cr:profiler ---"

# Static block — auto-derives notebook name and sidecar path at runtime.
# Guarded by CR_BATCH_EXECUTION so interactive sessions have zero overhead.
# Written as ruff-compliant Python (no lambdas, no one-line compounds).
PROFILER_BLOCK = f"""\
{_SENTINEL_START}
if __import__('os').environ.get("CR_BATCH_EXECUTION") == "1":
    import json as _j, re as _r, os as _os
    _cr_nb = _os.path.splitext(_os.path.basename(_os.environ.get("PAPERMILL_OUTPUT_PATH", "")))[0]
    if _cr_nb:
        _cr_mp = _os.path.join(_os.getcwd(), f".cr_cell_metrics_{{_cr_nb}}.jsonl")
        open(_cr_mp, 'w').close()
        _cr_re = _r.compile(r"^#\\s*@cr:\\w+\\s+name='([^']+)'\\s+id=(\\w+)")
        def _cr_jc():
            return -1
        try:
            _s = __import__('pyspark.sql', fromlist=['SparkSession']).SparkSession.getActiveSession()
            if _s:
                def _cr_jc():  # noqa: F811
                    return _s._jsc.sc().dagScheduler().nextJobId().get()
        except Exception:
            pass
        def _cr_pre(info):
            info._cr_sj = _cr_jc()
        def _cr_post(r):
            sj = getattr(r.info, '_cr_sj', -1)
            sa = _cr_jc()
            m = _cr_re.match((r.info.raw_cell or '').split('\\n')[0])
            if m:
                with open(_cr_mp, 'a') as f:
                    f.write(_j.dumps({{"cell_name": m.group(1), "cell_id": m.group(2),
                                      "spark_jobs": (sa - sj) if sj >= 0 and sa >= 0 else None}}) + '\\n')
        get_ipython().events.register('pre_run_cell', _cr_pre)
        get_ipython().events.register('post_run_cell', _cr_post)
{_SENTINEL_END}"""


def has_profiler(nb_path: Path) -> bool:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        text = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        return _SENTINEL_START in text
    return False


def add_profiler(nb_path: Path) -> bool:
    """Add the profiler block to the first code cell. Returns True if modified."""
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        text = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        if _SENTINEL_START in text:
            return False
        cell["source"] = text.rstrip("\n") + "\n" + PROFILER_BLOCK + "\n"
        nb_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
        return True
    return False


def _strip_profiler_block(text: str) -> str:
    lines = text.split("\n")
    result, inside = [], False
    for line in lines:
        if line.strip() == _SENTINEL_START:
            inside = True
            continue
        if line.strip() == _SENTINEL_END:
            inside = False
            continue
        if not inside:
            result.append(line)
    return "\n".join(result).rstrip("\n") + "\n"


def remove_profiler(nb_path: Path) -> bool:
    """Remove the profiler block from the first code cell. Returns True if modified."""
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        text = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        if _SENTINEL_START not in text:
            return False
        cell["source"] = _strip_profiler_block(text)
        nb_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
        return True
    return False
