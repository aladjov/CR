"""Privacy scanner for cycle-close gate.

Scans a set of files (or a git diff) against an engagement-local denylist
and a fixed set of CRM-record-ID regexes. Exits 0 on clean, 1 on any hit.

Usage:
    python diagnostic_notebooks/check_privacy.py --diff HEAD
    python diagnostic_notebooks/check_privacy.py --paths path/a path/b
    python diagnostic_notebooks/check_privacy.py --denylist debug/eng_a/.check_privacy.yaml --diff HEAD

Denylist file (gitignored, stored under debug/<engagement>/) format:

    terms:
      - ExampleCorp
      - internal_catalog_name
    id_regexes:
      - '[0-9a-zA-Z]{15,18}'
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_ID_REGEXES = [r"\b[0-9a-zA-Z]{18}\b"]

STAGED_DEBUG_PATH_PATTERN = re.compile(r"^debug/")


@dataclass(frozen=True)
class Hit:
    path: Path
    line_no: int
    pattern: str
    snippet: str


def load_denylist(path: Path | None) -> tuple[list[str], list[re.Pattern]]:
    terms: list[str] = []
    id_regexes = list(DEFAULT_ID_REGEXES)
    if path is None:
        return terms, [re.compile(p) for p in id_regexes]
    data = yaml.safe_load(path.read_text()) or {}
    terms = [t for t in (data.get("terms") or []) if t]
    extra = data.get("id_regexes") or []
    id_regexes.extend(extra)
    return terms, [re.compile(p) for p in id_regexes]


def git_diff_paths(ref: str) -> list[Path]:
    result = subprocess.run(
        ["git", "diff", "--name-only", ref],
        check=True,
        capture_output=True,
        text=True,
    )
    return [Path(line) for line in result.stdout.splitlines() if line.strip()]


def scan_file(path: Path, terms: list[str], id_regexes: list[re.Pattern]) -> list[Hit]:
    if not path.is_file():
        return []
    hits: list[Hit] = []
    text = path.read_text(errors="replace")
    lowered_terms = [(t, t.lower()) for t in terms]
    for line_no, line in enumerate(text.splitlines(), start=1):
        lowered = line.lower()
        for original, needle in lowered_terms:
            if needle and needle in lowered:
                hits.append(Hit(path, line_no, f"term:{original}", line.strip()[:200]))
        for regex in id_regexes:
            for m in regex.finditer(line):
                hits.append(Hit(path, line_no, f"id:{regex.pattern}", m.group(0)))
    return hits


def scan_staged_debug_paths(diff_paths: list[Path]) -> list[Hit]:
    return [
        Hit(p, 0, "staged:debug_path", str(p))
        for p in diff_paths
        if STAGED_DEBUG_PATH_PATTERN.match(str(p))
    ]


def report(hits: list[Hit]) -> None:
    for hit in hits:
        loc = f"{hit.path}:{hit.line_no}" if hit.line_no else str(hit.path)
        print(f"  [{hit.pattern}] {loc}  {hit.snippet}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--diff", help="git ref; scans changed files")
    parser.add_argument("--paths", nargs="+", type=Path, help="explicit file paths to scan")
    parser.add_argument("--denylist", type=Path, help="path to .check_privacy.yaml (optional)")
    args = parser.parse_args(argv)

    if not args.diff and not args.paths:
        parser.error("provide --diff <ref> or --paths <paths>")

    terms, id_regexes = load_denylist(args.denylist)
    target_paths = list(args.paths) if args.paths else git_diff_paths(args.diff)
    staged_debug = scan_staged_debug_paths(target_paths) if args.diff else []

    all_hits: list[Hit] = list(staged_debug)
    for path in target_paths:
        if str(path).startswith("debug/"):
            continue
        all_hits.extend(scan_file(path, terms, id_regexes))

    if all_hits:
        print(f"check_privacy: {len(all_hits)} hit(s):")
        report(all_hits)
        return 1
    print(f"check_privacy: clean ({len(target_paths)} file(s) scanned)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
