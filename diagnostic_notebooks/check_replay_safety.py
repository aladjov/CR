"""Replay-safety scanner for cycle-close gate.

Given a git base-ref, lists every file that changed since that ref and
walks the static reach-chain: for each changed source file, finds every
module or notebook that imports / references it. The cycle doc author
compares the reach-chain against the stages it plans to skip; if any
skipped stage appears, the cycle must re-run those stages.

Usage:
    python diagnostic_notebooks/check_replay_safety.py --since HEAD~1
    python diagnostic_notebooks/check_replay_safety.py --since main --json out.json

The scanner is deliberately conservative: it flags any textual reference
(import path, module name, file-stem reference in notebook sources). It
does not resolve dynamic imports or string-loaded templates; if those
are used, treat the cycle as non-replay-safe.

Exit codes:
    0 — no changes, nothing to check
    1 — changes found; report printed (not a failure, just evidence)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

SOURCE_ROOTS = ("src", "exploration_notebooks", "diagnostic_notebooks")


@dataclass
class ReachChain:
    changed: Path
    referenced_by: list[Path] = field(default_factory=list)


def git_diff_paths(ref: str) -> list[Path]:
    result = subprocess.run(
        ["git", "diff", "--name-only", ref],
        check=True,
        capture_output=True,
        text=True,
    )
    return [Path(line) for line in result.stdout.splitlines() if line.strip()]


def iter_scan_files(repo_root: Path) -> list[Path]:
    files: list[Path] = []
    for root in SOURCE_ROOTS:
        base = repo_root / root
        if not base.exists():
            continue
        files.extend(p for p in base.rglob("*.py") if "__pycache__" not in p.parts)
        files.extend(base.rglob("*.ipynb"))
    return files


def module_path_for(changed: Path) -> str | None:
    if changed.suffix != ".py":
        return None
    parts = list(changed.parts)
    if "src" in parts:
        idx = parts.index("src")
        dotted = ".".join(parts[idx + 1 :]).removesuffix(".py")
        return dotted
    return None


def search_needles(changed: Path) -> list[str]:
    needles: list[str] = []
    stem = changed.stem
    if changed.suffix == ".py":
        mod = module_path_for(changed)
        if mod:
            needles.append(mod)
            tail = mod.split(".")[-1]
            if tail:
                needles.append(f"from .{tail}")
                needles.append(f"import {tail}")
        needles.append(stem)
    elif changed.suffix == ".ipynb":
        needles.append(stem)
        needles.append(str(changed))
    else:
        needles.append(stem)
    return [n for n in needles if n]


def file_contains_any(path: Path, needles: list[str]) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(errors="replace")
    return any(n in text for n in needles)


def build_reach_chains(changed_files: list[Path], scan_files: list[Path], repo_root: Path) -> list[ReachChain]:
    chains: list[ReachChain] = []
    changed_set = {str(p) for p in changed_files}
    for changed in changed_files:
        needles = search_needles(changed)
        if not needles:
            chains.append(ReachChain(changed=changed))
            continue
        referenced_by: list[Path] = []
        for scan_file in scan_files:
            if str(scan_file.relative_to(repo_root)) in changed_set:
                continue
            if file_contains_any(scan_file, needles):
                referenced_by.append(scan_file.relative_to(repo_root))
        chains.append(ReachChain(changed=changed, referenced_by=sorted(referenced_by)))
    return chains


def print_report(chains: list[ReachChain]) -> None:
    print(f"check_replay_safety: {len(chains)} changed file(s)")
    for chain in chains:
        print(f"\n  {chain.changed}")
        if not chain.referenced_by:
            print("    (no downstream references found)")
            continue
        for ref in chain.referenced_by:
            print(f"    -> {ref}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--since", required=True, help="git ref to diff against")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--json", type=Path, help="write machine-readable output here")
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    changed = git_diff_paths(args.since)
    if not changed:
        print("check_replay_safety: no changes since ref")
        return 0

    scan_files = iter_scan_files(repo_root)
    chains = build_reach_chains(changed, scan_files, repo_root)

    print_report(chains)

    if args.json:
        payload = [
            {"changed": str(c.changed), "referenced_by": [str(r) for r in c.referenced_by]}
            for c in chains
        ]
        args.json.write_text(json.dumps(payload, indent=2))

    return 1


if __name__ == "__main__":
    sys.exit(main())
