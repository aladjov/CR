"""Instantiate a fix-cycle notebook + doc for an engagement.

Copies the universal templates into the engagement tree and prepends the
engagement's `@cr:code_system` cell so `customer_retention.*` imports
resolve on Databricks Workspace-Repos clusters (no pip install needed).

Usage:
    python diagnostic_notebooks/new_cycle.py \\
        --engagement engagement_e4ad6e1b \\
        --cycle 2 \\
        --slug event_type_classification

Engagement setup (one-time, gitignored):

    cat > debug/<engagement>/.engagement.yaml <<EOF
    framework_repo_root: /Workspace/Repos/<user>/customer_retention
    EOF

The file is read at instantiation time; the workspace path never enters
a versioned file. Running this against a clean engagement tree creates
`debug/<engagement>/{cycles,fix_cycles}/` if missing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_engagement_config(engagement: str) -> dict:
    path = REPO_ROOT / "debug" / engagement / ".engagement.yaml"
    if not path.exists():
        raise SystemExit(
            f"missing {path} — create it with:\n"
            f"  framework_repo_root: /Workspace/Repos/<user>/customer_retention"
        )
    return yaml.safe_load(path.read_text()) or {}


def build_code_system_cell(framework_repo_root: str) -> dict:
    return {
        "cell_type": "code",
        "id": "cr-syspath",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# @cr:code_system name='framework_path' id=cr-syspath\n",
            "import sys\n",
            "\n",
            f'FRAMEWORK_REPO_ROOT = "{framework_repo_root}"\n',
            '_src = f"{FRAMEWORK_REPO_ROOT}/src"\n',
            "if _src not in sys.path:\n",
            "    sys.path.insert(0, _src)\n",
        ],
    }


def instantiate_cycle_notebook(
    nb_src: Path, nb_dst: Path, framework_repo_root: str,
) -> None:
    nb = json.loads(nb_src.read_text())
    existing_ids = {c["id"] for c in nb["cells"]}
    if "cr-syspath" not in existing_ids:
        nb["cells"] = [build_code_system_cell(framework_repo_root)] + nb["cells"]
    nb_dst.parent.mkdir(parents=True, exist_ok=True)
    nb_dst.write_text(json.dumps(nb, indent=1))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--engagement", required=True)
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing cycle notebook/doc")
    args = parser.parse_args(argv)

    config = load_engagement_config(args.engagement)
    framework_root = config.get("framework_repo_root")
    if not framework_root:
        raise SystemExit(
            f"debug/{args.engagement}/.engagement.yaml must define framework_repo_root"
        )

    nb_src = REPO_ROOT / "diagnostic_notebooks" / "cycle_template.ipynb"
    md_src = REPO_ROOT / "diagnostic_notebooks" / "cycle_template.md"
    eng_dir = REPO_ROOT / "debug" / args.engagement
    nb_dst = eng_dir / "cycles" / f"{args.cycle:03d}_{args.slug}.ipynb"
    md_dst = eng_dir / "fix_cycles" / f"{args.cycle:03d}_{args.slug}.md"

    for dst in (nb_dst, md_dst):
        if dst.exists() and not args.force:
            raise SystemExit(f"{dst} exists — pass --force to overwrite")

    instantiate_cycle_notebook(nb_src, nb_dst, framework_root)
    md_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(md_src, md_dst)

    rel_nb = nb_dst.relative_to(REPO_ROOT)
    rel_md = md_dst.relative_to(REPO_ROOT)
    print(f"wrote {rel_nb}")
    print(f"wrote {rel_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
