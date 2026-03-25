#!/usr/bin/env python3
"""Generate requirements.txt and requirements-databricks.txt from pyproject.toml.

Run manually:   python scripts/generate_requirements.py
Pre-commit hook: auto-runs on push (see .pre-commit-config.yaml)
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from customer_retention.integrations.requirements_generator import write_requirements  # noqa: E402

if __name__ == "__main__":
    req_path, dbr_path = write_requirements(_PROJECT_ROOT)
    print(f"Generated {req_path}")
    print(f"Generated {dbr_path}")
