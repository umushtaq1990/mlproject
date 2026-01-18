"""Pytest configuration helpers for the poc tests.

Ensure the project root is on sys.path so tests can import the package
using its package name (e.g. `poc.src...`). This mirrors how CI/dev
environments work when the project is installed or run from the repo root.
"""

from __future__ import annotations

import sys
from pathlib import Path

# project root is two levels up from poc/tests -> ML_Project
ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "poc" / "src"

# Insert project root first (so `import poc` works), then poc/src as a fallback
for p in (ROOT, SRC_DIR):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
