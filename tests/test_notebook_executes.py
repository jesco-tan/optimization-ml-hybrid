"""
End-to-end check: the canonical notebook runs without error.

Set SKIP_NOTEBOOK_E2E=1 to skip (faster local iteration).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "optimization_ml_hybrid_walkthrough.ipynb"
OUT_EXEC = ROOT / "outputs" / "_test_executed.ipynb"


def test_walkthrough_notebook_present() -> None:
    assert NOTEBOOK.is_file(), "Canonical notebook must exist at repo root"


@pytest.mark.skipif(
    os.environ.get("SKIP_NOTEBOOK_E2E", "").lower() in ("1", "true", "yes"),
    reason="SKIP_NOTEBOOK_E2E is set",
)
def test_walkthrough_notebook_executes() -> None:
    assert NOTEBOOK.is_file(), f"Missing {NOTEBOOK}"
    if OUT_EXEC.is_file():
        OUT_EXEC.unlink()

    subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            str(NOTEBOOK.name),
            "--output",
            OUT_EXEC.name,
            "--output-dir",
            str(ROOT / "outputs"),
        ],
        check=True,
        cwd=str(ROOT),
        timeout=900,
    )
    assert OUT_EXEC.is_file(), "nbconvert did not write executed notebook"
    OUT_EXEC.unlink(missing_ok=True)
