"""Filesystem locations, resolved from the environment.

Scripts in this directory import their paths from here instead of hardcoding
them, so a checkout can be pointed at a different tree by setting environment
variables rather than editing source:

  MRNA_GPT_ROOT      repository root           (default: the parent of this file's directory)
  MRNA_GPT_RUNS      training / generation runs (default: <ROOT>/runs)
  MRNA_GPT_DATA      input datasets             (default: <ROOT>/data)
  MRNA_GPT_EXTERNAL  third-party tools          (default: <ROOT>/external)

A relative value is resolved against ROOT, so MRNA_GPT_RUNS=scratch/runs and
MRNA_GPT_RUNS=/mnt/scratch/runs both work.

Third-party tools are never vendored here; `external()` only says where they are
expected to be installed.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = ["ROOT", "RUNS", "DATA", "EXTERNAL", "external"]


def _resolve(var: str, default: str, root: Path) -> Path:
    value = os.environ.get(var)
    if not value:
        return root / default
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


ROOT = Path(os.environ.get("MRNA_GPT_ROOT")
            or Path(__file__).resolve().parents[1]).expanduser()
RUNS = _resolve("MRNA_GPT_RUNS", "runs", ROOT)
DATA = _resolve("MRNA_GPT_DATA", "data", ROOT)
EXTERNAL = _resolve("MRNA_GPT_EXTERNAL", "external", ROOT)


def external(name: str, var: str | None = None) -> Path:
    """Where a separately installed third-party tool is expected to live.

    `var` names a tool-specific override (e.g. LINEARDESIGN_DIR); when it is
    unset the tool is looked for at <EXTERNAL>/<name>.
    """
    if var:
        value = os.environ.get(var)
        if value:
            return Path(value).expanduser()
    return EXTERNAL / name


