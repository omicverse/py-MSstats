"""I/O helpers — re-exports the converter for symmetry with the R package.

The actual converter lives in :mod:`pymsstats.pipeline` so the package's
top-level :func:`pymsstats.maxquant_to_msstats` can be a thin re-export.
"""
from __future__ import annotations

from .pipeline import CANONICAL_COLS, maxquant_to_msstats

__all__ = ["maxquant_to_msstats", "CANONICAL_COLS"]
