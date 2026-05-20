"""R-parity tests for the Tier-4 / v0.3 functions.

Checks ``check_repeated_design`` boolean agreement and
``sdrf_to_annotation`` structural agreement against R MSstats 4.14.2.

Skipped if the CMAP R env / MSstats package is unavailable.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
import pytest

from pymsstats import check_repeated_design, sdrf_to_annotation

CONDA_BIN = "/home/users/steorra/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV = "/scratch/users/steorra/env/CMAP"


def _run_r(script: str) -> str:
    out = subprocess.run(
        ["bash", "-lc",
         f"source {CONDA_BIN} && conda activate {CONDA_ENV} "
         f"&& Rscript -e {script!r}"],
        capture_output=True, text=True, timeout=180, check=False,
    )
    return out.stdout


def _r_available() -> bool:
    try:
        out = _run_r('suppressMessages(library(MSstats)); cat("OK")')
        return "OK" in out
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _r_available(),
    reason="CMAP R env or MSstats not installed.",
)


def test_check_repeated_design_r_parity():
    """check_repeated_design boolean agrees with R checkRepeatedDesign."""
    r_out = _run_r(
        'suppressMessages(library(MSstats)); '
        'cc <- data.frame(GROUP=c("A","A","B","B"), '
        'SUBJECT=c("s0","s1","s2","s3"), Protein="p"); '
        'tc <- data.frame(GROUP=c("A","A","B","B"), '
        'SUBJECT=c("s0","s1","s0","s1"), Protein="p"); '
        'cat(checkRepeatedDesign(list(ProteinLevelData=cc)), '
        'checkRepeatedDesign(list(ProteinLevelData=tc)))'
    )
    tokens = r_out.strip().split()
    r_cc = tokens[-2].upper() == "TRUE"
    r_tc = tokens[-1].upper() == "TRUE"

    cc = pd.DataFrame({"GROUP": ["A", "A", "B", "B"],
                       "SUBJECT": ["s0", "s1", "s2", "s3"]})
    tc = pd.DataFrame({"GROUP": ["A", "A", "B", "B"],
                       "SUBJECT": ["s0", "s1", "s0", "s1"]})
    assert check_repeated_design(cc) == r_cc
    assert check_repeated_design(tc) == r_tc
    assert r_cc is False and r_tc is True


def test_sdrf_to_annotation_r_parity():
    """sdrf_to_annotation column structure agrees with R SDRFtoAnnotation."""
    r_out = _run_r(
        'suppressMessages(library(MSstats)); '
        'sdrf <- data.frame(check.names=FALSE, '
        '`comment[data file]`=c("a.raw","b.raw","c.raw","d.raw"), '
        '`characteristics[disease]`=c("control","disease","control","disease"), '
        '`characteristics[biological replicate]`=c(1,2,3,4)); '
        'ann <- SDRFtoAnnotation(sdrf); '
        'cat(paste(colnames(ann), collapse=","), "|", '
        'paste(ann$Run, collapse=","), "|", '
        'paste(ann$Condition, collapse=","))'
    )
    parts = r_out.strip().split("|")
    r_cols = [c.strip() for c in parts[0].split(",")]
    r_runs = [c.strip() for c in parts[1].split(",")]
    r_cond = [c.strip() for c in parts[2].split(",")]

    sdrf = pd.DataFrame({
        "comment[data file]": ["a.raw", "b.raw", "c.raw", "d.raw"],
        "characteristics[disease]":
            ["control", "disease", "control", "disease"],
        "characteristics[biological replicate]": [1, 2, 3, 4],
    })
    ann = sdrf_to_annotation(sdrf)
    assert list(ann.columns) == r_cols
    assert list(ann["Run"]) == r_runs
    assert list(ann["Condition"]) == r_cond
