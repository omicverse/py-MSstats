"""Tests for the SDRF helpers and bundled-dataset stand-ins."""
from __future__ import annotations

import pandas as pd
import pytest

from pymsstats import (
    check_repeated_design,
    example_sdrf,
    extract_sdrf,
    load_dda_example,
    load_dia_example,
    make_peptides_dictionary,
    sdrf_to_annotation,
)
from tests._synthetic import mock_sdrf, synthetic_msstats


# -----------------------------------------------------------------------------
# example_sdrf
# -----------------------------------------------------------------------------
def test_example_sdrf_structure():
    sdrf = example_sdrf()
    assert isinstance(sdrf, pd.DataFrame)
    assert len(sdrf) > 0
    assert "comment[data file]" in sdrf.columns
    assert "characteristics[disease]" in sdrf.columns
    assert "characteristics[biological replicate]" in sdrf.columns


# -----------------------------------------------------------------------------
# sdrf_to_annotation
# -----------------------------------------------------------------------------
def test_sdrf_to_annotation_smoke():
    ann = sdrf_to_annotation(mock_sdrf())
    assert list(ann.columns) == ["Run", "Condition", "BioReplicate"]
    assert len(ann) > 0


def test_sdrf_to_annotation_on_example_sdrf():
    ann = sdrf_to_annotation(example_sdrf())
    assert set(ann.columns) == {"Run", "Condition", "BioReplicate"}
    # 8 samples x 4 runs = 32 rows
    assert len(ann) == len(example_sdrf())


def test_sdrf_to_annotation_missing_column_raises():
    bad = mock_sdrf().drop(columns=["characteristics[disease]"])
    with pytest.raises(ValueError, match="were not found"):
        sdrf_to_annotation(bad)


def test_sdrf_to_annotation_with_fraction():
    ann = sdrf_to_annotation(
        mock_sdrf(), fraction="comment[fraction identifier]")
    assert "Fraction" in ann.columns


# -----------------------------------------------------------------------------
# extract_sdrf (annotation -> SDRF)
# -----------------------------------------------------------------------------
def test_extract_sdrf_smoke():
    annot = pd.DataFrame({
        "Condition": ["A", "A", "B", "B"],
        "BioReplicate": ["b0", "b1", "b2", "b3"],
        "Run": ["r0", "r1", "r2", "r3"],
    })
    sdrf = extract_sdrf(annot)
    assert "comment[data file]" in sdrf.columns
    assert "characteristics[disease]" in sdrf.columns
    assert len(sdrf) == 4


def test_extract_sdrf_roundtrip():
    """extract_sdrf then sdrf_to_annotation recovers the annotation."""
    annot = pd.DataFrame({
        "Condition": ["A", "A", "B", "B"],
        "BioReplicate": ["b0", "b1", "b2", "b3"],
        "Run": ["r0", "r1", "r2", "r3"],
    })
    sdrf = extract_sdrf(annot)
    back = sdrf_to_annotation(sdrf)
    # column sets agree
    assert set(back.columns) == {"Run", "Condition", "BioReplicate"}
    merged = annot.merge(back, on=["Run", "Condition", "BioReplicate"])
    assert len(merged) == len(annot)


# -----------------------------------------------------------------------------
# check_repeated_design
# -----------------------------------------------------------------------------
def test_check_repeated_design_case_control_false():
    # each subject in exactly one group -> not repeated
    df = pd.DataFrame({
        "GROUP": ["A", "A", "B", "B"],
        "SUBJECT": ["s0", "s1", "s2", "s3"],
    })
    assert check_repeated_design(df) is False


def test_check_repeated_design_time_course_true():
    # subjects cross groups -> repeated
    df = pd.DataFrame({
        "GROUP": ["A", "A", "B", "B"],
        "SUBJECT": ["s0", "s1", "s0", "s1"],
    })
    assert check_repeated_design(df) is True


def test_check_repeated_design_accepts_dict():
    df = pd.DataFrame({
        "GROUP": ["A", "B"], "SUBJECT": ["s0", "s0"],
    })
    assert check_repeated_design({"ProteinLevelData": df}) is True


# -----------------------------------------------------------------------------
# make_peptides_dictionary
# -----------------------------------------------------------------------------
def test_make_peptides_dictionary():
    long_df, _ = synthetic_msstats(n_proteins=5, seed=0)
    d = make_peptides_dictionary(long_df)
    assert list(d.columns) == ["PeptideSequence", "ProteinName"]
    # one row per unique (peptide, protein)
    assert len(d) == long_df[["PeptideSequence", "ProteinName"]] \
        .drop_duplicates().shape[0]


# -----------------------------------------------------------------------------
# bundled-dataset stand-ins
# -----------------------------------------------------------------------------
def test_load_dda_example():
    df = load_dda_example(n_proteins=10)
    assert df["ProteinName"].nunique() == 10
    assert (df["FragmentIon"] == "NA").all()


def test_load_dia_example():
    df = load_dia_example(n_proteins=8)
    assert df["ProteinName"].nunique() == 8
    assert df["FragmentIon"].nunique() > 1
