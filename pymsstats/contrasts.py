"""Contrast-matrix builder — port of ``MSstats::MSstatsContrastMatrix``.

R signature::

    MSstatsContrastMatrix(contrasts, conditions, labels = NULL)

with dispatch on ``contrasts``:

* ``character`` — only ``"pairwise"`` is implemented in R; we extend it
  here to accept a single string with ``-``/``vs`` infix
  (``"groupB-groupA"`` or ``"groupB vs groupA"``).
* ``list`` — each element is a ``list(positive, negative)`` pair of
  condition names. Returns one contrast row per element, with coefficient
  ``+1/len(positive)`` and ``-1/len(negative)``.
* ``matrix`` / ``data.frame`` — passed through with column-name validation.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def _parse_single_contrast(s: str) -> tuple[list[str], list[str]]:
    """Parse a string like ``"groupB-groupA"`` or ``"groupB vs groupA"``.

    Multiple conditions on a side may be comma-separated, e.g.
    ``"groupB,groupC-groupA"`` averages B and C against A.
    """
    s = str(s).strip()
    if " vs " in s:
        pos, neg = s.split(" vs ", 1)
    elif "-" in s:
        # find the LAST hyphen so that names like 'group-1' still parse
        # but unambiguous A-B parses correctly. Fall back to first hyphen.
        # Use the first hyphen (canonical: pos-neg).
        pos, neg = s.split("-", 1)
    else:
        raise ValueError(
            f"cannot parse contrast string {s!r} — expected 'A-B' "
            "or 'A vs B' form"
        )
    pos_list = [t.strip() for t in pos.split(",") if t.strip()]
    neg_list = [t.strip() for t in neg.split(",") if t.strip()]
    return pos_list, neg_list


def _contrast_label(pos: Sequence[str], neg: Sequence[str]) -> str:
    return f"{','.join(pos)} vs {','.join(neg)}"


def msstats_contrast_matrix(
    contrasts,
    conditions: Sequence[str],
    labels: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Build a contrast matrix.

    Parameters
    ----------
    contrasts
        One of:

        * a single string ``"groupB-groupA"`` (or ``"groupB vs groupA"``);
        * the literal string ``"pairwise"`` (all pairwise A − B over
          ``conditions``);
        * a list of strings, each in the same single-string format;
        * a list of ``(positive_iter, negative_iter)`` tuples — multiple
          conditions per side are averaged ``+1/n_pos, -1/n_neg``;
        * a 2D array / DataFrame of shape ``(n_contrasts, n_conditions)``
          (passed through with column-name validation).
    conditions
        Ordered iterable of all condition names — defines the column order.
    labels
        Optional row labels. If ``None``, sensible default labels are
        generated (``"A vs B"`` for pair contrasts).

    Returns
    -------
    pd.DataFrame
        Shape ``(n_contrasts, n_conditions)`` with row labels = contrast
        names and column names = ``conditions``.
    """
    conditions = list(conditions)
    n = len(conditions)
    cond_index = {c: i for i, c in enumerate(conditions)}

    # Case 1: already a matrix-like
    if isinstance(contrasts, pd.DataFrame):
        if list(contrasts.columns) != conditions:
            if set(map(str, contrasts.columns)) != set(map(str, conditions)):
                raise ValueError(
                    f"contrast matrix columns {list(contrasts.columns)} "
                    f"do not match conditions {conditions}"
                )
            contrasts = contrasts.reindex(columns=conditions)
        if contrasts.index.name is None and (
                labels is None and contrasts.index.tolist()
                == list(range(len(contrasts)))):
            raise ValueError(
                "contrast DataFrame must have row labels (contrast names) — "
                "pass an indexed DataFrame or supply labels"
            )
        return contrasts.copy()
    if isinstance(contrasts, np.ndarray):
        arr = np.asarray(contrasts, dtype=float)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[1] != n:
            raise ValueError(
                f"contrast matrix has {arr.shape[1]} cols, expected {n} "
                f"(one per condition)"
            )
        if labels is None:
            row_labels = [f"contrast_{i}" for i in range(arr.shape[0])]
        else:
            row_labels = list(labels)
        return pd.DataFrame(arr, index=row_labels, columns=conditions)

    # Case 2: string
    if isinstance(contrasts, str):
        if contrasts == "pairwise":
            contrast_list = []
            for i in range(n):
                for j in range(i + 1, n):
                    contrast_list.append(([conditions[i]], [conditions[j]]))
        else:
            pos, neg = _parse_single_contrast(contrasts)
            contrast_list = [(pos, neg)]
    elif isinstance(contrasts, (list, tuple)):
        contrast_list = []
        for elem in contrasts:
            if isinstance(elem, str):
                pos, neg = _parse_single_contrast(elem)
                contrast_list.append((pos, neg))
            elif isinstance(elem, (list, tuple)) and len(elem) == 2:
                pos, neg = elem
                if isinstance(pos, str):
                    pos = [pos]
                if isinstance(neg, str):
                    neg = [neg]
                contrast_list.append((list(pos), list(neg)))
            else:
                raise ValueError(
                    f"don't know how to interpret contrast element {elem!r}"
                )
    else:
        raise TypeError(
            f"unsupported contrasts argument: {type(contrasts).__name__}"
        )

    # Build matrix
    M = np.zeros((len(contrast_list), n), dtype=float)
    for k, (pos, neg) in enumerate(contrast_list):
        for c in pos:
            if c not in cond_index:
                raise ValueError(
                    f"condition {c!r} not in conditions {conditions}")
            M[k, cond_index[c]] = 1.0 / len(pos)
        for c in neg:
            if c not in cond_index:
                raise ValueError(
                    f"condition {c!r} not in conditions {conditions}")
            M[k, cond_index[c]] = -1.0 / len(neg)
    if labels is None:
        row_labels = [_contrast_label(p, n_) for p, n_ in contrast_list]
    else:
        row_labels = list(labels)
    return pd.DataFrame(M, index=row_labels, columns=conditions)


__all__ = ["msstats_contrast_matrix"]
