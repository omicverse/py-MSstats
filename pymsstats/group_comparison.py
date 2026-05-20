"""
Per-protein group comparison — port of ``MSstats::groupComparison``.

The R reference fits, per protein:

* ``ABUNDANCE ~ GROUP`` if there are no technical replicates or only one
  subject (a fixed-effects linear model);
* ``ABUNDANCE ~ GROUP + (1|SUBJECT)`` (``lme4::lmer``) when there are
  repeated subjects, with the df estimated from the equivalent fixed-effect
  ``lm(ABUNDANCE ~ GROUP + SUBJECT)``.

For each contrast row :math:`L`:

* ``logFC = L β``
* ``SE = sqrt(L (X'X)^{-1} L' σ²)``
* ``t = logFC / SE``
* ``p = 2 · t_df(|t|)``  (Wald)
* adjusted p-values via Benjamini-Hochberg, **per contrast row**.

For v0.1 we expose the fixed-effects path (``lm``) AND a ``random_subject=True``
flag that switches to ``MixedLM(ABUNDANCE ~ GROUP, groups=SUBJECT)``.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats


def _bh(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvalues, dtype=float)
    mask = np.isfinite(p)
    out = np.full_like(p, np.nan)
    if not mask.any():
        return out
    good = p[mask]
    out[mask] = stats.false_discovery_control(np.clip(good, 0.0, 1.0), method="bh")
    return out


def _fit_one_protein_lm(
    df: pd.DataFrame,
    groups: Sequence[str],
    contrast_row: np.ndarray,
) -> dict:
    """Single-protein OLS fit + one-contrast Wald test.

    Mirrors the ``lm(ABUNDANCE ~ GROUP)`` branch of
    ``MSstats:::.fitModelForGroupComparison``.

    Parameters
    ----------
    df
        Long-format rows for one protein (columns ``GROUP``, ``ABUNDANCE``,
        optional ``SUBJECT``).
    groups
        Ordered list of all groups in the design (the contrast row is in
        this order). Used to map design columns.
    contrast_row
        1D array of length ``n_groups``, e.g. ``[-1, 1]`` for B - A.

    Returns
    -------
    dict
        ``log2FC, SE, df, t, pvalue, issue``.
    """
    df = df.dropna(subset=["ABUNDANCE", "GROUP"]).copy()
    df["GROUP"] = df["GROUP"].astype(str)

    present = sorted(df["GROUP"].unique(), key=lambda g: list(groups).index(g))
    if len(present) < 2:
        return {
            "log2FC": np.nan, "SE": np.nan, "df": np.nan, "t": np.nan,
            "pvalue": np.nan, "issue": "oneConditionMissing",
        }
    # One-hot design with no intercept: one column per group (same as
    # R's ``lm(ABUNDANCE ~ GROUP - 1)`` with the contrast referencing groups).
    # MSstats's R code uses ``lm(ABUNDANCE ~ GROUP)`` which fits an intercept
    # + (k-1) dummy columns; the contrast row in groupComparison is given
    # over the *group means*, so we match by using a cell-means design here.
    X_cols = []
    for g in groups:
        X_cols.append((df["GROUP"] == g).astype(float).to_numpy())
    X = np.column_stack(X_cols)
    y = df["ABUNDANCE"].to_numpy(dtype=float)

    # Drop empty columns (= groups not present), and the matching contrast entry.
    present_mask = X.sum(axis=0) > 0
    X_p = X[:, present_mask]
    c = np.asarray(contrast_row, dtype=float)[present_mask]
    # If contrast asks for a missing group, we can't test.
    full_contrast = np.asarray(contrast_row, dtype=float)
    if np.any((np.abs(full_contrast) > 0) & ~present_mask):
        return {
            "log2FC": np.nan, "SE": np.nan, "df": np.nan, "t": np.nan,
            "pvalue": np.nan, "issue": "completeMissing",
        }
    n, p = X_p.shape
    if n - p <= 0:
        return {
            "log2FC": np.nan, "SE": np.nan, "df": np.nan, "t": np.nan,
            "pvalue": np.nan, "issue": "insufficientDf",
        }
    # OLS via lstsq.
    XtX = X_p.T @ X_p
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ X_p.T @ y
    resid = y - X_p @ beta
    df_r = n - p
    sigma2 = float((resid @ resid) / df_r) if df_r > 0 else np.nan

    log2fc = float(c @ beta)
    var = float(c @ XtX_inv @ c) * sigma2
    if not np.isfinite(var) or var < 0:
        return {
            "log2FC": log2fc, "SE": np.nan, "df": df_r, "t": np.nan,
            "pvalue": np.nan, "issue": "nonEstimable",
        }
    se = float(np.sqrt(var))
    if se == 0:
        return {
            "log2FC": log2fc, "SE": 0.0, "df": df_r, "t": np.nan,
            "pvalue": np.nan, "issue": "zeroSE",
        }
    tval = log2fc / se
    pval = 2.0 * float(stats.t.sf(abs(tval), df_r))
    return {
        "log2FC": log2fc, "SE": se, "df": float(df_r),
        "t": tval, "pvalue": pval, "issue": None,
    }


def _fit_one_protein_mixed(
    df: pd.DataFrame,
    groups: Sequence[str],
    contrast_row: np.ndarray,
) -> dict:
    """Mixed path: ``ABUNDANCE ~ GROUP + (1|SUBJECT)`` via statsmodels.

    Mirrors MSstats's handling: try ``MixedLM`` for the variance / SE,
    fall back to a fixed-effects ``ABUNDANCE ~ GROUP + SUBJECT`` OLS if
    the LMM solver hits a singular Hessian (R MSstats falls back to the
    same OLS for the df estimate; we use it for the SE too in that case).
    """
    df = df.dropna(subset=["ABUNDANCE", "GROUP", "SUBJECT"]).copy()
    df["GROUP"] = df["GROUP"].astype(str)
    df["SUBJECT"] = df["SUBJECT"].astype(str)
    if df["SUBJECT"].nunique() <= 1 or df["GROUP"].nunique() < 2:
        return _fit_one_protein_lm(df, groups, contrast_row)

    present = sorted(df["GROUP"].unique(), key=lambda g: list(groups).index(g))
    present_mask = np.array([g in present for g in groups])
    full = np.asarray(contrast_row, dtype=float)
    if np.any((np.abs(full) > 0) & ~present_mask):
        return {
            "log2FC": np.nan, "SE": np.nan, "df": np.nan, "t": np.nan,
            "pvalue": np.nan, "issue": "completeMissing",
        }

    # Cell-means design, one column per *present* group.
    X_grp = np.column_stack([(df["GROUP"] == g).astype(float).to_numpy()
                             for g in present])
    y = df["ABUNDANCE"].to_numpy(dtype=float)
    sub = df["SUBJECT"].astype(str).to_numpy()
    c = full[present_mask]

    # MSstats's df_full = lm(ABUNDANCE ~ GROUP + SUBJECT).df.residual
    # = n - rank(GROUP design + SUBJECT dummies), absorbing intercept.
    n = len(y)
    n_subj = len(np.unique(sub))
    df_full = n - len(present) - n_subj + 1
    df_full = max(int(df_full), 1)

    # Try MixedLM first (matches R lme4::lmer).
    se = None
    log2fc = None
    issue = None
    try:
        import warnings
        from statsmodels.regression.mixed_linear_model import MixedLM
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = MixedLM(endog=y, exog=X_grp, groups=sub)
            fit = model.fit(method="lbfgs", reml=True, disp=False)
        beta = np.asarray(fit.fe_params, dtype=float)
        cov = np.asarray(fit.cov_params(), dtype=float)
        cov_fe = cov[:len(beta), :len(beta)]
        log2fc = float(c @ beta)
        var = float(c @ cov_fe @ c)
        if np.isfinite(var) and var >= 0:
            se = float(np.sqrt(var))
    except Exception:
        pass

    if se is None or not np.isfinite(se):
        # Fallback: lm(ABUNDANCE ~ GROUP + SUBJECT) (cell means).
        # Build a one-hot design with both group and subject blocks,
        # then drop one subject column (reference) to avoid singularity.
        subj_levels = sorted(np.unique(sub))[1:]  # drop first as reference
        X_sub = np.column_stack(
            [(sub == s).astype(float) for s in subj_levels]
        ) if subj_levels else np.empty((n, 0))
        X = np.hstack([X_grp, X_sub])
        try:
            XtX = X.T @ X
            XtX_inv = np.linalg.pinv(XtX)
            beta_full = XtX_inv @ X.T @ y
            resid = y - X @ beta_full
            df_r = n - np.linalg.matrix_rank(X)
            df_full = max(int(df_r), 1)
            sigma2 = float((resid @ resid) / df_full)
            c_full = np.concatenate([c, np.zeros(X_sub.shape[1])])
            log2fc = float(c_full @ beta_full)
            var = float(c_full @ XtX_inv @ c_full) * sigma2
            se = float(np.sqrt(var)) if var >= 0 else None
        except Exception:
            return _fit_one_protein_lm(df, groups, contrast_row)

    if se is None or not np.isfinite(se):
        return {
            "log2FC": log2fc if log2fc is not None else np.nan,
            "SE": np.nan, "df": float(df_full), "t": np.nan,
            "pvalue": np.nan, "issue": "nonEstimable",
        }
    if se == 0:
        return {
            "log2FC": log2fc, "SE": 0.0, "df": float(df_full), "t": np.nan,
            "pvalue": np.nan, "issue": "zeroSE",
        }
    tval = log2fc / se
    pval = 2.0 * float(stats.t.sf(abs(tval), df_full))
    return {
        "log2FC": log2fc, "SE": se, "df": float(df_full), "t": tval,
        "pvalue": pval, "issue": issue,
    }


def _detect_subject_design(
    df: pd.DataFrame,
    *,
    group_col: str = "GROUP",
    subject_col: str = "SUBJECT",
) -> str:
    """Reproduce MSstats's design dispatch.

    Returns ``'lm'`` if the simple ``lm(ABUNDANCE ~ GROUP)`` branch should
    be used (which is the case when each GROUP has exactly one subject,
    OR when SUBJECT is fully nested within GROUP and uniquely names a
    sample — equivalent to ``has_tech_replicates=False`` AND no
    subject crossing).

    Returns ``'mixed'`` otherwise (paired / repeated-measures design
    where the same subject contributes to several groups).
    """
    sub = df[[group_col, subject_col]].drop_duplicates()
    # subject crosses groups iff some SUBJECT appears in >1 GROUP
    n_groups_per_subject = sub.groupby(subject_col)[group_col].nunique()
    if (n_groups_per_subject > 1).any():
        return "mixed"
    return "lm"


def group_comparison(
    processed: pd.DataFrame,
    contrast_matrix: np.ndarray | pd.DataFrame,
    *,
    groups: Sequence[str] | None = None,
    random_subject: bool | str = "auto",
    protein_col: str = "Protein",
    abundance_col: str = "LogIntensities",
    group_col: str = "GROUP",
    run_col: str = "RUN",
    subject_col: str = "SUBJECT",
) -> pd.DataFrame:
    """LMM-/LM-based two-condition test on TMP-summarized data.

    Parameters
    ----------
    processed
        Output of :func:`pymsstats.data_process` — long-format with at
        least ``Protein, RUN, LogIntensities, GROUP, SUBJECT`` columns.
    contrast_matrix
        2D array of shape ``(n_contrasts, n_groups)``. Row sums should
        typically be 0 (e.g. ``[[-1, 1]]`` for B vs A).
        If a DataFrame with row/column labels, columns must match
        ``groups`` (or all groups in ``processed``).
    groups
        Ordered groups in the contrast. If None, inferred from
        ``processed[group_col]`` in sorted order, or from
        ``contrast_matrix.columns`` if it's a DataFrame.
    random_subject
        How to model SUBJECT.

        * ``"auto"`` (default, R-MSstats behaviour) — mirror MSstats's
          design dispatch: use a fixed-effects ``ABUNDANCE ~ GROUP`` model
          when SUBJECT is nested within GROUP (the typical LFQ case),
          otherwise switch to a mixed model ``ABUNDANCE ~ GROUP +
          (1|SUBJECT)`` (paired / repeated-measures design).
        * ``True`` — always fit the mixed model.
        * ``False`` — always fit ``ABUNDANCE ~ GROUP`` (ignores SUBJECT).
    protein_col, abundance_col, group_col, run_col, subject_col
        Column names.

    Returns
    -------
    pd.DataFrame
        Columns: ``Protein, Label, log2FC, SE, df, t, pvalue, adj.pvalue,
        issue``. One row per ``(Protein, contrast)``.
    """
    contrast_matrix = np.asarray(contrast_matrix) if not isinstance(
        contrast_matrix, pd.DataFrame) else contrast_matrix
    if isinstance(contrast_matrix, pd.DataFrame):
        contrast_groups = list(contrast_matrix.columns)
        contrast_labels = list(contrast_matrix.index.astype(str))
        C = contrast_matrix.to_numpy(dtype=float)
    else:
        C = np.atleast_2d(np.asarray(contrast_matrix, dtype=float))
        if groups is not None:
            contrast_groups = list(groups)
        else:
            contrast_groups = sorted(processed[group_col].astype(str).unique())
        contrast_labels = [f"contrast_{i}" for i in range(C.shape[0])]

    if C.shape[1] != len(contrast_groups):
        raise ValueError(
            f"contrast_matrix has {C.shape[1]} columns but {len(contrast_groups)} "
            "groups were provided"
        )

    df_proc = processed.copy()
    df_proc["GROUP"] = df_proc[group_col].astype(str)
    df_proc["ABUNDANCE"] = df_proc[abundance_col].astype(float)
    if subject_col in df_proc.columns:
        df_proc["SUBJECT"] = df_proc[subject_col].astype(str)
    else:
        df_proc["SUBJECT"] = df_proc[run_col].astype(str)
    df_proc["Protein"] = df_proc[protein_col].astype(str)

    if random_subject == "auto":
        mode = _detect_subject_design(df_proc)
    elif random_subject is True:
        mode = "mixed"
    else:
        mode = "lm"
    fit_fn = _fit_one_protein_mixed if mode == "mixed" else _fit_one_protein_lm

    rows = []
    for prot, sub in df_proc.groupby("Protein", sort=False):
        for label, c_row in zip(contrast_labels, C):
            r = fit_fn(sub, contrast_groups, c_row)
            rows.append({
                "Protein": prot,
                "Label": label,
                "log2FC": r["log2FC"],
                "SE": r["SE"],
                "df": r["df"],
                "t": r["t"],
                "pvalue": r["pvalue"],
                "issue": r["issue"],
            })
    res = pd.DataFrame(rows)
    # BH per contrast row.
    res["adj.pvalue"] = np.nan
    for label, idx in res.groupby("Label").groups.items():
        sl = res.loc[idx, "pvalue"].to_numpy()
        res.loc[idx, "adj.pvalue"] = _bh(sl)
    return res[["Protein", "Label", "log2FC", "SE", "df", "t",
                "pvalue", "adj.pvalue", "issue"]]
