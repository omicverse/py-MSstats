"""
Tukey median polish (TMP) feature-level → protein-level summarization.

Mirrors ``MSstats::: .fitTukey`` / ``MSstats:::median_polish_summary``.
The canonical R function is ``stats::medpolish``: iteratively subtract
the per-row median, then the per-column median, until the sum of |z|
either stops decreasing (within tolerance ``eps``) or ``maxiter`` is
reached.

For an MSstats protein we build a ``run × feature`` matrix of log2
abundances, run medpolish, and report the per-run summarized value as
``overall + col_effect[run]`` (the column effects ARE the runs after the
``RUN × FEATURE`` cast).

Wait — in MSstats's R code:

```R
wide = data.table::dcast(LABEL + RUN ~ FEATURE, data=input, value.var="newABUNDANCE")
tmp_fitted = median_polish_summary(as.matrix(wide[, features, with=FALSE]))
wide[, newABUNDANCE := tmp_fitted]
```

So ``tmp_fitted`` is a *vector* (length = nrow(wide) = n_runs). The R
helper :func:`median_polish_summary` is implemented in C++ and returns
``overall + row_effect[run]`` for every run. (Rows of `wide` are runs,
columns are features; medpolish returns ``r + t`` over rows.)

That is exactly what we reproduce here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def medpolish(
    X: np.ndarray,
    *,
    eps: float = 0.01,
    maxiter: int = 10,
    na_rm: bool = True,
) -> dict:
    """Tukey median polish — port of R ``stats::medpolish``.

    Parameters
    ----------
    X
        2D float matrix; NaN allowed (treated as missing if ``na_rm``).
    eps
        Convergence tolerance on the residual L1 sum, relative to itself
        (R default = 0.01).
    maxiter
        Iteration cap (R default = 10).
    na_rm
        Drop NaNs from each row / column median.

    Returns
    -------
    dict
        Keys ``overall`` (scalar), ``row`` (1D), ``col`` (1D),
        ``residuals`` (2D), ``converged`` (bool), ``iter`` (int).
    """
    Z = np.array(X, dtype=float, copy=True)
    nr, nc = Z.shape
    t = 0.0
    r = np.zeros(nr)
    c = np.zeros(nc)
    oldsum = 0.0
    converged = False
    iter_done = 0
    median = np.nanmedian if na_rm else np.median
    sumabs = (lambda z: np.nansum(np.abs(z))) if na_rm else (
        lambda z: np.sum(np.abs(z))
    )

    for it in range(maxiter):
        # Row sweep
        rdelta = median(Z, axis=1)
        rdelta = np.where(np.isnan(rdelta), 0.0, rdelta)
        Z = Z - rdelta[:, None]
        r = r + rdelta
        # Re-center column effects through the row median of c.
        delta = float(median(c)) if c.size else 0.0
        if np.isnan(delta):
            delta = 0.0
        c = c - delta
        t = t + delta

        # Column sweep
        cdelta = median(Z, axis=0)
        cdelta = np.where(np.isnan(cdelta), 0.0, cdelta)
        Z = Z - cdelta[None, :]
        c = c + cdelta
        delta = float(median(r)) if r.size else 0.0
        if np.isnan(delta):
            delta = 0.0
        r = r - delta
        t = t + delta

        newsum = float(sumabs(Z))
        iter_done = it + 1
        if newsum == 0 or abs(newsum - oldsum) < eps * newsum:
            converged = True
            break
        oldsum = newsum

    return {
        "overall": float(t),
        "row": r,
        "col": c,
        "residuals": Z,
        "converged": converged,
        "iter": iter_done,
    }


def tmp_summarize(
    long_df: pd.DataFrame,
    *,
    protein_col: str = "PROTEIN",
    run_col: str = "RUN",
    feature_col: str = "FEATURE",
    abundance_col: str = "ABUNDANCE",
) -> pd.DataFrame:
    """TMP-summarize a normalized long-format feature table.

    For each ``PROTEIN``:
      1. Pivot to a ``run × feature`` matrix of log abundances (NaN if
         absent).
      2. Run :func:`medpolish`.
      3. Per run, report ``overall + row_effect[run]`` — this is the
         protein-level log abundance for that run.

    Parameters
    ----------
    long_df
        Normalized long-format table with at least
        ``PROTEIN, RUN, FEATURE, ABUNDANCE`` columns. ``ABUNDANCE`` is
        log-scale (the result of ``log2 + equalize_medians``).
    protein_col, run_col, feature_col, abundance_col
        Column names.

    Returns
    -------
    pd.DataFrame
        Columns: ``Protein, RUN, LogIntensities, n_features, n_obs``.
        ``LogIntensities`` is the TMP-summarized log abundance for each
        ``(Protein, RUN)`` cell.
    """
    out_rows = []
    grouped = long_df.groupby(protein_col, sort=False)
    for prot, sub in grouped:
        # Pivot to run × feature.
        wide = sub.pivot_table(
            index=run_col, columns=feature_col, values=abundance_col,
            aggfunc="mean",   # MSstats's data.table dcast picks the first
            dropna=False,
        )
        runs = list(wide.index.astype(str))
        feats = list(wide.columns.astype(str))
        X = wide.to_numpy(dtype=float)
        n_feat = X.shape[1]
        n_obs = int(np.isfinite(X).sum())
        if X.size == 0 or not np.isfinite(X).any():
            continue
        if n_feat == 1:
            # Single feature → just return the column directly.
            tmp_fitted = X[:, 0]
        else:
            mp = medpolish(X)
            # MSstats's median_polish_summary returns the *row* fitted values
            # = overall + row_effect, evaluated for every row of the matrix.
            tmp_fitted = mp["overall"] + mp["row"]
        for run, val in zip(runs, tmp_fitted):
            if np.isnan(val):
                continue
            out_rows.append({
                "Protein": prot,
                "RUN": run,
                "LogIntensities": float(val),
                "n_features": n_feat,
                "n_obs": n_obs,
            })
    return pd.DataFrame(out_rows)


def linear_summarize(
    long_df: pd.DataFrame,
    *,
    protein_col: str = "PROTEIN",
    run_col: str = "RUN",
    feature_col: str = "FEATURE",
    abundance_col: str = "ABUNDANCE",
    equal_variance: bool = True,
) -> pd.DataFrame:
    """Linear-model summarization — port of ``MSstats:::.summarizeSingleLinear``.

    For each protein fit ``ABUNDANCE ~ RUN + FEATURE`` (cell-means
    parameterization on RUN, treatment-coded on FEATURE) and report the
    per-run estimate ``overall + RUN_effect``. Mirrors the legacy
    pre-TMP MSstats summarization (option ``summary_method='linear'``).

    For ``equal_variance=False`` (R default for the linear summary) the
    summary is identical — only the error variance differs (irrelevant for
    the per-run point estimates).

    Returns the same schema as :func:`tmp_summarize`.
    """
    out_rows = []
    for prot, sub in long_df.groupby(protein_col, sort=False):
        sub = sub.dropna(subset=[abundance_col])
        if len(sub) == 0:
            continue
        runs = list(sub[run_col].astype(str).unique())
        feats = list(sub[feature_col].astype(str).unique())
        if len(runs) == 0 or len(feats) == 0:
            continue
        if len(feats) == 1:
            # Same shortcut as TMP — single-feature → just the values
            run_means = sub.groupby(run_col)[abundance_col].mean()
            for run, val in run_means.items():
                if pd.isna(val):
                    continue
                out_rows.append({
                    "Protein": prot, "RUN": str(run),
                    "LogIntensities": float(val),
                    "n_features": 1, "n_obs": int(sub[abundance_col].notna().sum()),
                })
            continue
        # Cell-means design on RUN, treatment-coded on FEATURE (drop first feat).
        X_run = np.column_stack([
            (sub[run_col].astype(str) == r).astype(float).to_numpy()
            for r in runs
        ])
        X_feat = np.column_stack([
            (sub[feature_col].astype(str) == f).astype(float).to_numpy()
            for f in feats[1:]
        ]) if len(feats) > 1 else np.empty((len(sub), 0))
        X = np.hstack([X_run, X_feat])
        y = sub[abundance_col].to_numpy(dtype=float)
        try:
            XtX_inv = np.linalg.pinv(X.T @ X)
            beta = XtX_inv @ X.T @ y
        except Exception:
            continue
        # RUN coefficients give abundance at the reference feature;
        # MSstats reports the per-run estimate AT THE AVERAGE FEATURE, so we
        # add back the mean of the (treatment-coded) feature effects.
        run_coefs = beta[:len(runs)]
        feat_coefs = beta[len(runs):]
        # average feature effect = mean over all features of their effect;
        # reference feature has effect 0, the rest have feat_coefs entries.
        mean_feat_effect = (
            float(np.sum(feat_coefs)) / len(feats) if len(feats) > 1 else 0.0
        )
        run_estimates = run_coefs + mean_feat_effect
        n_obs = int(sub[abundance_col].notna().sum())
        for run, val in zip(runs, run_estimates):
            if not np.isfinite(val):
                continue
            out_rows.append({
                "Protein": prot, "RUN": str(run),
                "LogIntensities": float(val),
                "n_features": len(feats),
                "n_obs": n_obs,
            })
    return pd.DataFrame(out_rows)


def msstats_summarize(
    long_df: pd.DataFrame,
    *,
    method: str = "TMP",
    protein_col: str = "PROTEIN",
    run_col: str = "RUN",
    feature_col: str = "FEATURE",
    abundance_col: str = "ABUNDANCE",
    equal_variance: bool = True,
) -> pd.DataFrame:
    """Dispatch summarization — port of ``MSstats::MSstatsSummarize``.

    Methods:

    * ``'TMP'`` → :func:`tmp_summarize`
    * ``'linear'`` → :func:`linear_summarize`
    """
    if method == "TMP":
        return tmp_summarize(
            long_df,
            protein_col=protein_col, run_col=run_col,
            feature_col=feature_col, abundance_col=abundance_col,
        )
    if method == "linear":
        return linear_summarize(
            long_df,
            protein_col=protein_col, run_col=run_col,
            feature_col=feature_col, abundance_col=abundance_col,
            equal_variance=equal_variance,
        )
    raise ValueError(
        f"unknown summarization method={method!r} — choose 'TMP' or 'linear'"
    )


def quantification(
    processed: pd.DataFrame,
    type: str = "Sample",
    format: str = "matrix",
    *,
    protein_col: str = "Protein",
    abundance_col: str = "LogIntensities",
    group_col: str = "GROUP",
    subject_col: str = "SUBJECT",
) -> pd.DataFrame:
    """Per-protein log-intensity wide matrix.

    Port of ``MSstats::quantification``.

    Parameters
    ----------
    processed
        Output of :func:`pymsstats.data_process` — long-format with
        ``Protein, LogIntensities, GROUP, SUBJECT`` columns.
    type
        ``'Sample'`` → one column per ``GROUP_SUBJECT``; ``'Group'`` →
        one column per ``GROUP`` (median over subjects).
    format
        ``'matrix'`` (wide, default) or ``'long'``.

    Returns
    -------
    pd.DataFrame
    """
    t = str(type).upper()
    f = str(format).upper()
    if t not in ("SAMPLE", "GROUP"):
        raise ValueError(f"type must be 'Sample' or 'Group' (got {type!r})")
    if f not in ("MATRIX", "LONG"):
        raise ValueError(f"format must be 'matrix' or 'long' (got {format!r})")

    df = processed.dropna(subset=[abundance_col]).copy()
    if t == "SAMPLE":
        df["_col"] = (
            df[group_col].astype(str) + "_" + df[subject_col].astype(str)
        )
        wide = df.pivot_table(
            index=protein_col, columns="_col", values=abundance_col,
            aggfunc="median",
        ).reset_index().rename_axis(columns=None)
        if f == "LONG":
            out = wide.melt(id_vars=protein_col,
                            var_name="Group_Subject", value_name="LogIntensity")
            return out
        return wide
    # GROUP
    df = (
        df.groupby([protein_col, group_col, subject_col], as_index=False)
        [abundance_col].median()
    )
    df = (
        df.groupby([protein_col, group_col], as_index=False)
        [abundance_col].median()
    )
    wide = df.pivot_table(
        index=protein_col, columns=group_col, values=abundance_col,
        aggfunc="median",
    ).reset_index().rename_axis(columns=None)
    if f == "LONG":
        out = wide.melt(id_vars=protein_col,
                        var_name="Group", value_name="LogIntensity")
        return out
    return wide


# -----------------------------------------------------------------------------
# Accessors — ports of getProcessed / getSamplesInfo / getSelectedProteins
# -----------------------------------------------------------------------------
def get_processed(processed: pd.DataFrame) -> pd.DataFrame:
    """Return the rows where ``remove == True`` (R: ``getProcessed``).

    In R the *processed* dataset has a boolean ``remove`` column flagging
    features that were filtered out. We mirror that interface — if the
    ``remove`` column is missing or all-False, return ``None``-equivalent
    (empty DataFrame); otherwise return the removed rows for inspection.
    """
    if "remove" not in processed.columns:
        return processed.iloc[:0].copy()
    if not processed["remove"].any():
        return processed.iloc[:0].copy()
    return processed.loc[processed["remove"]].copy()


def get_samples_info(processed: pd.DataFrame,
                     *, group_col: str = "GROUP",
                     run_col: str = "RUN") -> pd.DataFrame:
    """Number of runs per group — port of ``getSamplesInfo``."""
    return (
        processed.groupby(group_col)[run_col].nunique()
        .reset_index().rename(columns={run_col: "NumRuns"})
    )


def get_selected_proteins(chosen_proteins, all_proteins) -> list:
    """Validate & resolve a protein selection — port of ``getSelectedProteins``.

    Accepts either a list of strings (protein names) or a list of 1-based
    integer indices (matches R semantics).
    """
    all_proteins = list(all_proteins)
    if isinstance(chosen_proteins, (str,)):
        chosen_proteins = [chosen_proteins]
    if all(isinstance(c, str) for c in chosen_proteins):
        missing = [c for c in chosen_proteins if c not in all_proteins]
        if missing:
            raise ValueError(
                f"proteins not in dataset: {missing}"
            )
        return list(chosen_proteins)
    if all(isinstance(c, (int, np.integer)) for c in chosen_proteins):
        if max(chosen_proteins) > len(all_proteins):
            raise ValueError(
                f"index out of range — dataset has {len(all_proteins)} proteins"
            )
        # R is 1-indexed
        return [all_proteins[i - 1] for i in chosen_proteins]
    raise TypeError(
        "chosen_proteins must be a list of strings OR a list of 1-based ints"
    )
