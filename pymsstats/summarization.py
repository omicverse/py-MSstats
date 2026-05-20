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
