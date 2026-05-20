"""Sample-size / power calculation — port of ``MSstats::designSampleSize``.

The R function takes the output of ``groupComparison`` (specifically the
``FittedModel`` list) and computes either:

* the number of biological replicates required to achieve a given power
  at a desired fold change (``numSample=TRUE``), or
* the power achieved by a fixed number of replicates at that FC
  (``power=TRUE``).

The Python port accepts the same processed-table object that
``group_comparison`` consumes — we re-derive the per-protein error
variance from the residuals of the same per-protein OLS fit, then run
the identical closed-form equations:

.. code-block:: text

    z_alpha = qnorm(1 - alpha/2)
    z_beta  = qnorm(power)
    n       = round(2 * (sigma_e + sigma_s) / (delta / (z_alpha + z_beta))^2)

with ``alpha = power * FDR / (1 + (1-FDR) * 99)`` (the multiple-testing
adjustment from R MSstats with ``m0/m1 = 99``).
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm


def _per_protein_variance(
    processed: pd.DataFrame,
    *,
    protein_col: str = "Protein",
    group_col: str = "GROUP",
    abundance_col: str = "LogIntensities",
    subject_col: str = "SUBJECT",
) -> pd.DataFrame:
    """Compute per-protein error variance from the processed long-format table.

    Mirrors ``MSstats:::.getVarComponent`` — for each protein fit
    ``LogIntensities ~ GROUP`` and take ``sigma_e^2 = SSE / (n - p)``.
    A second-level (subject) variance is reported when SUBJECT crosses
    GROUP (paired/repeated-measures design); for the canonical LFQ design
    it is NaN and treated as 0 downstream (matches R).

    Returns
    -------
    pd.DataFrame
        Columns ``Protein, Error, Subject``.
    """
    rows = []
    for prot, sub in processed.groupby(protein_col, sort=False):
        sub = sub.dropna(subset=[abundance_col, group_col])
        groups = sub[group_col].astype(str).unique()
        if len(groups) < 2 or len(sub) < 3:
            rows.append({"Protein": prot, "Error": np.nan, "Subject": np.nan})
            continue
        # Cell-means design: one column per group.
        X = np.column_stack([
            (sub[group_col].astype(str) == g).astype(float) for g in groups
        ])
        y = sub[abundance_col].to_numpy(dtype=float)
        XtX_inv = np.linalg.pinv(X.T @ X)
        beta = XtX_inv @ X.T @ y
        resid = y - X @ beta
        dof = len(y) - len(groups)
        sigma_e2 = float(resid @ resid / dof) if dof > 0 else np.nan
        # Subject variance — only meaningful in paired designs
        subj_var = np.nan
        if subject_col in sub.columns:
            n_subj = sub[subject_col].astype(str).nunique()
            if n_subj < len(sub):  # subjects repeat → paired
                # Estimate subject effect by averaging within-subject means
                # (a rough method-of-moments σ²_subject).
                sub_means = (
                    sub.groupby(subject_col)[abundance_col].mean()
                )
                if len(sub_means) >= 2:
                    subj_var = max(0.0, float(sub_means.var(ddof=1)) - sigma_e2)
        rows.append({"Protein": prot, "Error": sigma_e2, "Subject": subj_var})
    return pd.DataFrame(rows)


def _calculate_power(
    desired_fc: np.ndarray,
    fdr: float,
    delta: np.ndarray,
    median_sigma_e: float,
    median_sigma_s: float,
    num_sample: int,
) -> np.ndarray:
    """Port of ``MSstats:::.calculatePower``.

    Solves for ``power`` such that
    ``Φ⁻¹(power) + Φ⁻¹(1 − power·FDR/(1+(1-FDR)·99)/2) = t``
    where ``t = δ / sqrt(2·(σ_e + σ_s)/n)``.
    """
    m0_m1 = 99.0
    denom = np.sqrt(2.0 * (median_sigma_e / num_sample
                           + median_sigma_s / num_sample))
    t = delta / denom
    power_grid = np.arange(0.0, 1.01, 0.01)
    out = np.empty_like(t)
    for i, ti in enumerate(t):
        diff = (
            norm.ppf(power_grid)
            + norm.ppf(1.0 - power_grid * fdr
                       / (1.0 + (1.0 - fdr) * m0_m1) / 2.0)
            - ti
        )
        # absolute distance to 0; pick power closest
        order = np.argsort(np.abs(diff))
        out[i] = power_grid[order[0]]
    return out


def _get_num_sample(
    desired_fc: np.ndarray,
    power: float,
    alpha: float,
    delta: np.ndarray,
    median_sigma_e: float,
    median_sigma_s: float,
) -> np.ndarray:
    """Port of ``MSstats:::.getNumSample``."""
    z_alpha = norm.ppf(1.0 - alpha / 2.0)
    z_beta = norm.ppf(power)
    aa = (delta / (z_alpha + z_beta)) ** 2
    n = np.round(2.0 * (median_sigma_e + median_sigma_s) / aa, 0)
    return n


def design_sample_size(
    processed: pd.DataFrame,
    desired_fc: Sequence[float] | float = (1.25, 1.5),
    *,
    fdr: float = 0.05,
    num_sample: bool | int = True,
    power: bool | float = 0.9,
    protein_col: str = "Protein",
    group_col: str = "GROUP",
    abundance_col: str = "LogIntensities",
    subject_col: str = "SUBJECT",
) -> pd.DataFrame:
    """Sample-size / power calculation for an MSstats design.

    Port of ``MSstats::designSampleSize``.

    Parameters
    ----------
    processed
        Output of :func:`pymsstats.data_process` — long-format with
        ``Protein, RUN, LogIntensities, GROUP, SUBJECT`` columns.
        We re-derive per-protein variance from this directly (no need
        to keep fitted models around as R does).
    desired_fc
        Range of fold-changes to consider (linear scale). Default
        ``(1.25, 1.5)`` matches R.
    fdr
        Target FDR (default 0.05).
    num_sample
        ``True`` → solve for ``n``; integer → fix sample size and report
        the achieved power.
    power
        Target power. Float in (0, 1) when ``num_sample=True``. Either
        ``power`` or ``num_sample`` is fixed at a time.
    protein_col, group_col, abundance_col, subject_col
        Column names.

    Returns
    -------
    pd.DataFrame
        Columns: ``desiredFC, numSample, FDR, power, CV``. One row per
        fold-change on the grid ``2^seq(log2(fc_low), log2(fc_high), 0.025)``.
    """
    if isinstance(desired_fc, (int, float)):
        desired_fc = (float(desired_fc), float(desired_fc))
    fc_low, fc_high = sorted(map(float, desired_fc))
    # R: delta = log2(seq(fc_low, fc_high, 0.025)); desiredFC = 2^delta
    # The seq() is on the LINEAR fold-change scale.
    if fc_high > fc_low:
        desired_fc_grid = np.arange(fc_low, fc_high + 1e-9, 0.025)
    else:
        desired_fc_grid = np.array([fc_low])
    delta = np.log2(desired_fc_grid)

    var_comp = _per_protein_variance(
        processed,
        protein_col=protein_col,
        group_col=group_col,
        abundance_col=abundance_col,
        subject_col=subject_col,
    )
    median_sigma_e = float(np.nanmedian(var_comp["Error"]))
    sigma_s = var_comp["Subject"].dropna()
    median_sigma_s = float(np.nanmedian(sigma_s)) if len(sigma_s) > 0 else 0.0

    if not np.isfinite(median_sigma_e):
        raise ValueError(
            "could not compute median sigma_e — no proteins with usable "
            "residual variance (need ≥ 2 groups per protein and df > 0)"
        )

    if num_sample is True:
        if not isinstance(power, (int, float)) or not 0 < power < 1:
            raise ValueError(
                "when num_sample=True, power must be a float in (0, 1)"
            )
        m0_m1 = 99.0
        alpha = float(power) * fdr / (1.0 + (1.0 - fdr) * m0_m1)
        n = _get_num_sample(desired_fc_grid, float(power), alpha, delta,
                            median_sigma_e, median_sigma_s)
        cv = np.round(
            2.0 * (median_sigma_e / n + median_sigma_s / n) / desired_fc_grid,
            3,
        )
        return pd.DataFrame({
            "desiredFC": desired_fc_grid,
            "numSample": n.astype(int),
            "FDR": fdr,
            "power": float(power),
            "CV": cv,
        })
    if power is True:
        if not isinstance(num_sample, int):
            raise ValueError(
                "when power=True, num_sample must be a fixed integer"
            )
        p_out = _calculate_power(desired_fc_grid, fdr, delta,
                                 median_sigma_e, median_sigma_s,
                                 int(num_sample))
        cv = np.round(
            2.0 * (median_sigma_e / num_sample + median_sigma_s / num_sample)
            / desired_fc_grid, 3,
        )
        return pd.DataFrame({
            "desiredFC": desired_fc_grid,
            "numSample": int(num_sample),
            "FDR": fdr,
            "power": p_out,
            "CV": cv,
        })
    raise ValueError(
        "exactly one of num_sample / power must be True (the other fixed)"
    )


def design_sample_size_plots(result: pd.DataFrame, ax=None):
    """Plot the sample-size / power curve from :func:`design_sample_size`.

    Port of ``MSstats::designSampleSizePlots``. The R function plots
    ``numSample`` vs ``desiredFC`` (when the result varies sample size)
    or ``power`` vs ``desiredFC`` (when it varies power).

    Parameters
    ----------
    result
        Output of :func:`design_sample_size` — a table with at least
        ``desiredFC`` plus one of ``numSample`` / ``power`` varying.
    ax
        Optional matplotlib Axes; created if None.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt  # lazy import

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    if result["numSample"].nunique() > 1:
        index = "numSample"
    elif result["power"].nunique() > 1:
        index = "power"
    else:
        index = "numSample"

    x = result["desiredFC"].to_numpy(dtype=float)
    if index == "numSample":
        ax.plot(x, result["numSample"].to_numpy(dtype=float), lw=2, color="C0")
        ax.set_xlabel("Desired fold change")
        ax.set_ylabel("Minimal number of biological replicates")
        ax.set_title(
            f"FDR is {', '.join(map(str, result['FDR'].unique()))}\n"
            f"Statistical power is "
            f"{', '.join(map(str, result['power'].unique()))}"
        )
    else:
        ax.plot(x, result["power"].to_numpy(dtype=float), lw=2, color="C0")
        ax.set_xlabel("Desired fold change")
        ax.set_ylabel("Power")
        ax.set_title(
            f"Number of replicates is "
            f"{', '.join(map(str, result['numSample'].unique()))}\n"
            f"FDR is {', '.join(map(str, result['FDR'].unique()))}"
        )
    return ax


__all__ = ["design_sample_size", "design_sample_size_plots"]
