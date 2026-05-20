"""Matplotlib plotting helpers — Python ports of MSstats's ggplot2 plots.

Lazily imports ``matplotlib`` so that loading the package does not depend
on a GUI backend. The plots mirror, in spirit, the R outputs:

* :func:`theme_msstats` — sets matplotlib rcParams to match the
  ggplot2-based style used in MSstats (white background, larger axis
  labels, gray strip backgrounds).
* :func:`data_process_plots` — QC plots after ``dataProcess``:
    ``"ProfilePlot"``  per-(protein × run) abundance lines + protein-level
    summary (one figure per protein, but a single ``ax`` plotted when
    ``protein`` is specified);
    ``"QCPlot"``  global boxplot of log abundances per run;
    ``"ConditionPlot"``  mean ± SE of LogIntensities per condition.
* :func:`group_comparison_plots` — VolcanoPlot, Heatmap, ComparisonPlot.
* :func:`group_comparison_qc_plots` — diagnostic plots (residuals,
  Q-Q) on the per-protein test.
* :func:`model_based_qc_plots` — fitted-vs-residual diagnostic.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


def _get_mpl():
    """Lazy matplotlib import."""
    import matplotlib  # noqa: F401
    import matplotlib.pyplot as plt
    return matplotlib, plt


def theme_msstats(
    type: str = "QCPLOT",
    x_axis_size: int = 10,
    y_axis_size: int = 10,
    legend_size: int = 13,
):
    """Apply MSstats's ggplot2 theme to matplotlib's rcParams.

    Returns the matplotlib rcParams dict that was modified, so the caller
    can restore them later via ``plt.rcParams.update(...)``.
    """
    _, plt = _get_mpl()
    cfg = {
        "axes.facecolor":     "white",
        "axes.edgecolor":     "black",
        "axes.grid":          False,
        "axes.titlesize":     x_axis_size + 8,
        "axes.labelsize":     x_axis_size + 5,
        "xtick.labelsize":    x_axis_size,
        "ytick.labelsize":    y_axis_size,
        "xtick.color":        "black",
        "ytick.color":        "black",
        "legend.fontsize":    legend_size,
        "legend.loc":         "upper center" if type.upper() != "CONDITIONPLOT"
                              else "upper right",
        "figure.facecolor":   "white",
        "savefig.facecolor":  "white",
    }
    plt.rcParams.update(cfg)
    return cfg


# -----------------------------------------------------------------------------
# dataProcessPlots
# -----------------------------------------------------------------------------
def data_process_plots(
    processed: pd.DataFrame,
    type: str = "ProfilePlot",
    *,
    protein: Optional[str] = None,
    feature_data: Optional[pd.DataFrame] = None,
    ax=None,
):
    """QC plots of the ``dataProcess`` output.

    Parameters
    ----------
    processed
        Output of :func:`pymsstats.data_process` (long-format with
        ``Protein, RUN, LogIntensities, GROUP, SUBJECT``).
    type
        One of ``'ProfilePlot'``, ``'QCPlot'``, ``'ConditionPlot'``.
    protein
        Required for ``'ProfilePlot'`` — which protein to plot.
    feature_data
        Optional feature-level long-format table (for the ProfilePlot to
        show the underlying features as light lines).
    ax
        Optional matplotlib Axes; created if None.

    Returns
    -------
    matplotlib.axes.Axes
    """
    _, plt = _get_mpl()
    t = str(type).upper()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    if t == "QCPLOT":
        # Box plot of LogIntensities per run
        data = [
            grp["LogIntensities"].dropna().to_numpy()
            for _, grp in processed.groupby("RUN", sort=True)
        ]
        labels = [r for r, _ in processed.groupby("RUN", sort=True)]
        ax.boxplot(data, tick_labels=labels)
        ax.set_xlabel("Run")
        ax.set_ylabel("Log2 Intensity")
        ax.set_title("QC plot of summarized protein abundances")
        ax.tick_params(axis="x", rotation=45)
        return ax

    if t == "PROFILEPLOT":
        if protein is None:
            raise ValueError("ProfilePlot requires the protein argument")
        sub = processed.loc[processed["Protein"] == protein].sort_values("RUN")
        if feature_data is not None:
            fsub = feature_data.loc[
                feature_data.get("ProteinName", feature_data.get("PROTEIN"))
                == protein
            ]
            if len(fsub) > 0 and "ABUNDANCE" in fsub.columns:
                for _, fg in fsub.groupby(
                        fsub.get("FEATURE",
                                 fsub.get("PeptideSequence"))):
                    ax.plot(fg.get("RUN", fg.get("Run")),
                            fg["ABUNDANCE"], color="lightgray", lw=0.8)
        ax.plot(sub["RUN"], sub["LogIntensities"], marker="o", color="C0",
                label="Protein-level (TMP)")
        ax.set_title(f"Profile plot — {protein}")
        ax.set_xlabel("Run")
        ax.set_ylabel("Log2 Intensity")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
        return ax

    if t == "CONDITIONPLOT":
        # Mean ± SE of LogIntensities per condition
        agg = (
            processed.groupby("GROUP")["LogIntensities"]
            .agg(["mean", "sem", "count"]).reset_index()
        )
        ax.errorbar(agg["GROUP"], agg["mean"], yerr=agg["sem"],
                    fmt="o", capsize=4, color="C0")
        ax.set_title("Condition-level mean abundance")
        ax.set_xlabel("Condition")
        ax.set_ylabel("Mean Log2 Intensity")
        return ax

    raise ValueError(
        f"type={type!r} not recognised (use 'ProfilePlot', 'QCPlot', "
        "or 'ConditionPlot')"
    )


# -----------------------------------------------------------------------------
# groupComparisonPlots
# -----------------------------------------------------------------------------
def group_comparison_plots(
    result: pd.DataFrame,
    type: str = "VolcanoPlot",
    *,
    sig: float = 0.05,
    fc_cutoff: float = 1.0,
    ax=None,
    label_col: str = "Label",
    contrast: Optional[str] = None,
):
    """Volcano / heatmap / comparison plots on a ``groupComparison`` result.

    Parameters
    ----------
    result
        Output of :func:`pymsstats.group_comparison` (one row per
        protein × contrast with ``log2FC, pvalue, adj.pvalue`` columns).
    type
        ``'VolcanoPlot'``, ``'Heatmap'``, or ``'ComparisonPlot'``.
    sig
        Significance threshold on ``adj.pvalue`` for highlighting points
        / annotating cells.
    fc_cutoff
        Absolute log2-fold-change threshold for highlighting.
    contrast
        For multi-contrast results, restrict to a single contrast.
    """
    _, plt = _get_mpl()
    t = str(type).upper()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    res = result.copy()
    if contrast is not None:
        res = res.loc[res[label_col] == contrast]

    if t == "VOLCANOPLOT":
        # one contrast at a time — if multiple, pick the first
        if res[label_col].nunique() > 1:
            res = res.loc[res[label_col] == res[label_col].iloc[0]]
        x = res["log2FC"].to_numpy(dtype=float)
        y = -np.log10(np.clip(res["adj.pvalue"].astype(float), 1e-300, 1))
        sig_mask = (
            (res["adj.pvalue"] < sig).fillna(False)
            & (res["log2FC"].abs() > fc_cutoff)
        )
        ax.scatter(x[~sig_mask], y[~sig_mask], s=15, color="lightgray",
                   alpha=0.7, label="non-significant")
        ax.scatter(x[sig_mask], y[sig_mask], s=20, color="red",
                   alpha=0.85, label=f"adj.p<{sig}, |log2FC|>{fc_cutoff}")
        ax.axhline(-np.log10(sig), color="gray", ls="--", lw=0.7)
        ax.axvline(fc_cutoff, color="gray", ls="--", lw=0.7)
        ax.axvline(-fc_cutoff, color="gray", ls="--", lw=0.7)
        ax.set_xlabel("log2(fold change)")
        ax.set_ylabel("-log10(adj.pvalue)")
        ax.set_title("Volcano plot")
        ax.legend(fontsize=8)
        return ax

    if t == "HEATMAP":
        wide = res.pivot_table(
            index="Protein", columns=label_col, values="log2FC",
            aggfunc="mean",
        )
        if wide.empty:
            ax.text(0.5, 0.5, "(empty result)", ha="center", va="center",
                    transform=ax.transAxes)
            return ax
        im = ax.imshow(wide.to_numpy(), aspect="auto",
                       cmap="RdBu_r",
                       vmin=-max(abs(wide.values.min()), abs(wide.values.max())),
                       vmax=max(abs(wide.values.min()), abs(wide.values.max())))
        ax.set_yticks(range(len(wide.index)))
        ax.set_yticklabels(wide.index, fontsize=7)
        ax.set_xticks(range(len(wide.columns)))
        ax.set_xticklabels(wide.columns, rotation=45, ha="right")
        plt.colorbar(im, ax=ax, label="log2FC")
        ax.set_title("Heatmap of log2 fold changes")
        return ax

    if t == "COMPARISONPLOT":
        # One row per contrast, point=log2FC, error bars=SE
        for prot, sub in res.groupby("Protein"):
            ax.errorbar(sub[label_col], sub["log2FC"], yerr=sub["SE"],
                        fmt="o", capsize=3, label=prot, alpha=0.7)
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_xlabel("Contrast")
        ax.set_ylabel("log2FC ± SE")
        ax.set_title("Comparison plot")
        if res["Protein"].nunique() <= 8:
            ax.legend(fontsize=7)
        return ax

    raise ValueError(
        f"type={type!r} not recognised (use 'VolcanoPlot', 'Heatmap', or "
        "'ComparisonPlot')"
    )


# -----------------------------------------------------------------------------
# QC plots
# -----------------------------------------------------------------------------
def group_comparison_qc_plots(
    result: pd.DataFrame,
    *,
    ax=None,
):
    """Distribution of test statistics across proteins.

    Mirrors ``MSstats::groupComparisonQCPlots`` — a histogram of pvalues
    (and adj.pvalues overlaid) so the user can see whether the model is
    well-calibrated (a uniform tail of pvalues is the diagnostic).
    """
    _, plt = _get_mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    p = result["pvalue"].dropna()
    adj = result["adj.pvalue"].dropna()
    ax.hist(p, bins=40, color="C0", alpha=0.6, label="pvalue")
    ax.hist(adj, bins=40, color="C3", alpha=0.5, label="adj.pvalue")
    ax.set_xlabel("p-value")
    ax.set_ylabel("# proteins")
    ax.set_title("groupComparison QC — p-value histogram")
    ax.legend()
    return ax


def model_based_qc_plots(
    processed: pd.DataFrame,
    *,
    ax=None,
    protein: Optional[str] = None,
):
    """Per-protein residuals vs fitted abundance.

    Mirrors ``MSstats::modelBasedQCPlots`` — for each protein fits the
    same OLS ``LogIntensities ~ GROUP`` that group_comparison would, and
    plots residuals vs fitted values pooled across proteins (or one
    protein when ``protein`` is given).
    """
    _, plt = _get_mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    sub = (
        processed if protein is None
        else processed.loc[processed["Protein"] == protein]
    )
    fitted = []
    resid = []
    for prot, p_sub in sub.groupby("Protein", sort=False):
        groups = p_sub["GROUP"].astype(str)
        y = p_sub["LogIntensities"].astype(float).to_numpy()
        gm = p_sub.groupby("GROUP")["LogIntensities"].transform("mean").to_numpy()
        fitted.extend(gm.tolist())
        resid.extend((y - gm).tolist())
    fitted = np.asarray(fitted)
    resid = np.asarray(resid)
    ax.scatter(fitted, resid, s=10, alpha=0.5, color="C0")
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlabel("Fitted log2 intensity")
    ax.set_ylabel("Residual")
    title = "Model-based QC — residuals vs fitted"
    if protein is not None:
        title += f" ({protein})"
    ax.set_title(title)
    return ax


def save_plot(fig, path: str, *, width: float = 10.0, height: float = 10.0,
              dpi: int = 300, **kwargs):
    """Save a matplotlib figure to disk.

    Port of ``MSstats::savePlot`` — a thin wrapper around the R ``pdf()``
    device. R only supports a PDF device; this Python port supports any
    extension matplotlib can write (``.pdf``, ``.png``, ``.svg`` …),
    inferred from ``path``.

    Parameters
    ----------
    fig
        A matplotlib ``Figure`` (or an ``Axes`` — its parent figure is
        used).
    path
        Output file path; the extension determines the format.
    width, height
        Figure size in inches (applied before saving). R defaults to a
        square page.
    dpi
        Resolution for raster formats (PNG).
    **kwargs
        Forwarded to ``Figure.savefig``.

    Returns
    -------
    str
        The path written.
    """
    _get_mpl()
    # accept an Axes
    if hasattr(fig, "figure") and not hasattr(fig, "savefig"):
        fig = fig.figure
    if width is not None and height is not None:
        fig.set_size_inches(float(width), float(height))
    fig.savefig(path, dpi=dpi, bbox_inches="tight", **kwargs)
    return path


__all__ = [
    "theme_msstats",
    "data_process_plots",
    "group_comparison_plots",
    "group_comparison_qc_plots",
    "model_based_qc_plots",
    "save_plot",
]
