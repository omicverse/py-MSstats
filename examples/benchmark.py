"""Head-to-head speed+accuracy benchmark: R MSstats vs pymsstats.

Runs ``dataProcess → groupComparison`` on a synthetic LFQ dataset and
reports:

* wall-clock time for each step (R via ``Rscript``, Python via
  ``time.perf_counter``)
* Pearson r between Python and R for: LogIntensities (post-TMP),
  log2FC, pvalue, adj.pvalue.

The synthetic table is generated deterministically in NumPy and dumped
to TSV so both languages see identical input.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pymsstats import data_process, group_comparison


HERE = Path(__file__).parent
WORK = HERE / "compare_out"
CONDA_BIN = "/home/users/steorra/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV = "/scratch/users/steorra/env/CMAP"


def _make_dataset(n_proteins: int, n_features_per_protein: int = 3,
                  n_per_group: int = 4, seed: int = 42):
    rng = np.random.default_rng(seed)
    rows = []
    for prot_i in range(n_proteins):
        base = rng.normal(20.0, 1.5)
        is_de_i = rng.uniform() < 0.1
        delta = (rng.choice([-1, 1]) * 1.0) if is_de_i else 0.0
        for feat_i in range(n_features_per_protein):
            feat_offset = rng.normal(0, 0.5)
            for g in range(2):
                for s in range(n_per_group):
                    intensity = 2 ** rng.normal(
                        base + feat_offset + (delta if g == 1 else 0.0), 0.4,
                    )
                    rows.append({
                        "ProteinName":      f"prot_{prot_i:05d}",
                        "PeptideSequence":  f"PEP_{prot_i:05d}_{feat_i}",
                        "PrecursorCharge":  2,
                        "FragmentIon":      "NA",
                        "ProductCharge":    "NA",
                        "IsotopeLabelType": "L",
                        "Condition":        f"group{g}",
                        "BioReplicate":     f"group{g}_rep{s}",
                        "Run":              f"Run_{g}_{s}",
                        "Intensity":        float(intensity),
                    })
    return pd.DataFrame(rows)


def run_python(df: pd.DataFrame, runs: int = 3):
    """Run the full Python pipeline ``runs`` times; return mean wall-clock
    and the final result frame."""
    elapsed_dp = []
    elapsed_gc = []
    res = None
    for _ in range(runs):
        t0 = time.perf_counter()
        proc = data_process(df)
        t1 = time.perf_counter()
        res = group_comparison(
            proc, contrast_matrix=np.array([[-1.0, 1.0]]),
            groups=["group0", "group1"],
        )
        t2 = time.perf_counter()
        elapsed_dp.append(t1 - t0)
        elapsed_gc.append(t2 - t1)
    return {
        "dataProcess_s": float(np.mean(elapsed_dp)),
        "groupComparison_s": float(np.mean(elapsed_gc)),
        "total_s": float(np.mean(elapsed_dp) + np.mean(elapsed_gc)),
        "result": res,
        "processed": proc,
    }


def run_R(work: Path):
    cmd = (
        f"source {CONDA_BIN} && conda activate {CONDA_ENV} && Rscript "
        f"{Path(__file__).resolve().parent.parent / 'tests' / 'r_reference_driver.R'} "
        f"{work / 'msstats_long.tsv'} {work / 'R_out'}"
    )
    t0 = time.perf_counter()
    subprocess.run(["bash", "-lc", cmd], check=True,
                   capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    timing = pd.read_csv(work / "R_out" / "timing.tsv", sep="\t")
    return elapsed, timing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--n-proteins", type=int, default=500)
    ap.add_argument("--n-features-per-protein", type=int, default=3)
    ap.add_argument("--n-per-group", type=int, default=4)
    args = ap.parse_args()

    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)

    df = _make_dataset(args.n_proteins, args.n_features_per_protein,
                       args.n_per_group)
    df.to_csv(WORK / "msstats_long.tsv", sep="\t", index=False, na_rep="NA")
    n_runs = df["Run"].nunique()
    n_features = df.shape[0]
    print(f"Synthetic LFQ: {args.n_proteins} proteins × {n_runs} runs "
          f"({n_features} feature-rows)")

    print(f"\n--- Python timing (mean of {args.runs} runs) ---")
    py = run_python(df, runs=args.runs)
    print(f"  dataProcess        {py['dataProcess_s']*1000:8.2f} ms")
    print(f"  groupComparison    {py['groupComparison_s']*1000:8.2f} ms")
    print(f"  total              {py['total_s']*1000:8.2f} ms")

    print(f"\n--- R timing (single run) ---")
    r_total, r_timing = run_R(WORK)
    r_dp = float(r_timing["dataProcess"].iloc[0])
    r_gc = float(r_timing["groupComparison"].iloc[0])
    r_pipeline = r_dp + r_gc
    print(f"  dataProcess        {r_dp*1000:8.2f} ms")
    print(f"  groupComparison    {r_gc*1000:8.2f} ms")
    print(f"  pipeline           {r_pipeline*1000:8.2f} ms")
    print(f"  total wall (incl. Rscript startup) {r_total*1000:8.2f} ms")

    print(f"\nSpeedup (R pipeline / Python pipeline): "
          f"{r_pipeline / py['total_s']:.2f}×")
    print(f"Speedup incl. Rscript startup:          {r_total / py['total_s']:.2f}×")

    # Accuracy comparison
    r_run = pd.read_csv(WORK / "R_out" / "run_level.tsv", sep="\t")
    r_run["RUN"] = r_run["originalRUN"].astype(str)
    r_run = r_run.set_index(["Protein", "RUN"])["LogIntensities"]
    py_run = py["processed"].set_index(["Protein", "RUN"])["LogIntensities"]
    common = py_run.index.intersection(r_run.index)
    li_diff = (py_run.loc[common] - r_run.loc[common]).abs()
    li_r = float(np.corrcoef(py_run.loc[common], r_run.loc[common])[0, 1])

    r_res = pd.read_csv(WORK / "R_out" / "group_comp.tsv", sep="\t")
    py_res = py["result"].set_index("Protein")
    r_res2 = r_res.set_index("Protein")
    common_p = py_res.index.intersection(r_res2.index)

    def _pearson(col_py, col_r):
        a = py_res.loc[common_p, col_py].to_numpy(dtype=float)
        b = r_res2.loc[common_p, col_r].to_numpy(dtype=float)
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 3:
            return float("nan")
        return float(np.corrcoef(a[m], b[m])[0, 1])

    fc_r = _pearson("log2FC", "log2FC")
    se_r = _pearson("SE", "SE")
    p_r = _pearson("pvalue", "pvalue")
    adj_r = _pearson("adj.pvalue", "adj.pvalue")

    print(f"\n--- Accuracy (Python vs R) ---")
    print(f"  LogIntensities  max abs diff={li_diff.max():.3e}  Pearson r={li_r:.6f}")
    print(f"  log2FC          Pearson r={fc_r:.6f}")
    print(f"  SE              Pearson r={se_r:.6f}")
    print(f"  pvalue          Pearson r={p_r:.6f}")
    print(f"  adj.pvalue      Pearson r={adj_r:.6f}")

    summary = {
        "shape": [int(args.n_proteins), int(n_runs)],
        "py_time_ms": py["total_s"] * 1000,
        "py_dataProcess_ms": py["dataProcess_s"] * 1000,
        "py_groupComparison_ms": py["groupComparison_s"] * 1000,
        "r_time_ms_pipeline": r_pipeline * 1000,
        "r_time_ms_total": r_total * 1000,
        "r_dataProcess_ms": r_dp * 1000,
        "r_groupComparison_ms": r_gc * 1000,
        "speedup_pipeline": r_pipeline / py["total_s"],
        "speedup_with_startup": r_total / py["total_s"],
        "logintensities_max_abs_diff": float(li_diff.max()),
        "logintensities_pearson_r": li_r,
        "log2fc_pearson_r": fc_r,
        "SE_pearson_r": se_r,
        "pvalue_pearson_r": p_r,
        "adj_pvalue_pearson_r": adj_r,
    }
    (WORK / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nFull report → {WORK / 'summary.json'}")


if __name__ == "__main__":
    main()
