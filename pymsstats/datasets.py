"""Bundled example datasets.

R MSstats ships four example datasets — ``DDARawData``,
``DDARawData.Skyline``, ``DIARawData`` and ``SRMRawData``. These are
copyrighted Bioconductor data objects and are NOT redistributed with
pymsstats. Instead this module provides a synthetic generator that
returns a structurally identical MSstats long-format DataFrame so the
pipeline can be exercised end-to-end without the original data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


CANONICAL_COLS = [
    "ProteinName", "PeptideSequence", "PrecursorCharge",
    "FragmentIon", "ProductCharge", "IsotopeLabelType",
    "Condition", "BioReplicate", "Run", "Intensity",
]


def make_example_dataset(
    *,
    n_proteins: int = 30,
    n_features_per_protein: int = 3,
    n_per_group: int = 3,
    n_groups: int = 2,
    prop_de: float = 0.2,
    effect_size: float = 1.0,
    seed: int = 0,
    acquisition: str = "DDA",
) -> pd.DataFrame:
    """Synthetic MSstats long-format example dataset.

    A drop-in replacement for the R example datasets (``DDARawData`` /
    ``DIARawData`` / ``SRMRawData``). Returns a balanced label-free
    design with the 10 canonical MSstats columns.

    Parameters
    ----------
    n_proteins, n_features_per_protein, n_per_group, n_groups
        Design dimensions.
    prop_de
        Fraction of proteins given a true between-group effect.
    effect_size
        Log2 fold-change applied to differentially abundant proteins.
    seed
        RNG seed.
    acquisition
        ``'DDA'`` (one ``NA`` fragment per peptide), ``'DIA'`` /
        ``'SRM'`` (named transitions). Only affects the ``FragmentIon`` /
        ``ProductCharge`` columns.

    Returns
    -------
    pd.DataFrame
        MSstats long-format table.
    """
    rng = np.random.default_rng(seed)
    acq = str(acquisition).upper()
    n_frag = 1 if acq == "DDA" else 3
    rows = []
    for prot_i in range(n_proteins):
        base = rng.normal(20.0, 1.5)
        is_de = rng.uniform() < prop_de
        delta = (rng.choice([-1, 1]) * effect_size) if is_de else 0.0
        for feat_i in range(n_features_per_protein):
            feat_offset = rng.normal(0, 0.5)
            for frag_i in range(n_frag):
                if acq == "DDA":
                    frag_ion, prod_charge = "NA", "NA"
                else:
                    frag_ion, prod_charge = f"y{frag_i + 3}", 1
                for g in range(n_groups):
                    for s in range(n_per_group):
                        mu = base + feat_offset + (delta if g > 0 else 0.0)
                        intensity = 2 ** rng.normal(mu, 0.4)
                        rows.append({
                            "ProteinName":      f"Protein_{prot_i:03d}",
                            "PeptideSequence":  f"PEP_{prot_i:03d}_{feat_i}",
                            "PrecursorCharge":  2,
                            "FragmentIon":      frag_ion,
                            "ProductCharge":    prod_charge,
                            "IsotopeLabelType": "L",
                            "Condition":        f"Condition{g + 1}",
                            "BioReplicate":     f"C{g + 1}_rep{s + 1}",
                            "Run":              f"Run_{g + 1}_{s + 1}",
                            "Intensity":        float(intensity),
                        })
    return pd.DataFrame(rows, columns=CANONICAL_COLS)


def load_dda_example(**kwargs) -> pd.DataFrame:
    """Synthetic stand-in for the R ``DDARawData`` example dataset.

    See :func:`make_example_dataset` for keyword arguments.
    """
    kwargs.setdefault("acquisition", "DDA")
    return make_example_dataset(**kwargs)


def load_dia_example(**kwargs) -> pd.DataFrame:
    """Synthetic stand-in for the R ``DIARawData`` example dataset."""
    kwargs.setdefault("acquisition", "DIA")
    return make_example_dataset(**kwargs)


def load_srm_example(**kwargs) -> pd.DataFrame:
    """Synthetic stand-in for the R ``SRMRawData`` example dataset."""
    kwargs.setdefault("acquisition", "SRM")
    return make_example_dataset(**kwargs)


__all__ = [
    "make_example_dataset",
    "load_dda_example",
    "load_dia_example",
    "load_srm_example",
]
