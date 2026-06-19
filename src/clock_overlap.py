"""
Overlap analysis: do the CpGs my elastic-net clock selected match the published
Hannum (2013) and Horvath (2013) clock sites?

Why this matters: the model chose its CpGs from a top-variance shortlist with no
knowledge of the published clocks. If its picks overlap the known clock sites
more than chance predicts, that is evidence it rediscovered real age-related
biology rather than dataset noise.

Reference CpG lists come from the biolearn package data (Biomarkers of Aging
Consortium), which mirrors the published coefficient tables:
  Hannum 2013 (Mol Cell):      71 sites,  biolearn/data/Hannum.csv
  Horvath 2013 (Genome Biol):  353 sites, biolearn/data/Horvath1.csv
Source: https://github.com/bio-learn/biolearn  (we use only the CpGmarker column)

Reads:
  data/methylation_top.pkl   to recover the probe order (column names)
  results/clock.joblib       the fitted model, to find nonzero-coefficient CpGs

Writes:
  results/clock_overlap.json
  data/raw/Hannum2013.csv, data/raw/Horvath2013.csv   (cached downloads, gitignored)

Prints a summary including a hypergeometric enrichment test.
"""

import json
import os
import urllib.request

import numpy as np
import pandas as pd
import joblib
from scipy.stats import hypergeom

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
RESULTS_DIR = os.path.join(ROOT, "results")

# biolearn reference clock files (raw GitHub). These mirror the published
# coefficient tables. We read only the first column, CpGmarker.
REFS = {
    "Hannum2013": {
        "url": "https://raw.githubusercontent.com/bio-learn/biolearn/master/biolearn/data/Hannum.csv",
        "expected_n": 71,
    },
    "Horvath2013": {
        "url": "https://raw.githubusercontent.com/bio-learn/biolearn/master/biolearn/data/Horvath1.csv",
        "expected_n": 353,
    },
}


def load_reference(name, info):
    """Download (or reuse cached) a clock CSV and return its set of CpG markers."""
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, name + ".csv")
    if not os.path.exists(path):
        print("[overlap] downloading {} reference from biolearn".format(name))
        urllib.request.urlretrieve(info["url"], path)
    df = pd.read_csv(path)
    # First column is CpGmarker; drop an intercept row if one is present.
    markers = df.iloc[:, 0].astype(str)
    markers = markers[markers.str.lower() != "intercept"]
    cpgs = set(markers.tolist())
    if len(cpgs) != info["expected_n"]:
        print("[overlap] WARNING: {} has {} markers, expected {}".format(
            name, len(cpgs), info["expected_n"]))
    return cpgs


def main():
    # Recover the probe order and the fitted model.
    x = pd.read_pickle(os.path.join(DATA_DIR, "methylation_top.pkl"))
    probes = list(x.columns)
    bundle = joblib.load(os.path.join(RESULTS_DIR, "clock.joblib"))
    model = bundle["model"]
    coef = np.asarray(model.coef_).ravel()

    candidate_pool = set(probes)                              # probes the model could pick from
    selected = {p for p, c in zip(probes, coef) if c != 0.0}  # the CpGs it kept
    M = len(candidate_pool)
    N = len(selected)
    print("[overlap] candidate pool: {} probes; model selected: {}".format(M, N))

    report = {"n_candidate_pool": M, "n_selected": N, "clocks": {}}

    for name, info in REFS.items():
        ref = load_reference(name, info)
        ref_total = len(ref)
        # Only clock CpGs that survived into the candidate pool could be picked.
        ref_in_pool = ref & candidate_pool
        n_in_pool = len(ref_in_pool)
        overlap = selected & ref
        k = len(overlap)
        # Expected overlap if N picks were drawn at random from the pool, plus
        # the hypergeometric P(X >= k). Background is the candidate pool, which
        # conditions on what was actually selectable (the variance filter that
        # built the pool is unsupervised, so it does not peek at age).
        expected = (N * n_in_pool / M) if M else 0.0
        if n_in_pool > 0 and N > 0:
            p_value = float(hypergeom.sf(k - 1, M, n_in_pool, N))
        else:
            p_value = float("nan")
        fold = (k / expected) if expected > 0 else None

        report["clocks"][name] = {
            "clock_total_cpgs": ref_total,
            "clock_cpgs_in_candidate_pool": n_in_pool,
            "overlap_with_selected": k,
            "overlap_cpgs": sorted(overlap),
            "expected_overlap_by_chance": round(expected, 3),
            "fold_enrichment": round(fold, 2) if fold is not None else None,
            "hypergeometric_p": p_value,
        }

        print("\n[overlap] {}: {} clock CpGs, {} present in the pool".format(
            name, ref_total, n_in_pool))
        print("[overlap]   selected and clock share {} CpGs (expected about {:.2f} by chance)".format(
            k, expected))
        if fold is not None:
            print("[overlap]   fold enrichment {:.2f}x, hypergeometric p = {:.3g}".format(fold, p_value))
        if overlap:
            print("[overlap]   shared CpGs: {}".format(", ".join(sorted(overlap))))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "clock_overlap.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("\n[overlap] wrote results/clock_overlap.json")


if __name__ == "__main__":
    main()
