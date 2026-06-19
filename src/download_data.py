"""
Download GSE40279 (Hannum 2013 whole-blood 450K methylation) and reduce it to a
compact, model-ready matrix.

Outputs:
  data/methylation_top.pkl  samples x N_PROBES beta values (float32)
  data/meta.csv             one row per sample: sample, age, sex

Design notes:
  - The raw matrix is roughly 656 samples by 473k probes. We keep only the
    top-variance probes so the rest of the pipeline runs on a normal machine.
  - Variance ranking is unsupervised. It never looks at age, so it does not leak
    the label into the held-out test set.
  - The exact metadata key that holds age can vary by series. We search the
    sample characteristics for an age-like field rather than hardcoding a key.
    If parsing fails on first run, print one sample's characteristics and adjust
    AGE_PATTERNS below. Do not guess ages.
"""

import os
import re
import sys

import numpy as np
import pandas as pd

try:
    import GEOparse
except ImportError:
    sys.exit("GEOparse is not installed. Run: pip install -r requirements.txt")

GSE_ID = "GSE40279"
N_PROBES = 20000            # raise for more signal and slower runs
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")

# Regexes tried in order against each sample's characteristic strings.
AGE_PATTERNS = [
    re.compile(r"age\s*\(y\)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
    re.compile(r"\bage\b\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
]
SEX_PATTERN = re.compile(r"\b(gender|sex)\b\s*[:=]\s*([a-z]+)", re.IGNORECASE)


def parse_age(characteristics):
    """Return float age or None from a list of 'key: value' strings."""
    for item in characteristics:
        for pat in AGE_PATTERNS:
            m = pat.search(item)
            if m:
                return float(m.group(1))
    return None


def parse_sex(characteristics):
    for item in characteristics:
        m = SEX_PATTERN.search(item)
        if m:
            return m.group(2).strip().lower()
    return "unknown"


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    print(f"[download] fetching {GSE_ID} (this is a large download, be patient)")
    gse = GEOparse.get_GEO(geo=GSE_ID, destdir=RAW_DIR, silent=True)

    # Per-sample metadata.
    rows = []
    for gsm_name, gsm in gse.gsms.items():
        chars = gsm.metadata.get("characteristics_ch1", [])
        age = parse_age(chars)
        sex = parse_sex(chars)
        rows.append({"sample": gsm_name, "age": age, "sex": sex})
    meta = pd.DataFrame(rows).set_index("sample")

    missing = int(meta["age"].isna().sum())
    if missing == len(meta):
        sample0 = next(iter(gse.gsms.values()))
        print("[download] could not parse any ages. One sample's characteristics:")
        print("  ", sample0.metadata.get("characteristics_ch1"))
        sys.exit("Adjust AGE_PATTERNS in download_data.py to match, then rerun.")
    if missing:
        print(f"[download] WARNING: {missing} samples had no parseable age, dropping them")

    # Beta matrix: probes (rows) x samples (cols). pivot_samples uses the VALUE
    # column of each sample table, which for this series is the beta value.
    print("[download] building beta matrix")
    betas = gse.pivot_samples("VALUE").astype(np.float32)   # index=probe, cols=sample

    # Align to samples that have an age, transpose to samples x probes.
    keep = meta.dropna(subset=["age"]).index
    keep = [s for s in keep if s in betas.columns]
    betas = betas[keep].T                                   # samples x probes
    meta = meta.loc[keep]

    # Drop probes with any missing values, then keep the top-variance probes.
    betas = betas.dropna(axis=1)
    print(f"[download] {betas.shape[0]} samples, {betas.shape[1]} complete probes")
    variances = betas.var(axis=0)
    top = variances.sort_values(ascending=False).head(N_PROBES).index
    betas = betas[top]
    print(f"[download] reduced to top {betas.shape[1]} variable probes")

    os.makedirs(DATA_DIR, exist_ok=True)
    betas.to_pickle(os.path.join(DATA_DIR, "methylation_top.pkl"))
    meta.to_csv(os.path.join(DATA_DIR, "meta.csv"))
    print("[download] wrote data/methylation_top.pkl and data/meta.csv")


if __name__ == "__main__":
    main()
