"""
External validation: apply this clock, unchanged, to a second cohort (GSE87571).

The model is linear, so only its informative (nonzero-coefficient) sites affect the
prediction. We stream just those sites from the GEO series matrix and reconstruct
    age = intercept + sum_i ((beta_i - mean_i) / scale_i) * coef_i
over the sites present. Missing sites contribute nothing (training-mean imputation).
This is exact and avoids loading the full 485k-probe matrix.

Reads:
  data/methylation_top.pkl, results/clock.joblib
Writes:
  results/external_validation.json, results/external_validation.png
"""

import gzip
import json
import os
import re
import sys
import glob
import urllib.request

import numpy as np
import pandas as pd
import joblib
from scipy.stats import pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw_external")
RESULTS_DIR = os.path.join(ROOT, "results")
# Try these in order. Each is whole-blood 450K with age in the characteristics and
# betas in the series matrix (GSE87571 keeps betas in supplementary IDATs only, so
# it is not used here). First one that yields betas and ages is used.
EXT_CANDIDATES = ["GSE42861", "GSE41169"]

AGE_PATTERNS = [
    re.compile(r"age\s*\(y\)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
    re.compile(r"\bage\b\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
]


def parse_age(chars):
    for item in chars:
        for pat in AGE_PATTERNS:
            m = pat.search(item)
            if m:
                return float(m.group(1))
    return None


def matrix_url(gse):
    num = gse[3:]
    bucket = "GSE" + (num[:-3] if len(num) > 3 else "") + "nnn"
    return "https://ftp.ncbi.nlm.nih.gov/geo/series/{b}/{g}/matrix/{g}_series_matrix.txt.gz".format(b=bucket, g=gse)


def _progress(count, block, total):
    if total > 0 and count % 200 == 0:
        print("[extval]   downloaded {:.0f} MB".format(count * block / 1e6), end="\r")


def stream_series_matrix(path, wanted_probes, wanted_samples=None):
    wanted_probes = set(wanted_probes)
    betas, chars = {}, {}
    samples = None
    in_tab = False
    header = None
    keepidx, matsamples = [], []
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="latin-1", errors="ignore") as fh:
        for line in fh:
            if line.startswith("!Sample_geo_accession"):
                samples = [p.strip().strip('"') for p in line.rstrip("\n").split("\t")[1:]]
                continue
            if line.startswith("!Sample_characteristics_ch1"):
                vals = [p.strip().strip('"') for p in line.rstrip("\n").split("\t")[1:]]
                if samples:
                    for s, v in zip(samples, vals):
                        chars.setdefault(s, []).append(v)
                continue
            if line.startswith("!series_matrix_table_begin"):
                in_tab = True
                header = None
                continue
            if line.startswith("!series_matrix_table_end"):
                break
            if in_tab:
                parts = [p.strip().strip('"') for p in line.rstrip("\n").split("\t")]
                if header is None:
                    header = parts
                    matsamples = header[1:]
                    keepidx = [j for j, s in enumerate(matsamples)
                               if wanted_samples is None or s in wanted_samples]
                    continue
                cg = parts[0]
                if cg in wanted_probes:
                    for j in keepidx:
                        try:
                            betas.setdefault(matsamples[j], {})[cg] = float(parts[j + 1])
                        except (ValueError, IndexError):
                            continue
    return betas, chars


def stream_soft(path, wanted_probes, wanted_samples=None):
    wanted_probes = set(wanted_probes)
    betas, chars = {}, {}
    cur = None
    in_tab = False
    header_seen = False
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="latin-1", errors="ignore") as fh:
        for line in fh:
            if line.startswith("^SAMPLE"):
                cur = line.split("=", 1)[1].strip()
                in_tab = False
                header_seen = False
                continue
            if cur is None:
                continue
            if line.startswith("!Sample_characteristics_ch1"):
                if "=" in line:
                    chars.setdefault(cur, []).append(line.split("=", 1)[1].strip())
                continue
            if line.startswith("!sample_table_begin"):
                in_tab = True
                header_seen = False
                continue
            if line.startswith("!sample_table_end"):
                in_tab = False
                continue
            if in_tab:
                if not header_seen:
                    header_seen = True
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2 and parts[0] in wanted_probes:
                    try:
                        betas.setdefault(cur, {})[parts[0]] = float(parts[1])
                    except ValueError:
                        pass
    return betas, chars


def load_betas(raw_dir, gse, wanted_probes, wanted_samples=None):
    os.makedirs(raw_dir, exist_ok=True)
    mpath = os.path.join(raw_dir, "{}_series_matrix.txt.gz".format(gse))
    if not os.path.exists(mpath):
        url = matrix_url(gse)
        print("[extval] downloading series matrix: {}".format(url))
        try:
            urllib.request.urlretrieve(url, mpath, _progress)
            print()
        except Exception as e:
            print("\n[extval] series matrix download failed: {}".format(repr(e)[:120]))
    if os.path.exists(mpath):
        betas, chars = stream_series_matrix(mpath, wanted_probes, wanted_samples)
        if betas:
            return betas, chars
        print("[extval] series matrix had no usable table, trying cached SOFT")
    softs = glob.glob(os.path.join(raw_dir, "{}*.soft.gz".format(gse))) or \
        glob.glob(os.path.join(raw_dir, "*{}*soft*".format(gse)))
    if softs:
        print("[extval] streaming cached SOFT (slower): {}".format(os.path.basename(softs[0])))
        return stream_soft(softs[0], wanted_probes, wanted_samples)
    sys.exit("Could not obtain betas for {}.".format(gse))


def main():
    train_x = pd.read_pickle(os.path.join(DATA_DIR, "methylation_top.pkl"))
    train_probes = list(train_x.columns)
    bundle = joblib.load(os.path.join(RESULTS_DIR, "clock.joblib"))
    scaler, model = bundle["scaler"], bundle["model"]
    intercept = float(model.intercept_)

    info = {}
    for i, c in enumerate(model.coef_):
        if c != 0.0:
            sc = scaler.scale_[i] if scaler.scale_[i] != 0 else 1.0
            info[train_probes[i]] = (float(c), float(scaler.mean_[i]), float(sc))
    print("[extval] informative sites in the model: {}".format(len(info)))

    used_gse = None
    ages, preds, present_counts = [], [], []
    for gse in EXT_CANDIDATES:
        print("[extval] trying external cohort {}".format(gse))
        betas, chars = load_betas(RAW_DIR, gse, set(info.keys()), None)
        a, p, pc = [], [], []
        for sample, b in betas.items():
            age = parse_age(chars.get(sample, []))
            if age is None or not b:
                continue
            total = intercept
            present = 0
            for cg, (coef, mean, scale) in info.items():
                v = b.get(cg)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    total += ((v - mean) / scale) * coef
                    present += 1
            if present == 0:
                continue
            a.append(age); p.append(total); pc.append(present)
        if len(a) >= 20:                       # enough usable samples to be meaningful
            used_gse, ages, preds, present_counts = gse, a, p, pc
            print("[extval] using {}: {} usable samples".format(gse, len(a)))
            break
        print("[extval] {} unusable ({} samples with betas+age), trying next".format(gse, len(a)))

    if not ages:
        sys.exit("No candidate cohort yielded usable betas and ages: {}".format(EXT_CANDIDATES))

    ages = np.array(ages, dtype=float)
    preds = np.array(preds, dtype=float)
    mae = float(np.mean(np.abs(preds - ages)))
    rmse = float(np.sqrt(np.mean((preds - ages) ** 2)))
    r = float(pearsonr(preds, ages)[0])
    med_present = int(np.median(present_counts))

    out = {
        "external_dataset": "{} (whole blood, 450K)".format(used_gse),
        "n_samples": int(len(ages)),
        "external_mae_years": round(mae, 3),
        "external_rmse_years": round(rmse, 3),
        "external_pearson_r": round(r, 4),
        "informative_sites_present_median": med_present,
        "informative_sites_total": int(len(info)),
        "note": ("Model trained on GSE40279 applied with no refitting, using its "
                 "informative sites. Cohort may include disease cases; age is the target "
                 "regardless. Missing sites imputed at the training mean. "
                 "Cross-cohort normalization differences are expected to raise the "
                 "error relative to the within-cohort test."),
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "external_validation.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("[extval] external MAE {:.2f}  RMSE {:.2f}  r {:.3f}  (n={})".format(mae, rmse, r, len(ages)))
    print("[extval] informative sites present (median per sample): {} of {}".format(med_present, len(info)))
    print("[extval] wrote results/external_validation.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        lo = float(min(ages.min(), preds.min())) - 2
        hi = float(max(ages.max(), preds.max())) + 2
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([lo, hi], [lo, hi], "--", color="#888888", lw=1.2, label="perfect")
        ax.scatter(ages, preds, s=22, alpha=0.5, color="#1F3864")
        ax.set_xlabel("Real age (years)"); ax.set_ylabel("Predicted age (years)")
        ax.set_title("External cohort {}\nMAE {:.1f} yr, r {:.2f}, n {}".format(used_gse, mae, r, len(ages)))
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, "external_validation.png"), dpi=150)
        print("[extval] wrote results/external_validation.png")
    except Exception as e:
        print("[extval] skipped plot: {}".format(repr(e)[:120]))


if __name__ == "__main__":
    main()
