"""
Head-to-head benchmark: this clock vs the published Hannum 2013 clock.

Both clocks are scored on the same held-out 132 people from GSE40279. Our clock is
reapplied from the saved model. Hannum is applied from its published 71-CpG table
(a direct coefficient-times-methylation sum, no intercept, the standard form).

Reads:
  data/methylation_top.pkl, data/meta.csv, results/clock.joblib, ref/hannum_coefficients.csv
Writes:
  results/benchmark.json, results/benchmark.png

Data access:
  The Hannum sites sit mostly outside our top-variance 20k subset, so we read them
  from the GEO series matrix, which is one row per probe with samples as columns
  (about 485k rows, not the 485k-by-656 table GEOparse loads). We stream it once and
  keep only the 71 Hannum rows. The matrix is downloaded once to data/raw and reused.
  If it is unavailable we fall back to streaming the cached family SOFT file.
"""

import gzip
import json
import os
import sys
import glob
import urllib.request

import numpy as np
import pandas as pd
import joblib
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
RESULTS_DIR = os.path.join(ROOT, "results")
REF_DIR = os.path.join(ROOT, "ref")
SEED = 42
GSE_ID = "GSE40279"


def matrix_url(gse):
    num = gse[3:]
    bucket = "GSE" + (num[:-3] if len(num) > 3 else "") + "nnn"
    return "https://ftp.ncbi.nlm.nih.gov/geo/series/{b}/{g}/matrix/{g}_series_matrix.txt.gz".format(b=bucket, g=gse)


def _progress(count, block, total):
    if total > 0 and count % 200 == 0:
        mb = count * block / 1e6
        print("[bench]   downloaded {:.0f} MB".format(mb), end="\r")


def stream_series_matrix(path, wanted_probes, wanted_samples=None):
    """Single pass over a GEO series matrix. Returns (betas, chars)."""
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
    """Fallback: stream the family SOFT file. Returns (betas, chars)."""
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
                if wanted_samples is not None and cur not in wanted_samples:
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
        print("[bench] downloading series matrix: {}".format(url))
        try:
            urllib.request.urlretrieve(url, mpath, _progress)
            print()
        except Exception as e:
            print("\n[bench] series matrix download failed: {}".format(repr(e)[:120]))
    if os.path.exists(mpath):
        betas, chars = stream_series_matrix(mpath, wanted_probes, wanted_samples)
        if betas:
            return betas, chars
        print("[bench] series matrix had no usable table, trying cached SOFT")
    softs = glob.glob(os.path.join(raw_dir, "{}*.soft.gz".format(gse))) or \
        glob.glob(os.path.join(raw_dir, "*{}*soft*".format(gse)))
    if softs:
        print("[bench] streaming cached SOFT (slower): {}".format(os.path.basename(softs[0])))
        return stream_soft(softs[0], wanted_probes, wanted_samples)
    sys.exit("Could not obtain betas for {} (no series matrix and no cached SOFT).".format(gse))


def hannum_table():
    local = os.path.join(REF_DIR, "hannum_coefficients.csv")
    df = pd.read_csv(local) if os.path.exists(local) else None
    if df is None:
        import biolearn
        df = pd.read_csv(os.path.join(os.path.dirname(biolearn.__file__), "data", "Hannum.csv"))
    df.columns = [c.strip() for c in df.columns]
    cg_col = next((c for c in df.columns if "cpg" in c.lower() or c.lower() == "id"), df.columns[0])
    co_col = next((c for c in df.columns if "coef" in c.lower()), df.columns[-1])
    return dict(zip(df[cg_col].astype(str), df[co_col].astype(float)))


def main():
    x = pd.read_pickle(os.path.join(DATA_DIR, "methylation_top.pkl"))
    meta = pd.read_csv(os.path.join(DATA_DIR, "meta.csv"), index_col="sample")
    common = x.index.intersection(meta.index)
    x = x.loc[common]
    y = meta.loc[common, "age"].astype(float).values

    idx = np.arange(len(common))
    _, test_idx = train_test_split(idx, test_size=0.2, random_state=SEED)
    test_samples = list(common[test_idx])
    y_test = y[test_idx]
    print("[bench] reproduced held-out test set: {} people".format(len(test_samples)))

    bundle = joblib.load(os.path.join(RESULTS_DIR, "clock.joblib"))
    scaler, model = bundle["scaler"], bundle["model"]
    my_pred = model.predict(scaler.transform(x.loc[test_samples].values))
    my_mae = float(np.mean(np.abs(my_pred - y_test)))
    my_r = float(pearsonr(my_pred, y_test)[0])

    coeffs = hannum_table()
    print("[bench] Hannum clock CpGs: {}".format(len(coeffs)))
    betas, _ = load_betas(RAW_DIR, GSE_ID, set(coeffs.keys()), set(test_samples))
    sub = pd.DataFrame.from_dict(betas, orient="index").reindex(test_samples)
    present = [cg for cg in coeffs if cg in sub.columns]
    missing = [cg for cg in coeffs if cg not in sub.columns]
    print("[bench] Hannum CpGs present in cohort: {} of {} ({} absent)".format(
        len(present), len(coeffs), len(missing)))
    sub = sub[present].astype(float)
    sub = sub.fillna(sub.mean())
    coef_vec = np.array([coeffs[cg] for cg in present], dtype=float)
    hannum_pred = sub.values @ coef_vec
    han_mae = float(np.mean(np.abs(hannum_pred - y_test)))
    han_r = float(pearsonr(hannum_pred, y_test)[0])

    out = {
        "dataset": "GSE40279 (Hannum 2013, whole blood, 450K)",
        "test_n": int(len(y_test)),
        "this_clock": {"test_mae_years": round(my_mae, 3), "test_pearson_r": round(my_r, 4),
                       "n_cpgs": int(np.sum(model.coef_ != 0))},
        "hannum_2013": {"test_mae_years": round(han_mae, 3), "test_pearson_r": round(han_r, 4),
                        "n_cpgs_total": len(coeffs), "n_cpgs_present": len(present),
                        "n_cpgs_absent": len(missing)},
        "note": ("Both clocks scored on the identical held-out people. Hannum applied "
                 "as a direct coefficient-times-methylation sum with no intercept; "
                 "absent CpGs imputed at cohort mean."),
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "benchmark.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("[bench] this clock : MAE {:.2f}  r {:.3f}".format(my_mae, my_r))
    print("[bench] Hannum 2013: MAE {:.2f}  r {:.3f}".format(han_mae, han_r))
    print("[bench] wrote results/benchmark.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        lo = float(min(y_test.min(), my_pred.min(), hannum_pred.min())) - 2
        hi = float(max(y_test.max(), my_pred.max(), hannum_pred.max())) + 2
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([lo, hi], [lo, hi], "--", color="#888888", lw=1.2, label="perfect")
        ax.scatter(y_test, my_pred, s=26, alpha=0.6, color="#1F3864",
                   label="This clock (MAE {:.1f})".format(my_mae))
        ax.scatter(y_test, hannum_pred, s=26, alpha=0.5, color="#C8862C",
                   label="Hannum 2013 (MAE {:.1f})".format(han_mae))
        ax.set_xlabel("Real age (years)"); ax.set_ylabel("Predicted age (years)")
        ax.set_title("This clock vs Hannum 2013, same held-out people")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, "benchmark.png"), dpi=150)
        print("[bench] wrote results/benchmark.png")
    except Exception as e:
        print("[bench] skipped plot: {}".format(repr(e)[:120]))


if __name__ == "__main__":
    main()
