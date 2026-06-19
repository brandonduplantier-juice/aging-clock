"""
Export a small, self-contained data file for the Streamlit dashboard.

The deployed dashboard cannot load the large training matrix
(data/methylation_top.pkl) or the model pickle (results/clock.joblib), because
both are gitignored. This script distills everything the dashboard needs into one
committed JSON file.

Because the model is a linear elastic net on standardized features, a prediction
is exactly:
    age = intercept + sum_i ((beta_i - mean_i) / scale_i) * coef_i
so the dashboard reproduces the trained model with plain arithmetic, with no
scikit-learn at runtime.

Reads:
  data/methylation_top.pkl     to recover the CpG (probe) order
  results/clock.joblib         the fitted scaler and elastic-net model
  results/predictions.csv      held-out test predictions (age, predicted)
  results/metrics.json         headline metrics
  results/clock_overlap.json   optional, the published-clock overlap analysis

Writes:
  app_data/clock_app_data.json
"""

import json
import os

import numpy as np
import pandas as pd
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
APP_DIR = os.path.join(ROOT, "app_data")

# How many of the most influential sites the dashboard turns into sliders.
N_SLIDERS = 12


def main():
    # Recover the CpG order from the training matrix columns.
    x = pd.read_pickle(os.path.join(DATA_DIR, "methylation_top.pkl"))
    probes = list(x.columns)

    bundle = joblib.load(os.path.join(RESULTS_DIR, "clock.joblib"))
    scaler = bundle["scaler"]
    model = bundle["model"]
    coef = np.asarray(model.coef_, dtype=float).ravel()
    intercept = float(model.intercept_)
    mean = np.asarray(scaler.mean_, dtype=float).ravel()
    scale = np.asarray(scaler.scale_, dtype=float).ravel()

    if not (len(probes) == len(coef) == len(mean) == len(scale)):
        raise SystemExit("Length mismatch between probes, coefficients, and scaler arrays.")

    # Keep only the CpGs the elastic net actually used (nonzero coefficient).
    features = []
    for i, c in enumerate(coef):
        if c != 0.0:
            sc = scale[i] if scale[i] != 0 else 1.0
            features.append({
                "cg": str(probes[i]),
                "coef": float(c),
                "mean": float(mean[i]),
                "scale": float(sc),
                # Sensitivity of predicted age to this site across the full 0..1
                # methylation range: d(age)/d(beta) = coef / scale.
                "impact": float(c / sc),
            })
    # Sort by absolute impact so the dashboard surfaces the most influential first.
    features.sort(key=lambda f: abs(f["impact"]), reverse=True)

    # Held-out test predictions for the accuracy scatter.
    test = {"age": [], "predicted": []}
    pred_path = os.path.join(RESULTS_DIR, "predictions.csv")
    if os.path.exists(pred_path):
        pred = pd.read_csv(pred_path)
        test["age"] = [float(v) for v in pred["age"].tolist()]
        test["predicted"] = [float(v) for v in pred["predicted"].tolist()]

    # Headline metrics.
    metrics = {}
    mpath = os.path.join(RESULTS_DIR, "metrics.json")
    if os.path.exists(mpath):
        with open(mpath, encoding="utf-8") as f:
            metrics = json.load(f)

    # Optional published-clock overlap finding.
    overlap = None
    opath = os.path.join(RESULTS_DIR, "clock_overlap.json")
    if os.path.exists(opath):
        with open(opath, encoding="utf-8") as f:
            overlap = json.load(f)

    ages = test["age"] if test["age"] else [20.0, 100.0]
    out = {
        "intercept": intercept,
        "baseline_prediction": intercept,   # prediction when every site sits at its mean
        "n_features_total": int(len(coef)),
        "n_selected": int(len(features)),
        "n_sliders": int(min(N_SLIDERS, len(features))),
        "age_min": float(min(ages)),
        "age_max": float(max(ages)),
        "metrics": metrics,
        "overlap": overlap,
        "features": features,
        "test": test,
    }

    os.makedirs(APP_DIR, exist_ok=True)
    with open(os.path.join(APP_DIR, "clock_app_data.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("[export] wrote app_data/clock_app_data.json")
    print("[export] selected sites: {}, sliders: {}".format(out["n_selected"], out["n_sliders"]))
    print("[export] baseline (all-mean) predicted age: {:.1f} years".format(intercept))


if __name__ == "__main__":
    main()
