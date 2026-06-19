"""
Train a DNA methylation age clock with cross-validated elastic net.

Reads:
  data/methylation_top.pkl   samples x probes beta values
  data/meta.csv              sample, age, sex

Writes:
  results/metrics.json       test MAE, RMSE, Pearson r, n, n_features, n_selected
  results/predictions.csv    sample, age, predicted, residual (age acceleration)
  results/clock.joblib       fitted scaler + model

The residual (predicted minus actual age) is the age-acceleration estimate: a
positive value means the methylome looks older than the person's calendar age.
"""

import json
import os

import numpy as np
import pandas as pd
import joblib
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
SEED = 42


def main():
    x = pd.read_pickle(os.path.join(DATA_DIR, "methylation_top.pkl"))
    meta = pd.read_csv(os.path.join(DATA_DIR, "meta.csv"), index_col="sample")

    # Align rows.
    common = x.index.intersection(meta.index)
    x = x.loc[common]
    y = meta.loc[common, "age"].astype(float).values
    print(f"[train] {x.shape[0]} samples, {x.shape[1]} features")

    x_train, x_test, y_train, y_test = train_test_split(
        x.values, y, test_size=0.2, random_state=SEED
    )

    # Standardize on the training fold only, then apply to test.
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)

    # Cross-validated elastic net. The l1_ratio grid lets CV pick how sparse the
    # CpG signature should be; cv=5 selects alpha on the training fold only.
    model = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
        n_alphas=50,
        cv=5,
        max_iter=5000,
        random_state=SEED,
        n_jobs=-1,
    )
    print("[train] fitting elastic net (cross-validated, this can take a few minutes)")
    model.fit(x_train_s, y_train)

    pred = model.predict(x_test_s)
    mae = float(np.mean(np.abs(pred - y_test)))
    rmse = float(np.sqrt(np.mean((pred - y_test) ** 2)))
    r, _ = pearsonr(pred, y_test)
    n_selected = int(np.sum(model.coef_ != 0))

    metrics = {
        "dataset": "GSE40279 (Hannum 2013, whole blood, 450K)",
        "n_samples_total": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "n_test": int(len(y_test)),
        "test_mae_years": round(mae, 3),
        "test_rmse_years": round(rmse, 3),
        "test_pearson_r": round(float(r), 4),
        "n_cpgs_selected": n_selected,
        "chosen_l1_ratio": float(model.l1_ratio_),
        "chosen_alpha": float(model.alpha_),
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    out = pd.DataFrame(
        {"age": y_test, "predicted": pred, "residual": pred - y_test}
    )
    out.to_csv(os.path.join(RESULTS_DIR, "predictions.csv"), index=False)

    joblib.dump({"scaler": scaler, "model": model}, os.path.join(RESULTS_DIR, "clock.joblib"))

    print("[train] test MAE  : {:.2f} years".format(mae))
    print("[train] test RMSE : {:.2f} years".format(rmse))
    print("[train] test r    : {:.3f}".format(r))
    print("[train] CpGs used : {} of {}".format(n_selected, x.shape[1]))
    print("[train] wrote results/metrics.json, predictions.csv, clock.joblib")


if __name__ == "__main__":
    main()
