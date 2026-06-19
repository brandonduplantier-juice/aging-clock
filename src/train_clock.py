"""
Train a DNA methylation age clock with cross-validated elastic net.

In plain terms: this learns to predict a person's age from the methylation levels
at thousands of CpG sites, then measures how well it does on people it never saw
during training. The gap between predicted and real age (the residual) is our
estimate of "age acceleration."

Reads:
  data/methylation_top.pkl   samples x probes beta values (from download_data.py)
  data/meta.csv              one row per sample: sample, age, sex

Writes:
  results/metrics.json       test MAE, RMSE, Pearson r, and model details
  results/predictions.csv    per test sample: age, predicted, residual
  results/clock.joblib       the fitted scaler and model, saved for reuse
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

# Paths, resolved relative to this file so the script runs from anywhere.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
SEED = 42  # fixed so the split and the model come out the same on every run


def main():
    # Load the prepared feature matrix (rows = people, columns = CpG sites) and
    # the metadata table that holds each person's age.
    x = pd.read_pickle(os.path.join(DATA_DIR, "methylation_top.pkl"))
    meta = pd.read_csv(os.path.join(DATA_DIR, "meta.csv"), index_col="sample")

    # Keep only samples present in both tables, in a consistent order, so each
    # row of features lines up with the correct age.
    common = x.index.intersection(meta.index)
    x = x.loc[common]
    y = meta.loc[common, "age"].astype(float).values  # the target: chronological age
    print(f"[train] {x.shape[0]} samples, {x.shape[1]} features")

    # Hold out 20 percent of people as a test set the model never sees while
    # training. The test score is the honest measure of how well it generalizes.
    x_train, x_test, y_train, y_test = train_test_split(
        x.values, y, test_size=0.2, random_state=SEED
    )

    # Standardize features to mean 0 and spread 1. We fit the scaler on the
    # training data only and then apply it to the test data. Fitting it on
    # everything would leak test information into training.
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)

    # Elastic net regression with built-in cross-validation.
    #   l1_ratio grid: how much L1 (drives coefficients to zero, selecting CpGs)
    #                  versus L2 (smooth shrinkage that handles correlated CpGs).
    #   cv=5: split the training data into 5 folds to choose the penalty strength
    #         without ever touching the test set.
    # Elastic net is the standard tool here because we have far more features
    # (CpGs) than samples, and many of them move together.
    model = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
        cv=5,
        max_iter=5000,
        random_state=SEED,
        n_jobs=-1,  # use all CPU cores
    )
    print("[train] fitting elastic net (cross-validated, this can take a few minutes)")
    model.fit(x_train_s, y_train)

    # Predict ages for the held-out people and score the predictions.
    pred = model.predict(x_test_s)
    mae = float(np.mean(np.abs(pred - y_test)))            # average miss, in years
    rmse = float(np.sqrt(np.mean((pred - y_test) ** 2)))   # like MAE, punishes big misses
    r, _ = pearsonr(pred, y_test)                          # how well predictions track real age
    n_selected = int(np.sum(model.coef_ != 0))             # CpGs the model actually uses

    # Record everything needed to judge and reproduce the result.
    metrics = {
        "dataset": "GSE40279 (Hannum 2013, whole blood, 450K)",
        "n_samples_total": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "n_test": int(len(y_test)),
        "test_mae_years": round(mae, 3),
        "test_rmse_years": round(rmse, 3),
        "test_pearson_r": round(float(r), 4),
        "n_cpgs_selected": n_selected,
        "chosen_l1_ratio": float(model.l1_ratio_),  # the l1_ratio cross-validation picked
        "chosen_alpha": float(model.alpha_),        # the penalty strength it picked
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Per-sample predictions. The residual (predicted minus actual) is the
    # age-acceleration estimate: positive means the methylome looks older than
    # the person's calendar age.
    out = pd.DataFrame(
        {"age": y_test, "predicted": pred, "residual": pred - y_test}
    )
    out.to_csv(os.path.join(RESULTS_DIR, "predictions.csv"), index=False)

    # Save the scaler and model together so the clock can be reloaded and applied
    # to new data later without retraining.
    joblib.dump({"scaler": scaler, "model": model}, os.path.join(RESULTS_DIR, "clock.joblib"))

    print("[train] test MAE  : {:.2f} years".format(mae))
    print("[train] test RMSE : {:.2f} years".format(rmse))
    print("[train] test r    : {:.3f}".format(r))
    print("[train] CpGs used : {} of {}".format(n_selected, x.shape[1]))
    print("[train] wrote results/metrics.json, predictions.csv, clock.joblib")


if __name__ == "__main__":
    main()
