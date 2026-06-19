"""
Draw the two result figures from results/predictions.csv.

Outputs:
  results/predicted_vs_actual.png   how close predicted age is to real age
  results/residual_hist.png         the spread of age acceleration

Run train_clock.py first, since this reads the predictions it writes.
"""

import json
import os

import matplotlib

# Use the non-interactive "Agg" backend so the script saves PNG files without
# needing a display. This matters when running headless, with no GUI.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS_DIR = os.path.join(ROOT, "results")


def main():
    pred_path = os.path.join(RESULTS_DIR, "predictions.csv")
    if not os.path.exists(pred_path):
        raise SystemExit("results/predictions.csv not found. Run train_clock.py first.")
    df = pd.read_csv(pred_path)

    # Load the metrics if present, just to annotate the scatter title.
    metrics = {}
    mpath = os.path.join(RESULTS_DIR, "metrics.json")
    if os.path.exists(mpath):
        with open(mpath, encoding="utf-8") as f:
            metrics = json.load(f)

    # Figure 1: predicted age vs actual age. The dashed y = x line is perfect
    # prediction; points hugging it mean the clock is accurate.
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df["age"], df["predicted"], s=18, alpha=0.6, edgecolor="none")
    lo = min(df["age"].min(), df["predicted"].min()) - 2  # pad the axes a little
    hi = max(df["age"].max(), df["predicted"].max()) + 2
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1, linestyle="--", label="y = x")
    ax.set_xlabel("Chronological age (years)")
    ax.set_ylabel("Predicted age (years)")
    title = "Methylation age clock: predicted vs actual"
    if metrics:
        title += "\nMAE {:.2f}y, r {:.3f}, n_test {}".format(
            metrics.get("test_mae_years", float("nan")),
            metrics.get("test_pearson_r", float("nan")),
            metrics.get("n_test", "?"),
        )
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "predicted_vs_actual.png"), dpi=150)
    plt.close(fig)

    # Figure 2: the residual (predicted minus actual) distribution. This is the
    # age-acceleration signal. The dashed line at 0 means predicted matches real
    # age; people to the right look molecularly older than their calendar age.
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["residual"], bins=20, alpha=0.8, edgecolor="white")
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Age acceleration: predicted minus actual (years)")
    ax.set_ylabel("Samples")
    ax.set_title("Residual distribution (positive = looks older than calendar age)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "residual_hist.png"), dpi=150)
    plt.close(fig)

    print("[plots] wrote results/predicted_vs_actual.png and results/residual_hist.png")


if __name__ == "__main__":
    main()
