"""
Aging Clock: interactive DNA methylation biological-age demo.

This Streamlit app puts the trained elastic-net clock behind sliders. Each slider
is one of the most influential CpG sites the model selected; moving it changes
that site's methylation level and the predicted age updates live. Every other
selected site is held at its population average, so the starting prediction is
the average age in the training cohort.

The math is exactly the trained model:
    age = intercept + sum_i ((beta_i - mean_i) / scale_i) * coef_i
The coefficients, means, and scales come from app_data/clock_app_data.json, which
is exported from the fitted model by src/export_dashboard.py.

This is a research and portfolio demo, not a medical or diagnostic tool.
"""

import json
import os

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "app_data", "clock_app_data.json")

BLUE = "#2c7fb8"


@st.cache_data
def load_data():
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def predict(slider_values, slider_feats, intercept):
    """age = intercept + sum over shown sliders of ((v - mean) / scale) * coef.
    Selected sites not exposed as sliders stay at their mean and contribute zero."""
    total = intercept
    for f in slider_feats:
        v = slider_values.get(f["cg"])
        if v is None:
            continue
        total += ((v - f["mean"]) / f["scale"]) * f["coef"]
    return total


def main():
    st.set_page_config(page_title="Aging Clock", layout="wide")
    data = load_data()
    feats = data["features"]
    intercept = data["intercept"]
    n_sliders = data.get("n_sliders", min(12, len(feats)))
    slider_feats = feats[:n_sliders]
    metrics = data.get("metrics", {})

    st.title("Aging Clock: DNA Methylation Biological-Age Predictor")
    st.write(
        "An interactive demo of an elastic-net model that predicts age from DNA "
        "methylation. It was trained on 656 whole-blood samples and selected a "
        "sparse signature of {} CpG sites. Move the sliders to change methylation "
        "at the most influential sites and watch the predicted age respond. Every "
        "other site is held at its population average, so the starting value is "
        "the average age in the cohort.".format(data["n_selected"])
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test MAE", "{:.2f} yrs".format(metrics.get("test_mae_years", float("nan"))))
    c2.metric("Correlation r", "{:.3f}".format(metrics.get("test_pearson_r", float("nan"))))
    c3.metric("CpG sites used", "{}".format(data["n_selected"]))
    c4.metric("Held-out people", "{}".format(metrics.get("n_test", len(data["test"]["age"]))))

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Try the clock")
        st.caption(
            "Each slider is one CpG site (methylation 0 to 1). The label shows how "
            "higher methylation moves the predicted age."
        )
        slider_values = {}
        for f in slider_feats:
            lo = max(0.0, f["mean"] - 3 * f["scale"])
            hi = min(1.0, f["mean"] + 3 * f["scale"])
            if hi - lo < 0.05:  # guarantee a usable range
                lo = max(0.0, f["mean"] - 0.1)
                hi = min(1.0, f["mean"] + 0.1)
            direction = "higher = older" if f["coef"] > 0 else "higher = younger"
            slider_values[f["cg"]] = st.slider(
                "{}  ({})".format(f["cg"], direction),
                min_value=round(float(lo), 3),
                max_value=round(float(hi), 3),
                value=round(float(f["mean"]), 3),
                step=0.001,
            )
        age = predict(slider_values, slider_feats, intercept)
        delta = age - data["baseline_prediction"]
        st.metric(
            "Predicted biological age",
            "{:.1f} years".format(age),
            delta="{:+.1f} vs cohort average".format(delta),
        )

    with right:
        st.subheader("How accurate is it")
        test = data["test"]
        if test["age"]:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(test["age"], test["predicted"], s=20, alpha=0.6,
                       color=BLUE, edgecolor="none")
            lo = min(min(test["age"]), min(test["predicted"])) - 2
            hi = max(max(test["age"]), max(test["predicted"])) + 2
            ax.plot([lo, hi], [lo, hi], "--", color="black", lw=1, label="perfect")
            ax.set_xlabel("Actual age (years)")
            ax.set_ylabel("Predicted age (years)")
            ax.set_title("Held-out test set")
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig)
        st.caption(
            "Each point is a person the model never saw in training. Closer to the "
            "dashed line is a better prediction. MAE {:.2f} years, r {:.3f}.".format(
                metrics.get("test_mae_years", float("nan")),
                metrics.get("test_pearson_r", float("nan")),
            )
        )

    st.divider()
    st.subheader("What the model learned")
    bar = pd.DataFrame(
        {"cg": [f["cg"] for f in slider_feats],
         "impact (years, unmethylated to methylated)": [f["impact"] for f in slider_feats]}
    ).set_index("cg")
    st.bar_chart(bar, height=320)
    st.caption(
        "Impact is how far predicted age shifts as a site goes from fully "
        "unmethylated to fully methylated (coefficient divided by feature scale). "
        "Positive means higher methylation predicts older."
    )

    overlap = data.get("overlap")
    if overlap and overlap.get("clocks", {}).get("Hannum2013"):
        han = overlap["clocks"]["Hannum2013"]
        st.divider()
        st.subheader("Does it match known aging biology")
        k = han.get("overlap_with_selected")
        fold = han.get("fold_enrichment")
        p = han.get("hypergeometric_p")
        msg = (
            "The model picked its sites from a high-variance shortlist with no "
            "knowledge of the published clocks. Of the Hannum 2013 clock sites "
            "available to it, it independently selected {}".format(k)
        )
        if fold is not None and p is not None:
            msg += ", about {:.0f}x more overlap than chance (hypergeometric p = {:.2g})".format(fold, p)
        msg += ". That is evidence it rediscovered real age-related sites rather than fitting noise."
        st.write(msg)

    st.divider()
    st.caption(
        "Trained on GSE40279 (Hannum 2013, whole blood). Research and portfolio "
        "demo only, not a medical or diagnostic tool. Code: "
        "github.com/brandonduplantier-juice/aging-clock"
    )


if __name__ == "__main__":
    main()
