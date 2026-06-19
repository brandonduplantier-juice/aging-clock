"""
Aging Clock: interactive DNA methylation biological-age demo.

Controls live in the sidebar; results fill the main page. Each sidebar slider is
one influential CpG site. Moving it changes that site's methylation and the
predicted age updates live. Every other selected site is held at its population
average, so the starting prediction is the average age in the training cohort.

The math is exactly the trained model:
    age = intercept + sum_i ((beta_i - mean_i) / scale_i) * coef_i
The coefficients, means, and scales come from app_data/clock_app_data.json,
exported from the fitted model by src/export_dashboard.py.

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

NAVY = "#1F3864"


@st.cache_data
def load_data():
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def predict(slider_values, slider_feats, intercept):
    """age = intercept + sum over shown sliders of ((v - mean) / scale) * coef."""
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
    baseline = data["baseline_prediction"]
    n_sliders = data.get("n_sliders", min(12, len(feats)))
    slider_feats = feats[:n_sliders]
    metrics = data.get("metrics", {})
    mae = metrics.get("test_mae_years", float("nan"))
    r = metrics.get("test_pearson_r", float("nan"))
    n_test = metrics.get("n_test", len(data["test"]["age"]))

    # ---------- Sidebar: the controls ----------
    with st.sidebar:
        st.header("Adjust the DNA sites")
        st.caption(
            "Methylation is a chemical tag on DNA that shifts as a body ages. Each "
            "slider below is one DNA position, called a CpG site, set from 0 (no "
            "tag) to 1 (fully tagged). Move one and the estimate on the right "
            "updates. Everything starts at the population average."
        )
        slider_values = {}
        for i, f in enumerate(slider_feats, start=1):
            lo = max(0.0, f["mean"] - 3 * f["scale"])
            hi = min(1.0, f["mean"] + 3 * f["scale"])
            if hi - lo < 0.05:
                lo = max(0.0, f["mean"] - 0.1)
                hi = min(1.0, f["mean"] + 0.1)
            direction = "higher = older" if f["coef"] > 0 else "higher = younger"
            slider_values[f["cg"]] = st.slider(
                "Site {} ({})".format(i, direction),
                min_value=round(float(lo), 3),
                max_value=round(float(hi), 3),
                value=round(float(f["mean"]), 3),
                step=0.001,
                help="CpG site {}".format(f["cg"]),
            )

    age = predict(slider_values, slider_feats, intercept)
    delta = age - baseline

    # ---------- Main ----------
    st.title("Aging Clock")
    st.markdown("##### Estimating biological age from the chemical tags on DNA")
    st.write(
        "This is a working machine-learning model. It learned to read age from DNA "
        "methylation, the pattern of chemical tags that builds up and fades as a "
        "body ages, using blood samples from 656 people. Use the sliders on the "
        "left to change the methylation at the sites it relies on most, and watch "
        "the estimate move."
    )

    if abs(delta) < 0.05:
        delta_text = "Right at the cohort average of {:.0f} years".format(baseline)
    else:
        delta_text = "{:+.1f} years vs the cohort average of {:.0f}".format(delta, baseline)
    st.markdown(
        '<div style="background:linear-gradient(135deg,#1F3864,#33518A);'
        'color:#ffffff;padding:24px 28px;border-radius:14px;margin:6px 0 4px 0;">'
        '<div style="font-size:13px;letter-spacing:.08em;text-transform:uppercase;'
        'opacity:.8;">Estimated biological age</div>'
        '<div style="font-size:54px;font-weight:700;line-height:1.05;margin:2px 0;">'
        '{:.1f} <span style="font-size:22px;font-weight:500;opacity:.85;">years</span>'
        '</div><div style="font-size:15px;opacity:.92;">{}</div></div>'.format(age, delta_text),
        unsafe_allow_html=True,
    )
    st.caption(
        "The estimate starts at the average age of the people the model learned "
        "from. Each slider you move adds to or subtracts from that, the same way "
        "the model weighs each site."
    )

    st.divider()
    st.subheader("How good is the model")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Typical error", "{:.1f} yrs".format(mae),
              help="On average the estimate is off by about this many years. Lower is better.")
    c2.metric("Tracks real age", "{:.2f} / 1.0".format(r),
              help="How closely the estimate follows real age across people. 1.0 would be perfect.")
    c3.metric("DNA sites used", "{}".format(data["n_selected"]),
              help="The model narrowed thousands of candidate sites down to these.")
    c4.metric("Tested on", "{} people".format(n_test),
              help="People held out of training and used only to test the model.")

    st.divider()
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Estimate vs real age")
        test = data["test"]
        if test["age"]:
            fig, ax = plt.subplots(figsize=(5.4, 5.0))
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")
            ax.scatter(test["age"], test["predicted"], s=26, alpha=0.55,
                       color=NAVY, edgecolor="white", linewidth=0.4)
            lo = min(min(test["age"]), min(test["predicted"])) - 2
            hi = max(max(test["age"]), max(test["predicted"])) + 2
            ax.plot([lo, hi], [lo, hi], "--", color="#888888", lw=1.2, label="a perfect estimate")
            ax.set_xlabel("Real age (years)")
            ax.set_ylabel("Estimated age (years)")
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.grid(True, color="#EEEEEE", linewidth=0.8)
            ax.set_axisbelow(True)
            ax.legend(frameon=False, loc="upper left")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
        st.caption(
            "Each dot is one person the model never saw while learning. The closer "
            "a dot sits to the dashed line, the closer the estimate was to that "
            "person's real age."
        )

    with right:
        st.subheader("What the model pays attention to")
        bar = pd.DataFrame(
            {"site": ["Site {}".format(i) for i in range(1, len(slider_feats) + 1)],
             "years it can shift the estimate": [f["impact"] for f in slider_feats]}
        ).set_index("site")
        st.bar_chart(bar, height=360, color=NAVY)
        st.caption(
            "How far each site can move the estimate as it goes from no methylation "
            "to full. Bars above zero push the estimate older, below zero push it "
            "younger. These are the sliders on the left, in the same order."
        )

    overlap = data.get("overlap")
    if overlap and overlap.get("clocks", {}).get("Hannum2013"):
        han = overlap["clocks"]["Hannum2013"]
        st.divider()
        st.subheader("Does it match real biology")
        k = han.get("overlap_with_selected")
        fold = han.get("fold_enrichment")
        msg = (
            "The model chose its DNA sites on its own, without ever being shown the "
            "established published aging clocks. It still independently landed on "
            "{} of the sites used by the well-known Hannum clock".format(k)
        )
        if fold is not None:
            msg += ", about {:.0f} times more overlap than chance would give".format(fold)
        msg += (". In plain terms, it rediscovered real aging biology rather than "
                "memorizing noise in the data.")
        st.write(msg)

    with st.expander("How this works, in detail"):
        st.write(
            "The model is an elastic-net linear regression trained on GSE40279 "
            "(Hannum 2013), 656 whole-blood samples measured on the Illumina 450K "
            "array. From 20,000 candidate sites it kept a sparse signature of {} "
            "CpG sites. Because it is linear, an estimate is exactly the intercept "
            "plus, for each site, its methylation minus the training mean, divided "
            "by the training spread, times the site's weight.".format(data["n_selected"])
        )
        st.write(
            "This demo exposes the {} most influential sites as sliders and holds "
            "every other selected site at its population average, so the starting "
            "estimate equals the cohort average age. It shows how the estimate "
            "responds to its biggest levers, not a full personal readout.".format(len(slider_feats))
        )
        st.write(
            "Held-out accuracy: average error {:.2f} years, correlation with real "
            "age {:.3f}, measured on {} people kept out of training.".format(mae, r, n_test)
        )

    st.divider()
    st.caption(
        "Trained on GSE40279 (Hannum 2013, whole blood). Research and portfolio "
        "demo only, not a medical or diagnostic tool. "
        "Code: github.com/brandonduplantier-juice/aging-clock"
    )


if __name__ == "__main__":
    main()
