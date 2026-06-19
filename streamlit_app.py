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

import altair as alt
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "app_data", "clock_app_data.json")

NAVY = "#1F3864"
CHART_H = 360


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
    n_selected = data["n_selected"]

    # ---------- Sidebar: the controls ----------
    with st.sidebar:
        st.header("Adjust the DNA sites", help=(
            "These 12 sliders are the sites with the largest effect on the "
            "estimate. The model uses {} sites in total; the other {} stay at "
            "their average here so the page is not 12 plus sliders long."
        ).format(n_selected, n_selected - n_sliders))
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
            older = f["coef"] > 0
            direction = "higher = older" if older else "higher = younger"
            gene = (f.get("gene") or "").strip()
            if gene.lower() == "nan":
                gene = ""
            label = "Site {}: {} ({})".format(i, gene, direction) if gene \
                else "Site {} ({})".format(i, direction)
            gene_clause = " It sits in or near {}.".format(gene) if gene else ""
            help_text = (
                "CpG site {cg}.{gene_clause} Methylation here runs 0 (no chemical "
                "tag) to 1 (fully tagged). Raising it makes the model estimate {dir}. "
                "It starts at the population average of {mean:.2f}, and the slider "
                "spans the typical range seen across people ({lo:.2f} to {hi:.2f})."
            ).format(cg=f["cg"], gene_clause=gene_clause,
                     dir=("older" if older else "younger"),
                     mean=f["mean"], lo=lo, hi=hi)
            slider_values[f["cg"]] = st.slider(
                label,
                min_value=round(float(lo), 3),
                max_value=round(float(hi), 3),
                value=round(float(f["mean"]), 3),
                step=0.001,
                help=help_text,
            )

    age = predict(slider_values, slider_feats, intercept)
    delta = age - baseline

    # ---------- Main ----------
    st.title("Aging Clock", help=(
        "An interactive demo of a DNA methylation aging clock. Move the sliders on "
        "the left to see how the estimate responds to the sites the model weighs "
        "most. Built by Brandon Duplantier; trained on public data."))
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
    hero_title = (
        "Updates live as you move the sliders. It starts at {:.0f}, the average age "
        "of the 656 people the model trained on, because every site begins at its "
        "population average. Moving a slider adds that site's weighted effect."
    ).format(baseline)
    st.markdown(
        '<div title="{tip}" style="background:linear-gradient(135deg,#1F3864,#33518A);'
        'color:#ffffff;padding:24px 28px;border-radius:14px;margin:6px 0 4px 0;cursor:help;">'
        '<div style="font-size:13px;letter-spacing:.08em;text-transform:uppercase;'
        'opacity:.8;">Estimated biological age</div>'
        '<div style="font-size:54px;font-weight:700;line-height:1.05;margin:2px 0;">'
        '{age:.1f} <span style="font-size:22px;font-weight:500;opacity:.85;">years</span>'
        '</div><div style="font-size:15px;opacity:.92;">{delta}</div></div>'.format(
            tip=hero_title, age=age, delta=delta_text),
        unsafe_allow_html=True,
    )
    st.caption(
        "The estimate starts at the average age of the people the model learned "
        "from. Each slider you move adds to or subtracts from that, the same way "
        "the model weighs each site."
    )

    st.divider()
    st.subheader("How good is the model", help=(
        "All four numbers are measured on held-out people the model never trained "
        "on, not on the data it learned from, so they reflect performance on new "
        "people."))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Typical error", "{:.1f} yrs".format(mae), help=(
        "Mean absolute error. Across the {} held-out people, the average gap "
        "between the estimate and the person's real age. Lower is better. "
        "Published first-generation blood clocks reach about 3 to 4 years."
    ).format(n_test))
    c2.metric("Tracks real age", "{:.2f} / 1.0".format(r), help=(
        "Pearson correlation between estimated and real age across the held-out "
        "people. 1.0 would mean the estimate rises in perfect step with real age, "
        "0 would mean no relationship. 0.89 is a strong fit."))
    c3.metric("DNA sites used", "{}".format(n_selected), help=(
        "The model began with 20,000 candidate sites, the most variable ones, and "
        "the elastic-net penalty drove all but these {} to a weight of zero, so "
        "only these sites contribute to an estimate."
    ).format(n_selected))
    c4.metric("Tested on", "{} people".format(n_test), help=(
        "Randomly held out before training and never seen while the model learned, "
        "so their estimates are an honest measure of how it does on new people."))

    st.divider()
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Estimate vs real age", help=(
            "Each dot is one held-out person. On the dashed line the estimate was "
            "exact; above the line the model guessed too old, below it guessed too "
            "young. Hover a dot for its numbers."))
        test = data["test"]
        if test["age"]:
            df = pd.DataFrame({"Real age": test["age"], "Estimated age": test["predicted"]})
            df["Off by"] = df["Estimated age"] - df["Real age"]
            lo = min(min(test["age"]), min(test["predicted"])) - 2
            hi = max(max(test["age"]), max(test["predicted"])) + 2
            dom = [lo, hi]
            line_df = pd.DataFrame({"Real age": [lo, hi], "Estimated age": [lo, hi]})
            ref = alt.Chart(line_df).mark_line(
                strokeDash=[6, 4], color="#9AA3B2", size=1.5
            ).encode(
                x=alt.X("Real age:Q", scale=alt.Scale(domain=dom)),
                y=alt.Y("Estimated age:Q", scale=alt.Scale(domain=dom)),
            )
            pts = alt.Chart(df).mark_circle(
                size=70, opacity=0.5, color=NAVY
            ).encode(
                x=alt.X("Real age:Q", title="Real age (years)", scale=alt.Scale(domain=dom)),
                y=alt.Y("Estimated age:Q", title="Estimated age (years)", scale=alt.Scale(domain=dom)),
                tooltip=[alt.Tooltip("Real age:Q", title="Real age", format=".0f"),
                         alt.Tooltip("Estimated age:Q", title="Estimated age", format=".0f"),
                         alt.Tooltip("Off by:Q", title="Off by (years)", format="+.1f")],
            )
            chart = (ref + pts).properties(height=CHART_H).configure_view(strokeOpacity=0)
            st.altair_chart(chart, width='stretch')
        st.caption(
            "Each dot is one person the model never saw while learning. The closer "
            "a dot sits to the dashed line, the closer the estimate was to that "
            "person's real age. Hover a dot to see how far off it was."
        )

    with right:
        st.subheader("What the model pays attention to", help=(
            "The 12 sites with the largest effect on the estimate, the same ones on "
            "the sliders. Bar length is how many years the estimate moves as that "
            "site goes from no methylation to full. Hover a bar for its details."))
        ordered = ["Site {}".format(i) for i in range(1, len(slider_feats) + 1)]
        rows = []
        for i, f in enumerate(slider_feats, start=1):
            rows.append({
                "site": "Site {}".format(i),
                "impact": f["impact"],
                "effect": "higher methylation = older" if f["coef"] > 0 else "higher methylation = younger",
                "cg": f["cg"],
                "gene": (lambda g: g if g and g.lower() != "nan" else "not annotated")((f.get("gene") or "").strip()),
            })
        bdf = pd.DataFrame(rows)
        bars = alt.Chart(bdf).mark_bar(color=NAVY).encode(
            x=alt.X("site:N", sort=ordered, title=None, axis=alt.Axis(labelAngle=-40)),
            y=alt.Y("impact:Q", title="years it can shift the estimate"),
            tooltip=[alt.Tooltip("site:N", title="Slider"),
                     alt.Tooltip("gene:N", title="Gene"),
                     alt.Tooltip("cg:N", title="CpG site"),
                     alt.Tooltip("effect:N", title="Effect"),
                     alt.Tooltip("impact:Q", title="Max shift (years)", format="+.1f")],
        ).properties(height=CHART_H).configure_view(strokeOpacity=0)
        st.altair_chart(bars, width='stretch')
        st.caption(
            "How far each site can move the estimate as it goes from no methylation "
            "to full. Bars above zero push the estimate older, below zero push it "
            "younger. These are the sliders on the left, in the same order."
        )

    overlap = data.get("overlap")
    if overlap and overlap.get("clocks", {}).get("Hannum2013"):
        han = overlap["clocks"]["Hannum2013"]
        st.divider()
        st.subheader("Does it match real biology", help=(
            "A check that the chosen sites are genuine aging signal rather than "
            "noise the model happened to fit. It compares the model's sites against "
            "the published Hannum clock using a hypergeometric enrichment test."))
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
            "by the training spread, times the site's weight.".format(n_selected)
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
