"""
Gene-set enrichment on the clock's selected genes.

Takes the genes the clock landed on (from annotate_genes.py) and asks whether they
are enriched for known biological processes and pathways, using Enrichr through the
gseapy library. This turns "176 CpGs" into "the signature is enriched for X aging
processes", which is the biological payoff.

Reads:
  results/gene_annotation.csv

Writes:
  results/enrichment.csv     top enriched terms with adjusted p-values
  results/enrichment.png     bar chart of the top terms (if matplotlib available)

Notes:
  Enrichr is a web service, so this step needs network access. If it is unreachable
  the script prints a clear message and exits without writing partial results. It
  never fabricates terms. Background is the standard Enrichr background.
"""

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS_DIR = os.path.join(ROOT, "results")

# Libraries queried. GO biological process is the headline; the others add pathway
# and aging-specific context. Names are Enrichr library identifiers.
GENE_SETS = [
    "GO_Biological_Process_2021",
    "KEGG_2021_Human",
    "Aging_Perturbations_from_GEO_up",
    "Aging_Perturbations_from_GEO_down",
]
TOP_N = 15


def main():
    ann_path = os.path.join(RESULTS_DIR, "gene_annotation.csv")
    if not os.path.exists(ann_path):
        sys.exit("results/gene_annotation.csv not found. Run annotate_genes.py first.")
    ann = pd.read_csv(ann_path)

    genes = sorted({g for cell in ann["genes"].dropna() for g in str(cell).split(";") if g})
    print("[enrich] {} unique genes from the signature".format(len(genes)))
    if len(genes) < 5:
        sys.exit("Too few genes to run enrichment meaningfully. Stopping.")

    try:
        import gseapy
    except ImportError:
        sys.exit("gseapy is not installed. Run: pip install gseapy")

    try:
        enr = gseapy.enrichr(
            gene_list=genes,
            gene_sets=GENE_SETS,
            organism="human",
            outdir=None,            # do not dump gseapy's own files
            no_plot=True,
        )
    except Exception as e:
        print("[enrich] Enrichr request failed: {}".format(repr(e)[:200]))
        print("[enrich] This step needs internet access to Enrichr (maayanlab.cloud).")
        print("[enrich] Nothing written. Rerun when online.")
        return

    res = enr.results.copy()
    res = res.sort_values("Adjusted P-value").reset_index(drop=True)
    keep = ["Gene_set", "Term", "Overlap", "P-value", "Adjusted P-value", "Genes"]
    keep = [c for c in keep if c in res.columns]
    res[keep].to_csv(os.path.join(RESULTS_DIR, "enrichment.csv"), index=False)
    print("[enrich] wrote results/enrichment.csv ({} terms)".format(len(res)))

    sig = res[res["Adjusted P-value"] < 0.05]
    print("[enrich] {} terms significant at adjusted p < 0.05".format(len(sig)))
    for _, r in res.head(8).iterrows():
        print("    [{:.1e}] {} :: {}".format(r["Adjusted P-value"], r["Gene_set"], r["Term"]))

    # Optional bar chart of the strongest terms.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        plot_df = res.head(TOP_N).iloc[::-1]
        labels = [t[:55] for t in plot_df["Term"]]
        neglogp = -np.log10(plot_df["Adjusted P-value"].clip(lower=1e-300))
        fig, ax = plt.subplots(figsize=(8, 0.45 * len(plot_df) + 1.5))
        ax.barh(range(len(plot_df)), neglogp, color="#1F3864")
        ax.set_yticks(range(len(plot_df)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("-log10 adjusted p-value")
        ax.set_title("Enriched terms in the clock's gene signature")
        ax.axvline(-np.log10(0.05), color="#888888", ls="--", lw=1)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, "enrichment.png"), dpi=150)
        print("[enrich] wrote results/enrichment.png")
    except Exception as e:
        print("[enrich] skipped plot: {}".format(repr(e)[:120]))


if __name__ == "__main__":
    main()
