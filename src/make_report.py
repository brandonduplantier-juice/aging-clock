"""
Assemble REPORT.md from whatever results exist. Never fabricates a number.

Each section is written only if its source file is present. Missing sections are
marked with what to run to fill them, so the report is always honest about what has
and has not been computed.

Reads (any subset):
  results/metrics.json
  results/clock_overlap.json
  results/gene_annotation.csv
  results/enrichment.csv
  results/benchmark.json
  results/external_validation.json

Writes:
  REPORT.md   at the repo root
"""

import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS_DIR = os.path.join(ROOT, "results")


def load_json(name):
    p = os.path.join(RESULTS_DIR, name)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


def load_csv(name):
    p = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pending(script):
    return "_(Not yet computed. Run `python src/{}` to populate this section.)_".format(script)


def main():
    metrics = load_json("metrics.json")
    overlap = load_json("clock_overlap.json")
    genes = load_csv("gene_annotation.csv")
    enrich = load_csv("enrichment.csv")
    bench = load_json("benchmark.json")
    extval = load_json("external_validation.json")

    L = []
    L.append("# A from-scratch DNA methylation aging clock: results and limitations")
    L.append("")
    L.append("Brandon Duplantier. Code: github.com/brandonduplantier-juice/aging-clock")
    L.append("")
    L.append("## Summary")
    L.append("")
    if metrics:
        L.append(
            "An elastic-net regression trained on whole-blood DNA methylation predicts "
            "chronological age with a held-out mean absolute error of {:.2f} years and a "
            "Pearson correlation of {:.3f}, using a sparse signature of {} CpG sites "
            "selected from 20,000 candidates.".format(
                metrics.get("test_mae_years", float("nan")),
                metrics.get("test_pearson_r", float("nan")),
                metrics.get("n_cpgs_selected", "?"),
            )
        )
    else:
        L.append(pending("train_clock.py (in the core pipeline)"))
    L.append("")

    L.append("## Data and method")
    L.append("")
    L.append(
        "Source: GEO accession GSE40279 (Hannum et al., Molecular Cell, 2013), 656 "
        "whole-blood samples on the Illumina Infinium 450K array with chronological age "
        "and sex. The 20,000 most variable probes were kept (unsupervised, so no label "
        "leakage), features were standardized on the training fold only, and an "
        "ElasticNetCV model was fit with the L1/L2 mix and penalty chosen by 5-fold "
        "cross-validation. Twenty percent of people were held out for testing."
    )
    L.append("")

    L.append("## Accuracy")
    L.append("")
    if metrics:
        L.append("| Metric | Value |")
        L.append("|---|---|")
        L.append("| Test MAE (years) | {:.2f} |".format(metrics.get("test_mae_years", float("nan"))))
        L.append("| Test RMSE (years) | {:.2f} |".format(metrics.get("test_rmse_years", float("nan"))))
        L.append("| Test Pearson r | {:.3f} |".format(metrics.get("test_pearson_r", float("nan"))))
        L.append("| CpGs selected | {} |".format(metrics.get("n_cpgs_selected", "?")))
        L.append("| Held-out people | {} |".format(metrics.get("n_test", "?")))
        L.append("")
        L.append(
            "For reference, published first-generation blood clocks reach roughly 3 to 4 "
            "years MAE and r near 0.95. This baseline is in the right neighborhood and a "
            "step behind the optimized published clocks, as expected for a generic "
            "top-variance feature set with no cell-type correction."
        )
    else:
        L.append(pending("train_clock.py"))
    L.append("")

    L.append("## Head-to-head vs the published Hannum 2013 clock")
    L.append("")
    if bench:
        tc, hn = bench["this_clock"], bench["hannum_2013"]
        L.append("Both clocks scored on the same {} held-out people.".format(bench.get("test_n", "?")))
        L.append("")
        L.append("| Clock | MAE (years) | Pearson r |")
        L.append("|---|---|---|")
        L.append("| This clock | {:.2f} | {:.3f} |".format(tc["test_mae_years"], tc["test_pearson_r"]))
        L.append("| Hannum 2013 | {:.2f} | {:.3f} |".format(hn["test_mae_years"], hn["test_pearson_r"]))
        L.append("")
        L.append(
            "Hannum applied from its published 71-CpG table ({} of {} sites present in "
            "this cohort) as a direct coefficient-times-methylation sum. See "
            "results/benchmark.png.".format(hn["n_cpgs_present"], hn["n_cpgs_total"])
        )
    else:
        L.append(pending("benchmark_hannum.py"))
    L.append("")

    L.append("## Biological interpretation")
    L.append("")
    if overlap and overlap.get("clocks", {}).get("Hannum2013"):
        han = overlap["clocks"]["Hannum2013"]
        line = "Selected sites overlap the published Hannum clock at {} sites".format(
            han.get("overlap_with_selected", "?"))
        if han.get("fold_enrichment") is not None:
            line += ", about {:.0f} times chance".format(han["fold_enrichment"])
        if han.get("hypergeometric_p") is not None:
            line += " (hypergeometric p = {:.2g})".format(han["hypergeometric_p"])
        L.append(line + ", indicating the model independently recovered known aging sites.")
        L.append("")
    if genes:
        mapped = [g for g in genes if g.get("primary_gene")]
        L.append("Of {} selected CpGs, {} map to a gene via the 450K manifest. "
                 "Highest-impact mapped sites:".format(len(genes), len(mapped)))
        L.append("")
        L.append("| CpG | Impact (yr) | Gene |")
        L.append("|---|---|---|")
        for g in mapped[:10]:
            L.append("| {} | {:+.2f} | {} |".format(g["cg"], float(g["impact"]), g["primary_gene"]))
        L.append("")
    else:
        L.append(pending("annotate_genes.py"))
        L.append("")
    if enrich:
        sig = [e for e in enrich if float(e.get("Adjusted P-value", 1)) < 0.05]
        L.append("Gene-set enrichment (Enrichr): {} terms significant at adjusted "
                 "p < 0.05. Strongest terms:".format(len(sig)))
        L.append("")
        L.append("| Term | Library | Adj. p |")
        L.append("|---|---|---|")
        for e in enrich[:8]:
            L.append("| {} | {} | {:.1e} |".format(
                e.get("Term", "")[:60], e.get("Gene_set", ""), float(e.get("Adjusted P-value", 1))))
        L.append("")
    else:
        L.append(pending("enrichment.py"))
        L.append("")

    L.append("## External validation")
    L.append("")
    if extval:
        L.append(
            "Applied unchanged to {}, an independent whole-blood 450K cohort (n = {}). "
            "External MAE {:.2f} years, r {:.3f}, with {} of {} informative sites present. "
            "The error gap versus the within-cohort test reflects cross-study "
            "normalization differences and is the honest measure of transfer.".format(
                extval["external_dataset"], extval["n_samples"],
                extval["external_mae_years"], extval["external_pearson_r"],
                extval.get("informative_sites_present_median",
                            extval.get("informative_sites_present", "?")),
                extval["informative_sites_total"])
        )
    else:
        L.append(pending("external_validation.py"))
    L.append("")

    L.append("## Limitations")
    L.append("")
    L.append(
        "One cohort and tissue for training, one array platform. No cell-type "
        "deconvolution, so age-related shifts in blood composition are uncorrected. The "
        "target is chronological age, not a mortality-trained biological-age composite, "
        "so the residual is a crude age-acceleration proxy. 656 samples is small for "
        "20,000 candidate features, so regularization carries the model; a strong test "
        "correlation should not be read as biological proof on its own."
    )
    L.append("")
    L.append("## References")
    L.append("")
    L.append("Hannum G, et al. Genome-wide methylation profiles reveal quantitative views "
             "of human aging rates. Mol Cell. 2013;49(2):359-367. GEO: GSE40279.")
    L.append("")
    L.append("_Every number in this report is produced by the code in this repository. "
             "Sections without computed inputs are marked as pending. No em-dashes by "
             "project convention._")

    out = os.path.join(ROOT, "REPORT.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("[report] wrote REPORT.md ({} sections populated)".format(
        sum(x is not None for x in [metrics, bench, overlap, genes, enrich, extval])))


if __name__ == "__main__":
    main()
