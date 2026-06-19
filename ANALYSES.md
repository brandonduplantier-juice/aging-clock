# Portfolio-strengthening analyses

Four additions that turn the clock from a working model into a defensible piece of
computational biology, plus a report generator that writes them up.

## What each script does

- `src/annotate_genes.py` maps the selected CpGs to genes via the Illumina 450K
  manifest (GPL13534, downloaded once and cached). Writes `results/gene_annotation.csv`.
- `src/enrichment.py` runs gene-set enrichment (GO biological process, KEGG, and
  aging perturbation libraries) on those genes through Enrichr. Writes
  `results/enrichment.csv` and `results/enrichment.png`. Needs internet.
- `src/benchmark_hannum.py` scores this clock and the published Hannum 2013 clock on
  the exact same held-out people. Writes `results/benchmark.json` and `benchmark.png`.
- `src/external_validation.py` applies this clock unchanged to a second cohort
  (GSE87571) and reports transfer accuracy. Writes `results/external_validation.json`
  and `external_validation.png`.
- `src/make_report.py` assembles `REPORT.md` from whatever results exist. It never
  invents a number; sections without inputs are marked pending.

The dashboard also gains gene names: rerun `src/export_dashboard.py` after
`annotate_genes.py` and the sliders and bars show genes instead of bare cg IDs.

## Dependencies

GEOparse is already in requirements. Add `gseapy` for enrichment. The Hannum
coefficients are bundled in `ref/hannum_coefficients.csv`, so no biolearn or torch
is needed.

## Run order

Run the core pipeline (`run_all.py`) first so the trained model exists. Then:

```
pip install gseapy
python run_extras.py
python src/export_dashboard.py   # refresh dashboard data with gene labels
```

`run_extras.py` runs all four analyses then the report; each step is independent,
so one failing (for example enrichment with no internet) does not stop the rest.

## First-run checks

- annotate_genes: the GPL13534 manifest is a large one-time download. If the gene
  column is named differently, the script prints the available columns to adjust.
- benchmark: refetches the full GSE40279 beta matrix (cached under data/raw from the
  clock build, so usually fast) and reports how many Hannum CpGs were present. Sanity
  check that the Hannum MAE is in a believable range.
- external_validation: downloads GSE87571 (large). It reports how many informative
  sites were present in the new cohort; a low overlap means treat the transfer number
  with caution. A higher MAE than the within-cohort test is expected and is the point.
