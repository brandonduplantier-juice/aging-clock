# Aging Clock: a DNA methylation biological-age predictor

Predicts chronological age from whole-blood DNA methylation using penalized
regression, then treats the prediction residual (predicted age minus actual age)
as an estimate of biological age acceleration. This is the same modeling family
as the first-generation epigenetic clocks (Hannum 2013, Horvath 2013).

This project exists to demonstrate a real bioinformatics workflow end to end:
pulling a public omics dataset, handling a high-dimensional methylation matrix,
training and honestly evaluating a model, and interpreting the residual as a
biological-age signal. It is a portfolio and learning project, not a validated
clinical tool.

## Why this problem

Chronological age is trivially knowable. The useful quantity is biological age:
how old a person's tissue looks at the molecular level. DNA methylation at
specific CpG sites changes with age in a predictable enough way that a regression
model can recover age within a few years from blood alone. The residual (how much
older or younger the methylome looks than the calendar) is the aging signal that
downstream longevity work cares about.

## Data

Source: GEO accession GSE40279 (Hannum et al., "Genome-wide Methylation Profiles
Reveal Quantitative Views of Human Aging Rates," Molecular Cell, 2013).

- 656 whole-blood samples, Illumina Infinium 450K array, approximately 450,000
  CpG probes, with chronological age and sex metadata.
- Public, no access request required. Downloaded programmatically via GEOparse.

Citation: Hannum G, et al. Mol Cell. 2013;49(2):359-367. GEO: GSE40279.

The full beta-value matrix is large (656 samples by roughly 473k probes). To keep
the pipeline tractable on a normal machine, `download_data.py` keeps only the
top-variance probes (default 20,000). Variance ranking is unsupervised, so it
does not leak the age labels into the held-out test set. Preselecting the most
variable probes before penalized regression is standard practice on this dataset.

## Method

1. Download GSE40279, parse the beta matrix and the per-sample age and sex.
2. Keep the top-N most variable CpG probes (unsupervised filter).
3. Train/test split (80/20, fixed seed), standardize features on the training
   fold only.
4. Fit ElasticNetCV (cross-validated L1/L2 mix) to predict chronological age.
   Elastic net is the established choice here: it handles thousands of correlated
   probes and selects a sparse CpG signature, which is how the published clocks
   were built.
5. Evaluate on the held-out test set: mean absolute error in years, RMSE, and
   Pearson correlation between predicted and actual age.
6. Compute the residual (predicted minus actual) as the age-acceleration estimate
   and plot its distribution.

## Results

Metrics are written to `results/metrics.json` and `results/predictions.csv` when
you run the pipeline, and the plots to `results/`. Numbers are produced by your
run, not hardcoded here.

For context only, and not as a claim about this run: published first-generation
blood methylation clocks on 450K data typically reach a median or mean absolute
error in the low single digits of years (Hannum 2013). Treat that as the target
to compare your output against, not as a reported result.

## How to run

Requires Python 3.10+ and roughly 4 to 6 GB of free RAM for the download and
variance-filtering step (the raw matrix is large before it is reduced).

```
pip install -r requirements.txt
python run_all.py
```

Or run the stages individually:

```
python src/download_data.py     # fetch GSE40279, reduce to top-variance probes
python src/train_clock.py        # train elastic net, write metrics and predictions
python src/plots.py              # predicted-vs-actual and residual plots
```

Knobs worth changing: `N_PROBES` in `src/download_data.py` (more probes, slower,
usually slightly better), and the elastic-net `l1_ratio` grid in
`src/train_clock.py`.

## Limitations (read these)

- One cohort, one tissue, one array platform. A model trained on GSE40279 blood
  is not expected to transfer cleanly to other tissues, other arrays (EPIC v1/v2),
  or other populations. Cross-array probe mismatch alone can bias age estimates by
  many years.
- The residual is a crude age-acceleration proxy. Real biological-age clocks
  (PhenoAge, GrimAge) are trained against health and mortality outcomes, not just
  chronological age, and are better aging predictors than a pure age-regression
  residual. This project deliberately starts with the simpler chronological-age
  version.
- No batch-effect correction or cell-type deconvolution is done here. Blood cell
  composition shifts with age and can confound naive clocks. Adding cell-type
  adjustment is a clear next step.
- 656 samples is small for 20,000 features. Elastic net regularization is doing a
  lot of work; do not over-read a strong test correlation as proof of biological
  insight.

## Possible extensions

- Add cell-type deconvolution (for example Houseman-style estimation) and refit.
- Validate on a second public blood dataset (for example GSE87571) to test
  cross-cohort transfer honestly.
- Swap the chronological-age target for a PhenoAge-style composite to move toward
  a true biological-age clock.
- Report the selected CpG signature and check overlap with the published Hannum
  71-CpG clock.

## Notes

Data sources and the modeling approach are cited above. Result numbers come only
from running the code; nothing in this repo reports a metric that was not computed
from the data. No em-dashes anywhere by project convention.
