# Aging Clock: Lab Notebook and Learning Log

A living document. It explains what this project is, how every piece works, why
we made each choice, and where we are. We bump the version and add a changelog
row every time we change it, then commit. Git stores the real diffs, this header
keeps it readable.

Version: v0.3
Last updated: 2026-06-19
Owner: Brandon

## Version history

| Version | Date       | Change                                            |
|---------|------------|---------------------------------------------------|
| v0.1    | 2026-06-19 | First notebook. Project scaffolded, deps installed, not yet run. |
| v0.2    | 2026-06-19 | Added the Terms reference dictionary at the bottom. Loader rewritten to stream the SOFT file as float32 after the first run hit a MemoryError. |
| v0.3    | 2026-06-19 | First successful run. Recorded metrics (MAE 5.65y, r 0.892, 176 CpGs). Removed n_alphas from the model call (removed in scikit-learn 1.9). |

How to update this file: make your edits, bump the version number above, add one
row to this table describing what changed, save, then commit with a message like
"notebook v0.3: recorded first run metrics". When a new technical term shows up,
it goes in the Terms reference at the bottom so it is never lost. No em-dashes
anywhere.

## 1. North Star (why this project exists)

Long-term goal: build toward a biomedical longevity company.

Guiding hypothesis (Brandon's thesis): the first step of longevity is a "CRISPR
2.0" that gives full control over genes, not just turning them on or off, but
tuning them precisely and reversibly. This maps onto real research directions:
base and prime editing for precise sequence control, epigenetic editing for
graded and reversible expression control, and partial reprogramming for pushing
cells toward a younger state.

Where this project fits: control is the lever, but measurement is the gauge. You
cannot tell whether any intervention, including gene control, actually slows or
reverses aging without a way to measure biological age faster than waiting
decades for outcomes. An aging clock is that gauge. This project builds a simple
one from scratch so the measurement layer is something you understand and own,
not a black box. It is also the first real bioinformatics piece in the portfolio,
which is what makes you a credible applicant for computational-biology roles.

## 2. What this project is, in one paragraph

We take DNA methylation data from human blood, where each person is described by
the methylation level at roughly 450,000 sites in their genome, and we train a
model to predict their age from those levels. The model learns that certain sites
gain or lose methylation in a clock-like way as people get older. Once trained,
the gap between the age the model predicts and the person's real age becomes a
signal: if your methylome looks older than your birthday says, that is "age
acceleration," the quantity longevity research cares about.

## 3. Concepts you need (glossary in learning order)

DNA methylation: a chemical tag (a methyl group) added onto the DNA, almost
always on a cytosine base that is immediately followed by a guanine. It does not
change the genetic code, it changes how genes are read. Methylation patterns
shift with age in predictable ways, which is what makes a clock possible.

CpG site: the specific spot where methylation happens, a Cytosine followed by a
Guanine along the DNA strand (the "p" is the bond between them). Humans have about
28 million CpGs. The array used here measures about 450,000 of them.

Beta value: the methylation level at one CpG for one person, a number from 0
(no methylation) to 1 (fully methylated). These beta values are the raw features,
the inputs, our model learns from. Each person is a row of about 450,000 beta
values.

Epigenetic clock: a model, almost always a penalized linear regression, that
predicts age from a set of CpG beta values. The famous ones are the Hannum clock
(71 CpGs) and the Horvath clock (353 CpGs). Ours is the same idea, built fresh.

Chronological age vs biological age: chronological age is your calendar age.
Biological age is how old your body looks at the molecular level. The clock is
trained to predict chronological age, and the error it makes (predicted minus
actual) is our estimate of the biological-age signal.

Age acceleration: the residual, predicted age minus real age. Positive means the
methylome looks older than the calendar age (faster aging). This is the headline
output and the thing an intervention would try to lower.

Elastic net: the specific model we use. It is linear regression with two safety
mechanisms (penalties) bolted on. One penalty (L1, also called Lasso) pushes most
of the coefficients to exactly zero, which selects a small set of informative
CpGs and ignores the rest. The other (L2, also called Ridge) handles the fact
that many CpGs move together, so it shares weight smoothly among correlated sites
instead of picking one at random. The mix between the two is the l1_ratio.

Regularization: the general name for those penalties. We need them because we
have far more features (CpGs) than people. Without regularization the model would
memorize the training people perfectly and fail on anyone new. Regularization
forces it to keep things simple so it generalizes.

Train/test split: we fit the model on 80 percent of the people and then test it
on the other 20 percent that it never saw. The test score is the honest measure,
because it reflects new data, not memorization.

Data leakage: the cardinal sin, where information from the test set sneaks into
training and inflates the score. We avoid it in two places. We select CpGs by
variance only, which never looks at age, so it cannot leak the answer. And we
compute the feature scaling (mean and spread) using the training people only,
then apply it to the test people.

MAE (mean absolute error): the average size of the model's miss, in years. If MAE
is 4, the model is off by about 4 years on average. Lower is better. This is the
number to lead with.

RMSE (root mean squared error): similar to MAE but it punishes big misses more
heavily. Useful as a check that there are no large blowups.

Pearson r: the correlation between predicted and actual age, from 0 to 1. It says
how well the predictions track real age in rank order. High r with low MAE is the
goal.

## 4. The data

Accession: GSE40279, from the public Gene Expression Omnibus (GEO).
Source paper: Hannum et al., "Genome-wide Methylation Profiles Reveal Quantitative
Views of Human Aging Rates," Molecular Cell, 2013.

What it contains: 656 whole-blood samples, run on the Illumina Infinium 450K
array, so about 450,000 CpG beta values per person, plus each person's age and
sex. It is one of the standard datasets used to build and test blood methylation
clocks, which is why we use it. It is public and needs no access request.

## 5. How the pipeline works, file by file

run_all.py: the conductor. Runs the three stages below in order. It skips the
download if the data file already exists, so reruns are fast. Pass
--force-download to refetch.

src/download_data.py: gets the data ready.
  1. Downloads GSE40279 with the GEOparse library.
  2. Reads each sample's age and sex out of the metadata. It searches the text
     for an age field rather than assuming a fixed label, and if it cannot find
     ages it prints one sample's metadata and tells you what to fix. It never
     guesses ages.
  3. Builds the big matrix of beta values, people as rows, CpGs as columns.
  4. Drops CpGs that have any missing values, then keeps only the most variable
     CpGs (default 20,000). Variance selection is unsupervised, it ignores age,
     so it shrinks the problem to a tractable size without leaking the answer.
  5. Saves a compact matrix (data/methylation_top.pkl) and the ages and sex
     (data/meta.csv).

src/train_clock.py: builds and grades the clock.
  1. Loads the matrix and the ages, lines them up.
  2. Splits 80/20 into train and test with a fixed random seed so results
     reproduce.
  3. Standardizes the features using the training set only.
  4. Fits ElasticNetCV. The CV part means it tries several penalty strengths and
     L1/L2 mixes using cross-validation inside the training data and keeps the
     best, so we are not hand-tuning.
  5. Predicts ages on the held-out test people and computes MAE, RMSE, and r.
  6. Computes the residual (predicted minus actual) as age acceleration.
  7. Saves results/metrics.json, results/predictions.csv, and the fitted model.

src/plots.py: draws two pictures from the predictions.
  1. predicted_vs_actual.png: a scatter of predicted age against real age, with
     the line y = x. Points hugging that line means a good clock.
  2. residual_hist.png: the distribution of age acceleration. Centered near zero,
     with a spread. People far out on either side are the fast or slow agers.

## 6. How to read the results

metrics.json is the scorecard. What "good" looks like, using published blood
clocks as the reference point, is a test MAE in the low single digits of years
and a Pearson r above about 0.9. Treat those as the target to compare against,
not as a promise. The number of CpGs the model actually kept (n_cpgs_selected)
tells you how sparse the signature is, which is interesting in its own right, the
published Hannum clock used only 71.

The residual histogram is the conceptual payoff. The clock predicting age well is
nice, but the residual is the part that matters for longevity. It is a first,
crude version of the age-acceleration measure that a real intervention would aim
to shift.

## 7. Decisions log (why we did it this way)

- Predict chronological age, not a mortality-trained target. The advanced clocks
  (PhenoAge, GrimAge) are trained against health and death outcomes and are
  better aging predictors, but they are more complex. Starting with chronological
  age is the clean, well-understood baseline. Upgrading the target is a planned
  next step.
- Elastic net, not a fancier model. It is the established method for this exact
  shape of problem (many correlated features, few samples) and it is what the
  published clocks used, so it is the honest baseline and it is interpretable.
- Top-variance probe preselection. The full matrix is heavy and most CpGs barely
  move between people. Keeping the most variable ones is standard, it is
  unsupervised so it does not leak, and it keeps the pipeline runnable on a normal
  machine.
- Fixed random seed. So the same run gives the same numbers, which matters for a
  portfolio piece someone might rerun.
- No cell-type correction yet. Blood cell composition shifts with age and can
  confound a naive clock. We left it out of v1 on purpose to keep the baseline
  simple, and flagged it as the clearest next improvement.

## 8. Current status

Ran end to end successfully on 2026-06-19. The clock works.

First-run results (held-out test set, 20 percent of 656 people):
- Test MAE: 5.65 years (average miss on people the model never saw)
- Test RMSE: 7.34 years
- Test Pearson r: 0.892
- CpGs the model actually used: 176 of 20,000

How to read this: the clock predicts age from blood methylation with an average
error under six years and a strong correlation. Published first-generation blood
clocks reach roughly 3 to 4 years and r around 0.95, so this baseline is in the
right neighborhood and clearly working, while being a bit less accurate than the
optimized published clocks. That gap is expected, because they used supervised
CpG selection and cell-type correction and we used a generic top-variance feature
set with no correction. Elastic net narrowing 20,000 candidates to a 176-CpG
signature on its own, the same order of magnitude as the Hannum 71, is a good
sign it found real structure.

Outputs are in results/: metrics.json, predictions.csv, clock.joblib,
predicted_vs_actual.png, residual_hist.png.

Notes from this run: pandas 3.0 worked once the loader bypassed GEOparse, so no
pin was needed. The age parser worked with no manual tweak. Training the
cross-validated elastic net is the slow step now, a few minutes.

## 9. Open questions (fill these as we go)

Answered by the first run:
- Our MAE was 5.65 years and r was 0.892, versus the published clocks at roughly
  3 to 4 years and r around 0.95. We are close but a step behind the optimized
  versions, as expected for a baseline.
- The model kept 176 CpGs. Whether any overlap the known Hannum 71 is still worth
  checking and would be a nice addition.
- pandas 3.0 worked once we stopped using GEOparse for the matrix, so no pin was
  needed.

Still open:
- Do any of our 176 CpGs overlap the published Hannum or Horvath clock sites.
- How much would more probes (raising N_PROBES) or cell-type correction improve
  the MAE.
- Does the residual (age acceleration) correlate with anything we have, such as
  sex.

## 10. Next steps

- Run the pipeline, paste metrics, record them here as v0.2.
- Then deepen this project or move to the next portfolio piece (NGS pipeline,
  single-cell aging analysis, longevity gene survival analysis), in that order.
- Eventually swap the chronological-age target for a biological-age target and
  add cell-type correction, which moves this from a teaching clock toward a real
  one.

## Terms reference (running dictionary)

The running dictionary. Section 3 explains the core ideas in learning order; this
is the quick lookup, kept broad on purpose, including the software and debugging
terms. When a new technical term comes up in our work, we add it here so it is
never lost. Plain definitions, one or two sentences each.

### Biology and the data

DNA methylation. A chemical tag (a methyl group) added to DNA, almost always on a
cytosine inside a CpG. It does not change the genetic code, it changes how genes
are read, and its pattern shifts with age.

CpG site. A spot where a cytosine (C) sits directly next to a guanine (G) along
the DNA. Methylation happens here. Humans have about 28 million.

Beta value. The methylation level at one CpG for one person, from 0
(unmethylated) to 1 (fully methylated). These numbers are what we model.

Epigenetics. Changes in how genes are used that do not alter the DNA sequence
itself. Methylation is one epigenetic mechanism.

Epigenetic clock. A model that predicts age from methylation levels at a chosen
set of CpGs.

Chronological age. Calendar age, the years since birth.

Biological age. How old the body looks at the molecular level, which can run
ahead of or behind calendar age.

Age acceleration. The gap between predicted and actual age (the residual).
Positive means the molecule looks older than the calendar.

Illumina 450K array. The lab chip (Infinium HumanMethylation450 BeadChip) that
measures methylation at about 450,000 CpGs at once.

Probe. One measurement spot on the array, here one CpG, named by an ID like
cg00000029.

Whole blood sample. The tissue these measurements come from, drawn blood with all
its cell types mixed together.

Cohort. A group of people studied together. Ours is the 656 people in GSE40279.

GEO (Gene Expression Omnibus). A public NCBI database where researchers deposit
genomics datasets.

GSE. A GEO Series, one full study or dataset. Ours is GSE40279.

GSM. A GEO Sample, one person or sample inside a series.

GPL. A GEO Platform, the instrument or array used (the 450K chip here).

SOFT file (family SOFT). A large GEO text file bundling a series' samples, their
data tables, and metadata. The one we downloaded is 2.7 GB compressed.

Series matrix file. A leaner GEO text file holding just the data matrix plus a
short header. An alternative to the family SOFT.

Cell-type deconvolution. Estimating the mix of blood cell types in a sample, used
to correct clocks because cell composition shifts with age. Not done in v1.

### Statistics and modeling

Feature. An input the model learns from. Here each CpG is one feature.

Feature matrix. The table of inputs, rows are samples and columns are features.

Target (label). What the model predicts. Here, age.

Regression. Predicting a number (age), as opposed to predicting a category.

Coefficient (weight). The number the model assigns to a feature, saying how much
it pushes the prediction up or down.

Elastic net. A regression that adds two penalties so it stays simple and stable
when there are many correlated features. Our model.

L1 penalty (Lasso). Pushes many coefficients to exactly zero, which selects a
small set of useful features.

L2 penalty (Ridge). Shrinks coefficients smoothly and shares weight among
correlated features.

Regularization. The general idea of penalizing complexity so the model does not
just memorize the training data.

l1_ratio. The dial that sets how much of the penalty is L1 versus L2.

alpha. The overall strength of the penalty. Higher means a simpler model.

Cross-validation (CV). Splitting the training data into folds to test settings
internally and keep the best, without touching the held-out test set.

Standardization (StandardScaler, z-score). Rescaling each feature to mean 0 and
spread 1 so no feature dominates just because of its units.

Train/test split. Fitting on part of the data and judging the model on a separate
part it never saw.

Held-out set. The test portion kept aside for an honest score.

Data leakage. When test information sneaks into training and inflates the score.
We avoid it by selecting features without looking at age and by scaling on the
training set only.

Overfitting. When a model memorizes the training data and then fails on new data.

Variance (statistical). How much a value spreads across samples. High-variance
CpGs carry more signal, which is why we keep them.

Unsupervised vs supervised. Unsupervised steps ignore the target (our variance
filter). Supervised steps use it (the model fit).

MAE (mean absolute error). The average size of the model's miss, in years. Lower
is better. The headline accuracy number.

RMSE (root mean squared error). Like MAE but it punishes big misses more heavily.

Pearson correlation (r). How well predictions track the real values, from 0 to 1.

Residual. Predicted minus actual. Our age-acceleration signal.

Random seed. A fixed number that makes random steps (like the split) come out the
same every run, so results reproduce.

Sparse model. A model that ends up using only a few of the available features.

### Software and data tooling

Python. The programming language the project is written in.

Library (package). Reusable code you install and import. Ours include numpy,
pandas, scikit-learn, scipy, matplotlib, joblib, and GEOparse.

numpy. Fast numerical arrays and math.

pandas. Tables (DataFrames) for handling data.

scikit-learn. Machine-learning models, including the elastic net.

scipy. Scientific and statistical functions (the Pearson r here).

matplotlib. Plotting.

joblib. Saving and loading models and objects to disk.

GEOparse. A library for downloading and parsing GEO files. We stopped using its
matrix-building step because it ran out of memory, and now stream the file
ourselves.

DataFrame. A pandas table with labeled rows and columns.

Series. A single labeled column in pandas.

numpy array. A grid of numbers, more memory-efficient than a plain Python list.

dtype (data type). What kind of value an array or column holds (float, integer,
text).

float32 vs float64. Decimal numbers stored in 4 versus 8 bytes. float32 uses half
the memory, which is why we use it for the matrix.

object dtype. A catch-all type that stores general Python objects, often strings.
It is very memory-heavy, and it is what caused the first crash.

NaN. Short for "Not a Number," the marker for a missing value.

MemoryError. The error a program throws when it asks for more RAM than the machine
can give. The first run hit this.

gzip (.gz). A compression format. The GEO file arrives gzipped.

pickle (.pkl). A Python format that saves an object (our matrix) to disk exactly
as it sits in memory.

CSV. A plain-text table, values separated by commas or tabs.

JSON. A plain-text format for structured data. Our metrics.json uses it.

Streaming (line-by-line parsing). Reading a file a piece at a time instead of
loading it all at once, which keeps memory low. This is how the fixed loader
works.

Vectorized operation. Doing math on a whole array at once in fast compiled code,
instead of looping value by value in Python.

Traceback (stack trace). The error report Python prints showing exactly where a
crash happened. The MemoryError traceback is how we found the bad step.

Virtual environment (venv). An isolated Python setup per project so its packages
do not clash with other projects. Ours lives in the .venv folder.

pip. The tool that installs Python packages.

requirements.txt. The list of packages this project needs.

### Project and workflow

Git. The version-control tool that tracks every change to the files.

Repository (repo). The project folder that Git is tracking.

Commit. A saved snapshot of changes in Git, with a message describing it.

.gitignore. A file listing things Git should not track, like the large data files.

README. The front-page document that explains a project.

Lab notebook. This document, the running record of what we are doing and why.

PowerShell. The Windows command-line shell you run the setup commands in.

Pipeline. A sequence of steps run in order. Ours is download, train, plot, run by
run_all.py.

LF vs CRLF. Two ways to mark the end of a line in a text file (Linux and Mac use
LF, Windows uses CRLF). The Git warnings about this are harmless.

BOM (UTF-8 no-BOM). A hidden marker some editors place at the very start of a text
file. We write files without it (no-BOM) to avoid stray characters showing up.

## Notes

Every factual claim here traces to the data, the cited paper, or the code in this
repo. Where something is an expectation rather than a measured result, it says so.
No em-dashes by project convention.
