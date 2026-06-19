# Aging Clock: Lab Notebook and Learning Log

A living document. It explains what this project is, how every piece works, why
we made each choice, and where we are. We bump the version and add a changelog
row every time we change it, then commit. Git stores the real diffs, this header
keeps it readable.

Version: v0.1
Last updated: 2026-06-19
Owner: Brandon

## Version history

| Version | Date       | Change                                            |
|---------|------------|---------------------------------------------------|
| v0.1    | 2026-06-19 | First notebook. Project scaffolded, deps installed, not yet run. |

How to update this file: make your edits, bump the version number above, add one
row to this table describing what changed, save, then commit with a message like
"notebook v0.2: recorded first run metrics". No em-dashes anywhere.

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

Scaffold built, correct dependencies installed in the project's virtual
environment (numpy, pandas, scikit-learn, scipy, matplotlib, joblib, GEOparse).
Not yet run. Next action: run the pipeline and record the first metrics here.

Watch items on first run: the download is large and slow; GEOparse against the
very new pandas 3.0 may need pandas pinned below 3.0; and the age parser may need
a one-line regex tweak if the metadata label differs, in which case the script
tells you exactly what to change.

## 9. Open questions (fill these as we go)

- What MAE and r did our run actually get, and how does that compare to the
  published clocks.
- How many CpGs did the model keep, and do any overlap the known Hannum 71.
- Does pinning pandas matter on this machine or did 3.0 work.

## 10. Next steps

- Run the pipeline, paste metrics, record them here as v0.2.
- Then deepen this project or move to the next portfolio piece (NGS pipeline,
  single-cell aging analysis, longevity gene survival analysis), in that order.
- Eventually swap the chronological-age target for a biological-age target and
  add cell-type correction, which moves this from a teaching clock toward a real
  one.

## Notes

Every factual claim here traces to the data, the cited paper, or the code in this
repo. Where something is an expectation rather than a measured result, it says so.
No em-dashes by project convention.
