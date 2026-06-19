# A from-scratch DNA methylation aging clock: results and limitations

Brandon Duplantier. Code: github.com/brandonduplantier-juice/aging-clock

## Summary

An elastic-net regression trained on whole-blood DNA methylation predicts chronological age with a held-out mean absolute error of 5.65 years and a Pearson correlation of 0.892, using a sparse signature of 176 CpG sites selected from 20,000 candidates.

## Data and method

Source: GEO accession GSE40279 (Hannum et al., Molecular Cell, 2013), 656 whole-blood samples on the Illumina Infinium 450K array with chronological age and sex. The 20,000 most variable probes were kept (unsupervised, so no label leakage), features were standardized on the training fold only, and an ElasticNetCV model was fit with the L1/L2 mix and penalty chosen by 5-fold cross-validation. Twenty percent of people were held out for testing.

## Accuracy

| Metric | Value |
|---|---|
| Test MAE (years) | 5.65 |
| Test RMSE (years) | 7.34 |
| Test Pearson r | 0.892 |
| CpGs selected | 176 |
| Held-out people | 132 |

For reference, published first-generation blood clocks reach roughly 3 to 4 years MAE and r near 0.95. This baseline is in the right neighborhood and a step behind the optimized published clocks, as expected for a generic top-variance feature set with no cell-type correction.

## Head-to-head vs the published Hannum 2013 clock

Both clocks scored on the same 132 held-out people.

| Clock | MAE (years) | Pearson r |
|---|---|---|
| This clock | 5.65 | 0.892 |
| Hannum 2013 | 5.55 | 0.950 |

Hannum applied from its published 71-CpG table (71 of 71 sites present in this cohort) as a direct coefficient-times-methylation sum. See results/benchmark.png.

## Biological interpretation

Selected sites overlap the published Hannum clock at 3 sites, about 85 times chance (hypergeometric p = 2.7e-06), indicating the model independently recovered known aging sites.

Of 176 selected CpGs, 109 map to a gene via the 450K manifest. Highest-impact mapped sites:

| CpG | Impact (yr) | Gene |
|---|---|---|
| cg04875128 | +18.64 | OTUD7A |
| cg08128734 | -9.84 | RASSF5 |
| cg00573770 | -5.84 | ZEB2 |
| cg01074797 | -5.14 | PDZK1IP1 |
| cg07584066 | +4.89 | DHX40 |
| cg05898618 | -4.69 | KCNQ1 |
| cg24471254 | +4.48 | ACTL6B |
| cg12934382 | +4.07 | GRM2 |
| cg14296767 | -3.99 | HLA-L |
| cg02939078 | +3.63 | TCFL5 |

Gene-set enrichment (Enrichr): 0 terms significant at adjusted p < 0.05. Strongest terms:

| Term | Library | Adj. p |
|---|---|---|
| dicarboxylic acid catabolic process (GO:0043649) | GO_Biological_Process_2021 | 5.0e-02 |
| aspartate metabolic process (GO:0006531) | GO_Biological_Process_2021 | 2.0e-01 |
| regulation of T-helper cell differentiation (GO:0045622) | GO_Biological_Process_2021 | 2.5e-01 |
| fatty-acyl-CoA biosynthetic process (GO:0046949) | GO_Biological_Process_2021 | 2.5e-01 |
| cellular amino acid catabolic process (GO:0009063) | GO_Biological_Process_2021 | 2.5e-01 |
| regulation of dendritic spine morphogenesis (GO:0061001) | GO_Biological_Process_2021 | 2.5e-01 |
| regulation of intracellular protein transport (GO:0033157) | GO_Biological_Process_2021 | 2.5e-01 |
| positive regulation of transmembrane receptor protein serine | GO_Biological_Process_2021 | 2.5e-01 |

## External validation

Applied unchanged to GSE42861 (whole blood, 450K), an independent whole-blood 450K cohort (n = 689). External MAE 6.36 years, r 0.884, with 176 of 176 informative sites present. The error gap versus the within-cohort test reflects cross-study normalization differences and is the honest measure of transfer.

## Limitations

One cohort and tissue for training, one array platform. No cell-type deconvolution, so age-related shifts in blood composition are uncorrected. The target is chronological age, not a mortality-trained biological-age composite, so the residual is a crude age-acceleration proxy. 656 samples is small for 20,000 candidate features, so regularization carries the model; a strong test correlation should not be read as biological proof on its own.

## References

Hannum G, et al. Genome-wide methylation profiles reveal quantitative views of human aging rates. Mol Cell. 2013;49(2):359-367. GEO: GSE40279.

_Every number in this report is produced by the code in this repository. Sections without computed inputs are marked as pending. No em-dashes by project convention._
