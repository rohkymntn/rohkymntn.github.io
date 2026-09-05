# Blood methylation and skin microbiome age prediction

Kun Hyung Roh · executed September 5, 2026

This is a newly executed public-data reconstruction of the concept in the 2023 W3PHIAI abstract “Integration of Blood Methylome and Skin Microbiome Data in Machine Learning Algorithms for Accurate Age Prediction.” The original experiment's code and data were unavailable. These results must not be attributed to the 2023 experiment.

## Results

| Evaluation | MAE (years) | R² |
| --- | ---: | ---: |
| Blood, plate-held-out nested CV, 360 controls | 4.21 | 0.91 |
| Blood, independent external cohort, 274 control women | 6.03 | -1.38 |
| Skin, participant-held-out nested CV, 339 participant identifiers / 1,798 samples | 9.17 | 0.44 |
| Skin, leave-one-study-out, fixed forest | 18.35 | -0.89 |

The cohorts are unpaired. **No validated blood-plus-skin model or measured integration gain is claimed.** The paired-data utility is a tested input guardrail, not an evaluated fusion model.

The primary independent blood test retains two highly incomplete samples and their extreme predictions. The missingness-filtered sensitivity analysis is explicitly post hoc. Skin errors weight each participant equally, preventing repeatedly sampled donors from dominating the reported metric. See the report for baselines, conditional bootstrap intervals, cohort dependence and limits on interpretation.

## Reproduce

Python 3.12 was used. Install the recorded dependency versions in an isolated environment:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/download.py
python src/prepare.py
python src/analyze.py
python src/diagnostics.py
python src/figures.py
python src/report.py
python src/test_analysis.py
```

The raw public inputs total approximately 230 MB. The GEO and author-repository URLs are in `data/sources.json`. Downloaded input hashes are in `results/audit.json`. Public source files can change; a checksum mismatch stops the downloader rather than silently substituting different data. No original participant-level private data are needed.

The analysis does not access raw sequencing or IDAT files. It uses released processed abundance and beta-value matrices. The skin source table was already globally prevalence-filtered without age labels. All additional feature filtering, imputation, selection and scaling occur inside training folds.

## Files

- `index.html`: standalone report; open locally or serve this directory.
- `src/prepare.py`: GEO and microbial loaders, identity/QC audit, eligibility rules.
- `src/analyze.py`: nested validation, held-out predictions, baselines and label shuffles.
- `src/diagnostics.py`: locked-prediction QC sensitivity and matched error comparisons.
- `src/figures.py`: three multi-panel figures as vector SVG/PDF and 400-dpi PNG.
- `src/paired_fusion.py`: refuses invalid pairing by unrelated or duplicate IDs.
- `src/test_analysis.py`: ten integrity checks, including fold isolation and metric recomputation.
- `results/`: executed predictions, model-selection records, coefficients, feature annotation and metric tables.
- `figures/`: final figure exports.
- `PLAN.md`: analysis design and prespecified model grids, followed by explicitly labeled exploratory diagnostics.

For complete regeneration, source data are downloaded into `data/raw` and processed arrays are cached in `data`. These large third-party files are not republished in the code archive. The code archive includes the report, executed derived results, source manifest and figures.

## Sources and scope

Huang et al., mSystems 2020, doi:10.1128/mSystems.00630-19, and the linked author repository; GSE41037; GSE19711; GPL8490. Full source attribution is in the report and manifest. Original datasets remain subject to their source terms. Newly written project code is provided for reproducibility; third-party datasets are not relicensed here.

Chronological-age prediction does not establish biological age, health benefit, causality or clinical validity. This reconstruction is not a peer-reviewed paper.
