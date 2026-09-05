# Blood methylation and skin microbiome age prediction

Kun Hyung Roh · executed September 5, 2026

This is a newly executed public-data reconstruction of the concept in the 2023 W3PHIAI abstract “Integration of Blood Methylome and Skin Microbiome Data in Machine Learning Algorithms for Accurate Age Prediction.” The original experiment's code and data were unavailable. These results must not be attributed to the 2023 experiment.

## Principal integration result

An exploratory follow-up on the retained skin cohort evaluated metadata-adjusted microbial prediction under three repeated nested participant-held-out validations. The combined model reduced participant-weighted MAE from 9.71 to 8.74 years versus metadata alone (10.0%; paired reduction 0.97 years, conditional 95% bootstrap interval 0.66–1.28). Relative to microbiome alone, the reduction was 0.41 years (0.10–0.74). Both average differences were positive in all three repetitions.

Metadata comprises study, body site, and sex. This result is **not blood–skin fusion accuracy**. Effects varied by study and the aggregate benefit versus metadata was driven by study 10317. The experiment was specified before running this follow-up, but uses previously analyzed data and is not independent confirmation. All repeats and negative subgroup effects are retained.

The design, selected parameters, all held-out predictions, participant errors and study-stratified conditional bootstrap summaries are in `results/integration_*`. Run `modal run src/modal_integration.py`, `python src/summarize_integration.py`, and `python src/figure_integration.py` to reproduce. No further model search was performed after these results.

## Component results

| Evaluation | MAE (years) | R² |
| --- | ---: | ---: |
| Blood, plate-held-out nested CV, 360 controls | 4.21 | 0.91 |
| Blood, independent external cohort, 274 control women | 6.03 | -1.38 |
| Skin, participant-held-out nested CV, 339 participant identifiers / 1,798 samples | 9.17 | 0.44 |
| Skin, leave-one-study-out, fixed forest | 18.35 | -0.89 |

The cohorts are unpaired. **No validated blood-plus-skin model or measured integration gain is claimed.** A fitted late-fusion implementation is provided in `models/late_fusion_age_model.joblib`. Its independently trained ridge and random-forest components are combined with a prespecified 0.5/0.5 prediction rule. Component fitting ran on Modal CPU. The implementation is tested, but joint predictive accuracy remains unestimated without paired test participants.

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
pip install modal==1.5.5
modal run src/modal_fusion.py
python src/report.py
python src/test_analysis.py
python -m unittest discover -s src -p 'test_fusion.py'
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
- `src/fusion_model.py`: equal-weight prediction-level fusion with feature-schema and linkage checks.
- `src/modal_fusion.py`: bounded Modal CPU job for component fitting and serialization.
- `src/test_fusion.py`: nine software checks of fusion arithmetic, subject alignment, schemas, provenance and weights.
- `models/late_fusion_age_model.joblib`: fitted component regressors plus fusion operator (downloaded separately from the report).
- `results/fusion_model_manifest.json`: execution provenance and explicit absence of a paired accuracy estimate.
- `src/test_analysis.py`: ten integrity checks, including fold isolation and metric recomputation.
- `results/`: executed predictions, model-selection records, coefficients, feature annotation and metric tables.
- `figures/`: final figure exports.
- `PLAN.md`: analysis design and prespecified model grids, followed by explicitly labeled exploratory diagnostics.

For complete regeneration, source data are downloaded into `data/raw` and processed arrays are cached in `data`. These large third-party files are not republished in the code archive. The code archive includes the report, executed derived results, source manifest and figures.

## Sources and scope

Huang et al., mSystems 2020, doi:10.1128/mSystems.00630-19, and the linked author repository; GSE41037; GSE19711; GPL8490. Full source attribution is in the report and manifest. Original datasets remain subject to their source terms. Newly written project code is provided for reproducibility; third-party datasets are not relicensed here.

Chronological-age prediction does not establish biological age, health benefit, causality or clinical validity. This reconstruction is not a peer-reviewed paper.

## Paired inference

```python
import sys, joblib
sys.path.insert(0, "src")
model = joblib.load("models/late_fusion_age_model.joblib")
# Frames must have fitted feature columns and verified shared participant/visit indices.
predictions = model.predict_paired(blood_frame, skin_frame,
    pairing_provenance="Documented shared cohort and specimen linkage")
```

Only load a serialized model from a trusted source. The downloadable archive contains source code; the fitted model is a separate download. No synthetic pairs or biological fusion accuracy are reported.
