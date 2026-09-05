# Public-data reconstruction of the 2023 age-prediction concept

Run date: 2026-09-05. The original code and participant-level paired data are unavailable. This is a new analysis, not a recovery of the 2023 experiment.

1. Train a blood methylation age model on healthy, source-QC-passing adults in GSE41037; test generalization to control women in GSE19711. Use only processed 27K beta values. Report the target definition difference (age at recruitment externally).
2. Train a skin microbial age model on the publicly released Huang et al. matrix. Prefix participant identifiers by study. Exclude study 11052 from the primary benchmark because all its 177 samples have one subject label despite discrepant ages 31/36. Keep the exclusion in the audit trail.
3. Fit feature filtering, imputation, selection and scaling only on training partitions. Use outer plate-held-out methylation CV, outer participant-held-out skin CV, and nested inner validation. Score skin predictions with equal total weight per participant.
4. Audit random-sample versus participant and study splits using the same prespecified random forest. Add a study/site/sex-only baseline, age-median baselines, group-resampled uncertainty, age-stratified errors, label-shuffle controls and exploratory feature summaries.
5. Do not join independent people by age or row number. No empirical combined-model accuracy can be estimated from these unpaired data. Include a mathematical explanation of the missing joint-error covariance and a tested paired-data interface for future use.
6. Produce reproducible code, saved predictions/metrics, source checksums, publication-style vector figures and a portfolio report. Publish the completed project to the requested GitHub Pages site, preserving unrelated local drafts. Update the resume only with results supported by the executed analysis.

Model grids (declared before results): methylation top 100/500 training-correlated CpGs, ridge alpha 10/100/1000; microbiome Hellinger transform with training prevalence >=1%, 160-tree RF, sqrt feature subsampling, minimum leaf 1/3/5. Fixed RF diagnostics use leaf 3. Seed 20260905. Ten shuffled-label runs are descriptive negative controls, not a high-resolution significance test.
