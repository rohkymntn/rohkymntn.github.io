# Complementary molecular information for age prediction and metabolic health

Research protocol draft · Kun Hyung Roh · 5 September 2026

## Central hypothesis

Methylation, microbial composition, and circulating proteins capture partly distinct age-associated variation. Their incremental predictive value may depend on assay quality, age range, and biological context. Cross-omic discordance may associate with metabolic health after chronological age is accounted for. These are testable hypotheses, not established findings.

The study combines three objectives: improve out-of-sample prediction, identify reproducible biological sources of complementarity, and test relevance to health beyond chronological age. Chronological-age accuracy alone does not establish a biological-age biomarker or clinical utility.

## Cohort feasibility

| Resource | Verified evidence | Role and outstanding requirements |
|---|---|---|
| HI-SEED | Public metadata and genus/species tables have 123 unique, exactly matched participant IDs, ages 17–82; methylation-derived clock scores, HbA1c and BMI are available. | Immediate paired clock-score–gut pilot. Raw methylation accessions GSE273166/GSE222131 require specimen-linkage verification. Not skin microbiome and not proteomics. |
| Lifelines-DEEP | Primary paired EWAS reports 616 blood-methylation/16S participants and 683 methylation/shotgun participants. Official DEEP catalogue also lists proteomics. | Preferred candidate for three-modality analysis. Complete three-way overlap, protein assay, covariates, consent and access remain to be verified. The approximately 1,500-person cohort is not an established complete-case sample size. |
| Netherlands Twin Register | Primary EWAS reports 296 paired methylation/microbiome participants. | Potential replication, subject to access. Preserve family groups in all partitions. Proteomics overlap not verified. |
| MGB-ABC / OMICmAge | Published report includes 1,789 proteomic participants, 1,475 with matched methylation; linked clinical information is controlled. | Potential methylation–protein validation. Microbiome not documented. Public PRIDE proteomics alone does not supply all linked covariates. |
| Stanford iPOP / iHMP | Official portal distributes longitudinal multi-omics; published studies include microbial and proteomic measurements. | Investigate skin–protein complementarity and longitudinal stability. Methylation and exact matched sample counts have not been established here. |

Sources: [HI-SEED study](https://pmc.ncbi.nlm.nih.gov/articles/PMC13365576/), [HI-SEED data](https://doi.org/10.6084/m9.figshare.26487670), [paired Lifelines-DEEP/NTR EWAS](https://pmc.ncbi.nlm.nih.gov/articles/PMC11660878/), [DEEP catalogue](https://wiki.lifelines.nl/doku.php?id=deep), [OMICmAge report](https://www.nature.com/articles/s43587-026-01073-7), [iPOP data portal](https://med.stanford.edu/ipop/data-download.html).

## Aim 1: Incremental prediction and reliability

Compare methylation, microbiome and proteomics individually; all three pairwise combinations; and the three-way combination in exactly the same held-out participants. Distinguish raw methylation models from models using previously developed clock scores. Begin with ridge/elastic-net and tree baselines, equal-weight averaging, and cross-fitted linear stacking. Evaluate a reliability-aware fusion model only after these baselines are fixed: weights may depend on assay missingness or training-derived uncertainty, never on held-out age or errors.

Primary endpoint: paired difference in participant-weighted MAE between the prespecified fusion model and an inner-validation-selected unimodal comparator, evaluated once in a locked external cohort. Report confidence intervals and R², calibration slope, age-stratified bias, and performance under missing modalities. Secondary analyses examine modality cost and whether fusion permits lower-cost assays at similar error. Sample-size and power calculations must use the actual paired overlap and a meaningful precision target; there is no current power claim.

Split participants, families and repeated visits before all preprocessing. Perform feature selection, imputation, normalization, batch adjustment, tuning and stacking within training folds. Use nested validation for development; reserve an external cohort before model comparisons. Record all attempted comparisons and correct secondary hypothesis families for multiplicity.

## Aim 2: Biological complementarity

Estimate modality-specific age residuals using cross-fitting. Age adjustment must also be learned in training partitions. Test whether residual covariance and conditional feature importance replicate across cohorts. Evaluate measured gene/protein pathways and microbial functional profiles where available; 16S taxonomic assignments alone do not establish biochemical activity.

Prioritize stable modules over isolated high-ranking features. Use held-out permutation importance with correlated-feature sensitivity analyses, selection stability, and pathway backgrounds defined by measured analytes. For methylation, account for unequal probe coverage and correlation. For cross-omic links, distinguish gene annotation from measured expression or protein abundance. A reproducible association is not evidence of a causal mechanism.

## Aim 3: Health relevance beyond chronological age

Use HbA1c and BMI in HI-SEED as exploratory metabolic correlates. Larger cohorts may support inflammation, frailty or incident outcomes, contingent on access and adequate event counts. Compare an age/sex baseline with the same baseline plus cross-fitted modality residuals or discordance. Model BMI as a covariate only when it is not itself the endpoint. Assess incremental held-out prediction and calibration, not just an in-sample association P value. Survival analyses are optional only when real longitudinal outcomes are available and scientifically justified.

Potential conceptual contribution: identifying when modalities disagree, whether those disagreements replicate, and whether they improve health characterization beyond a more accurate chronological-age estimate. Generic multi-omics fusion and microbiome prediction of methylation clocks already exist; novelty is not established by combining modalities alone.

## Public pilot audit and analysis gate

The downloaded HI-SEED tables contain 123 participants with no missing values in the released metadata and exact ID agreement between metadata, genus and species tables. Horvath scores span 10.84–75.57, and DunedinPACE spans 0.656–1.682. The column labelled GrimAge2 spans −23.94 to 20.32; its definition or transformation must be resolved before treating it as raw predicted age. Exclude every AgeAccel column from chronological-age prediction because residualization used the outcome.

A pilot may compare Horvath score alone, microbial composition alone and their combination. This would evaluate a pretrained-clock–gut model, not a raw blood methylome–skin model. At n=123, use a small prespecified model family, nested participant validation, uncertainty estimates and explicit exploratory status. Establish whether published clock construction used these participants before interpreting generalization. The prespecified pilot has now been executed on Modal (three repeated nested validations). Calibrated Horvath MAE was 4.87 years, microbial MAE 12.59, and combined ridge MAE 9.53. Fusion degraded prediction relative to the clock by 4.66 years (conditional 95% interval 3.35–6.03). All models and outcomes are retained in results/hiseed_summary.json. The next proposed experiment would test residual microbial information while protecting the unpenalized clock component; this is a new exploratory specification, not an executed positive finding.

## Figure plan

1. Cohort inclusion, age/sex distributions, exact modality-overlap counts, missingness and model architecture. Mark development versus external validation.
2. Matched unimodal, pairwise and triple-model comparisons, paired uncertainty, external calibration and missing-modality sensitivity. Retain non-improving comparisons.
3. Stable cross-omic modules and measured pathways, followed by age-adjusted health associations in held-out participants. Show multiple-testing results and replication status.

Current public figures contain only completed analyses. The new methylation annotation analysis tested 1,333 Reactome pathways with 10,000 random measured-probe sets; none passed FDR <0.05. The retained positive skin microbiome–metadata result is a 9.97% aggregate MAE reduction relative to metadata alone, with heterogeneous cohort effects. Neither result establishes three-omic complementarity or clinical utility.

## Reproducibility and compute

Modal is available for prespecified CPU model sweeps and, if data size warrants it, GPU models. Compute capacity does not replace paired specimens or access authorization. Record data releases, hashes, participant eligibility, preprocessing, split seeds, model configurations and all outcomes. Keep a locked confirmatory evaluation separate from exploratory iteration.

Architecture styling adapts conventions from [figures4papers](https://github.com/ChenLiu-1996/figures4papers), commit 565e6b97a9609e14ac07bee83dcb94589034fe27. The Matplotlib diagram and statistical figures are original code; outputs are editable SVG/PDF and high-resolution PNG.

Access status: the investigator reports no existing Lifelines-DEEP or BIOS access. Controlled-cohort work is contingent on a future approved application.

## Executed clock-preserving model extension

The proposed residual experiment has now been run with three calibration options, training-only clock orthogonalization, linear/RBF kernel ridge corrections and a clock-only fallback. In 15 outer folds across the same three repetitions, protected fusion achieved MAE 4.848 years versus 4.851 for selected calibration alone. The primary paired gain was 0.004 years (conditional 95% interval −0.061 to 0.070); microbial benefit was not detected. The post hoc engineering comparison against early fusion showed a 49.1% MAE reduction (9.529 to 4.848 years). This supports recovery from early-fusion degradation, not superiority over the clock or methodological novelty. All choices, predictions and outcomes are recorded in protected_design.json and protected_summary.json. Further testing requires independent data to support a scientific discovery claim.
