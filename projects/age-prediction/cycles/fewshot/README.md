# Few-shot adaptation of methylation age predictors across cohorts

Independent research cycle initiated 5 September 2026. This study is separate from the W3PHIAI abstract and the earlier fusion experiments.

## Hypothesis

A neural representation trained to correct a source age predictor using a small labelled support set can reduce external-cohort prediction error relative to support-tuned classical recalibration and retraining. Domain-level feature perturbations may improve adaptation when target assays omit source probes. Both are hypotheses to test, not assumed benefits.

## Prior art and scope

Differentiable ridge-regression meta-learning was introduced by [Bertinetto et al., ICLR 2019](https://arxiv.org/abs/1805.08136). [Transfer-learning epigenetic clocks](https://pubmed.ncbi.nlm.nih.gov/42368478/) and [semi-supervised methylation clocks with fewer labels](https://www.scitepress.org/Papers/2023/116244/116244.pdf) also exist. Our implementation combines standard components in a new experiment. We have not established priority, a novel biological mechanism, or suitability for publication in Nature.

## Data and partitioning

Source: GSE41037 healthy adult controls and GSE19711 control women. Of 634 source samples, 542 train feature selection, scaling, a ridge source clock, and the neural model. The 92 samples on discovery plate H form a source-only validation domain. Source selection retains 1,024 CpGs. Alpha selection uses source-group CV within source-selected features; those development scores are not unbiased performance estimates.

New external cohorts: [GSE40279](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE40279) (656 whole-blood samples, 450K) and [GSE37008](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE37008) (99 deposited PBMC arrays, 27K; 92 retained participants). In the processed releases, 834 and 809 of the source-selected probes have observations, respectively. Entirely unavailable probes are source-median imputed; samples with more than 1% missingness among the observed selected probes are excluded. Technical-replicate identifiers are audited before partitioning; the retained PBMC matrix has no duplicated donor identifiers after QC.

GSE40279: fixed random support pool 128, locked query set 528. GSE37008: support pool 32, query set 60. These partitions are independent of age stratification. Evaluate 8, 16 and 32 labelled support participants, with 20 seeded support draws per budget. The primary endpoint uses 16 labels. At 32 shots in PBMC, the entire support pool is reused; the baseline CV assignments can still vary.

## Model

The source-selected, standardized CpGs enter an MLP with dimensions 1,024 → 128 → 32, LayerNorm, GELU and tanh. Concatenating a learned scaling of the source-clock prediction supplies a clock coordinate. Within an episode, ridge regression fits source-clock residuals in the learned representation, including a support-fitted intercept. Backpropagation through the ridge solve trains the encoder and regularization strength using disjoint episode queries. The source-clock inputs for training episodes are group-out-of-fold predictions.

Train 2,000 episodes for each of three seeds, with and without task-level Gaussian feature shifts, multiplicative scales and feature masking. Every 100 steps, 12 fixed 16-shot episodes from the source-only validation domain select the best checkpoint. The three seeds are ensembled within each variant. Perturbations affect the encoder inputs while source-clock predictions stay fixed; this is representation robustness training, not simulation of a complete new assay.

## Baselines and inference

Compare frozen source ridge, support median, affine recalibration, support-only ridge, and linear/RBF residual kernel adaptation. All methods receive the same source-selected features and support labels. Four-fold support-only CV selects each baseline's regularization, and also selects the primary classical comparator across families. External query labels do not select baselines or neural checkpoints. Report every baseline separately, including any that perform better than the CV-selected comparator.

Primary contrasts: augmented neural model versus selected classical baseline at 16 shots, separately in each cohort. Require improvement with Holm-adjusted P <0.05 in both to meet the stated replication criterion. Intervals bootstrap query participants after averaging each participant's errors over support draws. These are conditional estimates, not full retraining uncertainty or proof of broad population generalization. Record all secondary contrasts separately with their own multiplicity correction.

## Reproduction

Use Python 3.12; NumPy 2.5.2, PyTorch 2.8.0, pandas 3.0.5, scikit-learn 1.9.0, SciPy 1.18.1; Matplotlib 3.11.1 for figures. Modal training uses at most two T4 containers (2 CPUs, 8 GB RAM, 20-minute timeout each), six bounded runs. Evaluation uses CPU containers.

The archive includes the source-selected numeric matrices, model weights and every external prediction for direct rerunning. Full GEO matrices are not redistributed. To reconstruct preprocessing, run src/download.py and src/prepare_source.py, then src/prepare_targets.py. Source preparation uses existing parent-project source caches when present and otherwise reads the downloaded matrices. Run Modal training, Modal evaluation, summarization, the assay-matched diagnostic and figures in that order. Source/query identifiers, hashes and design records are retained.

The assay-matched source-ridge diagnostic was added after the primary evaluation. It refits only on probes observed in the target release, selects alpha with source-group CV, and never uses external ages for fitting. Its whole-blood MAE (6.09 years) is lower than the neural model; in PBMCs its MAE is 7.89 years. This is a post hoc diagnostic, not an independently confirmed method.

## Preprocessing correction

An initial PBMC run treated normalized M-values as beta values and is invalidated. The corrected run uses beta = 1/(1+2**(-M)), as appropriate to the documented GEO sample-table M-value definition. All neural weights, baseline settings and participant partitions remained frozen. Current results, figures and archives contain the corrected evaluation. The invalidated run is retained locally as an error record, not scientific evidence; results/scale_correction.json and input_units_audit.json document the correction.
