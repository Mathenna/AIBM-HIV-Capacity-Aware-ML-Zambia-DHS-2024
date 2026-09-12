# Audit of `FINAL_ANALYSIS_CODE_ZAMBIA_DHS_2024(1).zip`

## Purpose

This note records the audit of the recovered final-analysis archive used to prepare the public journal reproducibility release.

Unlike the older `thesis_claude.zip`, this archive is closely aligned with the corrected thesis and reproduces the main final-analysis results. It nevertheless contained one important inconsistency that had to be resolved before public release.

## What was verified from the recovered archive

The recovered code was executed against the authorised local Zambia DHS 2024 IR/MR/AR files. The following headline results were reproduced:

- analytic sample: 25,491 respondents;
- HIV biomarker positive: 2,245;
- cluster-exclusive train/validation/test: 15,334 / 5,141 / 5,016;
- positive counts: 1,357 / 440 / 448;
- zero cluster and household overlap across the three partitions;
- selected model: weighted XGBoost;
- test ROC-AUC: 0.7972;
- test PR-AUC: 0.3694;
- test Brier score: 0.0684;
- repeated-development mean-F2 optimum: 24.3%;
- mean F2 at that capacity: 0.4991;
- one-standard-error F2 capacity: 21.3%;
- maximum-mean-F1 capacity: 9.9%;
- maximum-mean-Youden-J capacity: 26.3%;
- fixed-threshold sensitivity optimum: 0.1218.

The primary frozen final-test probabilities also reproduced the reported 24.3% operating-point result:

- selected: 1,218 of 5,016;
- TP / FP / FN / TN: 281 / 937 / 167 / 3,631;
- recall: 0.6272;
- precision: 0.2307;
- specificity: 0.7949;
- F2: 0.4668;
- final selected-record boundary: approximately 0.1234.

## Critical inconsistency identified

The recovered `capacity_selection.py` included an additional path that, after selecting the capacity on the 80% development sample, refitted weighted XGBoost on the complete training + validation development set and recalibrated it before scoring the held-out test partition.

That refit produced a second final-test result that differed from the final thesis/manuscript result. In the audit run, the 80%-refit path produced approximately:

- TP / FP / FN / TN: 290 / 928 / 158 / 3,640;
- recall: 0.6473;
- precision: 0.2381;
- boundary: approximately 0.1226.

This creates two incompatible final-test estimands if left unresolved.

## Publication analytical contract

The public release uses one final-test contract only:

1. the primary model is selected and calibrated under the original cluster-exclusive 60/20/20 development design;
2. its final-test probabilities are frozen;
3. the combined training + validation 80% development sample is used for repeated capacity selection only;
4. the selected nominal capacity is then applied to the frozen primary test probabilities;
5. the model is **not** refitted on the complete 80% development sample before final-test evaluation.

This is the contract implemented in the public `src/capacity_selection.py` and it preserves the locked final-thesis/manuscript result of 281 true positives at 24.3% capacity.

## Additional publication extensions reproduced

The audited release also reproduces or verifies the following aggregate publication evidence:

- final-test calibration intercept, slope and 10-bin ECE;
- five-fold weighted-XGBoost model-stability summary;
- pooled outer-fold subgroup recall for overall, youth, male and rural groups;
- domain-only and leave-one-domain-out test performance;
- exact capacity criteria and fixed-threshold sensitivity.

The automated verifier checks 80 locked aggregate quantities. All 80 checks passed in the audit run used to prepare the repository.

## `scale_pos_weight` clarification

The recovered source uses related but not identical weighting conventions across analyses. Repeated capacity validation and complete-cohort outer subgroup validation recompute `N_negative/N_positive` within the relevant outer/development training subset. The locked five-fold candidate-model stability table is reproduced using the primary-training selected value 10.2999 while the grouped outer partition itself uses random state 24101766.

These are documented separately in `docs/REPRODUCIBILITY_NOTES.md` rather than being described as one identical fold-weighting procedure.

## Stacked-domain limitation

The final thesis reports a secondary stacked-domain sensitivity. The exact historical implementation that produced the reported stacked-domain metrics was not present in the recovered archive. The repository therefore provides only an optional best-effort reconstruction and deliberately excludes that analysis from locked-result verification.

The public release does not fabricate missing historical code.

## Data-protection review

The archive supplied for this audit did not contain the Zambia DHS `.dta` files, but running the analysis locally generates respondent-level or model-state artefacts that should not be committed publicly, including split assignments, per-record probability arrays and fitted model files. The repository `.gitignore` excludes these artefacts together with raw DHS source files.

## Release decision

The recovered archive was suitable as the basis for the final publication code after the final-test refit conflict was removed, configuration was made operational, locked-result checks were added, and public outputs were restricted to aggregate non-identifiable evidence.
