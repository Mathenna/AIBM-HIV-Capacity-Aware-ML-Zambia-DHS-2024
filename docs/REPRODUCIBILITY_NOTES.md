# Reproducibility notes for the publication release

## 1. Final-test analytical contract

The publication release uses a single final-test estimand.

The primary model is selected and calibrated using the original 60/20/20 cluster-exclusive design. The final test probabilities produced by that frozen primary model are retained. The combined training + validation (80%) development sample is used only for repeated operating-point selection. After the nominal capacity is selected, it is applied to the frozen test probabilities; the model is not refitted on all 80% development records before final-test evaluation.

This reproduces the locked reported result at 24.3% capacity: 1,218 selected, 281 TP, 937 FP, 167 FN and 3,631 TN; recall 0.6272, precision 0.2307 and boundary 0.1234.

The recovered `FINAL_ANALYSIS_CODE_ZAMBIA_DHS_2024(1).zip` contained a conflicting 80%-development refit path that generated a different final-test confusion matrix. That path is not used in the publication release because it conflicts with the frozen-model results reported throughout the final thesis and manuscript.

## 2. `scale_pos_weight` nuance

The recovered source material contained two related conventions:

- repeated capacity validation and full-cohort outer subgroup validation recompute `scale_pos_weight = N_negative / N_positive` within each outer/development training subset;
- the locked five-fold candidate-model stability table is reproduced when the selected XGBoost configuration is refitted with the primary-training selected value `13,977 / 1,357 = 10.2999`, while the outer folds themselves are generated with `random_state=24101766`.

The publication code records these analyses separately rather than presenting them as one identical weighting procedure.

## 3. Stacked-domain sensitivity

The final thesis reports a secondary stacked-domain sensitivity (four domain-specific weighted-XGBoost base learners plus a logistic meta-learner), with validation ROC-AUC/PR-AUC 0.7809/0.3349 and test ROC-AUC/PR-AUC 0.7867/0.3503.

The exact implementation that produced those four values was **not present** in `FINAL_ANALYSIS_CODE_ZAMBIA_DHS_2024(1).zip`. A best-effort reconstruction can be invoked with:

```bash
python src/publication_extensions.py --include-stacked-reconstruction
```

but it is deliberately excluded from `verify_locked_results.py` because exact numerical agreement cannot be guaranteed from the recovered source archive. The release does not fabricate the missing implementation.

For journal submission, the safest options are either:

1. locate the exact original stacked-domain implementation before claiming complete code-level reproducibility for that sensitivity analysis; or
2. remove or soften that exploratory stacked-domain result in the manuscript while retaining the fully reproducible primary, capacity, subgroup, calibration and domain-ablation analyses.

## 4. Public release versus historical source archive

The recovered historical core script also generated local respondent-level split assignments, probability arrays, fitted model objects, bootstrap replicates, diagnostic figures and additional post-hoc analyses. The public release provides the compact audited implementation required to reproduce the locked publication results without distributing those restricted or unnecessary local artefacts.

The public workflow is:

```bash
python src/analysis_pipeline.py
```

It reproduces the frozen primary model, repeated capacity selection, calibration diagnostics, outer-fold subgroup evidence and domain analyses, then executes the locked-result verifier.

## 5. Restricted outputs

The following locally generated artefacts are respondent-level or model-state artefacts and are intentionally excluded from the public repository:

- DHS `.dta` source files;
- split assignments containing cluster/household/line identifiers;
- per-record calibrated/base probability arrays;
- fitted `.joblib`/pickle model objects;
- cluster-bootstrap replicate files;
- long per-fold/per-capacity record-level or model-state artefacts that are not needed for the aggregate public evidence.

Only audited aggregate tables are deposited publicly.

## 6. Verification status

The corrected publication implementation was rerun against the authorised local Zambia DHS 2024 IR/MR/AR files. The locked verifier checks 80 aggregate values spanning sample construction, partitions, discrimination, calibration, operating-point selection, confusion counts, subgroup recall, outer-fold subgroup pooling and domain analyses. All locked checks passed in the audit run used to prepare this release.
