# Revision validation status

Verified locally on 29 September 2026. **13 automated tests passed** in 21.048 seconds. `pip check` found no broken requirements. All eight declared real-data analyses completed, each with eight model families, five cluster-exclusive outer folds and 1,000 conditional cluster-bootstrap replicates.

Runtime: Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, scikit-learn 1.8.0, XGBoost 3.1.3, PyYAML 6.0.3, joblib 1.5.3 and threadpoolctl 3.6.0.

## Checks performed

- Exact global and proportional subgroup budgets, including zero, full and small capacities.
- Stable tied-score selection under row reordering, input validation, confusion arithmetic and F2.
- Eligibility independent of the biomarker outcome and Stata label/linkage checks.
- Fold-local class weights, cloning, preprocessing, grouped tuning and monotonic calibration.
- Synthetic end-to-end nested evaluation and all-eight-family prediction smoke checks.
- Optimised cumulative capacity selection against the 491-point reference search.
- Integer-multiplicity bootstrap against explicit resampling with ties, unequal clusters, survey weights and proportional policies.
- Real-data bootstrap equivalence on five identical draws for each of four policies; maximum observed absolute difference below 8e-12.
- Completed aggregate results: source hashes, counts, pooled reconciliation, equal budgets, positive calibration slopes and 1,000 valid overall bootstrap replicates.

The complete cohort has 25,491 respondents and 2,245 positive outcomes; the prior-positive-excluded cohort has 24,136 and 972. Sparse joint stratification warnings were retained, while both outcome classes and cluster separation were checked. Seven singleton stratum/fold blocks in the full cohort cannot vary under the conditional bootstrap.

## Limits

This supports a tested implementation and completed internal analysis. It does not establish external validity, prospective clinical benefit or a novel algorithm. The bootstrap conditions on fitted predictions and does not repeat development. SHAP was not regenerated and its historical findings are not retained in the revised manuscript. A separate historical reproduction passed 28 of 80 locked checks; its computational discrepancy remains unresolved. See the companion package for the preserved evidence and revised results.

## Pre-upload verification on 30 September 2026

The unchanged revised implementation passed all 13 automated tests again in 20.676 seconds before this repository update. Synthetic sparse-class warnings were emitted; all tests completed successfully. The full real-data analysis was not rerun for this code deposit.
