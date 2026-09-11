# Reproducibility design

This document records the methodological controls used in the submitted thesis/manuscript.

## Data construction

- Women’s and men’s individual recode records were harmonised and linked to HIV biomarker records.
- The final analytic cohort contained **25,491 respondents**, including **2,245 HIV-positive biomarker cases**.
- Survey-specific special values were interpreted from the Zambia DHS documentation rather than treated as literal measurements.
- Age at first sex was represented using both a numeric component and a sexual-debut status component so that special codes such as no prior sex and inconsistent response were not treated as ages.

## Group-preserving validation

- Complete sampling clusters were assigned to development or evaluation partitions.
- The primary design used an approximate **60/20/20 train/validation/test split**.
- No cluster or household was allowed to cross the primary partition boundaries.
- Hyperparameter tuning and preprocessing were contained within grouped training folds.

## Candidate models

Five model families were compared under a common analytical contract:

1. Logistic regression
2. Class-weighted random forest
3. Gradient boosting
4. Subsampled gradient boosting
5. Weighted XGBoost

PR-AUC was the primary model-selection metric because HIV biomarker positivity was the minority outcome.

## Selected model and calibration

The selected model was weighted XGBoost. Its selected configuration within the reported search grid used:

- `n_estimators = 400`
- `max_depth = 3`
- `learning_rate = 0.04`
- `min_child_weight = 5`

Class weighting altered the raw probability scale, so the selected model was calibrated using grouped out-of-fold **Platt scaling** before probability reporting and capacity-based selection.

## Capacity selection

- The original 80% development sample was repeatedly repartitioned using cluster-grouped folds.
- Ten repeated grouped validations were evaluated.
- Exact top-k capacities from **1.0% to 50.0%** were evaluated in increments of **0.1 percentage points**.
- The prespecified primary criterion was mean **F2**, giving greater weight to recall than precision.
- The selected nominal capacity was **24.3%**.
- The final held-out test set was used only after the capacity had been selected on development data.

## Final held-out results

At 24.3% nominal capacity on the final test partition:

- N = 5,016
- selected = 1,218
- true positives = 281
- false positives = 937
- false negatives = 167
- true negatives = 3,631
- recall = 0.6272
- precision = 0.2307
- specificity = 0.7949
- F2 = 0.4668
- yield enrichment ≈ 2.58×

## Subgroup audit

Subgroup analyses were reported for sex, residence, age, wealth, region and education. They were used to identify disparities in predictive/operational performance, not to claim demographic parity or fairness.

A particularly important finding was low recall among respondents aged 15–24 years.

## SHAP

- TreeExplainer
- interventional feature perturbation
- 200-record training background
- model output: raw XGBoost margin
- all 5,016 final-test respondents explained

SHAP values were interpreted as model-attribution quantities only. They were not treated as causal effects or intervention recommendations.

## Prior-positive self-report sensitivity

A publication-specific sensitivity analysis excluded respondents who self-reported a positive result at their most recent prior HIV test. This analysis was designed to test how eligibility definition changed the prediction task. It does **not** establish an undiagnosed-HIV cohort.

Machine-readable aggregate outputs for this sensitivity are provided under `outputs/prior_status_sensitivity/`.

## Current repository limitation

The thesis reproducibility manifest references an original `analysis_pipeline.py`, `analysis_config.yaml`, complete grid-search outputs, bootstrap replicates and additional result tables. Those exact artefacts are not present in this chat session. This repository therefore documents and exposes the available non-restricted materials without pretending to reconstruct missing source code.
