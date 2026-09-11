# Capacity-Aware and Subgroup-Audited Machine Learning for HIV Biomarker-Status Prediction Using Zambia DHS 2024

Reproducibility and publication-support repository for the manuscript:

**Capacity-Aware and Subgroup-Audited Machine Learning for HIV Biomarker-Status Prediction Using Zambia DHS 2024**

## Authors

1. **Mathenna A/P Karunanethe** — School of Computer Sciences, Universiti Sains Malaysia
2. **Vaithegy Doraisamy** — School of Computer Sciences, Universiti Sains Malaysia
3. **Sudersen Lekshmikanth** — Computtify Tech; Industrial Advisor, Politeknik METrO Tasek Gelugor

**Corresponding author:** Vaithegy Doraisamy — vaithegy@usm.my

## Study overview

This study evaluates a cluster-aware machine-learning workflow for retrospective prediction of **contemporaneous HIV biomarker status** and capacity-constrained prioritisation using Zambia Demographic and Health Survey (ZDHS) 2024 data.

The primary analysis linked women's and men's respondent records to HIV biomarker records and produced an analytic cohort of **25,491 respondents**, including **2,245 HIV-positive biomarker cases**. Five model families were compared under cluster-exclusive development. The selected model was a Platt-calibrated, class-weighted XGBoost classifier.

Headline held-out test results for the primary analysis were:

- ROC-AUC: **0.7972**
- PR-AUC: **0.3694**
- Brier score: **0.0684**
- Validation-selected nominal capacity: **24.3%**, chosen by maximum mean F2 across repeated cluster-grouped development validations
- Test-set recall at 24.3% capacity: **0.6272**
- Test-set precision at 24.3% capacity: **0.2307**
- Yield enrichment: **2.58×**

Subgroup auditing showed materially lower recall among some groups, particularly respondents aged 15–24 years. The study therefore treats subgroup performance as a safety and reporting issue rather than claiming fairness or deployment readiness.

## Important interpretation boundary

This repository and manuscript concern **retrospective HIV biomarker-status prediction**. The model is **not**:

- an HIV diagnostic test,
- a prospective HIV-acquisition risk model,
- a clinical decision-support system,
- a substitute for routine or universal access to testing,
- evidence of causal effects from SHAP attributions,
- externally or prospectively validated for deployment.

## Data source and access

The underlying microdata are from the **Zambia Demographic and Health Survey 2024** and were obtained through authorised access from The DHS Program.

**Raw DHS microdata are not included in this repository and must not be redistributed.** Researchers wishing to reproduce the study must independently request access from The DHS Program and comply with its terms of use.

This repository intentionally excludes all respondent-level IR, MR and AR source files and related DHS distribution files.

See [`docs/data_access.md`](docs/data_access.md) for details.

## Analytical design

Key elements of the primary workflow include:

- codebook-compliant harmonisation of women's and men's DHS variables;
- 17 primary raw predictors grouped into four analytical domains;
- cluster-exclusive 60/20/20 training/validation/test partitioning;
- grouped hyperparameter tuning using PR-AUC as the primary refit metric;
- class-weighted candidate models for the imbalanced outcome;
- out-of-fold Platt calibration for the selected weighted XGBoost model;
- repeated cluster-grouped capacity validation across exact nominal capacities from 1.0% to 50.0% in 0.1-percentage-point increments;
- F2 as the prespecified primary capacity-selection criterion;
- subgroup audits by sex, residence, age, wealth, region and education;
- cluster-bootstrap uncertainty for final test metrics;
- TreeSHAP with interventional feature perturbation on the uncalibrated XGBoost raw-margin scale;
- a prior-positive self-report eligibility sensitivity analysis reported separately.

## Software environment

The thesis-recorded final analytical environment was:

- Python 3.13.5
- pandas 2.2.3
- NumPy 2.3.5
- SciPy 1.17.0
- scikit-learn 1.8.0
- XGBoost 3.1.3
- SHAP 0.50.0
- Matplotlib 3.10.8
- joblib 1.5.3
- PyYAML 6.0.3

The primary random state was **24101765**. Five-fold paired model stability used `random_state=24101766`, and repeated capacity validation used seeds **24101765–24101774**.

## Repository contents

The repository contains only materials that are safe to redistribute, including:

- documentation and environment specifications;
- aggregate analysis outputs;
- prior-status sensitivity outputs;
- publication figures;
- reproducibility notes.

The final thesis reproducibility manifest references `analysis_pipeline.py`, `analysis_config.yaml`, complete grid-search outputs, bootstrap replicates and additional result tables. A subsequently supplied archive (`thesis_claude.zip`) was audited and found to contain an **earlier pre-correction pipeline** rather than the final analysis. It also contains restricted DHS `.dta` files and therefore has not been uploaded wholesale.

See [`docs/thesis_claude_archive_audit.md`](docs/thesis_claude_archive_audit.md) for the detailed comparison.

Accordingly, this repository should not yet be described as a fully rerunnable end-to-end implementation of the final manuscript. Final analysis code should only be deposited once it has been verified to reproduce the corrected thesis results.

## Reproduction

1. Request and obtain authorised ZDHS 2024 data access from The DHS Program.
2. Keep all DHS microdata outside the Git repository.
3. Reconstruct the IR/MR/AR source paths locally.
4. Use the software versions in `requirements.txt` / `environment/environment_notes.md`.
5. Review `docs/reproducibility.md` for design and seed controls.
6. Use aggregate outputs in `outputs/` to cross-check publication results.

## Ethics and responsible use

The research used de-identified secondary survey data. The authors had no direct contact with participants and collected no new identifiable data. The original ZDHS survey used its own ethics and informed-consent procedures. No separate Universiti Sains Malaysia ethics approval or exemption reference number was issued for this secondary analysis.

The analytical outputs should not be used to deny routine HIV testing or to make individual clinical decisions.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). Publication details and DOI will be updated after journal publication / archival release.

## Repository status

**Pre-publication reproducibility release.** The manuscript has been prepared for submission to *AI Biology & Medicine*. Repository contents may be updated before the archival release used for the final DOI.
