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

- an HIV diagnostic test;
- a prospective HIV-acquisition risk model;
- a clinical decision-support system;
- a substitute for routine or universal access to testing;
- evidence that SHAP attributions are causal effects;
- externally or prospectively validated for deployment.

## Data source and access

The underlying microdata are from the **Zambia Demographic and Health Survey 2024** and were obtained through authorised access from The DHS Program.

**Raw DHS microdata are not included in this repository and must not be redistributed.** Researchers wishing to reproduce the study must independently request access from The DHS Program and comply with its terms of use.

This repository intentionally excludes respondent-level IR, MR and AR source files and related restricted DHS distribution files.

See [`docs/data_access.md`](docs/data_access.md).

## Analytical design

Key elements include:

- codebook-compliant harmonisation of women's and men's DHS variables;
- 17 primary raw predictors grouped into four analytical domains;
- cluster-exclusive 60/20/20 training/validation/test partitioning;
- grouped hyperparameter tuning using PR-AUC as the primary refit metric;
- class-weighted candidate models for the imbalanced outcome;
- grouped out-of-fold Platt calibration for the selected weighted XGBoost model;
- repeated cluster-grouped capacity validation from 1.0% to 50.0% in 0.1-percentage-point increments;
- F2 as the primary capacity-selection criterion;
- subgroup auditing by sex, residence, age, wealth, region and education;
- cluster-aware uncertainty and outer-fold subgroup stability analyses;
- SHAP interpreted as post-hoc attribution rather than causal evidence;
- a prior-positive self-report eligibility sensitivity analysis reported separately.

## Verified publication code

The recovered final-analysis archive was audited against the corrected thesis and rerun against the authorised local Zambia DHS files. A conflict in the recovered capacity script was identified: it could refit the model on the full 80% development sample before the final test and therefore generate a second, inconsistent final-test estimand.

The public publication contract resolves that conflict explicitly:

> The combined training + validation sample is used for repeated **capacity selection**, but the selected capacity is applied to the **frozen primary final-test probabilities**. The model is not refitted on the full 80% development sample for final-test evaluation.

Under this contract the code reproduces the locked reported result: **1,218 selected; TP/FP/FN/TN = 281/937/167/3,631; recall 0.6272; precision 0.2307; boundary 0.1234**.

The release includes an automated verifier covering **80 locked aggregate checks**. All checks passed in the audit run used to prepare the repository.

Code: [`src/`](src/)  
Configuration: [`config/analysis_config.yaml`](config/analysis_config.yaml)  
Reproducibility notes: [`docs/REPRODUCIBILITY_NOTES.md`](docs/REPRODUCIBILITY_NOTES.md)  
Final-analysis archive audit: [`docs/final_analysis_archive_audit.md`](docs/final_analysis_archive_audit.md)  
Audited aggregate snapshots: [`outputs/reproducibility/`](outputs/reproducibility/)  
Release/Zenodo checklist: [`docs/RELEASE_CHECKLIST.md`](docs/RELEASE_CHECKLIST.md)

## Reproduction

After receiving independent DHS authorisation, place the following files locally under `data/`:

```text
data/ZMIR81FL.dta
data/ZMMR81FL.dta
data/ZMAR81FL.dta
```

Then, from the repository root:

```bash
pip install -r requirements.txt
python src/analysis_pipeline.py
```

A successful run ends with:

```text
All locked-result checks passed.
```

DHS microdata, per-record probability arrays, split assignments, fitted model objects and other respondent-level/model-state artefacts are excluded by `.gitignore`.

## Software environment

The final analytical environment was:

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

## Recovered-source limitation

The exact historical code that generated the exploratory stacked-domain sensitivity was not present in the recovered final-analysis ZIP. A best-effort reconstruction is available as an optional analysis, but it is **not** included in locked-result verification. The repository does not fabricate the missing implementation. See [`docs/REPRODUCIBILITY_NOTES.md`](docs/REPRODUCIBILITY_NOTES.md).

## Ethics and responsible use

The research used de-identified secondary survey data. The authors had no direct contact with participants and collected no new identifiable data. The original ZDHS survey used its own ethics and informed-consent procedures. No separate Universiti Sains Malaysia ethics approval or exemption reference number was issued for this secondary analysis.

The analytical outputs should not be used to deny routine HIV testing or to make individual clinical decisions.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). Publication details and archival DOI will be updated after the journal/Zenodo release.

## Repository status

**Pre-publication reproducibility release.** The primary publication analysis is code-verified against the locked corrected thesis/manuscript results. The remaining archival steps are final author review, GitHub `v1.0.0` release creation, Zenodo DOI assignment, and insertion of the permanent DOI into the manuscript and repository metadata.
