# Publication analysis code

This directory contains the audited reproducibility implementation used for the journal release of:

**Capacity-Aware and Subgroup-Audited Machine Learning for HIV Biomarker-Status Prediction Using Zambia DHS 2024**

## Files

- `analysis_pipeline.py` — top-level publication reproducibility workflow.
- `analysis_common.py` — DHS linkage, feature construction, cluster-exclusive splitting, preprocessing, calibration and evaluation utilities distilled from the recovered final-analysis source.
- `publication_primary_run.py` — reproduces the frozen selected weighted-XGBoost model and primary final-test probabilities.
- `capacity_selection.py` — ten repeated cluster-grouped development validations, exact capacity search and fixed-threshold sensitivity. The selected capacity is applied to the **frozen** primary test probabilities; no 80% final-test refit is performed.
- `publication_extensions.py` — calibration diagnostics, XGBoost outer-fold stability, pooled subgroup validation, domain-only and leave-one-domain-out analyses, with optional best-effort stacked-domain reconstruction.
- `verify_locked_results.py` — checks locked aggregate thesis/manuscript results and exits non-zero on material disagreement.
- `core_model_pipeline.py` — compatibility shim for the audited public workflow. The larger recovered historical retuning/SHAP driver is not silently recreated in the public release.

Configuration is stored in `../config/analysis_config.yaml`.

## Required local data

Obtain the following Zambia DHS 2024 files independently through The DHS Program and place them in the repository's local `data/` directory:

- `ZMIR81FL.dta`
- `ZMMR81FL.dta`
- `ZMAR81FL.dta`

The files are restricted and are excluded by `.gitignore`.

## Run

From the repository root:

```bash
pip install -r requirements.txt
python src/analysis_pipeline.py
```

The workflow writes local results under `results/`. Respondent-level and model-state artefacts are ignored by Git. Only separately audited aggregate snapshots are stored under `outputs/reproducibility/`.

A successful verified run ends with:

```text
All locked-result checks passed.
```

## Locked primary results

The release reproduces the locked final-thesis/manuscript primary evidence, including:

- analytic N = 25,491; HIV-positive = 2,245;
- train/validation/test = 15,334 / 5,141 / 5,016;
- test ROC-AUC = 0.7972;
- test PR-AUC = 0.3694;
- Brier score = 0.0684;
- selected capacity = 24.3%;
- selected test records = 1,218;
- TP/FP/FN/TN = 281/937/167/3,631;
- recall = 0.6272; precision = 0.2307;
- cohort-specific boundary = 0.1234.

## Important reproducibility note

The exact historical code that produced the exploratory stacked-domain sensitivity was not recovered in the supplied final-analysis archive. A best-effort reconstruction is available only with `--include-stacked-reconstruction` and is deliberately excluded from the locked verifier. See `docs/REPRODUCIBILITY_NOTES.md`.
