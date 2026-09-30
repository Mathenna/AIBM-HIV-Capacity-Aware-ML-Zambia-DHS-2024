# AIBM post-review analysis revision

**Status: revised analysis code deposited on 30 September 2026; real-data reanalysis completed on 29 September 2026.** Eight declared cohort/predictor analyses compare eight model families across five outer folds, with 1,000 conditional cluster-bootstrap replicates per analysis. The revised manuscript and aggregate results are supplied separately in the companion AIBM_reanalysis_20260929 submission package. This repository update deposits the revised code, tests and configurations; it does not include the companion manuscript or revised aggregate-result package. This is post-review internal validation; journal acceptance and clinical deployment validity are not established.

The target is AI Biology & Medicine. Read [the post-review protocol](docs/REVISION_PROTOCOL.md) before running. The paper should remain an internal evaluation of contemporaneous HIV biomarker ranking, with eligibility and subgroup consequences central to its conclusions.

## What changed

- Exact-budget subgroup allocation using largest remainders, including correct handling of zero capacity and deterministic score ties.
- Weighted and unweighted XGBoost, ordinary/spline logistic regression, demographic spline regression, random forest and two boosting comparators.
- Class weights computed within every fit, including inner tuning folds.
- Grouped hyperparameter search inside each calibration fold, then outer held-out evaluation of all families and the model-selection procedure.
- Prior-positive eligibility restriction read from V861/MV861 with label checks; no restriction based on HIV03.
- Fixed workload comparisons, training-selected F2 policies, survey-weighted metrics, common-age and no-testing-history sensitivities.
- Cluster resampling with top-k reselection, paired model differences, subgroup disparity intervals and proportional-allocation intervals.
- Fresh-run output directories, runtime/config/source/input hashes, convergence failures surfaced, aggregate-only outputs.

## Installation and tests

Use a local virtual environment. Python 3.12 is used for revision tests; runtime versions are recorded. The original study reported Python 3.13.5, so this does not certify reproduction in its exact historical runtime.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-revision.txt
.venv\Scripts\python -m unittest discover -s tests -v
```

Tests include a synthetic nested-model workflow. Synthetic success verifies software behaviour only, not scientific validity or expected DHS performance. See `TEST_STATUS.md` for the observed local validation status.

## Local data preflight

Keep the three authorised files in a local folder; no upload is needed. Use an output directory that does not already exist.

```powershell
.venv\Scripts\python src/revision_pipeline.py --data-dir "D:\AuthorisedDHS" --output "results/preflight_01" --preflight-only
```

The folder must contain `ZMIR81FL.dta`, `ZMMR81FL.dta` and `ZMAR81FL.dta`. Preflight checks linkage, prior-status labels, eligible ages, survey design fields and file/split provenance. Inspect the actual survey codebook and counts too; automated checks do not certify all recode semantics.

## Revised model comparison

```powershell
.venv\Scripts\python src/revision_pipeline.py --data-dir "D:\AuthorisedDHS" --output "results/revision_main_01"
```

The default evaluates full predictors for the all-eligible and prior-positive-excluded cohorts. It uses five outer folds, with nested tuning and calibration. It can take substantial time; all eight families are trained repeatedly. Console messages show cohort, outer fold and model.

For the common-age and testing-history sensitivities:

```powershell
.venv\Scripts\python src/revision_pipeline.py --data-dir "D:\AuthorisedDHS" --output "results/revision_sensitivities_01" --cohorts all restricted all_15_49 restricted_15_49 --feature-sets full no_testing_history
```

This full command repeats the main analyses as part of a single configuration-consistent run. Do not run both commands unless needed; use the main run to establish feasibility and decide the declared sensitivity scope before inspecting new results.

## Result files

Each cohort/feature analysis exports fold performance, pooled fixed-capacity performance, subgroup counts, nested search summaries, selected configurations, training capacity curves, calibration diagnostics/bins, conditional cluster intervals, paired model comparisons, equity-policy intervals and paired recall disparities. Default result outputs contain no per-person predictions or fitted model files.

An optional `--checkpoint-dir` saves **private record-level probability arrays** to resume expensive fits. Keep that folder outside any repository, publication folder or archive. Use checkpoints only with the same pinned Python/library environment; the cache fingerprint covers data, configuration and model-development source, but is not an environment-migration mechanism. For a clean reproduction use a new cache or omit the option. Completed cache files are marked only after their arrays and search records have been written.

The bootstrap uses exact integer multiplicities instead of materialising repeated respondents. It recalculates budgets and boundary selections, and matches the explicit implementation in `revision_policy.py`. The capacity search similarly sorts once and evaluates cumulative positive counts. Neither optimisation changes the declared method. `tests/test_revision.py` includes equivalence tests.

`selected_pipeline` evaluates family selection made inside each outer-training sample. It does not mean that one globally chosen model was tested independently after looking at every outer fold. The five outer folds reuse the historical data assignment; call this post-review nested internal validation, never a new untouched test or external validation.

The bootstrap intervals condition on fitted OOF scores, resample PSUs within strata and folds, and reselect each fold's exact budget. They do not include complete model-development uncertainty. Singleton stratum/fold blocks are counted in `bootstrap_design.json` and require interpretation.

## Historical files

The original public `src/analysis_*.py`, `publication_*.py`, `capacity_selection.py`, `core_model_pipeline.py`, `verify_locked_results.py`, `config/analysis_config.yaml`, old aggregate `outputs/`, and original repository documentation are retained for provenance. Their claims about 80 checks apply only to the authors' historical audit. They are not new validation of this revision.

[Historical README](docs/HISTORICAL_README.md) preserves the original description. `CITATION.cff` also describes that historical release, not an already published revision. Do not submit the entire folder as a final journal package without curating the intended source/result set.

The v2 workflow lives exclusively in `src/revision_*.py`, `config/revision_config.yaml` and `tests/test_revision.py`. It intentionally does not silently alter the old implementation or force agreement with previously reported results.

## Completed analysis and author review

The authorised Zambia DHS files were used locally. Cohort linkage, recode labels and frequencies, eight nested comparisons, conditional uncertainty, all sensitivity analyses, tables, figures and manuscript have been checked. The executable source hashes match the final run manifests. Thirteen automated tests passed; the optimised bootstrap also agreed with explicit resampling on actual out-of-fold predictions.

The supplied main manuscript retains the factual V7 ethics statement and DHS access permission date of 17 March 2026. No new institutional approval or exemption is asserted. SHAP findings were omitted because that explanation analysis was not regenerated. The historical code reproduced cohort counts but passed only 28 of 80 locked output checks; the discrepancy is disclosed rather than hidden.

This repository update deposits the revised analysis code, tests, configurations and documentation. Before journal submission, all authors should verify and approve the revised claims, affiliations and contribution/declaration statements. Cite the exact Git commit used for the revised code; the companion aggregate results remain supplied separately with the submission. This code deposit is not a journal submission or a statement that all authors have approved the manuscript. External validation remains necessary before deployment claims, but is reported honestly as a limitation of this retrospective article.

## Reproduce the reported configuration

`config/revision_real_run.yaml` records the actual four-worker configuration; each fitted estimator uses one thread. Use the pinned requirements and Python 3.12.14. A clean full run can be launched with:

```powershell
python src/revision_pipeline.py --data-dir "D:\AuthorisedDHS" --output "results/revision_complete_01" --config config/revision_real_run.yaml --cohorts all restricted all_15_49 restricted_15_49 --feature-sets full no_testing_history
```

The delivered runs were split into main, common-age and no-history jobs with the same settings. Private resumable checkpoints were used in the same runtime; a clean reproduction needs no checkpoint. The individual manifests retain actual run boundaries and hashes. The companion package's `verification` directory contains completed aggregate checks and real-data optimisation equivalence evidence.

The companion `reporting_source` folder preserves the scripts and text used to generate the reported figures, tables and Word documents. Those scripts retain the original workspace-relative run paths to make their provenance explicit; the model-fitting CLI above is portable. Runtime packages for optional figure generation are in `requirements-reporting.txt`.
