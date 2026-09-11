# Audit of uploaded `thesis_claude.zip`

## Purpose

This note records the publication-reproducibility review of the `thesis_claude.zip` archive supplied after the final thesis had already been corrected and submitted.

## Conclusion

The archive **must not be used as the final publication pipeline without revision**. It is an earlier, pre-correction analytical codebase dated principally July 2026 and does not reproduce the final corrected thesis methods or headline results.

The archive is retained outside this repository as source material for possible code reconstruction. It is **not uploaded wholesale** because it contains restricted DHS microdata and obsolete analytical logic.

## Restricted content identified

The archive contains respondent-level Zambia DHS source files under `data/zambia/`, including:

- `ZMAR81FL.dta`
- `ZMIR81FL.dta`
- `ZMMR81FL.dta`

These are restricted DHS microdata and are excluded from GitHub.

The archive also contains draft thesis DOCX files, unpacked Word internals, local Claude settings, cache files, compiled Python bytecode and other working material that is not appropriate for the publication repository.

## Evidence that the archive predates the final corrected analysis

The final corrected thesis reports a cluster-aware pipeline with the primary random state `24101765`, grouped model development, Platt-calibrated weighted XGBoost, and a validation-selected exact nominal capacity of 24.3%.

By contrast, the uploaded archive contains an older `config.yaml` and run outputs with materially different settings and results:

- `random_state: 2026`
- person-level stratified 60/20/20 splitting rather than the final cluster-exclusive grouped design
- model selection recorded as `gradient_boosting`
- test ROC-AUC approximately `0.7880`
- test PR-AUC approximately `0.2669`
- selected probability threshold `0.15`
- an earlier 20% capacity constraint and coarse threshold sweep
- earlier actionability / domain-response logic that was later removed or narrowed during the post-viva corrections

These values are inconsistent with the final corrected thesis headline results:

- weighted XGBoost
- test ROC-AUC `0.7972`
- test PR-AUC `0.3694`
- test Brier score `0.0684`
- exact nominal capacity `24.3%` selected by mean F2 across repeated cluster-grouped development validations
- 1,218 of 5,016 test respondents selected
- recall `0.6272`
- precision `0.2307`

## Publication decision

The following archive components were **not** copied to the publication repository as final analysis code:

- `run_pipeline.py`
- `config.yaml`
- `src/hhml/*`
- threshold/actionability scripts
- legacy model outputs and figures
- any raw DHS data

This prevents a misleading situation in which the public repository claims to reproduce the final manuscript while actually implementing an earlier version of the study.

## Safe use of the archive

The archive may still be useful as a development reference for:

- variable harmonisation logic;
- basic package/module organization;
- test scaffolding;
- plotting utilities;
- reconstruction of the final pipeline.

Any reconstructed publication code must be checked against the final corrected thesis specification and should reproduce the final headline metrics before it is labelled as the final reproducibility pipeline.

## Required next step

Either:

1. locate the exact final `analysis_pipeline.py`, `analysis_config.yaml`, `run_metadata.json` and associated final outputs referenced in Appendix F of the corrected thesis; **or**
2. reconstruct a clean publication pipeline from the final thesis specification and validate it against the final reported outputs before depositing it here.

Until then, the repository should be described as containing **publication-supporting aggregate outputs and reproducibility documentation**, not a complete rerunnable end-to-end implementation.
