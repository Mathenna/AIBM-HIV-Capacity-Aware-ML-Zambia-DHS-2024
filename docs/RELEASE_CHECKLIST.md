# Publication / Zenodo release checklist

Use this checklist before creating the archival software release associated with the manuscript.

## 1. Scientific reproducibility

- [x] Primary analytic cohort reproduces N = 25,491 and HIV-positive n = 2,245.
- [x] Cluster-exclusive train/validation/test counts reproduce 15,334 / 5,141 / 5,016.
- [x] Frozen weighted-XGBoost test ROC-AUC reproduces 0.7972.
- [x] Frozen weighted-XGBoost test PR-AUC reproduces 0.3694.
- [x] Frozen weighted-XGBoost Brier score reproduces 0.0684.
- [x] Repeated grouped validation selects 24.3% by maximum mean F2.
- [x] Final 24.3% test result reproduces 281 TP, recall 0.6272 and precision 0.2307.
- [x] One-standard-error, F1, Youden and fixed-threshold sensitivity results reproduce the reported operating region.
- [x] Key subgroup recall values reproduce the manuscript/thesis.
- [x] Final-test calibration intercept/slope/ECE reproduce the reported values.
- [x] Pooled outer-fold youth, male and rural recall reproduce the reported values.
- [x] Domain-only and leave-one-domain-out results reproduce the reported values.
- [x] Automated locked-result verifier passes all 80 checks.

## 2. Restricted-data protection

- [x] No Zambia DHS `.dta` files are committed.
- [x] `.gitignore` excludes DHS data formats and raw-data directories.
- [x] Split assignments containing cluster/household/line identifiers are excluded.
- [x] Per-record calibrated/base probability arrays are excluded.
- [x] Fitted `.joblib`, pickle and model-state artefacts are excluded.
- [x] Credentials, tokens and local environment files are excluded.
- [ ] Re-check `git ls-files` immediately before archival release for accidental restricted files.

## 3. Documentation

- [x] Root README explains the analytical scope and claim boundaries.
- [x] Data-access instructions explain independent DHS authorisation requirements.
- [x] Configuration is stored in `config/analysis_config.yaml`.
- [x] Reproducibility notes document the frozen-final-test analytical contract.
- [x] Older `thesis_claude.zip` is documented as a pre-correction archive and is not deposited.
- [x] `FINAL_ANALYSIS_CODE_ZAMBIA_DHS_2024(1).zip` audit and the 80%-refit conflict are documented.
- [x] CRediT-compatible author order and repository citation metadata are recorded.
- [ ] Update README/CITATION.cff with the final article DOI after acceptance/publication.

## 4. Known limitation to resolve or disclose

- [ ] Decide final manuscript treatment of the stacked-domain architectural sensitivity. The exact historical stacked implementation was not recovered. Either locate it before archival release or retain the current explicit limitation / soften the manuscript claim.

## 5. Author approval

Before the archival release, all authors should review the deposited code and documentation and approve the public release:

- [ ] Mathenna A/P Karunanethe
- [ ] Vaithegy Doraisamy
- [ ] Sudersen Lekshmikanth

## 6. GitHub archival release

Recommended first archival tag:

```text
v1.0.0
```

Recommended release title:

```text
AIBM manuscript reproducibility release v1.0.0
```

Recommended release notes should state that:

- this is the code and aggregate reproducibility package for the AIBM manuscript;
- raw Zambia DHS 2024 microdata are not redistributed;
- users must obtain authorised DHS access independently;
- the selected 24.3% operating point is an analytical capacity scenario rather than a clinical threshold or measured programme constraint;
- the model is not externally/prospectively validated or deployment-ready.

## 7. Zenodo

After the GitHub release is final:

- [ ] Connect the GitHub repository to Zenodo.
- [ ] Archive release `v1.0.0`.
- [ ] Confirm title, author order and affiliations.
- [ ] Confirm software/reproducibility-package resource type.
- [ ] Confirm licence metadata.
- [ ] Obtain the version-specific DOI and concept DOI.
- [ ] Insert the DOI in the manuscript Data and Code Availability section.
- [ ] Add the DOI to `CITATION.cff` and README.
- [ ] Preserve the exact archived release used for journal submission.

## 8. Journal submission

- [ ] Ensure the manuscript links to the public GitHub repository and/or Zenodo DOI.
- [ ] Ensure the Data and Code Availability statement explicitly says DHS microdata cannot be redistributed.
- [ ] Confirm all authors approve the exact submitted manuscript and code release.
- [ ] Retain a local copy of the release ZIP and Zenodo metadata for the submission record.
