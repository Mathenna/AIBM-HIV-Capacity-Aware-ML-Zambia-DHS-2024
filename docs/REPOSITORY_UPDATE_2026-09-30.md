# Revised analysis code deposit

This update adds the tested post-review implementation used for the revised article, together with configurations, pinned dependencies, tests and method documentation. The executable revision files are unchanged from the supplied AIBM_revision_v2_code.zip submission archive.

Run `src/revision_pipeline.py` using `requirements-revision.txt` and `config/revision_real_run.yaml`. See the root README for installation, authorised data access and reproduction commands. The configuration label `post_review_v2_candidate` is retained exactly as recorded in the completed runs; it is a provenance identifier.

The historical implementation and aggregates remain available for provenance. Their original claims and citation metadata do not describe the revised analysis. The historical baseline is commit `449c586d2086d84660636ca84037f4545e421429`. The revised source implements eight model families, grouped nested tuning and calibration, outcome-independent prior-status eligibility, exact-budget proportional allocation and conditional cluster-bootstrap comparisons. All 13 automated tests passed before upload.

Only source code, tests, configuration and documentation are added or updated here. Restricted DHS microdata, respondent identifiers, probability checkpoints, fitted models, credentials and local environment files are excluded. The revised manuscript, figures and aggregate-result package are supplied separately for journal submission.

This deposit does not create a journal publication, DOI, archival tag or GitHub Release. It does not establish external validation or clinical deployment readiness. Use this update's immutable commit URL when identifying the revised code in the manuscript.
