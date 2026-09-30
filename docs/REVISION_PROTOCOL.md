# Post-review analysis protocol

Target journal: AI Biology & Medicine, confirmed by the user on 28 September 2026.

Status: specified after examining the historical results, before the revised DHS rerun. This is not prospective preregistration. The historical result set remains an auditable record. Any revised numbers must be generated from authorised data and reconciled before changing the paper.

## Scientific question

How do prior-status eligibility, predictive algorithm, and a fixed outreach workload affect internal discrimination, calibration, biomarker-positive retrieval and subgroup access in Zambia DHS 2024?

The primary outcome remains contemporaneous HIV biomarker status. Neither a prior-negative self-report nor absence of a prior-positive self-report establishes undiagnosed infection. The analysis estimates retrospective prioritisation, not new diagnoses, causal risk or health benefit.

## Cohorts and predictor sensitivities

Run all eligible linked respondents and respondents excluding self-reported prior-positive V861/MV861=1. Eligibility is independent of HIV03. Confirm code 1 labels in both Stata recodes before running. Report missing/declined/other self-report categories explicitly.

Use common outer cluster assignments across the two cohorts. The revised code preserves the historical five-fold cluster assignment on the full cohort before restriction and corrects age-group labels afterward. It does not create a new untouched test. Count overlap and group separation are checked in code; source/split hashes are recorded locally.

Primary predictor comparisons use the full 17-predictor specification. Repeat with health/testing-history features excluded. Separately restrict both sexes to ages 15–49 to examine age-composition effects on sex differences. Do not choose the cohort or predictor specification that produces the best headline score after seeing results.

The standard run defaults to the two cohorts with full features. The all-sensitivities command covers the four age/eligibility cohorts and both full/no-testing-history features. It can be computationally substantial; estimate runtime from initial folds.

## Algorithms

Evaluate demographic spline logistic regression, ordinary logistic regression, spline logistic regression with all predictors, weighted random forest, gradient boosting, subsampled gradient boosting, unweighted XGBoost and fold-weighted XGBoost. The demographic comparator uses age, sex, residence and region. This is a small interpretable comparator, not a causal model.

Keep the historically plausible XGBoost regularisation and use a compact common grid for weighted/unweighted variants. Weight ratios are calculated inside each estimator fit, including each tuning fold. Spline terms allow nonlinear continuous effects; categorical variables remain one-hot encoded. Missing indicators are included, and every transformation is fitted within its own training sample.

No deep-learning model or synthetic oversampling is added by default. The question is whether more complexity or class weighting adds value, not whether every possible model family can be fitted.

## Nested internal validation

Each of five outer folds is evaluated after all relevant development decisions are made using the other folds. Within each outer-training sample:

1. Generate grouped out-of-fold base predictions. For each calibration holdout, perform a separate grouped inner hyperparameter search using only that calibration fold's training clusters.
2. Fit a Platt calibrator on these cross-fitted logits. Require a positive slope; stop on convergence failures rather than silently export invalid results.
3. Independently tune the final base estimator using the complete outer-training sample and grouped inner CV, then refit there.
4. Compute each family's training cross-fitted average precision. Families within 0.005 of the best are eligible; choose the first in the explicitly recorded preference list, favouring simpler models.
5. Select an exploratory F2 capacity from cross-fitted training scores only, resolving ties toward smaller workload. A monotonic calibrator preserves ranking; calibration is not claimed to improve top-k retrieval.
6. Apply frozen development choices to the held-out outer fold.

This evaluates both each family and the complete selection procedure. The calibrated model may differ between outer folds. There is no single claimed universal winning model until comparative evidence is interpreted. Average precision is explicitly named; it is not trapezoidal PR-curve area.

Outer-fold evaluation is nested with respect to these executable tuning and selection steps. It is still internal and post-review: analyst choices reflect previous knowledge of the same survey. Reusing these data does not repair the absence of external validation.

## Policy evaluation

Report performance at fixed capacities 5%, 10%, 15%, 20%, 24.3% and 30%. The historical 24.3% is a reference, not a newly externally justified service budget. F2-selected capacity is an additional training-selected policy, with per-fold values reported.

Exact budget is floor(C*N). For proportional sex, residence and age allocation, distribute that global integer budget using largest remainders of K*n_group/N. Ties in remainder use sorted group labels; ties in scores use stable hash keys derived locally from identifiers and the seed. This can change historical tied selections; reconcile explicitly.

Always show recall, precision, selected count, missed positive count and enrichment. Report random-selection expectations and model differences at equal capacity. National survey-weighted metrics are a sensitivity; policy budgets continue to count people, not survey-weight totals.

No participant's low model rank implies denial of routine testing. The paper must not present an F2 optimum as a measured cost-effective testing policy.

## Uncertainty

Resample complete PSUs within survey strata and outer folds; recompute each fold's top-k list in every resample. Use common bootstrap draws across models for paired differences and across subgroup allocation policies for comparability. Report numbers of valid replicates and contributing positive clusters. Singleton stratum/fold cells are reported because the conditional bootstrap cannot vary those clusters.

These intervals condition on the fitted cross-validated scores. They do not include repeated model-development uncertainty and are not a full design-based national variance estimator. A repeated complete nested fit or external cohort provides complementary evidence. Do not describe the old repeated-holdout SD/sqrt(10) as an independent-sample standard error.

Primary uncertainty outputs cover the historical 24.3% reference, with stratified allocation and paired male–female, rural–urban and youth–35–44 recall gaps. Other capacity-specific uncertainty can be prespecified through the configuration before rerunning.

## Reporting decisions

Keep the historical primary results clearly labelled and explain deviations in revised results. No code should be tuned to force new outputs to match 80 old numbers.

Use a result-to-source manifest and fresh output directories. Runtime versions, resolved config and input/source hashes accompany each run. The default outputs are aggregates; no raw data, per-person predictions or fitted models are included in the delivery.

Retain SHAP only as a secondary descriptive analysis if its claimed findings can be regenerated and checked. The v2 core does not depend on SHAP. Delete the claim of an empirical explanation-layer ablation: unchanged selection is a property of the pipeline. Add no new SHAP conclusions without a rerun of the explanatory analysis.

External validation and genuine programme costs/capacity remain outside this internal analysis. The manuscript retains the factual supplied ethics statement and DHS access date; it does not invent a new approval. Author review and approval precede submission.

## Before declaring the manuscript ready

- Validate raw codebook semantics and reproduce eligibility/linkage counts.
- Complete real-data runs and resolve all failures; inspect any changed result instead of altering a test to hide it.
- Inspect calibration, comparator differences, subgroup counts and uncertainty, and weighted/common-age/no-history sensitivities.
- Decide whether the empirical findings justify a focused internal-evaluation article; avoid claiming clinical deployment readiness.
- Regenerate every retained table/figure and update the abstract, results, limitations, repository and reporting checklist together.
- Check official AI Biology & Medicine author instructions and retain an accurate account of ethics and data permission.
- Obtain all authors' approval of the final scientific claims and submission files.

## Primary technical references

- [Grouped stratified validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)
- [Spline features](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.SplineTransformer.html)
- [XGBoost parameters](https://xgboost.readthedocs.io/en/stable/parameter.html)
- [TRIPOD+AI reporting](https://www.bmj.com/content/bmj/385/bmj-2023-078378.full.pdf)
- [PROBAST+AI assessment](https://www.bmj.com/content/388/bmj-2024-082505)
