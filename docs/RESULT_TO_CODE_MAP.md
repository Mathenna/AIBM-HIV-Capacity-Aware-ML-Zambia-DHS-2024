# Revision evidence map

All new results require `src/revision_pipeline.py` with the recorded resolved configuration and authorised local data. The historical output directory remains provenance only.

| Claim or result | Implementation | Generated evidence |
|---|---|---|
| Linkage and prior-status eligibility | `revision_data.py` | `linkage.csv`, `prior_status_counts.csv`, manifest label checks |
| Age-appropriate cohort and sex eligibility | `revision_data.py` | preflight assertions, cohort counts in manifest |
| Grouped tuning and calibration | `revision_models.py` | `nested_search_results.csv`, `selection_provenance.json` |
| Weighted versus unweighted XGBoost | `revision_models.py` | fold and pooled performance, paired model intervals |
| Interpretable comparator performance | `revision_models.py` | same comparison outputs |
| Exact-budget allocation | `revision_policy.py` | fold/pooled policy tables and tests |
| Fixed versus training-selected F2 capacity | `revision_pipeline.py` | `training_capacity_curves.csv`, capacity-role columns in fold tables |
| Calibration | `revision_pipeline.py` | `calibration_diagnostics.csv`, `calibration_bins.csv` |
| Subgroup retrieval | `revision_pipeline.py` | `fold_subgroups.csv`, positive-cluster counts |
| Conditional model uncertainty | `revision_bootstrap.py` (optimised), `revision_policy.py` (reference) | `conditional_cluster_intervals.csv`, `paired_model_intervals.csv` |
| Equity-policy uncertainty | `revision_pipeline.py` | `equity_policy_intervals.csv`, `paired_recall_disparities.csv` |
| Survey-weight sensitivity | `revision_policy.py` | `hiv_weighted` rows in the relevant outputs |
| Repeatable provenance | `revision_data.py`, `revision_pipeline.py` | input, ordered-cohort, fold, source and config hashes; actual runtime versions |

The companion result package supplies the completed figures, tables and revised manuscript. No new SHAP, external-validation or clinical-utility evidence is claimed. The supplied factual ethics statement and DHS permission date are retained without inventing a new approval.

The result set evaluates a revised procedure. Historical numbers should be reconciled in a separate old-versus-new table with reasons for differences, not enforced as targets of the new analysis.
