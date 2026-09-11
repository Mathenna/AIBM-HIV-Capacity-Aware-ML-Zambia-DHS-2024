#!/usr/bin/env python3
"""Verify publication outputs against the locked final-thesis numbers.

The script exits non-zero if a critical result differs beyond the stated tolerance.
It intentionally checks aggregate outputs only and writes no respondent-level data.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

SRC_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_ROOT.parent
RESULTS = PROJECT_ROOT / "results"

checks: list[dict] = []


def check(name: str, observed, expected, tol: float = 0.0) -> None:
    if hasattr(observed, "item"):
        observed = observed.item()
    if hasattr(expected, "item"):
        expected = expected.item()
    if isinstance(expected, str):
        ok = str(observed) == expected
    else:
        ok = abs(float(observed) - float(expected)) <= tol
    checks.append({"check": name, "observed": observed, "expected": expected, "tolerance": float(tol), "pass": bool(ok)})


sample = pd.read_csv(RESULTS / "core" / "tables" / "analytic_sample_summary.csv").iloc[0]
check("analytic_n", sample["analytic_n"], 25491)
check("positive_n", sample["positive_n"], 2245)

part = pd.read_csv(RESULTS / "core" / "tables" / "cluster_exclusive_partition_summary.csv").set_index("partition")
for p, n, pos in [("Training", 15334, 1357), ("Validation", 5141, 440), ("Test", 5016, 448)]:
    check(f"{p}_n", part.loc[p, "n"], n)
    check(f"{p}_positive", part.loc[p, "positives"], pos)

overlap = pd.read_csv(RESULTS / "core" / "tables" / "partition_overlap_check.csv").iloc[0]
for c in overlap.index:
    check(c, overlap[c], 0)

primary = (RESULTS / "core" / "metadata" / "primary_model.txt").read_text(encoding="utf-8").strip()
check("primary_model", primary, "Weighted XGBoost")

perf = pd.read_csv(RESULTS / "core" / "tables" / "final_test_probability_performance.csv")
perf = perf.loc[perf["analysis"].eq("Unweighted analytic sample")].iloc[0]
check("test_roc_auc", perf["roc_auc"], 0.7972, 5e-5)
check("test_pr_auc", perf["pr_auc"], 0.3694, 5e-5)
check("test_brier", perf["brier_score"], 0.0684, 5e-5)

capagg = pd.read_csv(RESULTS / "capacity_selection" / "exact_capacity_repeated_split_summary.csv")
best = capagg.sort_values(["mean_f2", "capacity"], ascending=[False, True]).iloc[0]
check("selected_capacity", best["capacity"], 0.243, 5e-10)
check("mean_f2", best["mean_f2"], 0.4991, 5e-5)
check("mean_recall", best["mean_recall"], 0.6777, 5e-5)
check("mean_precision", best["mean_precision"], 0.2431, 5e-5)
check("mean_boundary", best["mean_boundary"], 0.1178, 5e-5)

criteria = pd.read_csv(RESULTS / "capacity_selection" / "operating_point_criteria_summary.csv").set_index("criterion")
check("one_se_capacity", criteria.loc["One-standard-error F2", "capacity"], 0.213, 5e-10)
check("f1_capacity", criteria.loc["Maximum mean F1", "capacity"], 0.099, 5e-10)
check("youden_capacity", criteria.loc["Maximum mean Youden J", "capacity"], 0.263, 5e-10)

threshold = pd.read_csv(RESULTS / "capacity_selection" / "fixed_threshold_sensitivity_optimum.csv").iloc[0]
check("fixed_threshold", threshold["threshold"], 0.1218, 5e-10)
check("fixed_threshold_mean_workload", threshold["mean_selected_proportion"], 0.2318, 5e-5)

final = pd.read_csv(RESULTS / "capacity_selection" / "final_test_exact_capacity_primary_summary.csv").iloc[0]
for name, exp in [("selected", 1218), ("tp", 281), ("fp", 937), ("fn", 167), ("tn", 3631)]:
    check(f"final_{name}", final[name], exp)
check("final_recall", final["recall"], 0.6272, 5e-5)
check("final_precision", final["precision"], 0.2307, 5e-5)
check("final_specificity", final["specificity"], 0.7949, 5e-5)
check("final_f2", final["f2"], 0.4668, 5e-5)
check("final_boundary", final["boundary"], 0.1234, 5e-5)

sub = pd.read_csv(RESULTS / "core" / "tables" / "selected_capacity_subgroup_audit.csv")
for variable, group, exp in [
    ("sex", "Female", 0.7273), ("sex", "Male", 0.4506),
    ("residence", "Urban", 0.7077), ("residence", "Rural", 0.5160),
    ("age_group", "15–24", 0.1379),
]:
    row = sub.loc[sub["variable"].eq(variable) & sub["group"].astype(str).eq(group)].iloc[0]
    check(f"subgroup_recall_{variable}_{group}", row["recall"], exp, 5e-5)

ext = RESULTS / "publication_extensions"
cal = pd.read_csv(ext / "final_test_calibration_diagnostics.csv").iloc[0]
check("calibration_intercept", cal["calibration_intercept"], 0.0220, 5e-4)
check("calibration_slope", cal["calibration_slope"], 1.0292, 5e-4)
check("ece", cal["ece_10_quantile_bins"], 0.0081, 5e-4)

stab = pd.read_csv(ext / "repeated_grouped_model_performance.csv").set_index("model")
check("stability_xgb_pr", stab.loc["Weighted XGBoost", "mean_pr_auc"], 0.3675, 5e-4)
check("stability_xgb_roc", stab.loc["Weighted XGBoost", "mean_roc_auc"], 0.8036, 5e-4)

pooled = pd.read_csv(ext / "outer_grouped_subgroup_validation_folds.csv").set_index("group")
for group, tp, pos, recall in [("Overall",1471,2245,0.6552),("15–24",23,286,0.0804),("Male",430,792,0.5429),("Rural",525,962,0.5457)]:
    check(f"pooled_{group}_tp", pooled.loc[group, "true_positives_selected"], tp)
    check(f"pooled_{group}_positive", pooled.loc[group, "positive_cases"], pos)
    check(f"pooled_{group}_recall", pooled.loc[group, "pooled_recall"], recall, 5e-5)

domain = pd.read_csv(ext / "domain_only_leave_one_out_test_performance.csv").set_index("domain")
locked_domain = {
    "Demographic": (0.7331,0.1844,0.7692,0.3462,-0.0233),
    "Structural/socioeconomic": (0.6226,0.1356,0.7802,0.3428,-0.0266),
    "Behavioural/relationship": (0.7194,0.2089,0.7894,0.3511,-0.0183),
    "Health/testing history": (0.6623,0.1989,0.7852,0.2717,-0.0978),
}
for d, vals in locked_domain.items():
    for col, exp in zip(["domain_only_roc_auc","domain_only_pr_auc","without_domain_roc_auc","without_domain_pr_auc","delta_pr_auc"], vals):
        check(f"domain_{d}_{col}", domain.loc[d,col], exp, 5e-4)

report = {"all_pass": all(c["pass"] for c in checks), "checks": checks}
(RESULTS / "reproducibility_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
pd.DataFrame(checks).to_csv(RESULTS / "reproducibility_verification.csv", index=False)
failed = [c for c in checks if not c["pass"]]
print(pd.DataFrame(checks).to_string(index=False))
if failed:
    print("\nFAILED CHECKS:")
    print(pd.DataFrame(failed).to_string(index=False))
    sys.exit(1)
print("\nAll locked-result checks passed.")
