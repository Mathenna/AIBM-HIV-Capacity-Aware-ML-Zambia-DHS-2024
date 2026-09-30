#!/usr/bin/env python3
"""Post-review INTERNAL validation. No historical performance is promised.

Usage: python src/revision_pipeline.py --data-dir D:/authorised/DHS --output results/revision_run
Every output directory must be new. Restricted records and models are not saved.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import hashlib
from pathlib import Path
import platform
import warnings
from threadpoolctl import threadpool_limits
import numpy as np
import pandas as pd
import yaml
from scipy.special import logit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

import analysis_common as legacy
from revision_data import load_cohort, restrict_cohort, sha256
from revision_models import develop, calibrate
from revision_policy import select_exact, metrics, interval_summary
from revision_bootstrap import bootstrap_policy_fast as bootstrap_policy

ROOT = Path(__file__).resolve().parent.parent


def dump_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str, ensure_ascii=False), encoding="utf-8")


def capacity_from_training(y, scores, keys, grid):
    # Rank once; every capacity selects a prefix of exactly the same order.
    # Validate through the reference selector before taking cumulative counts.
    select_exact(scores, 0., tie_keys=keys)
    y, scores, keys = np.asarray(y), np.asarray(scores), np.asarray(keys)
    if y.shape != scores.shape or not np.isin(y, [0, 1]).all():
        raise ValueError("invalid capacity-training labels")
    capacities = np.asarray(grid, float)
    if not np.isfinite(capacities).all() or ((capacities<0)|(capacities>1)).any():
        raise ValueError("invalid capacity grid")
    order = np.lexsort((keys, -scores))
    positive_prefix = np.r_[0, np.cumsum(y[order])]
    k = np.floor(capacities * len(y)).astype(int)
    tp = positive_prefix[k]
    denominator = 5*tp + 4*(y.sum()-tp) + k-tp
    f2 = np.divide(5*tp, denominator, out=np.zeros(len(k), float), where=denominator>0)
    table = pd.DataFrame({"capacity": capacities, "f2": f2})
    best = table.sort_values(["f2", "capacity"], ascending=[False, True]).iloc[0]
    return float(best.capacity), table


def calibration_diagnostics(y, p):
    # Evaluation diagnostic only, NEVER used to update held-out probabilities.
    z = logit(np.clip(p, 1e-6, 1-1e-6)).reshape(-1, 1)
    fit = LogisticRegression(C=1e6, max_iter=5000).fit(z, y)
    bins = pd.qcut(pd.Series(p), q=10, duplicates="drop")
    grouped = pd.DataFrame({"y": np.asarray(y), "p": p, "bin": bins}).groupby("bin", observed=True).agg(
        n=("y", "size"), observed=("y", "mean"), predicted=("p", "mean"))
    ece = float((grouped.n / grouped.n.sum() * (grouped.observed - grouped.predicted).abs()).sum())
    return {"calibration_intercept": float(fit.intercept_[0]),
            "calibration_slope": float(fit.coef_[0,0]), "ece_10_quantile_bins": ece}, grouped.reset_index()


def evaluate_analysis(df, config, output, models=None, smoke=False, checkpoint_dir=None):
    output.mkdir(parents=True, exist_ok=False)
    models = models or config["models"]
    numeric, categorical = list(legacy.EXPANDED_NUMERIC), list(legacy.EXPANDED_CATEGORICAL)
    if config.get("feature_set") == "no_testing_history":
        numeric = [v for v in numeric if legacy.DOMAIN_MAP[v] != "Health/testing history"]
        categorical = [v for v in categorical if legacy.DOMAIN_MAP[v] != "Health/testing history"]
    grid_cfg = config["capacity_grid"]
    grid = np.round(np.arange(grid_cfg["start"], grid_cfg["stop"] + 1e-9, grid_cfg["step"]), 6)
    capacities = config["fixed_capacities"]
    perf, subgroups, tuning, selections, calibration, bin_rows, curve_rows = [], [], [], [], [], [], []
    predictions = df.copy()
    for name in models + ["selected_pipeline"]:
        predictions[name] = np.nan
    selected_capacity = np.full(len(df), np.nan)
    for outer in sorted(df.outer_fold.unique()):
        tr = df.loc[df.outer_fold.ne(outer)].reset_index(drop=True)
        hold_idx = np.flatnonzero(df.outer_fold.eq(outer))
        ho = df.iloc[hold_idx].reset_index(drop=True)
        if set(tr.cluster) & set(ho.cluster):
            raise AssertionError("outer cluster leakage")
        if tr.y.nunique() != 2 or ho.y.nunique() != 2:
            raise ValueError("outer partition lacks outcome class")
        candidates = []
        for name in models:
            print(f"{output.name}: outer={outer}, model={name}", flush=True)
            cache = None
            if checkpoint_dir is not None:
                fingerprint = hashlib.sha256()
                fingerprint.update(pd.util.hash_pandas_object(tr, index=True).values.tobytes())
                fingerprint.update(pd.util.hash_pandas_object(ho, index=True).values.tobytes())
                fingerprint.update(json.dumps([config, name, smoke], sort_keys=True).encode())
                fingerprint.update((Path(__file__).parent / "revision_models.py").read_bytes())
                cache = Path(checkpoint_dir) / fingerprint.hexdigest()
            if cache is not None and (cache / "complete.json").exists():
                saved = json.loads((cache / "complete.json").read_text())
                with np.load(cache / "private_probabilities.npz", allow_pickle=False) as arrays:
                    pbase, p, oof = arrays["pbase"], arrays["p"], arrays["oof"]
                search_table = pd.read_csv(cache / "search.csv")
                params, platt_slope = saved["parameters"], saved["platt_slope"]
                print("  reused matching local checkpoint", flush=True)
            else:
                base, features, cal, oof, search_table, params = develop(
                    tr, name, numeric, categorical, config, config["seed"] + 1000 * int(outer), smoke)
                pbase = base.predict_proba(ho[features])[:, 1]
                p = calibrate(pbase, cal)
                platt_slope = float(cal.coef_[0,0])
                if cache is not None:
                    cache.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(cache / "private_probabilities.npz", pbase=pbase, p=p, oof=oof)
                    search_table.to_csv(cache / "search.csv", index=False)
                    dump_json(cache / "complete.json", {"parameters": params, "platt_slope": platt_slope})
            predictions.loc[hold_idx, name] = p
            cstar, curve = capacity_from_training(tr.y, oof, tr.tie_key, grid)
            curve["outer_fold"], curve["model"] = outer, name
            curve_rows.append(curve)
            ap = float(average_precision_score(tr.y, oof))
            candidates.append({"model": name, "training_crossfit_ap": ap, "capacity": cstar,
                               "platt_slope": platt_slope, "parameters": params})
            search_table["outer_fold"], search_table["model"] = outer, name
            tuning.append(search_table)
            diag, bins = calibration_diagnostics(ho.y, p)
            calibration.append({"outer_fold": outer, "model": name, **diag,
                                "raw_brier": float(np.mean((ho.y.to_numpy() - pbase)**2)),
                                "calibrated_brier": float(np.mean((ho.y.to_numpy() - p)**2))})
            bins["outer_fold"], bins["model"] = outer, name
            bin_rows.append(bins)
            # The tuned F2 policy is evaluated once on the outer holdout.
            for c, role in [(float(c), "fixed") for c in capacities] + [(cstar, "training_selected_f2")]:
                s = select_exact(p, c, tie_keys=ho.tie_key)
                for weighting in ["unweighted", "hiv_weighted"]:
                    w = None if weighting == "unweighted" else ho.hiv_weight
                    perf.append({"outer_fold": outer, "model": name, "capacity": c,
                                 "policy": "global", "capacity_role": role, "weighting": weighting,
                                 **metrics(ho.y, p, s, w)})
        best_ap = max(r["training_crossfit_ap"] for r in candidates)
        eligible = {r["model"] for r in candidates
                    if r["training_crossfit_ap"] >= best_ap - config["family_selection_ap_tolerance"]}
        chosen = next(m for m in config["family_preference"] if m in eligible)
        selected = next(r for r in candidates if r["model"] == chosen)
        selections.append({"outer_fold": int(outer), "chosen_model": chosen, "candidates": candidates})
        predictions.loc[hold_idx, "selected_pipeline"] = predictions.loc[hold_idx, chosen].to_numpy()
        selected_capacity[hold_idx] = selected["capacity"]
        p = predictions.loc[hold_idx, "selected_pipeline"].to_numpy()
        for c, role in [(float(c), "fixed") for c in capacities] + [(selected["capacity"], "training_selected_f2")]:
            for policy, groups in [("global", None), ("sex_proportional", ho.sex),
                                   ("residence_proportional", ho.residence), ("age_proportional", ho.age_group)]:
                s = select_exact(p, c, groups, ho.tie_key)
                for weighting in ["unweighted", "hiv_weighted"]:
                    w = None if weighting == "unweighted" else ho.hiv_weight
                    perf.append({"outer_fold": outer, "model": "selected_pipeline", "capacity": c,
                                 "policy": policy, "capacity_role": role, "weighting": weighting,
                                 **metrics(ho.y, p, s, w)})
                for var in ["sex", "residence", "age_group", "wealth", "education", "region"]:
                    for group, part in ho.groupby(var, observed=True, sort=True):
                        idx = part.index.to_numpy()
                        for weighting in ["unweighted", "hiv_weighted"]:
                            w = None if weighting == "unweighted" else part.hiv_weight
                            subgroups.append({"outer_fold": outer, "variable": var, "group": str(group),
                                "capacity": c, "capacity_role": role, "policy": policy, "weighting": weighting,
                                "positive_clusters": int(part.loc[part.y.eq(1), "cluster"].nunique()),
                                **metrics(part.y, p[idx], s[idx], w)})
        # Analytical random-selection baseline; do not simulate noisy random labels.
        for c in capacities:
            k = int(np.floor(float(c) * len(ho)))
            perf.append({"outer_fold": outer, "model": "random_expectation", "capacity": c,
                         "capacity_role": "fixed", "policy": "global", "weighting": "unweighted",
                         "n": len(ho), "positive_n": int(ho.y.sum()), "selected_n": k,
                         "recall": k/len(ho), "precision": float(ho.y.mean()),
                         "tp": k*float(ho.y.mean()), "yield_enrichment": 1.})
    if predictions[models + ["selected_pipeline"]].isna().any().any():
        raise AssertionError("incomplete OOF evaluation")
    pooled = []
    for name in models + ["selected_pipeline"]:
        policies = [("global", None)]
        if name == "selected_pipeline":
            policies += [("sex_proportional", "sex"), ("residence_proportional", "residence"),
                         ("age_proportional", "age_group")]
        for c in capacities:
            for policy, allocation in policies:
                s = np.zeros(len(predictions), bool)
                for _, ho in predictions.groupby("outer_fold", sort=True):
                    idx = ho.index.to_numpy()
                    s[idx] = select_exact(ho[name], float(c),
                                         None if allocation is None else ho[allocation], ho.tie_key)
                for weighting in ["unweighted", "hiv_weighted"]:
                    w = None if weighting == "unweighted" else predictions.hiv_weight
                    pooled.append({"model": name, "policy": policy, "capacity": c,
                                   "weighting": weighting,
                                   **metrics(predictions.y, predictions[name], s, w)})
    pd.DataFrame(pooled).to_csv(output / "pooled_fixed_performance.csv", index=False)
    pd.DataFrame(perf).to_csv(output / "fold_performance.csv", index=False)
    pd.DataFrame(subgroups).to_csv(output / "fold_subgroups.csv", index=False)
    pd.concat(tuning).to_csv(output / "nested_search_results.csv", index=False)
    pd.concat(curve_rows).to_csv(output / "training_capacity_curves.csv", index=False)
    pd.DataFrame(calibration).to_csv(output / "calibration_diagnostics.csv", index=False)
    pd.concat(bin_rows).to_csv(output / "calibration_bins.csv", index=False)
    dump_json(output / "selection_provenance.json", selections)
    # All models use the same sampled clusters, so differences can be paired.
    bootstrap_capacity = config["bootstrap_capacity"]
    reps, meta = bootstrap_policy(predictions, {m: m for m in models + ["selected_pipeline"]},
                                  bootstrap_capacity, config["bootstrap_iterations"], config["seed"] + 909)
    interval_summary(reps).to_csv(output / "conditional_cluster_intervals.csv", index=False)
    paired = []
    overall = reps.loc[reps.group.eq("Overall")]
    for weighting in ["unweighted", "hiv_weighted"]:
        for metric in ["recall", "precision", "average_precision", "roc_auc", "brier"]:
            wide = overall.loc[overall.weighting.eq(weighting)].pivot(index="replicate", columns="model", values=metric)
            if "xgb_weighted" in wide:
                for comparator in [m for m in models if m != "xgb_weighted"]:
                    delta = (wide.xgb_weighted - wide[comparator]).dropna()
                    paired.append({"reference": "xgb_weighted", "comparator": comparator,
                        "weighting": weighting, "metric": metric, "capacity": bootstrap_capacity,
                        "difference_direction": "reference minus comparator; lower Brier is better",
                        "valid_replicates": len(delta), "lower": delta.quantile(.025), "upper": delta.quantile(.975)})
    pd.DataFrame(paired).to_csv(output / "paired_model_intervals.csv", index=False)
    # Equity policy uncertainty and subgroup gaps use complete-cohort selections.
    policy_intervals = []
    for policy, column in [("sex_proportional", "sex"), ("residence_proportional", "residence"), ("age_proportional", "age_group")]:
        pr, _ = bootstrap_policy(predictions, {"selected_pipeline": "selected_pipeline"}, bootstrap_capacity,
                                 config["bootstrap_iterations"], config["seed"] + 909, allocation_column=column)
        summary = interval_summary(pr)
        summary["policy"] = policy
        policy_intervals.append(summary)
    pd.concat(policy_intervals).to_csv(output / "equity_policy_intervals.csv", index=False)
    disparity = []
    for weighting in ["unweighted", "hiv_weighted"]:
        part = reps.loc[reps.model.eq("selected_pipeline") & reps.weighting.eq(weighting)]
        wide = part.pivot(index="replicate", columns="group", values="recall")
        for left, right in [("sex=Male", "sex=Female"), ("residence=Rural", "residence=Urban"),
                            ("age_group=15–24", "age_group=35–44")]:
            if left in wide and right in wide:
                delta = (wide[left] - wide[right]).dropna()
                disparity.append({"left": left, "right": right, "weighting": weighting,
                                  "metric": "recall difference left minus right", "valid_replicates": len(delta),
                                  "lower": delta.quantile(.025), "upper": delta.quantile(.975)})
    pd.DataFrame(disparity).to_csv(output / "paired_recall_disparities.csv", index=False)
    dump_json(output / "bootstrap_design.json", meta)
    return {"n": len(df), "positive_n": int(df.y.sum()), "clusters": int(df.cluster.nunique()),
            "models": models, "status": "synthetic_smoke_only" if smoke else "analysis_complete_pending_scientific_review"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/revision_config.yaml")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohorts", nargs="+", default=["all", "restricted"])
    parser.add_argument("--feature-sets", nargs="+", choices=["full", "no_testing_history"], default=["full"])
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, help="Optional PRIVATE local probability cache; never publish this directory")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if cfg["outer_folds"] != 5:
        raise ValueError("v2 preserves the historical five outer cluster folds")
    warnings.filterwarnings("error", category=ConvergenceWarning)
    df, linkage, eligibility, provenance = load_cohort(args.data_dir, cfg["seed"])
    args.output.mkdir(parents=True, exist_ok=False)
    linkage.to_csv(args.output / "linkage.csv", index=False)
    eligibility.to_csv(args.output / "prior_status_counts.csv", index=False)
    (args.output / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    manifest = {"status": "preflight_complete" if args.preflight_only else "running",
        "started_utc": datetime.now(timezone.utc).isoformat(), "data": provenance,
        "python": platform.python_version(),
        "software": {p: importlib.metadata.version(p) for p in ["numpy","pandas","scipy","scikit-learn","xgboost","PyYAML"]},
        "code_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
        "config_sha256": sha256(args.config), "cohorts": args.cohorts, "feature_sets": args.feature_sets,
        "interpretation": "post-review nested internal validation; original data already examined; no external/untouched claim"}
    dump_json(args.output / "run_manifest.json", manifest)
    if args.preflight_only:
        print("Local-data preflight complete. No models trained.")
        return
    results = {}
    try:
        for cohort in args.cohorts:
            data = restrict_cohort(df, cohort)
            for feature_set in args.feature_sets:
                label = f"{cohort}__{feature_set}"
                with threadpool_limits(limits=1):
                    results[label] = evaluate_analysis(data, cfg | {"feature_set": feature_set}, args.output / label,
                                                      checkpoint_dir=args.checkpoint_dir)
    except Exception as exc:
        manifest["status"], manifest["error"] = "failed", str(exc)
        dump_json(args.output / "run_manifest.json", manifest)
        raise
    manifest.update(status="analysis_complete_pending_scientific_review", results=results,
                    completed_utc=datetime.now(timezone.utc).isoformat())
    dump_json(args.output / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()
