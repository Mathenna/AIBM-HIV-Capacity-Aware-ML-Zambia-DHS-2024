#!/usr/bin/env python3
"""Publication-focused analyses omitted from the original bundled scripts.

This module reconstructs the analyses explicitly reported in the final corrected
thesis and needed to audit the journal manuscript:
- five-fold paired model-stability analysis on the 80% development sample;
- pooled 24.3% subgroup recall across those outer folds;
- final-test calibration intercept, slope and 10-quantile-bin ECE;
- domain-only and leave-one-domain-out XGBoost discrimination;
- stacked-domain architectural sensitivity.

No respondent-level output is written. Only aggregate tables are saved.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

SRC_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_ROOT.parent
PIPE = SRC_ROOT / "core_model_pipeline.py"
spec = importlib.util.spec_from_file_location("cp_ext", PIPE)
cp = importlib.util.module_from_spec(spec)
sys.modules["cp_ext"] = cp
assert spec.loader is not None
spec.loader.exec_module(cp)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def joint_label(df: pd.DataFrame) -> pd.Series:
    return (
        df["y"].astype(str) + "_" + df["sex"].astype(str) + "_" +
        df["residence"].astype(str) + "_" + df["age_group"].astype(str)
    )


def feature_types(features: list[str]) -> tuple[list[str], list[str]]:
    numeric = [f for f in features if f in cp.EXPANDED_NUMERIC]
    categorical = [f for f in features if f in cp.EXPANDED_CATEGORICAL]
    return numeric, categorical


def xgb_pipeline(features: list[str], pos_weight: float, seed: int) -> Pipeline:
    num, cat = feature_types(features)
    return Pipeline([
        ("preprocess", cp.build_preprocessor(num, cat, False)),
        ("model", XGBClassifier(
            n_estimators=400, learning_rate=0.04, max_depth=3, min_child_weight=5.0,
            scale_pos_weight=float(pos_weight), subsample=0.80, colsample_bytree=0.80,
            reg_lambda=5.0, reg_alpha=0.05, eval_metric="logloss", tree_method="hist",
            random_state=seed, n_jobs=-1,
        )),
    ])


def selected_models(features: list[str], y_train: pd.Series, seed: int) -> dict[str, Pipeline]:
    num, cat = feature_types(features)
    pos_weight = float((len(y_train) - y_train.sum()) / y_train.sum())
    return {
        "Logistic Regression": Pipeline([
            ("preprocess", cp.build_preprocessor(num, cat, True)),
            ("model", LogisticRegression(C=0.1, class_weight=None, max_iter=3000, solver="liblinear", random_state=seed)),
        ]),
        "Class-weighted Random Forest": Pipeline([
            ("preprocess", cp.build_preprocessor(num, cat, False)),
            ("model", RandomForestClassifier(
                n_estimators=120, max_depth=12, min_samples_leaf=15, max_features="sqrt",
                class_weight="balanced_subsample", random_state=seed, n_jobs=1,
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("preprocess", cp.build_preprocessor(num, cat, False)),
            ("model", GradientBoostingClassifier(
                n_estimators=250, max_depth=2, learning_rate=0.04, min_samples_leaf=10,
                subsample=1.0, random_state=seed,
            )),
        ]),
        "Subsampled Gradient Boosting": Pipeline([
            ("preprocess", cp.build_preprocessor(num, cat, False)),
            ("model", GradientBoostingClassifier(
                n_estimators=250, max_depth=2, learning_rate=0.04, min_samples_leaf=10,
                subsample=0.70, random_state=seed,
            )),
        ]),
        "Weighted XGBoost": xgb_pipeline(features, pos_weight, seed),
    }


def exact_top_k(scores: np.ndarray, capacity: float) -> np.ndarray:
    k = max(1, min(len(scores), int(np.floor(capacity * len(scores)))))
    order = np.argsort(-scores, kind="mergesort")
    selected = np.zeros(len(scores), dtype=bool)
    selected[order[:k]] = True
    return selected


def model_stability_and_pooled_subgroups(
    df: pd.DataFrame,
    stability_seed: int,
    primary_seed: int,
    capacity: float,
    outdir: Path,
    full_stability: bool = False,
) -> None:
    """Reproduce the two distinct outer-fold analyses reported in the thesis."""
    features = cp.EXPANDED_NUMERIC + cp.EXPANDED_CATEGORICAL
    primary_weight = 13977 / 1357

    dev = df.loc[~df["partition"].eq("Test")].reset_index(drop=True)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=stability_seed)
    rows: list[dict] = []
    for fold, (fit_idx, hold_idx) in enumerate(splitter.split(dev, joint_label(dev), groups=dev["cluster"])):
        tr = dev.iloc[fit_idx].reset_index(drop=True)
        ho = dev.iloc[hold_idx].reset_index(drop=True)
        Xtr, ytr = tr[features], tr["y"].reset_index(drop=True)
        Xho, yho = ho[features], ho["y"].to_numpy()
        models = selected_models(features, ytr, primary_seed)
        models["Weighted XGBoost"] = xgb_pipeline(features, primary_weight, primary_seed)
        if not full_stability:
            models = {"Weighted XGBoost": models["Weighted XGBoost"]}
        for name, model in models.items():
            model.fit(Xtr, ytr)
            scores = model.predict_proba(Xho)[:, 1]
            rows.append({
                "fold": fold, "model": name,
                "pr_auc": float(average_precision_score(yho, scores)),
                "roc_auc": float(roc_auc_score(yho, scores)),
                "heldout_n": len(ho), "heldout_positive": int(yho.sum()),
            })

    fold_df = pd.DataFrame(rows)
    summary = fold_df.groupby("model").agg(
        mean_pr_auc=("pr_auc", "mean"), sd_pr_auc=("pr_auc", "std"),
        mean_roc_auc=("roc_auc", "mean"), sd_roc_auc=("roc_auc", "std"),
    ).reset_index().sort_values("mean_pr_auc", ascending=False)
    paired = []
    if full_stability:
        xgb = fold_df.loc[fold_df["model"].eq("Weighted XGBoost"), ["fold", "pr_auc"]].rename(columns={"pr_auc": "xgb_pr_auc"})
        for comparator in [m for m in fold_df["model"].unique() if m != "Weighted XGBoost"]:
            c = fold_df.loc[fold_df["model"].eq(comparator), ["fold", "pr_auc"]].rename(columns={"pr_auc": "comp_pr_auc"})
            d = xgb.merge(c, on="fold")
            diff = d["xgb_pr_auc"] - d["comp_pr_auc"]
            paired.append({
                "comparator": comparator,
                "mean_difference": float(diff.mean()), "sd": float(diff.std(ddof=1)),
                "minimum": float(diff.min()), "maximum": float(diff.max()),
                "xgboost_better_folds": int((diff > 0).sum()), "folds": len(diff),
            })

    pooled = {g: {"tp": 0, "positive": 0} for g in ["Overall", "15–24", "Male", "Rural"]}
    full_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=primary_seed)
    for fit_idx, hold_idx in full_splitter.split(df, joint_label(df), groups=df["cluster"]):
        tr = df.iloc[fit_idx].reset_index(drop=True)
        ho = df.iloc[hold_idx].reset_index(drop=True)
        ytr = tr["y"].reset_index(drop=True)
        yho = ho["y"].to_numpy()
        pos_weight = float((len(ytr) - ytr.sum()) / ytr.sum())
        model = xgb_pipeline(features, pos_weight, primary_seed)
        model.fit(tr[features], ytr)
        scores = model.predict_proba(ho[features])[:, 1]
        selected = exact_top_k(scores, capacity)
        masks = {
            "Overall": np.ones(len(ho), dtype=bool),
            "15–24": ho["age_group"].astype(str).eq("15–24").to_numpy(),
            "Male": ho["sex"].eq("Male").to_numpy(),
            "Rural": ho["residence"].eq("Rural").to_numpy(),
        }
        for group, mask in masks.items():
            pooled[group]["tp"] += int((selected[mask] & (yho[mask] == 1)).sum())
            pooled[group]["positive"] += int((yho[mask] == 1).sum())

    pooled_rows = [{
        "group": group,
        "true_positives_selected": v["tp"],
        "positive_cases": v["positive"],
        "pooled_recall": v["tp"] / v["positive"] if v["positive"] else np.nan,
    } for group, v in pooled.items()]

    fold_df.to_csv(outdir / "repeated_grouped_split_results.csv", index=False)
    summary.to_csv(outdir / "repeated_grouped_model_performance.csv", index=False)
    pd.DataFrame(paired).to_csv(outdir / "paired_pr_auc_comparison.csv", index=False)
    pd.DataFrame(pooled_rows).to_csv(outdir / "outer_grouped_subgroup_validation_folds.csv", index=False)


def calibration_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    z = logit(np.clip(p, 1e-9, 1 - 1e-9))

    def nll(theta: np.ndarray) -> float:
        q = expit(theta[0] + theta[1] * z)
        q = np.clip(q, 1e-12, 1 - 1e-12)
        return float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).sum())

    opt = minimize(nll, x0=np.array([0.0, 1.0]), method="BFGS")
    intercept, slope = map(float, opt.x)
    bins = pd.qcut(pd.Series(p), q=10, duplicates="drop")
    tmp = pd.DataFrame({"y": y, "p": p, "bin": bins})
    grouped = tmp.groupby("bin", observed=True).agg(n=("y", "size"), obs=("y", "mean"), pred=("p", "mean"))
    ece = float(((grouped["n"] / len(tmp)) * (grouped["obs"] - grouped["pred"]).abs()).sum())
    return {"calibration_intercept": intercept, "calibration_slope": slope, "ece_10_quantile_bins": ece}


def domain_analysis(df: pd.DataFrame, seed: int, outdir: Path) -> None:
    tr = df.loc[df["partition"].eq("Training")].reset_index(drop=True)
    te = df.loc[df["partition"].eq("Test")].reset_index(drop=True)
    ytr = tr["y"].reset_index(drop=True)
    yte = te["y"].to_numpy()
    domains = list(dict.fromkeys(cp.DOMAIN_MAP.values()))
    domain_features = {d: [f for f, dom in cp.DOMAIN_MAP.items() if dom == d] for d in domains}
    full_features = cp.EXPANDED_NUMERIC + cp.EXPANDED_CATEGORICAL
    rows = []
    for d in domains:
        only = domain_features[d]
        without = [f for f in full_features if f not in set(only)]
        vals = {}
        for label, features in [("domain_only", only), ("without_domain", without)]:
            pos_weight = float((len(ytr) - ytr.sum()) / ytr.sum())
            model = xgb_pipeline(features, pos_weight, seed)
            model.fit(tr[features], ytr)
            p = model.predict_proba(te[features])[:, 1]
            vals[f"{label}_roc_auc"] = float(roc_auc_score(yte, p))
            vals[f"{label}_pr_auc"] = float(average_precision_score(yte, p))
        rows.append({"domain": d, **vals})
    result = pd.DataFrame(rows)
    full_p_path = PROJECT_ROOT / "results" / "core" / "analysis" / "test_probabilities_calibrated.npy"
    if full_p_path.exists():
        full_pr = float(average_precision_score(yte, np.load(full_p_path)))
        result["delta_pr_auc"] = result["without_domain_pr_auc"] - full_pr
    result.to_csv(outdir / "domain_only_leave_one_out_test_performance.csv", index=False)


def stacked_domain_analysis(df: pd.DataFrame, seed: int, outdir: Path) -> None:
    tr = df.loc[df["partition"].eq("Training")].reset_index(drop=True)
    va = df.loc[df["partition"].eq("Validation")].reset_index(drop=True)
    te = df.loc[df["partition"].eq("Test")].reset_index(drop=True)
    ytr = tr["y"].reset_index(drop=True)
    yv, yt = va["y"].to_numpy(), te["y"].to_numpy()
    domains = list(dict.fromkeys(cp.DOMAIN_MAP.values()))
    domain_features = {d: [f for f, dom in cp.DOMAIN_MAP.items() if dom == d] for d in domains}
    oof = np.zeros((len(tr), len(domains)), dtype=float)
    val_meta = np.zeros((len(va), len(domains)), dtype=float)
    test_meta = np.zeros((len(te), len(domains)), dtype=float)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    splits = list(splitter.split(tr, joint_label(tr), groups=tr["cluster"]))

    for j, domain in enumerate(domains):
        features = domain_features[domain]
        for fit_idx, hold_idx in splits:
            yfit = ytr.iloc[fit_idx]
            pos_weight = float((len(yfit) - yfit.sum()) / yfit.sum())
            model = xgb_pipeline(features, pos_weight, seed)
            model.fit(tr.iloc[fit_idx][features], yfit)
            oof[hold_idx, j] = model.predict_proba(tr.iloc[hold_idx][features])[:, 1]
        pos_weight = float((len(ytr) - ytr.sum()) / ytr.sum())
        final = xgb_pipeline(features, pos_weight, seed)
        final.fit(tr[features], ytr)
        val_meta[:, j] = final.predict_proba(va[features])[:, 1]
        test_meta[:, j] = final.predict_proba(te[features])[:, 1]

    meta = LogisticRegression(max_iter=3000, solver="lbfgs", random_state=seed)
    meta.fit(oof, ytr)
    pv = meta.predict_proba(val_meta)[:, 1]
    pt = meta.predict_proba(test_meta)[:, 1]
    pd.DataFrame([{
        "architecture": "Stacked-domain model",
        "validation_roc_auc": float(roc_auc_score(yv, pv)),
        "validation_pr_auc": float(average_precision_score(yv, pv)),
        "test_roc_auc": float(roc_auc_score(yt, pt)),
        "test_pr_auc": float(average_precision_score(yt, pt)),
    }]).to_csv(outdir / "stacked_domain_architectural_sensitivity.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "analysis_config.yaml")
    parser.add_argument("--core-output", type=Path, default=PROJECT_ROOT / "results" / "core")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "publication_extensions")
    parser.add_argument("--full-stability", action="store_true", help="Also refit all four comparator model families across the five stability folds (slower).")
    parser.add_argument("--include-stacked-reconstruction", action="store_true", help="Run a best-effort reconstruction of the exploratory stacked-domain sensitivity; the exact original implementation was not present in the recovered code archive.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    seed = int(cfg["random_states"]["primary_split_and_tuning"])
    stability_seed = int(cfg["random_states"]["five_fold_model_stability"])
    capacity = float(cfg["capacity_selection"]["selected_nominal_capacity"])

    ir = PROJECT_ROOT / cfg["paths"]["ir_path"]
    mr = PROJECT_ROOT / cfg["paths"]["mr_path"]
    ar = PROJECT_ROOT / cfg["paths"]["ar_path"]
    df, _ = cp.read_data(ir, mr, ar)
    df = cp.assign_partition(df, seed=seed)

    model_stability_and_pooled_subgroups(df, stability_seed, seed, capacity, args.output, full_stability=args.full_stability)
    domain_analysis(df, seed, args.output)
    if args.include_stacked_reconstruction:
        stacked_domain_analysis(df, seed, args.output)

    ptest = np.load(args.core_output / "analysis" / "test_probabilities_calibrated.npy")
    ytest = df.loc[df["partition"].eq("Test"), "y"].to_numpy()
    cal = calibration_metrics(ytest, ptest)
    pd.DataFrame([cal]).to_csv(args.output / "final_test_calibration_diagnostics.csv", index=False)
    (args.output / "publication_extensions_summary.json").write_text(json.dumps(cal, indent=2), encoding="utf-8")
    print(json.dumps(cal, indent=2), flush=True)


if __name__ == "__main__":
    main()
