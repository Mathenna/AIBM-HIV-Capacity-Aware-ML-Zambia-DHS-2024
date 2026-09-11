#!/usr/bin/env python3
"""Repeated cluster-grouped capacity selection for the publication release.

Analytical contract
-------------------
The held-out test probabilities are generated once by ``core_model_pipeline.py``
from the frozen primary model selected using the 60/20/20 development design.
The 80% development sample is used only to select the nominal workload/capacity
through repeated grouped validation. After C* is chosen, it is applied to the
already-frozen test probabilities. The final test model is NOT refitted on the
80% development sample.

This resolves an ambiguity in the dissertation text while preserving the locked
reported final-test results (281 TP at 24.3% capacity).
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

SRC_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_ROOT.parent
PIPE = SRC_ROOT / "core_model_pipeline.py"
spec = importlib.util.spec_from_file_location("cp", PIPE)
cp = importlib.util.module_from_spec(spec)
sys.modules["cp"] = cp
assert spec.loader is not None
spec.loader.exec_module(cp)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fbeta_from_counts(tp: int, fp: int, fn: int, beta: float = 2.0) -> float:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    b2 = beta * beta
    denom = b2 * precision + recall
    return ((1 + b2) * precision * recall / denom) if denom else 0.0


def evaluate_capacity(y: np.ndarray, scores: np.ndarray, c: float) -> dict:
    n = len(y)
    k = max(1, min(n, int(math.floor(c * n))))
    order = np.argsort(-scores, kind="mergesort")
    selected = np.zeros(n, dtype=bool)
    selected[order[:k]] = True
    tp = int((selected & (y == 1)).sum())
    fp = int((selected & (y == 0)).sum())
    fn = int((~selected & (y == 1)).sum())
    tn = int((~selected & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    f2 = fbeta_from_counts(tp, fp, fn, beta=2.0)
    boundary = float(scores[order[k - 1]])
    return {
        "capacity": float(c), "selected": k, "boundary": boundary,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall": recall, "precision": precision, "specificity": specificity,
        "youden_j": recall + specificity - 1.0,
        "f1": f1, "f2": f2,
    }


def evaluate_threshold(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    selected = scores >= threshold
    tp = int((selected & (y == 1)).sum())
    fp = int((selected & (y == 0)).sum())
    fn = int((~selected & (y == 1)).sum())
    tn = int((~selected & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "threshold": float(threshold),
        "selected": int(selected.sum()),
        "selected_proportion": float(selected.mean()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "specificity": specificity,
        "f2": fbeta_from_counts(tp, fp, fn, beta=2.0),
    }


def make_xgb_estimator(pos_weight: float, seed: int, cfg: dict) -> Pipeline:
    mcfg = cfg["models"]["candidates"]["weighted_xgboost"]
    n_estimators = int(max(mcfg["n_estimators"]))
    max_depth = int(max(mcfg["max_depth"]))
    return Pipeline([
        ("preprocess", cp.build_preprocessor(cp.EXPANDED_NUMERIC, cp.EXPANDED_CATEGORICAL, False)),
        ("model", XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=float(mcfg["learning_rate"]),
            max_depth=max_depth,
            min_child_weight=float(mcfg["min_child_weight"]),
            scale_pos_weight=float(pos_weight),
            subsample=float(mcfg["subsample"]),
            colsample_bytree=float(mcfg["colsample_bytree"]),
            reg_lambda=float(mcfg["reg_lambda"]),
            reg_alpha=float(mcfg["reg_alpha"]),
            eval_metric="logloss",
            tree_method=str(mcfg["tree_method"]),
            random_state=int(cfg["random_states"]["primary_split_and_tuning"]),
            n_jobs=-1,
        )),
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "analysis_config.yaml")
    parser.add_argument("--core-output", type=Path, default=PROJECT_ROOT / "results" / "core")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "capacity_selection")
    args = parser.parse_args()

    cfg = load_config(args.config)
    args.output.mkdir(parents=True, exist_ok=True)

    ir = PROJECT_ROOT / cfg["paths"]["ir_path"]
    mr = PROJECT_ROOT / cfg["paths"]["mr_path"]
    ar = PROJECT_ROOT / cfg["paths"]["ar_path"]
    base_seed = int(cfg["random_states"]["primary_split_and_tuning"])
    seeds = [int(s) for s in cfg["random_states"]["repeated_capacity_validation"]]
    cap_cfg = cfg["capacity_selection"]["exact_capacity_grid"]
    cap_grid = np.round(np.arange(float(cap_cfg["start"]), float(cap_cfg["stop"]) + 1e-12, float(cap_cfg["step"])), 3)

    df, _ = cp.read_data(ir, mr, ar)
    df = cp.assign_partition(df, seed=base_seed)
    dev = df.loc[~df["partition"].eq("Test")].reset_index(drop=True)
    test = df.loc[df["partition"].eq("Test")].reset_index(drop=True)
    features = cp.EXPANDED_NUMERIC + cp.EXPANDED_CATEGORICAL

    all_rows: list[dict] = []
    split_opt: list[dict] = []
    threshold_rows: list[dict] = []
    threshold_grid = np.round(np.arange(0.0500, 0.2000 + 1e-12, 0.0001), 4)

    for rep, seed in enumerate(seeds, start=1):
        joint = (
            dev["y"].astype(str) + "_" + dev["sex"].astype(str) + "_" +
            dev["residence"].astype(str) + "_" + dev["age_group"].astype(str)
        )
        outer = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
        train_idx, val_idx = list(outer.split(dev, joint, groups=dev["cluster"]))[-1]
        tr = dev.iloc[train_idx].reset_index(drop=True)
        va = dev.iloc[val_idx].reset_index(drop=True)
        Xtr = tr[features]
        ytr = tr["y"].reset_index(drop=True)
        gtr = tr["cluster"].reset_index(drop=True)
        Xv = va[features]
        yv = va["y"].to_numpy()
        pos_weight = float((len(ytr) - ytr.sum()) / ytr.sum())
        estimator = make_xgb_estimator(pos_weight, seed, cfg)
        joint_tr = (
            ytr.astype(str) + "_" + tr["sex"].astype(str) + "_" +
            tr["residence"].astype(str) + "_" + tr["age_group"].astype(str)
        )
        inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
        inner_splits = list(inner.split(Xtr, joint_tr, groups=gtr))
        base, cal, _ = cp.platt_fit(estimator, Xtr, ytr, inner_splits)
        pv = cp.apply_calibrator(base.predict_proba(Xv)[:, 1], cal)

        split_rows: list[dict] = []
        for c in cap_grid:
            row = evaluate_capacity(yv, pv, float(c))
            row.update(seed=seed, validation_n=len(va), validation_positive=int(yv.sum()))
            all_rows.append(row)
            split_rows.append(row)
        best = sorted(split_rows, key=lambda r: (-r["f2"], r["capacity"]))[0]
        split_opt.append(best)

        for threshold in threshold_grid:
            trow = evaluate_threshold(yv, pv, float(threshold))
            trow.update(seed=seed, validation_n=len(va), validation_positive=int(yv.sum()))
            threshold_rows.append(trow)

        print(rep, seed, "best capacity", best["capacity"], "f2", best["f2"], "boundary", best["boundary"], flush=True)

    all_df = pd.DataFrame(all_rows)
    agg = all_df.groupby("capacity").agg(
        mean_f2=("f2", "mean"), sd_f2=("f2", "std"), min_f2=("f2", "min"), max_f2=("f2", "max"),
        mean_f1=("f1", "mean"), mean_recall=("recall", "mean"), mean_precision=("precision", "mean"),
        mean_specificity=("specificity", "mean"), mean_youden_j=("youden_j", "mean"),
        mean_boundary=("boundary", "mean"), sd_boundary=("boundary", "std"), mean_selected=("selected", "mean"),
    ).reset_index()

    best_row = agg.sort_values(["mean_f2", "capacity"], ascending=[False, True]).iloc[0]
    cstar = float(best_row["capacity"])
    se_at_best = float(best_row["sd_f2"] / np.sqrt(len(seeds)))
    one_se_cutoff = float(best_row["mean_f2"] - se_at_best)
    one_se_row = agg.loc[agg["mean_f2"] >= one_se_cutoff].sort_values("capacity").iloc[0]
    f1_row = agg.sort_values(["mean_f1", "capacity"], ascending=[False, True]).iloc[0]
    youden_row = agg.sort_values(["mean_youden_j", "capacity"], ascending=[False, True]).iloc[0]

    tdf = pd.DataFrame(threshold_rows)
    tagg = tdf.groupby("threshold").agg(
        mean_f2=("f2", "mean"), sd_f2=("f2", "std"),
        mean_recall=("recall", "mean"), mean_precision=("precision", "mean"),
        mean_selected_proportion=("selected_proportion", "mean"), mean_selected=("selected", "mean"),
    ).reset_index()
    tbest = tagg.sort_values(["mean_f2", "threshold"], ascending=[False, True]).iloc[0]

    print("\nCstar", cstar, "mean F2", best_row["mean_f2"], "mean boundary", best_row["mean_boundary"], flush=True)

    ptest_path = args.core_output / "analysis" / "test_probabilities_calibrated.npy"
    if not ptest_path.exists():
        raise FileNotFoundError(
            f"Frozen primary test probabilities not found: {ptest_path}. "
            "Run core_model_pipeline.py before capacity_selection.py."
        )
    ptest = np.load(ptest_path)
    ytest = test["y"].to_numpy()
    if len(ptest) != len(test):
        raise RuntimeError(f"Frozen test probability length {len(ptest)} != test N {len(test)}")

    final = evaluate_capacity(ytest, ptest, cstar)
    final.update(
        test_n=len(test), test_positive=int(ytest.sum()),
        selected_capacity_from_repeated_validation=cstar,
        final_test_probability_source="frozen primary 60/20/20 model; no 80% development refit",
    )

    criteria = pd.DataFrame([
        {"criterion": "Maximum mean F2", "capacity": cstar, "score": float(best_row["mean_f2"]), "mean_recall": float(best_row["mean_recall"]), "mean_precision": float(best_row["mean_precision"]), "mean_boundary": float(best_row["mean_boundary"])},
        {"criterion": "One-standard-error F2", "capacity": float(one_se_row["capacity"]), "score": float(one_se_row["mean_f2"]), "mean_recall": float(one_se_row["mean_recall"]), "mean_precision": float(one_se_row["mean_precision"]), "mean_boundary": float(one_se_row["mean_boundary"])},
        {"criterion": "Maximum mean F1", "capacity": float(f1_row["capacity"]), "score": float(f1_row["mean_f1"]), "mean_recall": float(f1_row["mean_recall"]), "mean_precision": float(f1_row["mean_precision"]), "mean_boundary": float(f1_row["mean_boundary"])},
        {"criterion": "Maximum mean Youden J", "capacity": float(youden_row["capacity"]), "score": float(youden_row["mean_youden_j"]), "mean_recall": float(youden_row["mean_recall"]), "mean_precision": float(youden_row["mean_precision"]), "mean_boundary": float(youden_row["mean_boundary"])},
    ])

    all_df.to_csv(args.output / "exact_capacity_repeated_split_long.csv", index=False)
    agg.to_csv(args.output / "exact_capacity_repeated_split_summary.csv", index=False)
    pd.DataFrame(split_opt).to_csv(args.output / "exact_capacity_split_specific_optima.csv", index=False)
    pd.DataFrame([final]).to_csv(args.output / "final_test_exact_selected_capacity.csv", index=False)
    pd.DataFrame([final]).to_csv(args.output / "final_test_exact_capacity_primary_summary.csv", index=False)
    criteria.to_csv(args.output / "operating_point_criteria_summary.csv", index=False)
    tdf.to_csv(args.output / "fixed_threshold_repeated_split_long.csv", index=False)
    tagg.to_csv(args.output / "fixed_threshold_repeated_split_summary.csv", index=False)
    pd.DataFrame([{
        "threshold": float(tbest["threshold"]),
        "mean_f2": float(tbest["mean_f2"]),
        "mean_recall": float(tbest["mean_recall"]),
        "mean_precision": float(tbest["mean_precision"]),
        "mean_selected_proportion": float(tbest["mean_selected_proportion"]),
        "mean_selected": float(tbest["mean_selected"]),
    }]).to_csv(args.output / "fixed_threshold_sensitivity_optimum.csv", index=False)

    keys = sorted(set([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, round(cstar, 3), max(0.01, round(cstar - 0.01, 3)), min(0.50, round(cstar + 0.01, 3))]))
    agg[agg["capacity"].round(3).isin(keys)].to_csv(args.output / "exact_capacity_key_comparisons.csv", index=False)

    print(criteria.to_string(index=False), flush=True)
    print("Fixed-threshold optimum:", tbest.to_dict(), flush=True)
    print("Final frozen-test result:", final, flush=True)


if __name__ == "__main__":
    main()
