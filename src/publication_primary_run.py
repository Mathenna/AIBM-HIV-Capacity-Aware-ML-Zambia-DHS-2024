#!/usr/bin/env python3
"""Fast reproduction of the locked primary publication model.

This script uses the hyperparameters selected by the grouped grid search reported in
the final thesis. It is intended for reproducibility verification of the publication's frozen
primary model. The historical full-retuning/SHAP driver is not part of the audited
public release; see ``docs/REPRODUCIBILITY_NOTES.md``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

import analysis_common as cp

SRC_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_ROOT.parent


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "analysis_config.yaml")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "core")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed = int(cfg["random_states"]["primary_split_and_tuning"])
    args.output.mkdir(parents=True, exist_ok=True)
    paths = cp.ensure_dirs(args.output)

    ir = PROJECT_ROOT / cfg["paths"]["ir_path"]
    mr = PROJECT_ROOT / cfg["paths"]["mr_path"]
    ar = PROJECT_ROOT / cfg["paths"]["ar_path"]
    df, linkage = cp.read_data(ir, mr, ar)
    df = cp.assign_partition(df, seed)
    partition_summary, overlap = cp.verify_partition(df)
    linkage.to_csv(paths["tables"] / "data_linkage_and_exclusions.csv", index=False)
    partition_summary.to_csv(paths["tables"] / "cluster_exclusive_partition_summary.csv", index=False)
    pd.DataFrame([overlap]).to_csv(paths["tables"] / "partition_overlap_check.csv", index=False)
    pd.DataFrame([{
        "analytic_n": len(df), "positive_n": int(df["y"].sum()), "negative_n": int((df["y"] == 0).sum()),
        "unweighted_positive_proportion": float(df["y"].mean()),
        "hiv_weighted_positive_proportion": float(np.average(df["y"], weights=df["hiv_weight"])),
        "clusters": int(df["cluster"].nunique()), "households": int(df["household_id"].nunique()),
    }]).to_csv(paths["tables"] / "analytic_sample_summary.csv", index=False)

    tr = df.loc[df["partition"].eq("Training")].reset_index(drop=True)
    te = df.loc[df["partition"].eq("Test")].reset_index(drop=True)
    features = cp.EXPANDED_NUMERIC + cp.EXPANDED_CATEGORICAL
    ytr = tr["y"].reset_index(drop=True)
    pos_weight = float((len(ytr) - ytr.sum()) / ytr.sum())
    model = Pipeline([
        ("preprocess", cp.build_preprocessor(cp.EXPANDED_NUMERIC, cp.EXPANDED_CATEGORICAL, False)),
        ("model", XGBClassifier(
            n_estimators=400, learning_rate=0.04, max_depth=3, min_child_weight=5.0,
            scale_pos_weight=pos_weight, subsample=0.80, colsample_bytree=0.80,
            reg_lambda=5.0, reg_alpha=0.05, eval_metric="logloss", tree_method="hist",
            random_state=seed, n_jobs=-1,
        )),
    ])
    joint = ytr.astype(str) + "_" + tr["sex"].astype(str) + "_" + tr["residence"].astype(str) + "_" + tr["age_group"].astype(str)
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    splits = list(cv.split(tr[features], joint, groups=tr["cluster"]))
    base, calibrator, _ = cp.platt_fit(model, tr[features], ytr, splits)
    p_base = base.predict_proba(te[features])[:, 1]
    p = cp.apply_calibrator(p_base, calibrator)
    y = te["y"].to_numpy()

    pd.DataFrame([
        {"analysis": "Unweighted analytic sample", **cp.metric_row(y, p)},
        {"analysis": "HIV-weighted sensitivity", **cp.metric_row(y, p, te["hiv_weight"].to_numpy())},
    ]).to_csv(paths["tables"] / "final_test_probability_performance.csv", index=False)
    np.save(paths["analysis"] / "test_probabilities_calibrated.npy", p)
    np.save(paths["analysis"] / "test_probabilities_base.npy", p_base)
    (paths["metadata"] / "primary_model.txt").write_text("Weighted XGBoost", encoding="utf-8")

    selected = cp.select_top_capacity(p, float(cfg["capacity_selection"]["selected_nominal_capacity"]))
    subgroup_frames = []
    for variable in ["sex", "residence", "age_group", "wealth", "region", "education"]:
        subgroup_frames.append(cp.subgroup_table(te, y, p, selected, variable))
    pd.concat(subgroup_frames, ignore_index=True).to_csv(paths["tables"] / "selected_capacity_subgroup_audit.csv", index=False)

    summary = {
        "primary_model": "Weighted XGBoost",
        "primary_training_scale_pos_weight": pos_weight,
        "test_metrics": cp.metric_row(y, p),
        "selected_capacity": cp.classification_metrics(y, selected),
        "test_boundary": float(p[np.argsort(-p, kind="mergesort")[int(np.floor(float(cfg['capacity_selection']['selected_nominal_capacity']) * len(p))) - 1]]),
    }
    (paths["metadata"] / "headline_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
