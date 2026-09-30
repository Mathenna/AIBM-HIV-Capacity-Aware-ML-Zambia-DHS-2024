"""Exact count budgets and cluster-aware conditional policy uncertainty.

These functions intentionally contain no model fitting or publication constants.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def budget(n: int, capacity: float) -> int:
    if not np.isfinite(capacity) or not 0 <= capacity <= 1:
        raise ValueError("capacity must be finite and in [0, 1]")
    return math.floor(float(capacity) * n)


def select_exact(scores, capacity: float, groups=None, tie_keys=None) -> np.ndarray:
    """Select floor(capacity*N); optionally use proportional subgroup quotas.

    Largest remainders allocate residual places. Quota ties use sorted group
    labels; score ties use supplied unique stable keys, otherwise row position.
    Allocation uses no outcome information. No subgroup is guaranteed one place.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("scores must be a finite one-dimensional array")
    n = len(scores)
    k = budget(n, capacity)
    keys = np.arange(n) if tie_keys is None else np.asarray(tie_keys)
    if keys.shape != (n,) or len(np.unique(keys)) != n:
        raise ValueError("tie_keys must have one unique key per record")
    selected = np.zeros(n, dtype=bool)
    if groups is None:
        selected[np.lexsort((keys, -scores))[:k]] = True
        return selected
    g = pd.Series(groups).reset_index(drop=True)
    if len(g) != n or g.isna().any():
        raise ValueError("groups must match scores and cannot be missing")
    # Study subgroup labels are strings; first convert to avoid categorical
    # unobserved levels and pandas-version-dependent groupby defaults.
    g = g.astype(str).to_numpy()
    names, counts = np.unique(g, return_counts=True)
    # Quotas sum to the GLOBAL integer budget, not to independently rounded C*n_g.
    ideals = (k * counts / n) if n else np.array([], dtype=float)
    quotas = np.floor(ideals).astype(int)
    remaining = k - int(quotas.sum())
    order = np.lexsort((names, -(ideals - quotas)))
    quotas[order[:remaining]] += 1
    for name, quota in zip(names, quotas):
        idx = np.flatnonzero(g == name)
        selected[idx[np.lexsort((keys[idx], -scores[idx]))[:quota]]] = True
    if int(selected.sum()) != k:
        raise AssertionError("quota conservation failed")
    return selected


def metrics(y, p, selected, weights=None) -> dict:
    y, p, s = np.asarray(y, int), np.asarray(p, float), np.asarray(selected, bool)
    if not (y.shape == p.shape == s.shape) or not np.isin(y, [0, 1]).all():
        raise ValueError("binary labels, probabilities and selections must align")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("invalid probabilities")
    w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    if w.shape != y.shape or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("invalid evaluation weights")
    tp, fp = w[s & (y == 1)].sum(), w[s & (y == 0)].sum()
    fn, tn = w[~s & (y == 1)].sum(), w[~s & (y == 0)].sum()
    div = lambda a, b: float(a / b) if b > 0 else float("nan")
    prevalence = div(tp + fn, w.sum())
    precision = div(tp, tp + fp)
    has_both = (tp + fn > 0) and (fp + tn > 0)
    return {
        "n": len(y), "positive_n": int(y.sum()), "selected_n": int(s.sum()),
        "selected_fraction": div(s.sum(), len(s)),
        "tp": float(tp), "fp": float(fp), "fn": float(fn), "tn": float(tn),
        "positive_fraction": prevalence,
        "recall": div(tp, tp + fn), "precision": precision,
        "specificity": div(tn, tn + fp), "f2": div(5 * tp, 5 * tp + 4 * fn + fp),
        "yield_enrichment": div(precision, prevalence),
        "roc_auc": float(roc_auc_score(y, p, sample_weight=w)) if has_both else float("nan"),
        "average_precision": float(average_precision_score(y, p, sample_weight=w)) if tp + fn > 0 else float("nan"),
        "brier": float(brier_score_loss(y, p, sample_weight=w)),
    }


def bootstrap_policy(df, score_columns: dict[str, str], capacity: float,
                     iterations=1000, seed=24101765, strata_column="strata",
                     allocation_column=None, tie_column="tie_key"):
    """Resample PSUs within strata and outer folds, then RESELECT top-k.

    Scores are fixed OOF predictions. Each outer fold is a separate deployment
    cohort; its budget is recalculated on its bootstrap sample. This estimates
    conditional evaluation uncertainty, not full training/selection uncertainty.
    Use common draws for all models. Returns replicates for paired comparisons.
    """
    if iterations < 1:
        raise ValueError("iterations must be positive")
    df = df.reset_index(drop=True)
    needed = ["cluster", "outer_fold", "y", "hiv_weight", tie_column]
    if strata_column is not None:
        needed.append(strata_column)
    if df[needed].isna().any().any():
        raise ValueError("missing bootstrap design variable")
    if df.groupby("cluster")["outer_fold"].nunique().max() != 1:
        raise ValueError("cluster crosses outer folds")
    if strata_column and df.groupby("cluster")[strata_column].nunique().max() != 1:
        raise ValueError("cluster crosses strata")
    block_cols = ["outer_fold"] + ([strata_column] if strata_column else [])
    blocks = []
    for _, block in df.groupby(block_cols, sort=True, observed=True):
        blocks.append([g.index.to_numpy() for _, g in block.groupby("cluster", sort=True)])
    singleton_blocks = sum(len(block) == 1 for block in blocks)
    rng = np.random.default_rng(seed)
    rows = []
    for rep in range(iterations):
        idx = np.concatenate([g for block in blocks
                              for g in [block[j] for j in rng.integers(0, len(block), len(block))]])
        b = df.iloc[idx].reset_index(drop=True)
        # Repeated cluster draws duplicate records. Composite keys retain stable
        # respondent tie order and resolve duplicate copies deterministically.
        keys = np.array([f"{v}_{i:012d}" for i, v in enumerate(b[tie_column].astype(str))])
        for model, score_col in score_columns.items():
            sel = np.zeros(len(b), dtype=bool)
            for _, fold in b.groupby("outer_fold", sort=True):
                pos = fold.index.to_numpy()
                alloc = None if allocation_column is None else fold[allocation_column]
                sel[pos] = select_exact(fold[score_col], capacity, alloc, keys[pos])
            masks = {"Overall": np.ones(len(b), bool)}
            for var in ["sex", "residence", "age_group"]:
                if var in b:
                    masks.update({f"{var}={v}": b[var].astype(str).eq(v).to_numpy()
                                  for v in sorted(b[var].astype(str).unique())})
            for group, mask in masks.items():
                for weighting in ["unweighted", "hiv_weighted"]:
                    w = None if weighting == "unweighted" else b.loc[mask, "hiv_weight"]
                    vals = metrics(b.loc[mask, "y"], b.loc[mask, score_col], sel[mask], w)
                    rows.append({"replicate": rep, "model": model, "group": group,
                                 "weighting": weighting, "capacity": capacity, **vals})
    return pd.DataFrame(rows), {"singleton_stratum_fold_blocks": singleton_blocks,
        "estimand": "conditional on fitted OOF scores; stratified PSU bootstrap within outer folds; top-k reselected"}


def interval_summary(replicates):
    rows = []
    for key, part in replicates.groupby(["model", "group", "weighting", "capacity"], sort=True):
        for metric in ["recall", "precision", "roc_auc", "average_precision", "brier", "f2"]:
            vals = part[metric].dropna()
            rows.append(dict(zip(["model", "group", "weighting", "capacity"], key)) |
                        {"metric": metric, "valid_replicates": len(vals),
                         "lower": vals.quantile(.025), "upper": vals.quantile(.975)})
    return pd.DataFrame(rows)
