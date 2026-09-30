"""Model families with fold-local weights and grouped nested development."""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import parallel_backend
from scipy.special import logit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from xgboost import XGBClassifier


class FoldWeightedXGB(ClassifierMixin, BaseEstimator):
    """Compute Nnegative/Npositive INSIDE every fit, including GridSearchCV."""
    def __init__(self, weighted=True, n_estimators=400, max_depth=3,
                 learning_rate=.04, min_child_weight=5., subsample=.8,
                 colsample_bytree=.8, reg_lambda=5., reg_alpha=.05,
                 random_state=24101765, n_jobs=1):
        self.weighted = weighted
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        y = np.asarray(y, int)
        if set(np.unique(y)) != {0, 1}:
            raise ValueError("training fit needs both binary classes")
        self.scale_pos_weight_ = float((len(y) - y.sum()) / y.sum()) if self.weighted else 1.
        params = self.get_params().copy()
        params.pop("weighted")
        self.model_ = XGBClassifier(**params, scale_pos_weight=self.scale_pos_weight_,
                                    tree_method="hist", eval_metric="logloss")
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        self.n_features_in_ = self.model_.n_features_in_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.model_.predict(X)


def grouped_splits(df, n_splits, seed):
    """Validate every split, including both-class support and full OOF coverage."""
    joint = df[["y", "sex", "residence", "age_group"]].astype(str).agg("_".join, axis=1)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(splitter.split(df, joint, groups=df.cluster))
    seen = np.zeros(len(df), int)
    for train, hold in splits:
        if set(df.iloc[train].cluster) & set(df.iloc[hold].cluster):
            raise AssertionError("cluster overlap")
        if df.iloc[train].y.nunique() != 2 or df.iloc[hold].y.nunique() != 2:
            raise ValueError("a grouped fold lacks a class; revise fold count before analysis")
        seen[hold] += 1
    if not (seen == 1).all():
        raise AssertionError("invalid OOF coverage")
    return splits


def estimator_and_grid(name, numeric, categorical, seed, grids, smoke=False):
    if name == "demographic_spline":
        numeric, categorical = ["age"], ["sex", "residence", "region"]
    spline = name in ["spline_logistic", "demographic_spline"]
    numeric_steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if spline:
        numeric_steps.append(("spline", SplineTransformer(n_knots=4, degree=3,
                              include_bias=False, extrapolation="linear")))
    if spline or name == "logistic":
        numeric_steps.append(("scale", StandardScaler()))
    preprocess = ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), numeric),
        ("missing", MissingIndicator(features="all"), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ], verbose_feature_names_out=True)
    if name in ["logistic", "spline_logistic", "demographic_spline"]:
        learner = LogisticRegression(solver="lbfgs", max_iter=5000, random_state=seed)
    elif name in ["xgb_weighted", "xgb_unweighted"]:
        learner = FoldWeightedXGB(weighted=name == "xgb_weighted", random_state=seed)
    elif name == "random_forest":
        learner = RandomForestClassifier(n_estimators=120, class_weight="balanced_subsample",
                                          max_features="sqrt", n_jobs=1, random_state=seed)
    elif name in ["gradient_boosting", "subsampled_gradient_boosting"]:
        learner = GradientBoostingClassifier(learning_rate=.04, min_samples_leaf=10,
                    subsample=.7 if name.startswith("subsampled") else 1., random_state=seed)
    else:
        raise ValueError(f"unknown model {name}")
    grid = {"model__" + k: v for k, v in grids[name].items()}
    if smoke:
        grid = {k: [v[0]] for k, v in grid.items()}
        if hasattr(learner, "n_estimators"):
            learner.set_params(n_estimators=8)
            grid["model__n_estimators"] = [8]
    return Pipeline([("preprocess", preprocess), ("model", learner)]), grid


def tune(df, name, numeric, categorical, config, seed, smoke=False):
    estimator, grid = estimator_and_grid(name, numeric, categorical, seed, config["grids"], smoke)
    splits = grouped_splits(df, config["tuning_folds"], seed)
    search = GridSearchCV(estimator, grid, scoring="average_precision", refit=True,
                          cv=splits, n_jobs=config["search_jobs"], error_score="raise")
    features = ["age", "sex", "residence", "region"] if name == "demographic_spline" else numeric + categorical
    with parallel_backend("threading"):
        search.fit(df[features], df.y)
    # Only aggregate candidate results are exported.
    table = pd.DataFrame(search.cv_results_)[["params", "mean_test_score", "std_test_score", "rank_test_score"]]
    return search.best_estimator_, features, search.best_params_, table


def develop(df, name, numeric, categorical, config, seed, smoke=False):
    """Tune independently inside EACH OOF calibration fold, then tune full fit.

    No outer held-out records enter tuning, imputation, calibration or capacity
    selection. Calibration is prespecified for all families, not test-selected.
    """
    oof = np.full(len(df), np.nan)
    rows, selected_parameters = [], []
    for fold, (train, hold) in enumerate(grouped_splits(df, config["calibration_folds"], seed + 1)):
        base, features, params, table = tune(df.iloc[train].reset_index(drop=True), name,
                                          numeric, categorical, config, seed + 100 + fold, smoke)
        oof[hold] = base.predict_proba(df.iloc[hold][features])[:, 1]
        table["fit_role"] = f"calibration_fold_{fold}"
        rows.append(table)
        selected_parameters.append({"fit_role": f"calibration_fold_{fold}", "parameters": params})
    if not np.isfinite(oof).all():
        raise AssertionError("missing cross-fitted calibration probability")
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
    calibrator.fit(logit(np.clip(oof, 1e-6, 1-1e-6)).reshape(-1, 1), df.y)
    if float(calibrator.coef_[0, 0]) <= 0:
        raise ValueError("non-positive Platt slope; stop rather than silently reverse ranks")
    base, features, params, table = tune(df, name, numeric, categorical, config, seed, smoke)
    table["fit_role"] = "outer_training_full"
    rows.append(table)
    selected_parameters.append({"fit_role": "outer_training_full", "parameters": params})
    return base, features, calibrator, oof, pd.concat(rows, ignore_index=True), selected_parameters


def calibrate(probabilities, calibrator):
    return calibrator.predict_proba(logit(np.clip(probabilities, 1e-6, 1-1e-6)).reshape(-1, 1))[:, 1]
