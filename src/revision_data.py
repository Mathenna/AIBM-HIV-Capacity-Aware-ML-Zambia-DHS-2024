"""Local DHS access, prior-status eligibility and provenance for revision v2."""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import analysis_common as legacy


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ordered_digest(values):
    return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()


def load_cohort(data_dir, seed):
    data_dir = Path(data_dir)
    files = [data_dir / name for name in ["ZMIR81FL.dta", "ZMMR81FL.dta", "ZMAR81FL.dta"]]
    missing = [p.name for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError("Authorised local DHS data required: " + ", ".join(missing))
    df, linkage = legacy.read_data(*files)
    # Preserve the original outer assignment before correcting display labels.
    # Reuse is for comparability and is NOT a newly untouched external test.
    df = legacy.assign_partition(df, seed)
    legacy.verify_partition(df)
    if not df.age.between(15, 59).all() or (df.loc[df.sex.eq("Female"), "age"] > 49).any():
        raise ValueError("age outside declared Zambia survey eligibility; inspect release/codebook")
    df["age_group"] = pd.cut(df.age, bins=[14,24,34,44,54,59],
                              labels=["15–24","25–34","35–44","45–54","55–59"]).astype(str)
    # Correct the status label for truly missing numeric raw values.
    df.loc[df.age_first_sex_raw.isna(), "sexual_debut_status"] = "Not reported"
    prior = []
    label_evidence = {}
    for path, prefix, sex in [(files[0], "v", "Female"), (files[1], "mv", "Male")]:
        cols = [prefix + suffix for suffix in ["001", "002", "003", "861"]]
        raw = pd.read_stata(path, columns=cols, convert_categoricals=False)
        labelled = pd.read_stata(path, columns=[prefix + "861"], convert_categoricals=True)
        positive_labels = labelled.loc[raw[prefix + "861"].eq(1), prefix + "861"].astype(str).unique()
        if len(positive_labels) != 1 or "positive" not in positive_labels[0].lower():
            raise ValueError(f"Cannot confirm {prefix}861 code 1 = positive from embedded labels")
        label_evidence[sex] = {"code_1_label": str(positive_labels[0])}
        raw.columns = ["cluster", "household", "line", "prior_result"]
        raw["sex"] = sex
        prior.append(raw)
    df = df.merge(pd.concat(prior, ignore_index=True), on=["cluster","household","line","sex"],
                  how="left", validate="one_to_one", indicator=True)
    if not df._merge.eq("both").all():
        raise ValueError("prior-status linkage incomplete")
    df = df.drop(columns="_merge")
    if df[["cluster", "household", "line", "strata", "hiv_weight"]].isna().any().any():
        raise ValueError("missing key, survey design or HIV weight")
    if (df.hiv_weight <= 0).any():
        raise ValueError("non-positive HIV weight; inspect rather than silently discard")
    # Recode files may use sex-specific numeric stratum labels. Verify cluster
    # consistency before bootstrap rather than silently pool incompatible strata.
    if df.groupby("cluster").strata.nunique().max() != 1:
        raise ValueError("stratum differs within cluster; reconcile IR/MR design fields")
    df["outer_fold"] = df["fold"].astype(int)
    keys = df[["cluster", "household", "line"]].astype(str).agg("|".join, axis=1)
    if keys.duplicated().any():
        raise ValueError("duplicate analytic identifier")
    # Hash-based tie order avoids systematic preference for IR before MR rows.
    df["tie_key"] = [hashlib.sha256(f"{seed}|{k}".encode()).hexdigest() for k in keys]
    provenance = {"input_sha256": {p.name: sha256(p) for p in files},
                  "ordered_cohort_digest": ordered_digest(keys),
                  "ordered_fold_digest": ordered_digest(keys + "|" + df.outer_fold.astype(str)),
                  "prior_status_label_evidence": label_evidence,
                  "note": "local provenance; no respondent identifiers exported"}
    eligibility = pd.crosstab(df.prior_result.fillna("missing"), df.y).reset_index()
    eligibility.columns = ["prior_result", "biomarker_negative", "biomarker_positive"]
    return df, linkage, eligibility, provenance


def restrict_cohort(df, cohort):
    if cohort not in {"all", "restricted", "all_15_49", "restricted_15_49"}:
        raise ValueError(f"unknown cohort {cohort}")
    result = df
    if cohort.startswith("restricted"):
        # Eligibility uses only self-report, never contemporaneous biomarker y.
        result = result.loc[~result.prior_result.eq(1)]
    if cohort.endswith("15_49"):
        result = result.loc[result.age.le(49)]
    return result.reset_index(drop=True)
