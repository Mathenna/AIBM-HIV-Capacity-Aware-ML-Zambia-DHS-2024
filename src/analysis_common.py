#!/usr/bin/env python3
"""Shared data construction, preprocessing, calibration and evaluation utilities.

This module is the compact publication implementation distilled from the recovered
final-analysis source. It intentionally excludes restricted DHS data and writes no
respondent-level output by itself.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import beta
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 24101765
CAPACITIES = [0.05, 0.10, 0.15, 0.20, 0.243, 0.25, 0.30]
PRIMARY_CAPACITY = 0.243

REGION_MAP = {1:"Central",2:"Copperbelt",3:"Eastern",4:"Luapula",5:"Lusaka",6:"Muchinga",7:"Northern",8:"North Western",9:"Southern",10:"Western"}
RESIDENCE_MAP = {1:"Urban",2:"Rural"}
EDUCATION_MAP = {0:"No education",1:"Primary",2:"Secondary",3:"Higher"}
WEALTH_MAP = {1:"Poorest",2:"Poorer",3:"Middle",4:"Richer",5:"Richest"}
MARITAL_MAP = {0:"Never in union",1:"Married",2:"Living with partner",3:"Widowed",4:"Divorced",5:"Separated"}

EXPANDED_NUMERIC = ["age","age_first_sex","partners_last12m","lifetime_partners","months_since_hiv_test"]
EXPANDED_CATEGORICAL = [
    "sex","residence","region","education","wealth","marital_status","ever_tested_hiv",
    "sexual_debut_status","condom_last_sex","sti_last12m","received_hiv_result","circumcision_status",
]
DOMAIN_MAP = {
    "age":"Demographic","sex":"Demographic",
    "residence":"Structural/socioeconomic","region":"Structural/socioeconomic","education":"Structural/socioeconomic","wealth":"Structural/socioeconomic",
    "marital_status":"Behavioural/relationship","age_first_sex":"Behavioural/relationship","sexual_debut_status":"Behavioural/relationship",
    "condom_last_sex":"Behavioural/relationship","sti_last12m":"Behavioural/relationship","partners_last12m":"Behavioural/relationship",
    "lifetime_partners":"Behavioural/relationship","circumcision_status":"Behavioural/relationship",
    "ever_tested_hiv":"Health/testing history","months_since_hiv_test":"Health/testing history","received_hiv_result":"Health/testing history",
}


def ensure_dirs(root: Path) -> dict[str, Path]:
    paths = {"root":root,"tables":root/"tables","figures":root/"figures","models":root/"models","metadata":root/"metadata","analysis":root/"analysis"}
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def read_data(ir_path: Path, mr_path: Path, ar_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ir_cols = ["v001","v002","v003","v005","v012","v021","v022","v024","v025","v106","v190","v501","v531","v781","v761","v763a","v766b","v836","v826a","v828"]
    mr_cols = ["mv001","mv002","mv003","mv005","mv012","mv021","mv022","mv024","mv025","mv106","mv190","mv501","mv531","mv781","mv761","mv763a","mv766b","mv836","mv826a","mv828","mv483"]
    ar_cols = ["hivclust","hivnumb","hivline","hiv03","hiv05"]
    ir_raw = pd.read_stata(ir_path, columns=ir_cols, convert_categoricals=False)
    mr_raw = pd.read_stata(mr_path, columns=mr_cols, convert_categoricals=False)
    ar_raw = pd.read_stata(ar_path, columns=ar_cols, convert_categoricals=False)
    ir = ir_raw.rename(columns={c:c[1:] for c in ir_cols}); mr = mr_raw.rename(columns={c:c[2:] for c in mr_cols})
    ir["sex"]="Female"; mr["sex"]="Male"; ir["483"]=np.nan
    resp = pd.concat([ir,mr], ignore_index=True).rename(columns={
        "001":"cluster","002":"household","003":"line","005":"respondent_weight","012":"age","021":"psu","022":"strata",
        "024":"region_raw","025":"residence_raw","106":"education_raw","190":"wealth_raw","501":"marital_raw","531":"age_first_sex_raw",
        "781":"ever_tested_raw","761":"condom_last_sex_raw","763a":"sti_last12m_raw","766b":"partners_last12m_raw","836":"lifetime_partners_raw",
        "826a":"months_since_hiv_test_raw","828":"received_hiv_result_raw","483":"circumcision_raw",
    })
    ar = ar_raw.rename(columns={"hivclust":"cluster","hivnumb":"household","hivline":"line","hiv05":"hiv_weight_raw"})
    linkage_rows = [
        {"stage":"Women IR records","n":len(ir_raw)},{"stage":"Men MR records","n":len(mr_raw)},
        {"stage":"Combined respondent records","n":len(resp)},{"stage":"AR biomarker records","n":len(ar_raw)},
        {"stage":"Duplicate respondent identifiers","n":int(resp.duplicated(["cluster","household","line"]).sum())},
        {"stage":"Duplicate biomarker identifiers","n":int(ar.duplicated(["cluster","household","line"]).sum())},
    ]
    merged = resp.merge(ar,on=["cluster","household","line"],how="inner",validate="one_to_one")
    linkage_rows += [{"stage":"Respondent-biomarker linked records","n":len(merged)},{"stage":"Unmatched respondent records","n":len(resp)-len(merged)}]
    merged["y"] = np.where(merged["hiv03"]==0,0,np.where(merged["hiv03"].isin([1,2,3]),1,np.nan))
    invalid=int(merged["y"].isna().sum()); linkage_rows.append({"stage":"Excluded invalid/non-binary HIV03 results","n":invalid})
    df=merged.dropna(subset=["y"]).copy(); df["y"]=df["y"].astype(int)
    linkage_rows += [{"stage":"Final analytic records","n":len(df)},{"stage":"HIV-positive analytic records","n":int(df["y"].sum())},{"stage":"HIV-negative analytic records","n":int((df["y"]==0).sum())}]

    df["sexual_debut_status"] = np.select([df["age_first_sex_raw"].eq(0),df["age_first_sex_raw"].eq(97)],["Not had sex","Inconsistent"],default="Reported age")
    df["age_first_sex"] = df["age_first_sex_raw"].astype(float); df.loc[df["age_first_sex_raw"].isin([0,97]),"age_first_sex"] = np.nan
    df["region"] = df["region_raw"].map(REGION_MAP).fillna("Unknown")
    df["residence"] = df["residence_raw"].map(RESIDENCE_MAP).fillna("Unknown")
    df["education"] = df["education_raw"].map(EDUCATION_MAP).fillna("Unknown")
    df["wealth"] = df["wealth_raw"].map(WEALTH_MAP).fillna("Unknown")
    df["marital_status"] = df["marital_raw"].map(MARITAL_MAP).fillna("Unknown")
    df["ever_tested_hiv"] = df["ever_tested_raw"].map({0:"No",1:"Yes"}).fillna("Unknown")
    df["condom_last_sex"] = df["condom_last_sex_raw"].map({0:"No",1:"Yes"}).fillna("Not reported/not applicable")
    df["sti_last12m"] = df["sti_last12m_raw"].map({0:"No",1:"Yes",8:"Unknown"}).fillna("Unknown")
    df["received_hiv_result"] = df["received_hiv_result_raw"].map({0:"No",1:"Yes"}).fillna("Not tested/not reported")
    df["partners_last12m"] = pd.to_numeric(df["partners_last12m_raw"], errors="coerce"); df.loc[df["partners_last12m"].isin([98,99]),"partners_last12m"] = np.nan
    df["lifetime_partners"] = pd.to_numeric(df["lifetime_partners_raw"], errors="coerce"); df.loc[df["lifetime_partners"].isin([98,99]),"lifetime_partners"] = np.nan
    df["months_since_hiv_test"] = pd.to_numeric(df["months_since_hiv_test_raw"], errors="coerce"); df.loc[df["months_since_hiv_test"].isin([98,99]),"months_since_hiv_test"] = np.nan
    df["circumcision_status"]="Not applicable (female)"; male=df["sex"].eq("Male")
    df.loc[male & df["circumcision_raw"].eq(0),"circumcision_status"]="No"; df.loc[male & df["circumcision_raw"].eq(1),"circumcision_status"]="Yes"
    df.loc[male & ~df["circumcision_raw"].isin([0,1]),"circumcision_status"]="Unknown"
    df["respondent_weight"] = pd.to_numeric(df["respondent_weight"],errors="coerce")/1_000_000
    df["hiv_weight"] = pd.to_numeric(df["hiv_weight_raw"],errors="coerce")/1_000_000
    df["age_group"] = pd.cut(df["age"],bins=[14,24,34,44,54,65],labels=["15–24","25–34","35–44","45–54","55–65"],include_lowest=True)
    df["household_id"] = df["cluster"].astype(str)+"_"+df["household"].astype(str)
    return df, pd.DataFrame(linkage_rows)


def assign_partition(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    out=df.copy(); joint=out["y"].astype(str)+"_"+out["sex"].astype(str)+"_"+out["residence"].astype(str)+"_"+out["age_group"].astype(str)
    splitter=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed); folds=np.empty(len(out),dtype=np.int8)
    for fold,(_,test_idx) in enumerate(splitter.split(out,joint,groups=out["cluster"])): folds[test_idx]=fold
    out["fold"]=folds; out["partition"]=np.where(out["fold"]<=2,"Training",np.where(out["fold"]==3,"Validation","Test")); return out


def verify_partition(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str,int]]:
    summary=df.groupby("partition",observed=False).agg(n=("y","size"),positives=("y","sum"),prevalence=("y","mean"),clusters=("cluster","nunique"),households=("household_id","nunique"),female_n=("sex",lambda s:int((s=="Female").sum())),male_n=("sex",lambda s:int((s=="Male").sum()))).reset_index()
    sets={p:set(df.loc[df["partition"].eq(p),"cluster"]) for p in ["Training","Validation","Test"]}; hsets={p:set(df.loc[df["partition"].eq(p),"household_id"]) for p in ["Training","Validation","Test"]}
    overlap={"cluster_train_validation":len(sets["Training"]&sets["Validation"]),"cluster_train_test":len(sets["Training"]&sets["Test"]),"cluster_validation_test":len(sets["Validation"]&sets["Test"]),"household_train_validation":len(hsets["Training"]&hsets["Validation"]),"household_train_test":len(hsets["Training"]&hsets["Test"]),"household_validation_test":len(hsets["Validation"]&hsets["Test"])}
    if any(overlap.values()): raise RuntimeError(f"Partition leakage detected: {overlap}")
    return summary, overlap


def build_preprocessor(numeric: list[str], categorical: list[str], scale_numeric: bool) -> ColumnTransformer:
    num_steps: list[tuple[str,Any]]=[("imputer",SimpleImputer(strategy="median",add_indicator=True))]
    if scale_numeric: num_steps.append(("scaler",StandardScaler()))
    num_pipe=Pipeline(num_steps); cat_pipe=Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))])
    return ColumnTransformer([("numeric",num_pipe,numeric),("categorical",cat_pipe,categorical)],remainder="drop",verbose_feature_names_out=False)


def metric_row(y: np.ndarray, p: np.ndarray, sample_weight: np.ndarray|None=None) -> dict[str,float]:
    return {"roc_auc":float(roc_auc_score(y,p,sample_weight=sample_weight)),"pr_auc":float(average_precision_score(y,p,sample_weight=sample_weight)),"brier_score":float(brier_score_loss(y,p,sample_weight=sample_weight))}


def platt_fit(estimator: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, cv_splits: list[tuple[np.ndarray,np.ndarray]]) -> tuple[Pipeline,LogisticRegression,np.ndarray]:
    oof=np.empty(len(y_train),dtype=float)
    for fit_idx,hold_idx in cv_splits:
        m=clone(estimator); m.fit(X_train.iloc[fit_idx],y_train.iloc[fit_idx]); oof[hold_idx]=m.predict_proba(X_train.iloc[hold_idx])[:,1]
    calibrator=LogisticRegression(C=1e6,solver="lbfgs",max_iter=2000,random_state=SEED)
    calibrator.fit(logit(np.clip(oof,1e-6,1-1e-6)).reshape(-1,1),y_train)
    final_base=clone(estimator); final_base.fit(X_train,y_train); return final_base,calibrator,oof


def apply_calibrator(p: np.ndarray, calibrator: LogisticRegression) -> np.ndarray:
    return calibrator.predict_proba(logit(np.clip(p,1e-6,1-1e-6)).reshape(-1,1))[:,1]


def select_top_capacity(scores: np.ndarray, capacity: float, groups: pd.Series|None=None) -> np.ndarray:
    n=len(scores); selected=np.zeros(n,dtype=bool)
    if groups is None:
        k=max(1,int(math.floor(capacity*n))); order=np.lexsort((np.arange(n),-scores)); selected[order[:k]]=True; return selected
    gser=pd.Series(groups).reset_index(drop=True)
    for _,idx in gser.groupby(gser,dropna=False).groups.items():
        idx_arr=np.asarray(list(idx),dtype=int); k=max(1,int(math.floor(capacity*len(idx_arr)))); local_order=idx_arr[np.lexsort((idx_arr,-scores[idx_arr]))]; selected[local_order[:k]]=True
    return selected


def classification_metrics(y: np.ndarray, selected: np.ndarray) -> dict[str,float|int]:
    tn,fp,fn,tp=confusion_matrix(y,selected.astype(int),labels=[0,1]).ravel(); recall=tp/(tp+fn) if tp+fn else np.nan; precision=tp/(tp+fp) if tp+fp else np.nan; specificity=tn/(tn+fp) if tn+fp else np.nan
    return {"tp":int(tp),"fp":int(fp),"fn":int(fn),"tn":int(tn),"recall":float(recall),"precision":float(precision),"specificity":float(specificity),"false_negative_rate":float(1-recall),"load_n":int(selected.sum()),"load_fraction":float(selected.mean())}


def exact_binomial_ci(successes:int,trials:int,alpha:float=0.05)->tuple[float,float]:
    if trials==0:return np.nan,np.nan
    lower=0.0 if successes==0 else beta.ppf(alpha/2,successes,trials-successes+1); upper=1.0 if successes==trials else beta.ppf(1-alpha/2,successes+1,trials-successes); return float(lower),float(upper)


def subgroup_table(df_eval:pd.DataFrame,y:np.ndarray,scores:np.ndarray,selected:np.ndarray,variable:str)->pd.DataFrame:
    rows=[]; group_series=df_eval[variable].reset_index(drop=True)
    for group,idx in group_series.groupby(group_series,dropna=False).groups.items():
        idx_arr=np.asarray(list(idx),dtype=int); yy=y[idx_arr]; ss=selected[idx_arr]; m=classification_metrics(yy,ss); lo,hi=exact_binomial_ci(m["tp"],m["tp"]+m["fn"])
        rows.append({"variable":variable,"group":str(group),"n":len(idx_arr),"positive":int(yy.sum()),**m,"recall_ci_lower":lo,"recall_ci_upper":hi,"mean_score_positive":float(np.mean(scores[idx_arr][yy==1])) if yy.sum() else np.nan,"max_score_positive":float(np.max(scores[idx_arr][yy==1])) if yy.sum() else np.nan})
    return pd.DataFrame(rows)
