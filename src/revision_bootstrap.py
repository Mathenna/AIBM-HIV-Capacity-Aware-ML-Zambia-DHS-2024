"""Exact multiplicity representation of a fixed-score PSU bootstrap.

Resampling a cluster m times repeats each of its records m times. Keeping those
integer multiplicities avoids materialising duplicates and repeatedly sorting
unchanged scores. A boundary record can contribute only some of its copies to
the selected set. This is not survey-weight-based budget allocation.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd


def _summaries(y, p, selected_copies, multiplicity, survey_weights, rank):
    n = int(multiplicity.sum())
    positive_n = int(multiplicity @ y)
    selected_n = int(selected_copies.sum())
    w = multiplicity * survey_weights
    sw = selected_copies * survey_weights
    tp, fp = float(sw @ y), float(sw @ (1-y))
    positives, negatives = float(w @ y), float(w @ (1-y))
    fn, tn = positives-tp, negatives-fp
    div = lambda a,b: float(a/b) if b>0 else float('nan')
    prevalence, precision = div(positives,w.sum()), div(tp,tp+fp)
    order, ends = rank
    cum_tp = np.cumsum((w*y)[order])[ends]
    cum_fp = np.cumsum((w*(1-y))[order])[ends]
    # Each endpoint is the END of a tied-score block. Zero-multiplicity blocks
    # have zero increments and therefore cannot affect either area.
    dtp = np.diff(np.r_[0.,cum_tp])
    dfp = np.diff(np.r_[0.,cum_fp])
    ppv = np.divide(cum_tp,cum_tp+cum_fp,out=np.zeros_like(cum_tp),where=(cum_tp+cum_fp)>0)
    ap = div(np.sum(dtp*ppv),positives)
    auc = div(np.sum(dfp*(cum_tp+np.r_[0.,cum_tp[:-1]])*.5),positives*negatives)
    return dict(n=n,positive_n=positive_n,selected_n=selected_n,
        selected_fraction=div(selected_n,n),tp=tp,fp=fp,fn=fn,tn=tn,
        positive_fraction=prevalence,recall=div(tp,positives),precision=precision,
        specificity=div(tn,negatives),f2=div(5*tp,5*tp+4*fn+fp),
        yield_enrichment=div(precision,prevalence),roc_auc=auc,
        average_precision=ap,brier=div(np.sum(w*(y-p)**2),w.sum()))


def bootstrap_policy_fast(df, score_columns, capacity, iterations=1000,
                          seed=24101765, strata_column='strata',
                          allocation_column=None,tie_column='tie_key'):
    if iterations<1 or not np.isfinite(capacity) or not 0<=capacity<=1:
        raise ValueError('invalid bootstrap iterations or capacity')
    df=df.reset_index(drop=True)
    needed=['cluster','outer_fold','y','hiv_weight',tie_column]
    if strata_column is not None: needed.append(strata_column)
    if allocation_column is not None: needed.append(allocation_column)
    if df.empty or df[needed].isna().any().any(): raise ValueError('missing bootstrap design variable')
    if df.groupby('cluster').outer_fold.nunique().max()!=1: raise ValueError('cluster crosses outer folds')
    if strata_column and df.groupby('cluster')[strata_column].nunique().max()!=1:
        raise ValueError('cluster crosses strata')
    y=df.y.to_numpy(int); weights=df.hiv_weight.to_numpy(float)
    if not np.isin(y,[0,1]).all() or not np.isfinite(weights).all() or (weights<=0).any():
        raise ValueError('invalid labels or weights')
    keys=df[tie_column].astype(str).to_numpy()
    if len(np.unique(keys))!=len(keys): raise ValueError('original tie keys must be unique')
    block_cols=['outer_fold']+([strata_column] if strata_column else [])
    blocks=[[g.index.to_numpy() for _,g in b.groupby('cluster',sort=True)]
            for _,b in df.groupby(block_cols,sort=True,observed=True)]
    masks={'Overall':np.arange(len(df))}
    for var in ['sex','residence','age_group']:
        if var in df:
            labels=df[var].astype(str).to_numpy()
            masks.update({f'{var}={v}':np.flatnonzero(labels==v) for v in sorted(np.unique(labels))})
    layouts={}
    for model,column in score_columns.items():
        p=df[column].to_numpy(float)
        if not np.isfinite(p).all() or ((p<0)|(p>1)).any(): raise ValueError('invalid probabilities')
        ranks={}
        for name,idx in masks.items():
            order=np.argsort(-p[idx],kind='stable')
            ends=np.r_[np.flatnonzero(np.diff(p[idx][order])!=0),len(idx)-1]
            ranks[name]=(order,ends)
        policies=[]
        for _,fold in df.groupby('outer_fold',sort=True):
            idx=fold.index.to_numpy()
            if allocation_column is None:
                groups=[idx]
            else:
                names=sorted(fold[allocation_column].astype(str).unique())
                groups=[idx[fold[allocation_column].astype(str).to_numpy()==g] for g in names]
            orders=[g[np.lexsort((keys[g],-p[g]))] for g in groups]
            policies.append((idx,orders))
        layouts[model]=(p,ranks,policies)
    rng=np.random.default_rng(seed);rows=[]
    for rep in range(iterations):
        draws=np.concatenate([g for block in blocks for g in
                              [block[j] for j in rng.integers(0,len(block),len(block))]])
        mult=np.bincount(draws,minlength=len(df))
        for model,(p,ranks,policies) in layouts.items():
            selected=np.zeros(len(df),dtype=int)
            for idx,orders in policies:
                n=int(mult[idx].sum());k=math.floor(float(capacity)*n)
                counts=np.array([mult[o].sum() for o in orders])
                ideals=k*counts/n
                quotas=np.floor(ideals).astype(int)
                # orders follow lexicographically sorted group labels, exactly
                # as select_exact's secondary remainder tie-break.
                remainder_order=np.argsort(-(ideals-quotas),kind='stable')
                quotas[remainder_order[:k-int(quotas.sum())]]+=1
                for order,quota in zip(orders,quotas):
                    before=np.cumsum(mult[order])-mult[order]
                    selected[order]=np.minimum(mult[order],np.maximum(quota-before,0))
                if int(selected[idx].sum())!=k: raise AssertionError('bootstrap budget mismatch')
            for group,idx in masks.items():
                if mult[idx].sum()==0: continue
                for weighting in ['unweighted','hiv_weighted']:
                    w=np.ones(len(idx)) if weighting=='unweighted' else weights[idx]
                    vals=_summaries(y[idx],p[idx],selected[idx],mult[idx],w,ranks[group])
                    rows.append(dict(replicate=rep,model=model,group=group,weighting=weighting,
                                     capacity=capacity,**vals))
        if (rep+1)%100==0: print(f'  bootstrap {rep+1}/{iterations}',flush=True)
    return pd.DataFrame(rows),dict(singleton_stratum_fold_blocks=sum(len(b)==1 for b in blocks),
        estimand='conditional on fitted OOF scores; stratified PSU bootstrap within outer folds; top-k reselected',
        computation='integer cluster multiplicities; exact partial selection of boundary copies')
