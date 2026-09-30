"""Meaningful algorithm/integration checks; no DHS records required."""
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from revision_policy import select_exact, metrics, bootstrap_policy
from revision_bootstrap import bootstrap_policy_fast
from revision_models import FoldWeightedXGB, grouped_splits, develop, calibrate, estimator_and_grid
from revision_data import restrict_cohort, load_cohort
from revision_pipeline import evaluate_analysis, capacity_from_training


def synthetic_cohort(seed=91):
    rng = np.random.default_rng(seed)
    n = 600
    cluster = np.repeat(np.arange(60), 10)
    age = rng.integers(15, 60, n)
    sex = np.where(age > 49, "Male", rng.choice(["Male", "Female"], n))
    df = pd.DataFrame({"cluster": cluster, "outer_fold": cluster % 5,
        "strata": (cluster // 5) % 2, "age": age, "sex": sex,
        "residence": rng.choice(["Rural","Urban"], n),
        "region": rng.choice(["Central","Eastern","Lusaka"], n),
        "education": rng.choice(["Primary","Secondary"], n),
        "wealth": rng.choice(["Poorer","Richer"], n),
        "marital_status": rng.choice(["Married","Never in union"], n),
        "age_first_sex": rng.integers(14, 23, n).astype(float),
        "sexual_debut_status": "Reported age", "partners_last12m": rng.integers(0,3,n),
        "lifetime_partners": rng.integers(1,10,n), "months_since_hiv_test": rng.integers(0,50,n).astype(float),
        "ever_tested_hiv": rng.choice(["Yes","No"],n), "received_hiv_result": "Yes",
        "condom_last_sex": rng.choice(["Yes","No"],n), "sti_last12m": "No",
        "circumcision_status": np.where(sex == "Female", "Not applicable (female)", "No"),
        "hiv_weight": rng.uniform(.4,2.,n), "prior_result": rng.choice([1.,2.,np.nan],n),
        "tie_key": [f"record_{i:06d}" for i in range(n)]})
    df["age_group"] = pd.cut(df.age,[14,24,34,44,54,59],labels=["15–24","25–34","35–44","45–54","55–59"]).astype(str)
    # Strong signal ensures calibration slope tests are meaningful/stable.
    df["y"] = (rng.random(n) < (1 / (1 + np.exp(-(-4.5 + age*.10))))).astype(int)
    df.loc[rng.choice(n,40,False), "months_since_hiv_test"] = np.nan
    return df


class PolicyTests(unittest.TestCase):
    def test_fast_bootstrap_equals_explicit_resampling(self):
        df=synthetic_cohort()
        df=df.loc[(df.index%10)<(3+df.cluster%7)].reset_index(drop=True)
        df['a']=np.round(df.age/100,1)
        df['b']=np.linspace(.01,.99,len(df))
        keys=['replicate','model','group','weighting','capacity']
        for c,alloc in [(.243,None),(.243,'age_group'),(.243,'sex'),(.243,'residence'),(0,None),(1,'age_group')]:
            with self.subTest(capacity=c,allocation=alloc):
                a,_=bootstrap_policy(df,{'A':'a','B':'b'},c,9,77,allocation_column=alloc)
                b,_=bootstrap_policy_fast(df,{'A':'a','B':'b'},c,9,77,allocation_column=alloc)
                a=a.sort_values(keys).reset_index(drop=True)
                b=b.sort_values(keys).reset_index(drop=True)
                pd.testing.assert_frame_equal(a,b,check_dtype=False,atol=1e-10,rtol=1e-10)

    def test_prefix_capacity_curve_equals_reference_selections(self):
        df=synthetic_cohort();scores=np.round(df.age/100,1)
        grid=np.r_[0,np.arange(.01,.501,.001),1]
        c,curve=capacity_from_training(df.y,scores,df.tie_key,grid)
        expected=np.array([metrics(df.y,scores,select_exact(scores,g,tie_keys=df.tie_key))['f2'] for g in grid])
        np.testing.assert_allclose(curve.f2,expected,atol=1e-14,rtol=1e-14)
        self.assertEqual(c,float(grid[np.nanargmax(expected)]))

    def test_exact_age_quotas_and_boundaries(self):
        sizes = [2114,1324,992,503,83]
        groups = np.repeat(["a","b","c","d","e"], sizes)
        scores = np.linspace(0,1,sum(sizes))
        self.assertEqual(select_exact(scores,.243,groups).sum(),1218)
        for n in [0,1,3,10,5016]:
            for c in [0,.001,.01,.243,.5,1]:
                self.assertEqual(select_exact(np.ones(n),c).sum(),int(np.floor(n*c)))
        self.assertEqual(select_exact(np.ones(3),.1,["a","b","c"]).sum(),0)

    def test_ties_invariant_to_reordering_with_keys(self):
        scores=np.array([.5,.5,.5,.2]); keys=np.array(["c","b","a","d"])
        a=set(keys[select_exact(scores,.5,tie_keys=keys)])
        idx=np.array([2,3,0,1])
        b=set(keys[idx][select_exact(scores[idx],.5,tie_keys=keys[idx])])
        self.assertEqual(a,b)
        self.assertEqual(a,{"a","b"})

    def test_invalid_input_fails(self):
        for c in [-.1,1.1,float("nan")]:
            with self.assertRaises(ValueError): select_exact([.2,.8],c)
        with self.assertRaises(ValueError): select_exact([.2,np.nan],.5)
        with self.assertRaises(ValueError): select_exact([.2,.8],.5,["a"])

    def test_metric_arithmetic(self):
        y=np.r_[np.ones(281),np.zeros(937),np.ones(167),np.zeros(3631)]
        sel=np.arange(5016)<1218
        result=metrics(y,np.full(5016,.1),sel)
        self.assertAlmostEqual(result["recall"],281/448)
        self.assertAlmostEqual(result["f2"],5*281/(5*281+4*167+937))

    def test_outcome_independent_eligibility(self):
        df=synthetic_cohort(); out=restrict_cohort(df,"restricted")
        flip=df.copy();flip.y=1-flip.y
        self.assertEqual(list(out.tie_key),list(restrict_cohort(flip,"restricted").tie_key))
        self.assertFalse(out.prior_result.eq(1).any())
        self.assertTrue(restrict_cohort(df,"all_15_49").age.le(49).all())

    def test_bootstrap_reselects_exact_budget_and_paired_models(self):
        df=synthetic_cohort()
        # Unequal cluster sizes mean bootstrap N is variable.
        df=df.loc[(df.index%10)<(3+df.cluster%7)].copy()
        df["a"] = df.age/100
        df["b"] = df["a"]
        reps,_=bootstrap_policy(df,{"A":"a","B":"b"},.243,iterations=8,seed=77)
        overall=reps.loc[reps.group.eq("Overall") & reps.weighting.eq("unweighted")]
        wide=overall.pivot(index="replicate",columns="model",values="recall")
        self.assertTrue(np.allclose(wide.A,wide.B))
        # Five separately floored cohort budgets lie in [C*N - 5, C*N].
        self.assertTrue((overall.selected_n <= .243*overall.n).all())
        self.assertTrue((overall.selected_n > .243*overall.n-5).all())


class ModelTests(unittest.TestCase):
    def test_all_families_fit_and_ignore_unseen_categories(self):
        from analysis_common import EXPANDED_NUMERIC, EXPANDED_CATEGORICAL
        df=synthetic_cohort()
        cfg=yaml.safe_load((ROOT/'config/revision_config.yaml').read_text())
        for name in cfg['models']:
            with self.subTest(model=name):
                model,grid=estimator_and_grid(name,EXPANDED_NUMERIC,EXPANDED_CATEGORICAL,42,cfg['grids'],True)
                model.set_params(**{k:v[0] for k,v in grid.items()})
                features=['age','sex','residence','region'] if name=='demographic_spline' else EXPANDED_NUMERIC+EXPANDED_CATEGORICAL
                model.fit(df[features].iloc[:500],df.y.iloc[:500])
                ho=df[features].iloc[500:].copy();ho['region']='Previously unseen'
                p=model.predict_proba(ho)[:,1]
                self.assertTrue(np.isfinite(p).all())
                self.assertTrue(((p>=0)&(p<=1)).all())

    def test_fold_weight_is_recomputed_and_estimator_cloneable(self):
        X=np.arange(40).reshape(20,2); y=np.r_[np.zeros(15),np.ones(5)]
        model=FoldWeightedXGB(n_estimators=2,max_depth=1)
        model.fit(X,y);self.assertEqual(model.scale_pos_weight_,3.)
        other=clone(model).fit(X[10:],y[10:]);self.assertEqual(other.scale_pos_weight_,1.)
        plain=FoldWeightedXGB(weighted=False,n_estimators=2).fit(X,y)
        self.assertEqual(plain.scale_pos_weight_,1.)

    def test_grouped_development_and_monotonic_calibration(self):
        from analysis_common import EXPANDED_NUMERIC, EXPANDED_CATEGORICAL
        df=synthetic_cohort()
        for a,b in grouped_splits(df,3,42):
            self.assertFalse(set(df.iloc[a].cluster)&set(df.iloc[b].cluster))
        cfg=yaml.safe_load((ROOT/'config/revision_config.yaml').read_text())
        base,features,cal,oof,_,_=develop(df,"demographic_spline",EXPANDED_NUMERIC,
                                        EXPANDED_CATEGORICAL,cfg,42,smoke=True)
        self.assertTrue(np.isfinite(oof).all())
        p=calibrate(np.linspace(.001,.999,100),cal)
        self.assertTrue((np.diff(p)>0).all())

    def test_end_to_end_synthetic_smoke(self):
        cfg=yaml.safe_load((ROOT/'config/revision_config.yaml').read_text())
        cfg.update(bootstrap_iterations=2,fixed_capacities=[.243],
                   capacity_grid={"start":.2,"stop":.3,"step":.05})
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)/'synthetic'
            result=evaluate_analysis(synthetic_cohort(),cfg,dest,
                      models=["demographic_spline","xgb_unweighted","xgb_weighted"],smoke=True)
            self.assertEqual(result["status"],"synthetic_smoke_only")
            self.assertTrue((dest/'paired_model_intervals.csv').is_file())
            perf=pd.read_csv(dest/'fold_performance.csv')
            selected=perf.loc[perf.model.eq('selected_pipeline') & perf.weighting.eq('unweighted')]
            for _,g in selected.groupby(['outer_fold','capacity_role']):
                self.assertEqual(g.selected_n.nunique(),1)
            self.assertFalse(list(dest.glob('*.npy')))


class DataTests(unittest.TestCase):
    def test_stata_linkage_labels_and_age_correction(self):
        df=synthetic_cohort().reset_index(drop=True)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            for sex,prefix,filename in [('Female','v','ZMIR81FL.dta'),('Male','mv','ZMMR81FL.dta')]:
                part=df.loc[df.sex.eq(sex)]
                indices=part.index.to_numpy()
                raw=pd.DataFrame({prefix+'001':part.cluster.to_numpy()+1,
                    prefix+'002':indices//4+1,prefix+'003':indices%4+1,
                    prefix+'005':1_000_000,prefix+'012':part.age.to_numpy(),
                    prefix+'021':part.cluster.to_numpy()+1,prefix+'022':part.strata.to_numpy()+1,
                    prefix+'024':1,prefix+'025':1,prefix+'106':1,prefix+'190':1,
                    prefix+'501':1,prefix+'531':18,prefix+'781':1,prefix+'761':0,
                    prefix+'763a':0,prefix+'766b':1,prefix+'836':3,prefix+'826a':6,
                    prefix+'828':1,prefix+'861':part.prior_result.to_numpy()})
                if sex=='Male':raw[prefix+'483']=0
                raw.to_stata(path/filename,write_index=False,
                             value_labels={prefix+'861':{1:'Positive',2:'Negative'}})
            idx=df.index.to_numpy()
            pd.DataFrame({'hivclust':df.cluster+1,'hivnumb':idx//4+1,'hivline':idx%4+1,
                          'hiv03':df.y,'hiv05':1_000_000}).to_stata(path/'ZMAR81FL.dta',write_index=False)
            cohort,linkage,eligibility,provenance=load_cohort(path,24101765)
            self.assertEqual(len(cohort),len(df))
            self.assertEqual(cohort.y.sum(),df.y.sum())
            self.assertTrue(cohort.loc[cohort.age.ge(55),'age_group'].eq('55–59').all())
            self.assertEqual(len(restrict_cohort(cohort,'restricted')),int((~df.prior_result.eq(1)).sum()))
            self.assertEqual(len(provenance['input_sha256']),3)
            self.assertEqual(cohort.groupby('cluster').outer_fold.nunique().max(),1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
