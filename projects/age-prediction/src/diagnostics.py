"""Quality sensitivity and matched-control diagnostics, dated after primary results.
Missingness sensitivity is exploratory: primary external results retain every control.
"""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from threadpoolctl import threadpool_limits
from analyze import load,weights,donor_mae,evaluate,HellingerFilter,SEED,ROOT,RES,DATA

def main():
    # Inspect, but do not change, locked external predictions.
    p=pd.read_csv(RES/'blood_external_predictions.csv');Z=np.load(DATA/'external.npz')['X']
    p['missing_fraction']=np.isnan(Z).mean(axis=1);p['error']=p.prediction-p.age
    p['flag_over_1pct_missing']=p.missing_fraction>.01
    p[['sample_id','age','prediction','error','missing_fraction','flag_over_1pct_missing']].to_csv(RES/'external_quality_audit.csv',index=False)
    sensitivity=[]
    for limit in [1.,.01,.005]:
        q=p[p.missing_fraction<=limit]
        sensitivity.append({'maximum_missing_fraction':limit,'excluded_samples':len(p)-len(q),**evaluate(q)})
    pd.DataFrame(sensitivity).to_csv(RES/'external_qc_sensitivity.csv',index=False)
    # Paired donor bootstrap of predictive-error differences, not model-retraining CIs.
    def paired_ci(a,b):
        z=a.merge(b,on='sample_id',suffixes=('_a','_b'),validate='one_to_one')
        z['difference']=np.abs(z.prediction_a-z.age_a)-np.abs(z.prediction_b-z.age_b)
        d=z.groupby('group_a').difference.mean().to_numpy();rng=np.random.default_rng(SEED)
        bs=d[rng.integers(0,len(d),(5000,len(d)))].mean(axis=1)
        return {'difference_mae':float(d.mean()),'low':float(np.quantile(bs,.025)),'high':float(np.quantile(bs,.975)),'n_participants':len(d)}
    s=pd.read_csv(RES/'skin_cv_predictions.csv');meta=pd.read_csv(RES/'skin_metadata_predictions.csv');diag=pd.read_csv(RES/'skin_split_diagnostics.csv')
    c={'metadata_minus_microbiome':paired_ci(meta,s),'participant_minus_random':paired_ci(diag[diag.validation=='Participant split'],diag[diag.validation=='Random sample split'])}
    (RES/'paired_error_comparisons.json').write_text(json.dumps(c,indent=2))
    study=[]
    for name,d in [('Nested participant CV',s),('Leave study out',diag[diag.validation=='Leave study out'])]:
        for st,q in d.groupby('study'):study.append({'validation':name,'study':str(st),**evaluate(q)})
    pd.DataFrame(study).to_csv(RES/'skin_study_performance.csv',index=False)
    units=[]
    for name,d in diag.groupby('validation'):
        units.extend([{'validation':name,'weighting':'Equal samples','mae':mean_absolute_error(d.age,d.prediction)},{'validation':name,'weighting':'Equal participants','mae':donor_mae(d.age,d.prediction,d.group)}])
    pd.DataFrame(units).to_csv(RES/'skin_weighting_sensitivity.csv',index=False)
    observed=[]
    with threadpool_limits(limits=2):
        for typ in ['blood','skin']:
            X,m,_=load(typ);g=m.group.to_numpy();y=m.age.to_numpy();w=weights(g);cv=GroupKFold(5,shuffle=True,random_state=SEED)
            if typ=='blood':
                model=Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('select',SelectKBest(f_regression,k=500)),('scale',StandardScaler()),('ridge',Ridge(alpha=100.,solver='lsqr',tol=1e-5))])
            else:
                # Match the null: donor-mean target and exactly 80 trees.
                y=m.group.map(m.groupby('group').age.mean()).to_numpy()
                model=Pipeline([('composition',HellingerFilter()),('rf',RandomForestRegressor(n_estimators=80,max_features='sqrt',min_samples_leaf=3,n_jobs=2,random_state=SEED))])
            pred=np.empty(len(y))
            for tr,te in cv.split(X,y,g):
                model.fit(X.iloc[tr],y[tr],**({'rf__sample_weight':w[tr]} if typ=='skin' else {}));pred[te]=model.predict(X.iloc[te])
            observed.append({'modality':typ,'mae':mean_absolute_error(y,pred) if typ=='blood' else donor_mae(y,pred,g)})
    pd.DataFrame(observed).to_csv(RES/'permutation_observed_matched.csv',index=False)
    print('Quality sensitivity:',sensitivity);print('Paired comparisons:',c)
if __name__=='__main__':main()
