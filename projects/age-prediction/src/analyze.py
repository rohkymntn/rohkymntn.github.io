"""Reproducible independent age clocks, leakage audit, external validation.
No methylation/microbiome samples are paired or fused in this analysis.
"""
from pathlib import Path
import json, time, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator,TransformerMixin,clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold,KFold,GridSearchCV
from sklearn.metrics import mean_absolute_error,r2_score
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data';RES=ROOT/'results'
SEED=20260905

class HellingerFilter(TransformerMixin,BaseEstimator):
    def __init__(self,prevalence=.01):self.prevalence=prevalence
    def fit(self,X,y=None):
        a=np.asarray(X); self.keep_=(a>0).mean(axis=0)>=self.prevalence
        if not self.keep_.any(): raise ValueError('No microbial features survive training prevalence threshold')
        return self
    def transform(self,X):
        a=np.asarray(X,dtype=np.float64)
        # Normalize before filtering so denominator is the same sample-local total.
        return np.sqrt(a[:,self.keep_]/np.maximum(a.sum(axis=1,keepdims=True),1))

def load(name):
    f=np.load(DATA/f'{name}.npz');m=pd.read_csv(DATA/f'{name}_metadata.csv',dtype={'group':str,'study':str})
    return pd.DataFrame(f['X']),m,f['features']

def weights(groups):
    s=pd.Series(groups);return (1/s.map(s.value_counts())).to_numpy()

def donor_mae(y,p,groups):return mean_absolute_error(y,p,sample_weight=weights(groups))

def evaluate(frame):
    y=frame.age.to_numpy();p=frame.prediction.to_numpy()
    is_blood='plate' in frame.columns or 'ageatrecruitment' in frame.columns
    w=np.ones(len(y)) if is_blood else weights(frame.group)
    ae=np.abs(p-y);se=(p-y)**2
    g=pd.DataFrame({'ae':ae,'se':se,'bias':p-y,'group':frame.group}).groupby('group').agg(ae=('ae','mean'),total=('ae','sum'),n=('ae','size'))
    rng=np.random.default_rng(SEED);ix=rng.integers(0,len(g),(2000,len(g)))
    boots=(g.total.to_numpy()[ix].sum(axis=1)/g.n.to_numpy()[ix].sum(axis=1)) if is_blood else np.mean(g.ae.to_numpy()[ix],axis=1)
    return {'n_samples':len(y),'n_groups':len(g),'mae':float(np.average(ae,weights=w)),'mae_low':float(np.quantile(boots,.025)),'mae_high':float(np.quantile(boots,.975)),'rmse':float(np.sqrt(np.average(se,weights=w))),'r2':float(r2_score(y,p,sample_weight=w)),'spearman_sample_level':float(spearmanr(y,p).statistic),'bias':float(np.average(p-y,weights=w))}

def record(m,p,baseline,fold,model,validation):
    q=m.copy();q['prediction']=p;q['baseline']=baseline;q['fold']=fold;q['model']=model;q['validation']=validation
    return q

def blood():
    X,m,features=load('blood');Z,n,zfeatures=load('external')
    assert np.array_equal(features,zfeatures)
    y=m.age.to_numpy();groups=m.group.to_numpy()
    base=Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('select',SelectKBest(f_regression,k=500)),('scale',StandardScaler()),('ridge',Ridge(solver='lsqr',tol=1e-5))])
    grid={'select__k':[100,500],'ridge__alpha':[10.,100.,1000.]}
    # Primary internal test: whole assay plates held out, including all preprocessing.
    cv=GroupKFold(n_splits=5,shuffle=True,random_state=SEED)
    rows=[]; selected=[];params=[];all_features=[]
    for k,(tr,te) in enumerate(cv.split(X,y,groups)):
        assert not set(groups[tr])&set(groups[te])
        inner=list(GroupKFold(n_splits=3).split(X.iloc[tr],y[tr],groups[tr]))
        fit=GridSearchCV(base,grid,cv=inner,scoring='neg_mean_absolute_error',n_jobs=1).fit(X.iloc[tr],y[tr])
        rows.append(record(m.iloc[te],fit.predict(X.iloc[te]),np.median(y[tr]),k,'CpG ridge','Plate-held-out nested CV'))
        ix=fit.best_estimator_['select'].get_support();selected.extend(features[ix]);params.append({'fold':k,**fit.best_params_})
        all_features.append({'fold':k,'cpgs':features[ix].tolist()})
        print('Blood fold',k,'MAE',mean_absolute_error(y[te],rows[-1].prediction),'params',fit.best_params_,flush=True)
    pred=pd.concat(rows).sort_index();pred.to_csv(RES/'blood_cv_predictions.csv',index=False)
    full=GridSearchCV(base,grid,cv=list(cv.split(X,y,groups)),scoring='neg_mean_absolute_error',n_jobs=1).fit(X,y)
    # Only now evaluate external cohort: no adjustment using its ages.
    ext=record(n,full.predict(Z),np.median(y),-1,'CpG ridge','Independent external cohort')
    ext.to_csv(RES/'blood_external_predictions.csv',index=False)
    coef=full.best_estimator_['ridge'].coef_;keep=full.best_estimator_['select'].get_support()
    coeff=pd.DataFrame({'feature':features[keep],'coefficient_per_training_sd':coef,'cv_selection_count':[selected.count(c) for c in features[keep]]})
    coeff.sort_values('coefficient_per_training_sd',key=np.abs,ascending=False).to_csv(RES/'blood_coefficients.csv',index=False)
    (RES/'blood_model_selection.json').write_text(json.dumps({'outer':params,'final':full.best_params_,'selected_by_fold':all_features},indent=2))
    # Prespecified label-shuffle check; preprocessing and selection refitted after permutation.
    rng=np.random.default_rng(SEED);null=[]
    for perm in range(10):
        yp=rng.permutation(y);pr=np.empty(len(y))
        for tr,te in cv.split(X,yp,groups):
            fit=clone(base).set_params(select__k=500,ridge__alpha=100.).fit(X.iloc[tr],yp[tr]);pr[te]=fit.predict(X.iloc[te])
        null.append(mean_absolute_error(yp,pr))
    pd.DataFrame({'mae':null}).to_csv(RES/'blood_permutation.csv',index=False)
    return pred,ext

def skin():
    X,m,features=load('skin');y=m.age.to_numpy();groups=m.group.to_numpy();w=weights(groups)
    base=Pipeline([('composition',HellingerFilter()),('rf',RandomForestRegressor(n_estimators=160,max_features='sqrt',n_jobs=2,random_state=SEED))])
    grid={'rf__min_samples_leaf':[1,3,5]}
    cv=GroupKFold(n_splits=5,shuffle=True,random_state=SEED)
    def scorer(model,xx,yy):return -mean_absolute_error(yy,model.predict(xx),sample_weight=w[xx.index])
    rows=[];importances=[];params=[]
    for k,(tr,te) in enumerate(cv.split(X,y,groups)):
        assert not set(groups[tr])&set(groups[te])
        inner=list(GroupKFold(n_splits=3).split(X.iloc[tr],y[tr],groups[tr]))
        fit=GridSearchCV(base,grid,cv=inner,scoring=scorer,n_jobs=1).fit(X.iloc[tr],y[tr],rf__sample_weight=w[tr])
        # Weighted median baseline (each training participant receives equal weight).
        order=np.argsort(y[tr]);median=y[tr][order][np.searchsorted(np.cumsum(w[tr][order]),w[tr].sum()/2)]
        rows.append(record(m.iloc[te],fit.predict(X.iloc[te]),median,k,'Microbiome RF','Participant-held-out nested CV'))
        params.append({'fold':k,**fit.best_params_})
        # Exploratory training impurity scores, not causal or held-out importance.
        imp=np.zeros(len(features));imp[fit.best_estimator_['composition'].keep_]=fit.best_estimator_['rf'].feature_importances_;importances.append(imp)
        print('Skin fold',k,'donor MAE',donor_mae(y[te],rows[-1].prediction,groups[te]),fit.best_params_,flush=True)
    pred=pd.concat(rows).sort_index();pred.to_csv(RES/'skin_cv_predictions.csv',index=False)
    pd.DataFrame({'feature':features,'mean_training_impurity_importance':np.mean(importances,axis=0)}).sort_values('mean_training_impurity_importance',ascending=False).to_csv(RES/'skin_feature_importance.csv',index=False)
    (RES/'skin_model_selection.json').write_text(json.dumps(params,indent=2))
    # Fixed-estimator comparisons isolate split effects, rather than tuning each contrast.
    diagnostics=[];overlaps=[]
    splits=[('Random sample split',list(KFold(5,shuffle=True,random_state=SEED).split(X))),('Participant split',list(cv.split(X,y,groups))),('Leave study out',[(np.where(m.study!=s)[0],np.where(m.study==s)[0]) for s in sorted(m.study.unique())])]
    for label,folds in splits:
        for k,(tr,te) in enumerate(folds):
            fit=clone(base).set_params(rf__min_samples_leaf=3).fit(X.iloc[tr],y[tr],rf__sample_weight=w[tr])
            overlap=np.isin(groups[te],groups[tr]).mean();overlaps.append({'validation':label,'fold':k,'test_rows':len(te),'test_participant_overlap_fraction':float(overlap)})
            order=np.argsort(y[tr]);median=y[tr][order][np.searchsorted(np.cumsum(w[tr][order]),w[tr].sum()/2)]
            diagnostics.append(record(m.iloc[te],fit.predict(X.iloc[te]),median,k,'Fixed microbiome RF',label))
        print('Skin diagnostic complete',label,flush=True)
    diag=pd.concat(diagnostics);diag.to_csv(RES/'skin_split_diagnostics.csv',index=False);pd.DataFrame(overlaps).to_csv(RES/'skin_split_overlap.csv',index=False)
    # Metadata-only baseline: cannot include age, donor ID, or date.
    met=m[['study','site','sex']];met_rows=[]
    for k,(tr,te) in enumerate(cv.split(X,y,groups)):
        fit=Pipeline([('onehot',OneHotEncoder(handle_unknown='ignore')),('ridge',Ridge(alpha=10.,solver='lsqr'))]).fit(met.iloc[tr],y[tr],ridge__sample_weight=w[tr])
        met_rows.append(record(m.iloc[te],fit.predict(met.iloc[te]),0,k,'Study/site/sex only','Participant-held-out CV'))
    meta_pred=pd.concat(met_rows);meta_pred.to_csv(RES/'skin_metadata_predictions.csv',index=False)
    # Group-level age permutation preserves repeated-measure age trajectories within donor.
    # Predict each donor's mean age with fixed RF, shuffle mean age labels between donors.
    gm=m.groupby('group').age.mean();rng=np.random.default_rng(SEED);null=[]
    for perm in range(10):
        lookup=dict(zip(gm.index,rng.permutation(gm.to_numpy())));yp=m.group.map(lookup).to_numpy();pr=np.empty(len(y))
        for tr,te in cv.split(X,yp,groups):
            fit=clone(base).set_params(rf__n_estimators=80,rf__min_samples_leaf=3).fit(X.iloc[tr],yp[tr],rf__sample_weight=w[tr]);pr[te]=fit.predict(X.iloc[te])
        null.append(donor_mae(yp,pr,groups));print('Skin label permutation',perm,flush=True)
    pd.DataFrame({'mae':null}).to_csv(RES/'skin_permutation.csv',index=False)
    return pred,diag,meta_pred

def main():
    started=time.time()
    with threadpool_limits(limits=2):
        b,e=blood();s,diag,meta=skin()
    # Blood uncertainty is bootstrapped by plate internally, sample externally.
    rows=[]
    for name,p in [('Blood plate CV',b),('Blood external',e),('Skin participant CV',s),('Skin metadata only',meta)]:
        rows.append({'analysis':name,**evaluate(p)})
        if name!='Skin metadata only':
            q=p.copy();q['prediction']=q.baseline;rows.append({'analysis':name+' median baseline',**evaluate(q)})
    for label,p in diag.groupby('validation'):rows.append({'analysis':'Skin '+label,**evaluate(p)})
    metrics=pd.DataFrame(rows);metrics.to_csv(RES/'metrics.csv',index=False)
    sub=[]
    for name,p in [('blood_cv',b),('blood_external',e),('skin_cv',s)]:
        p=p.copy();p['age_band']=pd.cut(p.age,[18,30,40,50,60,70,101],right=False).astype(str)
        for col in ['age_band','sex']+(['study','site'] if name=='skin_cv' else []):
            for v,q in p.groupby(col):
                if len(q)>=3:sub.append({'analysis':name,'stratum':col,'level':v,**evaluate(q)})
    pd.DataFrame(sub).to_csv(RES/'subgroup_metrics.csv',index=False)
    (RES/'run.json').write_text(json.dumps({'seed':SEED,'analysis_date':'2026-09-05','runtime_seconds':time.time()-started,'warning':'2026 reconstruction, not recovered 2023 results. No matched blood/skin samples or validated multimodal model. Bootstrap intervals condition on fitted out-of-fold predictions and do not include full retraining uncertainty.'},indent=2))
    print(metrics.to_string(index=False),flush=True)
if __name__=='__main__':main()
