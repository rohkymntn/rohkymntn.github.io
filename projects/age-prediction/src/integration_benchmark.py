"""Exploratory follow-up: metadata-adjusted microbiome prediction.
Three repeated nested participant-held-out comparisons, defined before execution.
All candidates and repeats are retained regardless of their performance.
"""
from pathlib import Path
import json,time
import numpy as np,pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from threadpoolctl import threadpool_limits
from analyze import HellingerFilter,load,weights
ROOT=Path(__file__).resolve().parents[1]
SEEDS=[20260906,20260907,20260908]
LEAVES=[1,3,8]

def meta_model(kind):
    model=Ridge(alpha=10.,solver='lsqr') if kind=='ridge' else RandomForestRegressor(n_estimators=160,max_features=1.,min_samples_leaf=10,random_state=104,n_jobs=4)
    return Pipeline([('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False)),('reg',model)])
def rf(seed,leaf):return Pipeline([('hellinger',HellingerFilter()),('reg',RandomForestRegressor(n_estimators=160,max_features='sqrt',min_samples_leaf=leaf,random_state=seed,n_jobs=4))])
def mae(y,p,g):return mean_absolute_error(y,p,sample_weight=weights(g))

def run_repeat(seed):
    started=time.time();X,m,_=load('skin');X=X.to_numpy();y=m.age.to_numpy();g=m.group.to_numpy();meta=m[['study','site','sex']]
    w=weights(g);rows=[];selection=[]
    with threadpool_limits(limits=4):
      for fold,(tr,te) in enumerate(GroupKFold(5,shuffle=True,random_state=seed).split(X,y,g)):
        inner=list(GroupKFold(3).split(X[tr],y[tr],g[tr]))
        candidates={};preds={}
        # Select metadata-only architecture exclusively inside the training partition.
        for kind in ['ridge','forest']:
            out=np.empty(len(tr))
            for a,b in inner:
                fit=meta_model(kind).fit(meta.iloc[tr[a]],y[tr[a]],reg__sample_weight=w[tr[a]])
                out[b]=fit.predict(meta.iloc[tr[b]])
            candidates['metadata_'+kind]=mae(y[tr],out,g[tr])
        kind=min(['ridge','forest'],key=lambda k:candidates['metadata_'+k])
        # The augmented model always residualizes using ridge metadata predictions.
        # Ridge fitting and residual construction are repeated inside each inner fold.
        for leaf in LEAVES:
            out=np.empty(len(tr));micro=np.empty(len(tr))
            for a,b in inner:
                a,b=tr[a],tr[b]
                fitmeta=meta_model('ridge').fit(meta.iloc[a],y[a],reg__sample_weight=w[a])
                fit=rf(seed,leaf).fit(X[a],y[a]-fitmeta.predict(meta.iloc[a]),reg__sample_weight=w[a])
                out[np.isin(tr,b)]=0 # assignments below use exact fold ordering
                # indices in `tr` are sorted by position, as are GroupKFold outputs.
                ix=np.searchsorted(tr,b)
                out[ix]=fitmeta.predict(meta.iloc[b])+fit.predict(X[b])
                fm=rf(seed,leaf).fit(X[a],y[a],reg__sample_weight=w[a]);micro[ix]=fm.predict(X[b])
            candidates[f'combined_leaf{leaf}']=mae(y[tr],out,g[tr])
            candidates[f'microbiome_leaf{leaf}']=mae(y[tr],micro,g[tr])
        cl=min(LEAVES,key=lambda l:candidates[f'combined_leaf{l}']);ml=min(LEAVES,key=lambda l:candidates[f'microbiome_leaf{l}'])
        base=meta_model(kind).fit(meta.iloc[tr],y[tr],reg__sample_weight=w[tr])
        ridge=meta_model('ridge').fit(meta.iloc[tr],y[tr],reg__sample_weight=w[tr])
        combined=rf(seed,cl).fit(X[tr],y[tr]-ridge.predict(meta.iloc[tr]),reg__sample_weight=w[tr])
        micro=rf(seed,ml).fit(X[tr],y[tr],reg__sample_weight=w[tr])
        q=m.iloc[te].copy();q['repeat_seed']=seed;q['fold']=fold
        q['metadata_prediction']=base.predict(meta.iloc[te]);q['microbiome_prediction']=micro.predict(X[te]);q['combined_prediction']=ridge.predict(meta.iloc[te])+combined.predict(X[te]);rows.append(q)
        selection.append({'seed':seed,'fold':fold,'metadata_kind':kind,'combined_leaf':cl,'microbiome_leaf':ml,'inner_mae':candidates,'train_participants':len(set(g[tr])),'test_participants':len(set(g[te])),'participant_overlap':len(set(g[tr])&set(g[te]))})
        print('repeat',seed,'fold',fold,{col:round(mae(y[te],q[col],g[te]),3) for col in ['metadata_prediction','microbiome_prediction','combined_prediction']},flush=True)
    return pd.concat(rows).to_csv(index=False),selection,time.time()-started
