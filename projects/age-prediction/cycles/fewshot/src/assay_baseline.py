"""Post-evaluation diagnostic: source-only refitting on target-observed probes.
No external ages enter fitting/tuning. Added after primary results were inspected.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
R=Path(__file__).resolve().parents[1];s=np.load(R/'data/source.npz');tr=s['train'];rows=[]
for acc in ['GSE40279','GSE37008']:
 d=np.load(R/f'data/{acc}.npz');imputed=np.clip((s['impute']-s['mean'])/s['scale'],-8,8).astype('float32');available=np.any(abs(d['X']-imputed)>1e-6,axis=0)
 import json
 assert int(available.sum())==json.loads((R/f'results/{acc}_audit.json').read_text())['observed_selected_features']
 A=s['X'][tr][:,available];y=s['y'][tr];splits=list(GroupKFold(5).split(A,y,s['groups'][tr]));scores={}
 for alpha in [10.,100.,1000.]:
  scores[alpha]=np.mean(np.concatenate([abs(Ridge(alpha=alpha).fit(A[i],y[i]).predict(A[j])-y[j]) for i,j in splits]))
 alpha=min(scores,key=scores.get);fit=Ridge(alpha=alpha).fit(A,y);ix=np.flatnonzero(~d['support']);pred=fit.predict(d['X'][ix][:,available])
 for i,p in zip(ix,pred):rows.append({'cohort':acc,'participant':str(d['ids'][i]),'age':d['y'][i],'prediction':p,'absolute_error':abs(p-d['y'][i]),'alpha':alpha,'observed_probes':available.sum()})
pd.DataFrame(rows).to_csv(R/'results/assay_matched_source_predictions.csv',index=False)
print(pd.DataFrame(rows).groupby('cohort').absolute_error.mean())
