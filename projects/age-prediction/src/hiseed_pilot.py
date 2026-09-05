"""Prespecified exploratory paired clock-score / gut microbiome comparison."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold,GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,LinearRegression
ROOT=Path(__file__).resolve().parents[1]
def run(seed):
 m=pd.read_csv(ROOT/'data/candidates/Kunihiro2026_Metadata.csv').set_index('Sample_ID')
 g=pd.read_csv(ROOT/'data/candidates/Kunihiro2026_GenusOtu.csv').set_index('Sample_ID').loc[m.index]
 assert m.index.is_unique and g.index.is_unique and len(m)==123
 v=g.to_numpy(float);assert np.isfinite(v).all() and (v>=0).all() and (v.sum(1)>0).all()
 h=np.sqrt(v/v.sum(1,keepdims=True));clock=m[['Horvath']].to_numpy();y=m.Age.to_numpy()
 Xs={'clock':clock,'microbiome':h,'combined':np.column_stack([clock,h])}
 out=m[['Age','Sex','HbA1c','BMI']].copy();out['seed']=seed;choices=[]
 for fold,(tr,te) in enumerate(KFold(5,shuffle=True,random_state=seed).split(y)):
  out.loc[m.index[te],'fold']=fold
  out.loc[m.index[te],'median']=np.median(y[tr])
  for name,X in Xs.items():
   if name=='clock':model=LinearRegression().fit(X[tr],y[tr]);params={}
   else:
    model=GridSearchCV(make_pipeline(StandardScaler(),Ridge()),{'ridge__alpha':[.1,1,10,100,1000]},cv=KFold(3,shuffle=True,random_state=seed+fold+1),scoring='neg_mean_absolute_error',n_jobs=1).fit(X[tr],y[tr]);params=model.best_params_
   out.loc[m.index[te],name]=model.predict(X[te]);choices.append({'fold':fold,'model':name,'params':params})
 return out.reset_index().to_csv(index=False),choices
