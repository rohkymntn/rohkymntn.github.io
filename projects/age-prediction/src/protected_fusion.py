"""Exploratory unpenalized clock calibration with orthogonal microbial residuals.
Standard semiparametric building blocks; no claim of methodological novelty.
"""
import numpy as np
from sklearn.base import BaseEstimator,RegressorMixin
from sklearn.linear_model import LinearRegression,HuberRegressor,QuantileRegressor
from sklearn.kernel_ridge import KernelRidge
class ProtectedFusion(RegressorMixin,BaseEstimator):
 def __init__(self,calibration='ols',kernel='none',alpha=1.,strength=1.):
  self.calibration=calibration;self.kernel=kernel;self.alpha=alpha;self.strength=strength
 def fit(self,X,y):
  X=np.asarray(X);y=np.asarray(y)
  self.clock_= {'ols':LinearRegression(),'huber':HuberRegressor(alpha=0,max_iter=1000),'median':QuantileRegressor(quantile=.5,alpha=0,solver='highs')}[self.calibration]
  self.clock_.fit(X[:,:1],y)
  if self.kernel!='none':
   # Remove linear clock-associated variation from microbes within training only.
   self.projection_=LinearRegression().fit(X[:,:1],X[:,1:])
   h=X[:,1:]-self.projection_.predict(X[:,:1])
   self.residual_=KernelRidge(alpha=self.alpha,kernel=self.kernel,gamma=1.).fit(h,y-self.clock_.predict(X[:,:1]))
  return self
 def predict(self,X):
  X=np.asarray(X);p=self.clock_.predict(X[:,:1])
  if self.kernel!='none':p=p+self.strength*self.residual_.predict(X[:,1:]-self.projection_.predict(X[:,:1]))
  return p

def run(seed):
 from pathlib import Path
 import pandas as pd
 from sklearn.model_selection import KFold,GridSearchCV
 root=Path(__file__).resolve().parents[1]
 m=pd.read_csv(root/'data/candidates/Kunihiro2026_Metadata.csv').set_index('Sample_ID')
 g=pd.read_csv(root/'data/candidates/Kunihiro2026_GenusOtu.csv').set_index('Sample_ID').loc[m.index].to_numpy(float)
 assert m.index.is_unique and len(m)==123 and (g>=0).all()
 h=np.sqrt(g/g.sum(axis=1,keepdims=True));X=np.column_stack([m.Horvath,h]);y=m.Age.to_numpy()
 out=m[['Age']].copy();out['seed']=seed;choices=[]
 baseline={'calibration':['ols','huber','median'],'kernel':['none']}
 correction={'calibration':['ols','huber','median'],'kernel':['linear','rbf'],'alpha':[.1,1.,10.],'strength':[.25,1.]}
 for fold,(tr,te) in enumerate(KFold(5,shuffle=True,random_state=seed).split(X)):
  inner=list(KFold(3,shuffle=True,random_state=seed+fold+1).split(tr))
  out.loc[m.index[te],'fold']=fold
  out.loc[m.index[te],'ols']=ProtectedFusion().fit(X[tr],y[tr]).predict(X[te])
  for name,grid in [('calibration',baseline),('protected',[baseline,correction])]:
   search=GridSearchCV(ProtectedFusion(),grid,cv=inner,scoring='neg_mean_absolute_error',n_jobs=1).fit(X[tr],y[tr])
   out.loc[m.index[te],name]=search.predict(X[te]);choices.append({'fold':fold,'model':name,'parameters':search.best_params_,'inner_mae':-search.best_score_})
 return out.reset_index().to_csv(index=False),choices
