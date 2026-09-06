from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV,Ridge
from sklearn.model_selection import GroupKFold
R=Path(__file__).resolve().parents[1];P=R.parents[1]
# GSE41037 plate 7 is a source-only validation domain; keep external cohorts untouched.
if (P/'data/blood.npz').exists():
 x=np.load(P/'data/blood.npz');m=pd.read_csv(P/'data/blood_metadata.csv');z=np.load(P/'data/external.npz');n=pd.read_csv(P/'data/external_metadata.csv')
else:
 import gzip,csv
 def raw_source(acc):
  meta={}
  with gzip.open(R/f'data/{acc}_series_matrix.txt.gz','rt') as f:
   for row in csv.reader(f,delimiter='\t'):
    if not row:continue
    if row[0]=='!Sample_geo_accession':ids=row[1:]
    if row[0].startswith('!Sample_characteristics'):
     key=row[1].split(':',1)[0];meta[key]=[v.split(':',1)[1].strip() if ':' in v else v for v in row[1:]]
    if row[0]=='!series_matrix_table_begin':break
   frame=pd.read_csv(f,sep='\t',index_col=0,comment='!',na_values=['null','NA']).T.astype('float32')
  return frame,pd.DataFrame(meta,index=ids).loc[frame.index]
 a,m=raw_source('GSE41037');b,n=raw_source('GSE19711');m['age']=pd.to_numeric(m.age);n['age']=pd.to_numeric(n.ageatrecruitment,errors='coerce')
 keep=(m.diseasestatus=='1; control')&(m.used_in_analysis=='yes')&(m.age>=18);a=a.loc[keep];m=m.loc[keep].copy();m['group']=m.plate;m['sample_id']=m.index
 keep=n['sample type'].str.endswith('from Control')&(n.age>=18);b=b.loc[keep];n=n.loc[keep].copy();n['sample_id']=n.index
 common=a.columns.intersection(b.columns,sort=False);x={'X':a[common].to_numpy(),'features':common.to_numpy(str)};z={'X':b[common].to_numpy(),'features':common.to_numpy(str)}

assert np.array_equal(x['features'],z['features'])
X=np.vstack([x['X'],z['X']]);y=np.r_[m.age,n.age].astype(float)
groups=np.r_[('discovery:'+m.group.astype(str)).values,np.repeat('UK_control',len(n))]
ids=np.r_[m.sample_id,n.sample_id].astype(str)
# Choose a validation plate by sorted name, independent of model outcomes.
validation_group=sorted(set(groups)-{'UK_control'})[-1];valid=groups==validation_group
train=~valid
missing=np.mean(~np.isfinite(X[train]),0);eligible=missing<=.01
imp=SimpleImputer(strategy='median',keep_empty_features=True).fit(X[train]);filled=imp.transform(X)
scores=f_regression(filled[train],y[train])[0];scores[~eligible]=-np.inf;idx=np.argsort(scores)[-1024:]
scale=StandardScaler().fit(filled[train][:,idx]);A=scale.transform(filled[:,idx]).astype('float32');A=np.clip(A,-8,8)
cv=list(GroupKFold(5).split(A[train],y[train],groups[train]))
alpha_scores={}
for alpha in [10.,100.,1000.]:
 losses=[]
 for a,b in cv:
  xx=A[train];yy=y[train];model=Ridge(alpha=alpha).fit(xx[a],yy[a]);losses.extend(abs(model.predict(xx[b])-yy[b]))
 alpha_scores[str(alpha)]=float(np.mean(losses))
alpha=min([10.,100.,1000.],key=lambda a:alpha_scores[str(a)])
base=Ridge(alpha=alpha).fit(A[train],y[train]);basepred=base.predict(A)
oof=np.zeros(train.sum());tx=A[train];ty=y[train]
for a,b in cv:oof[b]=Ridge(alpha=alpha).fit(tx[a],ty[a]).predict(tx[b])
np.savez_compressed(R/'data/source.npz',X=A,y=y,train=train,groups=groups.astype(str),ids=ids,base=basepred,train_oof=oof,features=x['features'][idx],impute=imp.statistics_[idx],mean=scale.mean_,scale=scale.scale_,coef=base.coef_,intercept=base.intercept_,alpha=alpha)
pd.DataFrame({'sample_id':ids,'age':y,'group':groups,'partition':np.where(train,'source_train','source_validation')}).to_csv(R/'results/source_partitions.csv',index=False)
(R/'results/source_audit.json').write_text(json.dumps({'total_source':len(y),'source_training':int(train.sum()),'source_validation':int(valid.sum()),'validation_domain':validation_group,'features':1024,'alpha':alpha,'development_cv_mae':alpha_scores,'note':'Feature selection/scaling use source training only. Alpha CV occurs within these source-selected features; external evaluation remains independent, but these development CV numbers are not unbiased performance estimates.','source_features_sha256':hashlib.sha256('\n'.join(x['features'][idx]).encode()).hexdigest()},indent=2))
print((R/'results/source_audit.json').read_text())
