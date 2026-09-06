"""Locked-query few-shot benchmark. All baseline selection sees support labels only."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from meta_model import MetaAdapter
R=Path(__file__).resolve().parents[1]
def kernel_predict(K,y,s,q,alpha,offset=None):
 # Intercept fitted on support only, with correctly centered support/query kernels.
 A=K[np.ix_(s,s)];B=K[np.ix_(q,s)];col=A.mean(0);grand=A.mean();Ac=A-A.mean(1)[:,None]-col+grand;Bc=B-B.mean(1)[:,None]-col+grand
 t=y[s] if offset is None else y[s]-offset[s]
 pred=t.mean()+Bc@np.linalg.solve(Ac+alpha*np.eye(len(s)),t-t.mean())
 return pred if offset is None else pred+offset[q]
def configs():
 out=[('source',0),('median',0)]
 out += [('affine',a) for a in [.01,.1,1,10]]
 for kind in ['target_ridge','residual_linear','residual_rbf']:out += [(kind,a) for a in [.01,.1,1,10]]
 return out

def evaluate(acc):
 torch.set_num_threads(2)
 d=np.load(R/f'data/{acc}.npz');X=d['X'].astype(float);y=d['y'];base=d['base'];pool=np.flatnonzero(d['support']);query=np.flatnonzero(~d['support']);ids=d['ids']
 K=X@X.T/X.shape[1];diag=np.diag(K);rbf=np.exp(-np.maximum(diag[:,None]+diag[None,:]-2*K,0));clock=np.outer(base/20,base/20)
 def predict(c,s,q):
  kind,alpha=c
  if kind=='source':return base[q]
  if kind=='median':return np.repeat(np.median(y[s]),len(q))
  if kind=='affine':return kernel_predict(clock,y,s,q,alpha)
  return kernel_predict(rbf if kind=='residual_rbf' else K,y,s,q,alpha,None if kind=='target_ridge' else base)
 embeddings={};model_checks=[]
 for aug in [0,1]:
  embeddings[aug]=[]
  for seed in [41,42,43]:
   model=MetaAdapter(X.shape[1]);model.load_state_dict(torch.load(R/f'results/meta_{seed}_{aug}.pt',map_location='cpu',weights_only=True));model.eval()
   with torch.no_grad():h=model.embed(torch.tensor(X,dtype=torch.float32),torch.tensor(base,dtype=torch.float32)).numpy().astype(float)
   lam=float(model.log_lambda.exp().clamp(.001,100).detach());kh=h@h.T;embeddings[aug].append((kh,lam))
   # Numerical agreement between cached kernel inference and the trained module.
   si=pool[:8];qi=query[:5]
   with torch.no_grad():expected=model(torch.tensor(X[si],dtype=torch.float32),torch.tensor(y[si],dtype=torch.float32),torch.tensor(base[si],dtype=torch.float32),torch.tensor(X[qi],dtype=torch.float32),torch.tensor(base[qi],dtype=torch.float32)).numpy()
   actual=kernel_predict(kh,y,si,qi,lam,base);np.testing.assert_allclose(actual,expected,atol=.002,rtol=1e-4)
   model_checks.append({'seed':seed,'augmentation':aug,'max_cached_inference_difference':float(max(abs(actual-expected)))})
 rows=[];selections=[];cfg=configs()
 for k in [8,16,32]:
  for repeat in range(20):
   rng=np.random.default_rng(19001+repeat);s=rng.permutation(pool)[:k];assert not np.intersect1d(s,query).size
   splits=list(KFold(4,shuffle=True,random_state=802+repeat).split(s));scores=[]
   for c in cfg:
    losses=[]
    for tr,va in splits:losses.extend(abs(predict(c,s[tr],s[va])-y[s[va]]))
    scores.append(float(np.mean(losses)))
   best=int(np.argmin(scores));chosen=cfg[best]
   pred={'selected_baseline':predict(chosen,s,query),'source':base[query],'median':predict(('median',0),s,query)}
   for kind in ['affine','target_ridge','residual_linear','residual_rbf']:
    indexes=[i for i,c in enumerate(cfg) if c[0]==kind];j=min(indexes,key=lambda j:scores[j]);pred[kind]=predict(cfg[j],s,query)
   for aug,name in [(0,'meta_no_augmentation'),(1,'meta_augmented')]:pred[name]=np.mean([kernel_predict(Km,y,s,query,lam,base) for Km,lam in embeddings[aug]],axis=0)
   selections.append({'cohort':acc,'shots':k,'repeat':repeat,'selected_baseline':chosen,'support_ids':ids[s].tolist(),'support_cv_mae':scores[best]})
   for name,yp in pred.items():
    rows.extend({'cohort':acc,'shots':k,'repeat':repeat,'participant':str(ids[i]),'age':float(y[i]),'model':name,'prediction':float(p),'absolute_error':float(abs(y[i]-p))} for i,p in zip(query,yp))
 (R/f'results/{acc}_model_checks.json').write_text(json.dumps(model_checks,indent=2))
 return pd.DataFrame(rows).to_csv(index=False),selections,model_checks
