"""Episodic deep-kernel residual adaptation using differentiable ridge solves.
Builds on differentiable-solver meta-learning; novelty is not asserted.
"""
import math
import numpy as np
import torch
from torch import nn
class MetaAdapter(nn.Module):
 def __init__(self,p=1024):
  super().__init__();self.encoder=nn.Sequential(nn.Linear(p,128),nn.LayerNorm(128),nn.GELU(),nn.Linear(128,32),nn.Tanh());self.log_lambda=nn.Parameter(torch.tensor(-1.));self.log_clock_scale=nn.Parameter(torch.tensor(0.))
 def embed(self,x,base):return torch.cat([self.encoder(x)/math.sqrt(32),base[:,None]/20*torch.exp(self.log_clock_scale).clamp(.1,10)],1)
 def forward(self,xs,ys,bs,xq,bq):
  hs=self.embed(xs,bs);hq=self.embed(xq,bq);mu=hs.mean(0);hs=hs-mu;hq=hq-mu
  residual=(ys-bs)/20;offset=residual.mean();residual=residual-offset
  lam=torch.exp(self.log_lambda).clamp(.001,100)
  w=torch.linalg.solve(hs@hs.T+lam*torch.eye(len(xs),device=xs.device),residual)
  return bq+20*(offset+hq@hs.T@w)

def train(seed,augment):
 from pathlib import Path
 import io,json,time
 torch.set_num_threads(2);torch.manual_seed(seed);rng=np.random.default_rng(seed);start=time.time()
 r=Path(__file__).resolve().parents[1];d=np.load(r/'data/source.npz');tr=d['train'];dev='cuda' if torch.cuda.is_available() else 'cpu'
 X=torch.tensor(d['X'][tr],device=dev);y=torch.tensor(d['y'][tr],dtype=torch.float32,device=dev);b=torch.tensor(d['train_oof'],dtype=torch.float32,device=dev)
 V=torch.tensor(d['X'][~tr],device=dev);vy=torch.tensor(d['y'][~tr],dtype=torch.float32,device=dev);vb=torch.tensor(d['base'][~tr],dtype=torch.float32,device=dev)
 groups=d['groups'][tr];domains=[np.flatnonzero(groups==g) for g in np.unique(groups) if (groups==g).sum()>=24]
 vrng=np.random.default_rng(8801);vepisodes=[]
 for i in range(12):
  order=vrng.permutation(len(V));vepisodes.append((order[:16],order[16:]))
 model=MetaAdapter(X.shape[1]).to(dev);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001);trace=[];best=float('inf');best_state=None
 for step in range(1,2001):
  group=domains[rng.integers(len(domains))];order=rng.permutation(group);k=min(int(rng.choice([8,16,32])),len(group)//2);ii=order[:k];jj=order[k:k+24]
  xs=X[ii];xq=X[jj]
  if augment:
   shift=torch.tensor(rng.normal(0,.15,(1,X.shape[1])),dtype=torch.float32,device=dev);scale=torch.tensor(rng.lognormal(0,.1,(1,X.shape[1])),dtype=torch.float32,device=dev);mask=torch.tensor(rng.random((1,X.shape[1]))>.1,device=dev)
   xs=(xs*scale+shift)*mask;xq=(xq*scale+shift)*mask
  pred=model(xs,y[ii],b[ii],xq,b[jj]);loss=torch.nn.functional.smooth_l1_loss(pred/20,y[jj]/20,beta=.2)
  opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
  if step%100==0:
   model.eval()
   with torch.no_grad():score=float(np.mean([torch.mean(abs(model(V[i],vy[i],vb[i],V[j],vb[j])-vy[j])).item() for i,j in vepisodes]))
   trace.append({'step':step,'validation_mae':score,'train_episode_loss':float(loss),'lambda':float(model.log_lambda.exp())})
   if score<best:best=score;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
   model.train()
 out=io.BytesIO();torch.save(best_state,out)
 return out.getvalue(),{'seed':seed,'augment':augment,'best_validation_mae':best,'trace':trace,'seconds':time.time()-start,'device':dev}
