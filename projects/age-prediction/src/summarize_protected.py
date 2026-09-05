from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[1]
assert hashlib.sha256((R/'src/protected_fusion.py').read_bytes()).hexdigest()==json.loads((R/'results/protected_design.json').read_text())['code_sha256']
x=pd.concat([pd.read_csv(R/f'results/protected_predictions_{s}.csv') for s in [20260911,20260912,20260913]])
assert len(x)==369 and x.groupby('seed').Sample_ID.nunique().eq(123).all()
old=pd.read_csv(R/'results/hiseed_all_predictions.csv');check=x.merge(old,on=['Sample_ID','seed'],suffixes=('','_prior'))
np.testing.assert_allclose(check.ols,check.clock,rtol=1e-12,atol=1e-12);assert (check.fold==check.fold_prior).all()
for k in ['ols','calibration','protected']:x[k+'_error']=abs(x[k]-x.Age)
e=x.groupby('Sample_ID')[[k+'_error' for k in ['ols','calibration','protected']]].mean()
rng=np.random.default_rng(20260915);ix=rng.integers(0,len(e),(10000,len(e)));signs=rng.choice([-1,1],(10000,len(e)))
def ci(v):
 v=np.asarray(v);b=v[ix].mean(1);return {'mean':float(v.mean()),'low':float(np.quantile(b,.025)),'high':float(np.quantile(b,.975))}
s={k:ci(e[k+'_error']) for k in ['ols','calibration','protected']}
contrasts={'calibration_minus_protected':e.calibration_error-e.protected_error,'ols_minus_calibration':e.ols_error-e.calibration_error}
ps=[]
for k,v in contrasts.items():
 s[k]=ci(v);p=float((1+(abs((signs*np.asarray(v)).mean(1))>=abs(v.mean())).sum())/10001);s[k]['conditional_sign_flip_p']=p;ps.append(p)
order=np.argsort(ps);adj=np.maximum.accumulate(np.array(ps)[order]*np.arange(len(ps),0,-1));adj=np.minimum(adj,1)
for i,j in enumerate(order):s[list(contrasts)[j]]['holm_p']=float(adj[i])
choices=[c for seed in [20260911,20260912,20260913] for c in json.loads((R/f'results/protected_selection_{seed}.json').read_text()) if c['model']=='protected']
s['microbial_correction_selected_folds']=sum(c['parameters']['kernel']!='none' for c in choices);s['outer_folds']=15;s['n_participants']=123;s['limitations']='Exploratory adaptive reuse of the same cohort; uncertainty conditional on fitted predictions. Sign-flip tests assume exchangeability/symmetry of participant-level differences and do not account for shared training sets or prior research choices. Independent replication required.'
(R/'results/protected_summary.json').write_text(json.dumps(s,indent=2));x.to_csv(R/'results/protected_all_predictions.csv',index=False);e.to_csv(R/'results/protected_participant_errors.csv');x.groupby('seed')[[k+'_error' for k in ['ols','calibration','protected']]].mean().to_csv(R/'results/protected_by_repeat.csv')
print(json.dumps(s,indent=2))
