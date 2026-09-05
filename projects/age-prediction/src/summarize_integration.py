"""Summarize every prespecified repeat and comparison; conditional bootstrap intervals."""
from pathlib import Path
import json
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results'
SEEDS=[20260906,20260907,20260908]
frames=[]
for seed in SEEDS:
 q=pd.read_csv(R/f'integration_predictions_{seed}.csv',dtype={'group':str,'study':str})
 assert len(q)==1798 and q.group.nunique()==339
 assert q.groupby('group').fold.nunique().max()==1
 assert not q.sample_id.duplicated().any()
 selection=json.loads((R/f'integration_selection_{seed}.json').read_text())
 assert all(x['participant_overlap']==0 for x in selection['selection'])
 for name in ['metadata','microbiome','combined']:
  assert np.isfinite(q[name+'_prediction']).all()
  q[name+'_error']=np.abs(q[name+'_prediction']-q.age)
 frames.append(q)
q=pd.concat(frames,ignore_index=True)
assert q.groupby('sample_id').age.nunique().max()==1
q.to_csv(R/'integration_all_predictions.csv',index=False)
cols=['metadata_error','microbiome_error','combined_error']
p=q.groupby(['group','study'])[cols].mean().reset_index()
p['metadata_minus_combined']=p.metadata_error-p.combined_error
p['microbiome_minus_combined']=p.microbiome_error-p.combined_error
p.to_csv(R/'integration_participant_errors.csv',index=False)
rng=np.random.default_rng(20260909)
indices=np.concatenate([rng.choice(z.index.to_numpy(),size=(10000,len(z)),replace=True) for _,z in p.groupby('study')],axis=1)
summary={}
for col in cols+['metadata_minus_combined','microbiome_minus_combined']:
 v=p[col].to_numpy();boot=v[indices].mean(axis=1)
 summary[col]={'mean':float(v.mean()),'low':float(np.quantile(boot,.025)),'high':float(np.quantile(boot,.975))}
summary['relative_mae_reduction_vs_metadata']=float(p.metadata_minus_combined.mean()/p.metadata_error.mean())
summary['interpretation_scope']='Exploratory follow-up on previously analyzed cohorts. Bootstrap conditions on fitted predictions and resamples participants within study. No independently paired blood/skin evaluation.'
summary['n_participants']=len(p);summary['repeats']=3
(R/'integration_summary.json').write_text(json.dumps(summary,indent=2))
repeat=q.groupby(['repeat_seed','group'])[cols].mean().groupby('repeat_seed').mean();repeat['metadata_minus_combined']=repeat.metadata_error-repeat.combined_error;repeat['microbiome_minus_combined']=repeat.microbiome_error-repeat.combined_error
repeat.to_csv(R/'integration_by_repeat.csv')
rows=[]
for study,z in p.groupby('study'):
 for col in ['metadata_minus_combined','microbiome_minus_combined']:
  v=z[col].to_numpy();boot=v[rng.integers(0,len(v),size=(10000,len(v)))].mean(axis=1)
  rows.append({'study':study,'comparison':col,'n_participants':len(v),'mean':v.mean(),'low':np.quantile(boot,.025),'high':np.quantile(boot,.975)})
pd.DataFrame(rows).to_csv(R/'integration_by_study.csv',index=False)
print(json.dumps(summary,indent=2));print(repeat.to_string())
