from pathlib import Path
import json
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[1]
x=pd.concat([pd.read_csv(R/f'results/{c}_predictions.csv') for c in ['GSE40279','GSE37008']],ignore_index=True)
assert x.prediction.notna().all() and x.groupby(['cohort','shots','repeat','model']).size().nunique()==2
# Check every query participant appears once per model/draw and never as support.
for c in x.cohort.unique():
 d=np.load(R/f'data/{c}.npz');expected=set(d['ids'][~d['support']].astype(str));assert set(x[x.cohort==c].participant)==expected
 assert not expected.intersection(set(d['ids'][d['support']].astype(str)))
q=x.groupby(['cohort','shots','participant','model']).absolute_error.mean().unstack('model')
rng=np.random.default_rng(43210);rows=[];tests=[]
for (c,k),frame in q.groupby(level=['cohort','shots']):
 ix=rng.integers(0,len(frame),size=(10000,len(frame)))
 for model in frame.columns:
  v=frame[model].to_numpy();bs=v[ix].mean(1);rows.append({'cohort':c,'shots':k,'model':model,'query_n':len(v),'mae':v.mean(),'low':np.quantile(bs,.025),'high':np.quantile(bs,.975)})
 if k==16:
  signs=rng.choice([-1,1],size=(10000,len(frame)))
  for baseline in [v for v in frame.columns if v!='meta_augmented']:
   v=(frame[baseline]-frame.meta_augmented).to_numpy();bs=v[ix].mean(1);p=(1+(abs((signs*v).mean(1))>=abs(v.mean())).sum())/10001
   tests.append({'cohort':c,'baseline':baseline,'mae_reduction':v.mean(),'low':np.quantile(bs,.025),'high':np.quantile(bs,.975),'conditional_p':p,'primary':baseline=='selected_baseline'})
t=pd.DataFrame(tests);t['holm_p']=np.nan
# Two prespecified primary hypotheses; all other contrasts are a separate exploratory family.
for primary in [True,False]:
 idx=t.index[t.primary==primary];order=t.loc[idx].conditional_p.sort_values().index;p=t.loc[order].conditional_p.to_numpy();adjusted=np.minimum(1,np.maximum.accumulate(p*np.arange(len(p),0,-1)));t.loc[order,'holm_p']=adjusted
pd.DataFrame(rows).to_csv(R/'results/learning_curves.csv',index=False);t.to_csv(R/'results/primary_and_secondary_tests.csv',index=False);q.to_csv(R/'results/query_participant_losses.csv');x.groupby(['cohort','shots','repeat','model']).absolute_error.mean().to_csv(R/'results/support_draw_metrics.csv')
s={'primary_endpoint':'16-shot conditional MAE reduction, selected classical baseline minus augmented neural meta-adapter','primary_results':t[t.primary].to_dict('records'),'replication_criterion_met':bool(((t[t.primary].mae_reduction>0)&(t[t.primary].holm_p<.05)).all()),'total_locked_query_participants':588,'limitations':'Conditional participant-level bootstrap and sign-flip tests average errors across 20 overlapping support draws. They do not capture retraining uncertainty, uncertainty from choosing the support pool, or independent-cohort population inference. The 32-shot PBMC experiments reuse the entire 32-person pool; only CV partition assignment varies. Source datasets were previously studied, but external query cohorts were not used for model design or tuning.'}
(R/'results/summary.json').write_text(json.dumps(s,indent=2));print(json.dumps(s,indent=2));print(pd.DataFrame(rows).query('shots==16')[['cohort','model','mae']].to_string(index=False))
