"""Post hoc offset-only comparator: no neural fitting or query-label tuning."""
from pathlib import Path
import json,numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];rows=[]
for c in ['GSE40279','GSE37008']:
 d=np.load(R/f'data/{c}.npz');lookup={str(k):i for i,k in enumerate(d['ids'])};q=np.flatnonzero(~d['support'])
 for info in json.loads((R/f'results/{c}_support_selection.json').read_text()):
  if info['shots']!=16:continue
  s=np.array([lookup[k] for k in info['support_ids']]);pred=d['base'][q]+np.mean(d['y'][s]-d['base'][s])
  rows.extend({'cohort':c,'repeat':info['repeat'],'participant':str(d['ids'][i]),'prediction':p,'absolute_error':abs(p-d['y'][i])} for i,p in zip(q,pred))
pd.DataFrame(rows).to_csv(R/'results/posthoc_offset_only_predictions.csv',index=False)
