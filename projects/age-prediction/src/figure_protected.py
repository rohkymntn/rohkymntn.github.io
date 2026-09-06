"""Compare all completed paired-cohort models without concealing baselines."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import journal_style
R=Path(__file__).resolve().parents[1]
a=pd.read_csv(R/'results/protected_all_predictions.csv');b=pd.read_csv(R/'results/hiseed_all_predictions.csv')
x=a.merge(b[['Sample_ID','seed','combined']],on=['Sample_ID','seed'],validate='one_to_one');x['early_error']=abs(x.combined-x.Age)
e=x.groupby('Sample_ID')[['early_error','ols_error','calibration_error','protected_error']].mean();rng=np.random.default_rng(20260916);ix=rng.integers(0,len(e),(10000,len(e)))
v=(e.early_error-e.protected_error).to_numpy();boot=v[ix].mean(1)
z={'mean':float(v.mean()),'low':float(np.quantile(boot,.025)),'high':float(np.quantile(boot,.975)),'relative_mae_reduction':float(v.mean()/e.early_error.mean()),'status':'Post hoc engineering comparison against the earlier fusion implementation; not the prespecified microbial-complementarity endpoint.'}
(R/'results/protected_vs_early_fusion.json').write_text(json.dumps(z,indent=2))
plt.rcParams.update({'font.family':'Helvetica','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
f,ax=plt.subplots(1,2,figsize=(12,3.5),layout='constrained')
for j,(k,c) in enumerate([('early_error','#999999'),('ols_error','#3775BA'),('calibration_error','#42949E'),('protected_error','#0F4D92')]):
 v=e[k].to_numpy();bs=v[ix].mean(1);lo,hi=np.quantile(bs,[.025,.975]);ax[0].errorbar(v.mean(),j,xerr=[[v.mean()-lo],[hi-v.mean()]],color=c,fmt='o',capsize=4,markersize=8);ax[0].text(13.0,j,f'{v.mean():.2f}',ha='right',va='center')
ax[0].set(yticks=range(4),yticklabels=['Early feature fusion','OLS clock calibration','Selected clock calibration','Protected residual fusion'],ylim=(3.5,-.5),xlim=(0,13.5),xlabel='Mean absolute error (years)');ax[0].set_title('a  Prediction accuracy',loc='left',fontweight='bold')
s=json.loads((R/'results/protected_summary.json').read_text());q=s['calibration_minus_protected']
for j,(v,lo,hi,c) in enumerate([(z['mean'],z['low'],z['high'],'#0F4D92'),(q['mean'],q['low'],q['high'],'#42949E')]):ax[1].errorbar(v,j,xerr=[[v-lo],[hi-v]],fmt='o',color=c,capsize=4,markersize=8)
ax[1].axvline(0,color='#BBBBBB',ls='--',lw=1);ax[1].set(yticks=[0,1],yticklabels=['Versus early fusion','Versus selected clock'],ylim=(1.6,-.6),xlabel='MAE reduction with protected fusion (years)');ax[1].set_title('b  Paired improvement',loc='left',fontweight='bold')
f.suptitle('HI-SEED / 123 participants / three repeated nested validations',fontsize=12)
for ext in ['png','svg','pdf']:f.savefig(R/f'figures/protected_fusion_evaluation.{ext}',dpi=300,bbox_inches='tight')
print(json.dumps(z,indent=2))
