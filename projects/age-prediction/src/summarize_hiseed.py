from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import journal_style
R=Path(__file__).resolve().parents[1]
x=pd.concat([pd.read_csv(R/f'results/hiseed_predictions_{s}.csv') for s in [20260911,20260912,20260913]])
assert len(x)==369 and x.groupby('seed').Sample_ID.nunique().eq(123).all()
assert x[['clock','microbiome','combined','median']].notna().all().all()
for k in ['clock','microbiome','combined','median']:x[k+'_error']=abs(x[k]-x.Age)
e=x.groupby('Sample_ID')[[k+'_error' for k in ['clock','microbiome','combined','median']]].mean()
rng=np.random.default_rng(20260914);ix=rng.integers(0,len(e),size=(10000,len(e)))
def summary(v):
 b=np.asarray(v)[ix].mean(1)
 return {'mean':float(np.mean(v)),'low':float(np.quantile(b,.025)),'high':float(np.quantile(b,.975))}
s={k:summary(e[k+'_error']) for k in ['clock','microbiome','combined','median']}
s['clock_minus_combined']=summary(e.clock_error-e.combined_error);s['microbiome_minus_combined']=summary(e.microbiome_error-e.combined_error)
s['n_participants']=123;s['interpretation']='Exploratory paired Horvath-clock-score and gut-genus comparison; conditional bootstrap intervals; no external validation or raw methylome/skin/proteomic fusion.'
(R/'results/hiseed_summary.json').write_text(json.dumps(s,indent=2));x.to_csv(R/'results/hiseed_all_predictions.csv',index=False);e.to_csv(R/'results/hiseed_participant_errors.csv');x.groupby('seed')[[k+'_error' for k in ['clock','microbiome','combined','median']]].mean().to_csv(R/'results/hiseed_by_repeat.csv')
plt.rcParams.update({'font.family':'Helvetica','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
f,a=plt.subplots(1,2,figsize=(11,3.0),layout='constrained')
for j,(k,l,c) in enumerate([('clock','Calibrated Horvath','#0F4D92'),('microbiome','Gut microbiome','#42949E'),('combined','Combined','#8BCF8B')]):
 z=s[k];a[0].errorbar(z['mean'],j,xerr=[[z['mean']-z['low']],[z['high']-z['mean']]],fmt='o',color=c,capsize=4,markersize=9)
a[0].set(yticks=[0,1,2],yticklabels=['Calibrated Horvath','Gut microbiome','Combined'],xlabel='Mean absolute error (years)',ylim=(2.5,-.5));a[0].set_title('a  Paired prediction comparison',loc='left',fontweight='bold')
z=s['clock_minus_combined'];a[1].errorbar(z['mean'],0,xerr=[[z['mean']-z['low']],[z['high']-z['mean']]],fmt='o',color='#0F4D92',capsize=5,markersize=9);a[1].axvline(0,color='#888888',ls='--',lw=1);a[1].set(yticks=[],xlabel='MAE reduction versus calibrated Horvath (years)',ylim=(-1,1));a[1].set_title('b  Incremental microbial information',loc='left',fontweight='bold');a[1].text(.5,.88,f"{z['mean']:.2f} years (95% CI {z['low']:.2f} to {z['high']:.2f})",transform=a[1].transAxes,ha='center',fontsize=10)
f.suptitle('HI-SEED / 123 paired participants / repeated nested validation',fontsize=12)
for ext in ['png','svg','pdf']:f.savefig(R/f'figures/hiseed_paired_pilot.{ext}',dpi=300,bbox_inches='tight')
print(json.dumps(s,indent=2))
