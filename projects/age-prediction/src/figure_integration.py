"""Plot all prespecified integration comparisons, including source-study heterogeneity."""
import json
from pathlib import Path
import numpy as np,pandas as pd
from figures import plt,panel,save,BLUE,TEAL,GRAY,ORANGE
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results'
s=json.loads((R/'integration_summary.json').read_text());by=pd.read_csv(R/'integration_by_study.csv')
fig,axes=plt.subplots(1,2,figsize=(7.2,3.3),layout='constrained',gridspec_kw={'wspace':.18})
a=axes[0];panel(a,'a','Repeated nested validation')
for i,(key,label,c) in enumerate([('metadata_error','Metadata',GRAY),('microbiome_error','Microbiome',BLUE),('combined_error','Combined',TEAL)]):
 z=s[key];a.errorbar(z['mean'],i,xerr=[[z['mean']-z['low']],[z['high']-z['mean']]],fmt='o',ms=5,capsize=3,color=c)
 a.annotate(f"{z['mean']:.2f}",(z['mean'],i),xytext=(0,10),textcoords='offset points',ha='center',fontsize=7,color=c)
a.set(yticks=[0,1,2],yticklabels=['Metadata','Microbiome','Combined'],xlabel='Participant-weighted MAE (years)',ylim=(2.5,-.5));a.xaxis.grid(True,color='#E4E6E8',lw=.5);a.set_axisbelow(True)
a=axes[1];panel(a,'b','Incremental microbial information')
z=s['metadata_minus_combined'];rows=[('All studies',z)]+[(str(int(r.study)),r) for r in by[by.comparison=='metadata_minus_combined'].itertuples()]
for i,(label,z) in enumerate(rows):
 if isinstance(z,dict):v,lo,hi=z['mean'],z['low'],z['high']
 else:v,lo,hi=z.mean,z.low,z.high
 a.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='D' if i==0 else 'o',ms=5,capsize=3,color=TEAL if i==0 else GRAY)
a.axvline(0,color=GRAY,ls='--',lw=.8);a.set(yticks=range(len(rows)),yticklabels=[r[0] for r in rows],xlabel='MAE reduction versus metadata (years)',ylim=(len(rows)-.5,-.5));a.xaxis.grid(True,color='#E4E6E8',lw=.5);a.set_axisbelow(True)
save(fig,'fig4_metadata_microbiome_integration')
