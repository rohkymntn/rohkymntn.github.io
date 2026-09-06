"""Iteration 1 design proof. figures4papers palette + Nature final-size geometry.
No fits or inference changed. Source: integration summary/repeat/study outputs.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import journal_style
from matplotlib.ticker import MultipleLocator
R=Path(__file__).resolve().parents[1];O=R/'figures/design-review'
s=json.loads((R/'results/integration_summary.json').read_text());rep=pd.read_csv(R/'results/integration_by_repeat.csv');study=pd.read_csv(R/'results/integration_by_study.csv')
B='#0F4D92';T='#42949E';G='#A5A5A5';INK='#272727'
plt.rcParams.update({'font.family':'Helvetica','font.size':7,'axes.labelsize':7,'axes.titlesize':7,'axes.linewidth':.6,'axes.spines.top':False,'axes.spines.right':False,'xtick.major.width':.6,'ytick.major.width':.6,'xtick.major.size':2.5,'ytick.major.size':2.5,'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42,'text.color':INK,'axes.labelcolor':INK})
f=plt.figure(figsize=(183/25.4,105/25.4))
# Fixed physical layout: aligned axes, no oversized headings or unused legend column.
axes=[f.add_axes([.15,.57,.285,.32]),f.add_axes([.63,.57,.32,.32]),f.add_axes([.15,.13,.285,.28]),f.add_axes([.63,.13,.32,.28])]
for ax,l,title in zip(axes,'abcd',['Participant-held-out prediction','Consistency across repeats','Incremental microbial information','Source-study heterogeneity']):
 p=ax.get_position();f.text(p.x0-.055,p.y1+.045,l,fontweight='bold',fontsize=8);f.text(p.x0,p.y1+.045,title,fontsize=7)
a=axes[0]
for i,(key,col) in enumerate([('metadata_error',G),('microbiome_error',T),('combined_error',B)]):
 z=s[key];a.bar(i,z['mean'],width=.57,color=col,edgecolor=INK,linewidth=.45);a.errorbar(i,z['mean'],yerr=[[z['mean']-z['low']],[z['high']-z['mean']]],color=INK,capsize=2,lw=.6);a.text(i,z['high']+.24,f"{z['mean']:.2f}",ha='center',fontsize=7)
a.set(xticks=range(3),xticklabels=['Metadata','Microbiome','Combined'],ylim=(0,12),ylabel='MAE (years)',yticks=[0,4,8,12]);a.spines['bottom'].set_bounds(-.4,2.4)
a=axes[1]
for key,col,mark,name in [('metadata_error',G,'s','Metadata'),('microbiome_error',T,'^','Microbiome'),('combined_error',B,'o','Combined')]:
 vals=rep[key].to_numpy();a.plot(range(1,4),vals,color=col,marker=mark,ms=3.5,lw=.8);a.annotate(name,(3,vals[-1]),xytext=(5,0),textcoords='offset points',va='center',fontsize=6)
a.set(xlim=(.8,4.25),xticks=[1,2,3],ylim=(8.45,10),yticks=[8.5,9,9.5,10],xlabel='Validation repeat',ylabel='MAE (years)');a.spines['bottom'].set_bounds(1,3)
a=axes[2]
for i,(key) in enumerate(['metadata_minus_combined','microbiome_minus_combined']):
 z=s[key];a.errorbar(z['mean'],i,xerr=[[z['mean']-z['low']],[z['high']-z['mean']]],fmt='o',color=B,ms=4,capsize=2,lw=.8);a.text(z['mean'],i-.24,f"{z['mean']:.2f} [{z['low']:.2f}, {z['high']:.2f}]",ha='center',fontsize=6)
a.axvline(0,color=INK,ls=(0,(2,2)),lw=.6);a.set(yticks=[0,1],yticklabels=['vs metadata','vs microbiome'],ylim=(1.5,-.65),xlim=(-.08,1.48),xticks=[0,.5,1],xlabel='Combined-model MAE reduction (years)');a.spines['left'].set_visible(False);a.tick_params(axis='y',length=0)
a=axes[3]
rows=study[study.comparison=='metadata_minus_combined'].sort_values('study')
for i,r in enumerate(rows.itertuples()):
 a.errorbar(r.mean,i,xerr=[[r.mean-r.low],[r.high-r.mean]],fmt='o',color=B,ms=4,capsize=2,lw=.8)
a.axvline(0,color=INK,ls=(0,(2,2)),lw=.6);a.set(yticks=range(3),yticklabels=[f'{int(r.study)} (n = {int(r.n_participants)})' for r in rows.itertuples()],ylim=(2.5,-.5),xlim=(-1.2,2.2),xticks=[-1,0,1,2],xlabel='MAE reduction vs metadata (years)');a.spines['left'].set_visible(False);a.tick_params(axis='y',length=0)
for ext in ['png','pdf','svg']:f.savefig(O/f'iteration1_integration.{ext}',dpi=450,facecolor='white')
(O/'caption.txt').write_text('Microbiome–metadata integration improves age prediction within the sampled study mixture. a, Participant-weighted MAE for 339 participant identifiers; whiskers show conditional 95% participant-bootstrap intervals. b, All three repeated nested-validation estimates; repeats reuse participants and are not independent cohorts. c, Paired reductions relative to metadata and microbiome comparators; annotations give means and conditional 95% intervals. d, Study-specific reductions relative to metadata. Intervals use 10,000 paired participant resamples, stratified by study for pooled estimates, and condition on recorded predictions. Negative values indicate worse combined-model performance. These comparisons do not estimate blood–skin fusion accuracy.\n')
(O/'design-spec.json').write_text(json.dumps({'width_mm':183,'height_mm':105,'font_pt':7,'line_pt':.6,'palette_source':'ChenLiu-1996/figures4papers','status':'design proof; not published','inputs':['integration_summary.json','integration_by_repeat.csv','integration_by_study.csv']},indent=2))
print(O/'iteration1_integration.png')
