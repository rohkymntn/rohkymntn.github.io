"""Data-derived multi-panel research figures, with editable vector exports."""
from pathlib import Path
import json,csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
R=Path(__file__).resolve().parents[1];P=R.parents[1];F=R/'figures'
BLUE='#0F4D92';TEAL='#42949E';GRAY='#888888';ROSE='#B96475';GREEN='#7BBA79'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.titlesize':14,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'legend.fontsize':11,'pdf.fonttype':42,'svg.fonttype':'none'})
def panel(a,letter,title):a.set_title(f'{letter}  {title}',loc='left',fontweight='bold',pad=12)
def save(f,name):
 for ext in ['png','svg','pdf']:f.savefig(F/f'{name}.{ext}',dpi=350,bbox_inches='tight')
 plt.close(f)
curves=pd.read_csv(R/'results/learning_curves.csv');tests=pd.read_csv(R/'results/primary_and_secondary_tests.csv');draws=pd.read_csv(R/'results/support_draw_metrics.csv')
cohorts=['GSE40279','GSE37008'];names=['Whole blood · GSE40279','PBMC · GSE37008']
model_labels={'source':'Frozen source ridge','median':'Support median','affine':'Affine recalibration','target_ridge':'Target-only ridge','residual_linear':'Linear residual kernel','residual_rbf':'RBF residual kernel','selected_baseline':'CV-selected baseline','meta_no_augmentation':'Meta-adapter, no augmentation','meta_augmented':'Meta-adapter, augmented'}
# Figure 1: cohort structure and architecture.
f=plt.figure(figsize=(14,10),layout='constrained');gs=f.add_gridspec(2,2,height_ratios=[1,1.1]);a=f.add_subplot(gs[0,0]);panel(a,'a','Age distributions and sample roles')
s=pd.read_csv(R/'results/source_partitions.csv');sets=[('Source training',s[s.partition=='source_train'].age,GRAY),('Source validation',s[s.partition=='source_validation'].age,'#B6B6B6')]
for c,col in zip(cohorts,[BLUE,TEAL]):
 d=np.load(R/f'data/{c}.npz');sets.append((c+' query',d['y'][~d['support']],col))
for i,(name,y,col) in enumerate(sets):
 v=a.violinplot(y,positions=[i],orientation='horizontal',showextrema=False,widths=.7);v['bodies'][0].set(facecolor=col,edgecolor=col,alpha=.3);q=np.quantile(y,[.25,.5,.75]);a.plot(q[[0,2]],[i,i],color=col,lw=3);a.plot(q[1],i,'o',color=col,ms=5)
a.set(yticks=range(4),yticklabels=[f'{n} (n={len(y)})' for n,y,c in sets],xlabel='Age (years)',ylim=(3.6,-.6))
a=f.add_subplot(gs[0,1]);panel(a,'b','External support and query allocation')
for i,c in enumerate(cohorts):
 d=np.load(R/f'data/{c}.npz');ns=d['support'].sum();nq=(~d['support']).sum();a.barh(i,ns,color='#AADCA9',label='Support pool' if i==0 else None);a.barh(i,nq,left=ns,color=BLUE if i==0 else TEAL,label='Locked query' if i==0 else None);a.text(ns/2,i,str(ns),ha='center',va='center',fontsize=12);a.text(ns+nq/2,i,str(nq),ha='center',va='center',color='white',fontsize=12)
a.set(yticks=[0,1],yticklabels=cohorts,xlabel='Participants',ylim=(1.6,-.6));a.legend(loc='lower right');a.text(.03,.06,'Label budgets: 8, 16, 32\n20 support draws per budget',transform=a.transAxes,fontsize=12)
a=f.add_subplot(gs[1,:]);panel(a,'c','Episodic deep-kernel adaptation');a.set(xlim=(0,14),ylim=(0,5));a.axis('off')
def box(x,y,w,title,sub,col):
 a.add_patch(FancyBboxPatch((x,y),w,.88,boxstyle='round,pad=.025',facecolor=col,edgecolor='#CFCECE'))
 a.text(x+w/2,y+.60,title,ha='center',va='center',weight='bold',fontsize=12);a.text(x+w/2,y+.25,sub,ha='center',va='center',fontsize=12,color='#666666')
def ar(x,y,u,v):a.add_patch(FancyArrowPatch((x,y),(u,v),arrowstyle='-|>',mutation_scale=14,lw=1.7,color=BLUE))
box(.2,3.35,2.6,'Source methylation','Source-only CpG selection','#EAF0F8');box(3.5,3.35,3.0,'Neural encoder','1,024 → 128 → 32','#EAF0F8');box(7.3,3.35,2.7,'Support regression','Differentiable ridge solve','#DDF3DE');box(10.9,3.35,2.8,'Query predictions','Source age + residual','#EAF0F8')
ar(2.85,3.79,3.45,3.79);ar(6.55,3.79,7.25,3.79);ar(10.05,3.79,10.85,3.79)
box(.2,1.5,2.6,'Source clock','Group-out-of-fold residuals','#F4F4F4');box(3.5,1.5,3.0,'Domain episodes','Support / query separation','#EDF6F7');box(7.3,1.5,2.7,'Meta-training loss','Backpropagate through solve','#EDF6F7');box(10.9,1.5,2.8,'External evaluation','Frozen encoder; support fit','#F4F4F4')
ar(2.85,1.94,3.45,1.94);ar(5,2.41,5,3.30);ar(8.65,3.3,8.65,2.42);ar(7.25,1.94,6.55,1.94);ar(12.3,3.3,12.3,2.42)
a.text(7,.5,'Checkpoint selection: source validation only  ·  External query labels excluded from all model and baseline selection',ha='center',fontsize=11,color='#666666')
save(f,'figure1_cohorts_architecture')
# Figure 2: complete performance comparison.
f,ax=plt.subplots(3,2,figsize=(14,14),layout='constrained')
for j,(c,name) in enumerate(zip(cohorts,names)):
 a=ax[0,j];panel(a,'ab'[j],name+' learning curve')
 for model,col,style in [('meta_augmented',BLUE,'-'),('selected_baseline',TEAL,'-'),('source',GRAY,'--'),('affine',ROSE,':')]:
  z=curves[(curves.cohort==c)&(curves.model==model)].sort_values('shots');a.plot(z.shots,z.mae,style+'o',color=col,label=model_labels[model],lw=1.8,ms=5);a.fill_between(z.shots,z.low,z.high,color=col,alpha=.10)
 a.set(xlabel='Labelled support participants',ylabel='MAE (years)',xticks=[8,16,32]);a.legend(fontsize=10.5)
 a=ax[1,j];panel(a,'cd'[j],'All 16-shot comparators')
 z=curves[(curves.cohort==c)&(curves.shots==16)].set_index('model').loc[list(model_labels)]
 for i,(model,r) in enumerate(z.iterrows()):
  col=BLUE if model=='meta_augmented' else TEAL if model=='selected_baseline' else GRAY;a.errorbar(r.mae,i,xerr=[[r.mae-r.low],[r.high-r.mae]],fmt='o',color=col,capsize=3,ms=5)
 a.set(yticks=range(len(z)),yticklabels=[model_labels[m] for m in z.index],xlabel='MAE (years)',ylim=(len(z)-.4,-.6))
 a=ax[2,j];panel(a,'ef'[j],'16-shot neural prediction calibration')
 x=pd.read_csv(R/f'results/{c}_predictions.csv');x=x[(x.shots==16)&(x.model=='meta_augmented')].groupby('participant')[['age','prediction']].mean();a.scatter(x.age,x.prediction,s=13,color=BLUE,alpha=.45,edgecolors='none');lo=min(x.min())-3;hi=max(x.max())+3;a.plot([lo,hi],[lo,hi],ls='--',color=GRAY,lw=1);a.set(xlabel='Chronological age (years)',ylabel='Mean predicted age (years)',xlim=(lo,hi),ylim=(lo,hi))
save(f,'figure2_external_performance')
# Figure 3: hypothesis tests, ablations and failure analysis.
f,ax=plt.subplots(3,2,figsize=(14,13),layout='constrained');a=ax[0,0];panel(a,'a','Prespecified 16-shot contrasts')
for i,c in enumerate(cohorts):
 r=tests[(tests.cohort==c)&tests.primary].iloc[0];a.errorbar(r.mae_reduction,i,xerr=[[r.mae_reduction-r.low],[r.high-r.mae_reduction]],fmt='o',color=BLUE,capsize=4);a.text(.97,.85-i*.18,f'{c}: Holm P = {r.holm_p:.4f}',transform=a.transAxes,ha='right',fontsize=11)
a.axvline(0,ls='--',color=GRAY,lw=1);a.set(yticks=[0,1],yticklabels=cohorts,ylim=(1.7,-.6),xlabel='Selected-baseline MAE − neural MAE (years)')
a=ax[0,1];panel(a,'b','Support-set sensitivity')
for i,c in enumerate(cohorts):
 d=draws[(draws.cohort==c)&(draws.shots==16)].pivot(index='repeat',columns='model',values='absolute_error');v=d.selected_baseline-d.meta_augmented;a.scatter(np.repeat(i,len(v))+np.linspace(-.12,.12,len(v)),v,color=BLUE,alpha=.6,s=20);a.plot([i-.18,i+.18],[v.mean()]*2,color=TEAL,lw=3)
a.axhline(0,color=GRAY,ls='--',lw=1);a.set(xticks=[0,1],xticklabels=cohorts,ylabel='Per-draw MAE reduction (years)')
a=ax[1,0];panel(a,'c','Source-only checkpoint selection')
for aug,col in [(0,TEAL),(1,BLUE)]:
 for seed in [41,42,43]:
  t=pd.DataFrame(json.loads((R/f'results/meta_{seed}_{aug}.json').read_text())['trace']);a.plot(t.step,t.validation_mae,color=col,alpha=.65,lw=1,label=('No augmentation' if aug==0 else 'Augmented') if seed==41 else None)
a.set(xlabel='Training episodes',ylabel='Source-validation MAE (years)');a.legend()
a=ax[1,1];panel(a,'d','Probe availability in external assays')
for i,c in enumerate(cohorts):
 d=json.loads((R/f'results/{c}_audit.json').read_text());n=d['observed_selected_features'];a.barh(i,n,color=BLUE);a.barh(i,1024-n,left=n,color='#CFCECE');a.text(n/2,i,str(n)+' observed',ha='center',va='center',color='white',fontsize=12);a.text(n+(1024-n)/2,i,str(1024-n),ha='center',va='center',fontsize=12)
a.set(yticks=[0,1],yticklabels=cohorts,xlabel='Source-selected CpGs (total 1,024)',ylim=(1.5,-.5))
a=ax[2,0];panel(a,'e','Augmentation ablation')
for i,c in enumerate(cohorts):
 r=tests[(tests.cohort==c)&(tests.baseline=='meta_no_augmentation')].iloc[0];a.errorbar(r.mae_reduction,i,xerr=[[r.mae_reduction-r.low],[r.high-r.mae_reduction]],fmt='o',color=TEAL,capsize=4)
a.axvline(0,ls='--',color=GRAY,lw=1);a.set(yticks=[0,1],yticklabels=cohorts,xlabel='No-augmentation MAE − augmented MAE (years)',ylim=(1.5,-.5))
a=ax[2,1];panel(a,'f','Source-refitting diagnostic (post hoc)')
diag=pd.read_csv(R/'results/assay_matched_source_predictions.csv')
for i,c in enumerate(cohorts):
 for off,model,col in [(-.2,'source',GRAY),(0,'meta_augmented',BLUE)]:
  v=curves[(curves.cohort==c)&(curves.shots==16)&(curves.model==model)].iloc[0];a.bar(i+off,v.mae,width=.18,color=col,label=model_labels[model] if i==0 else None)
 v=diag[diag.cohort==c].absolute_error.mean();a.bar(i+.2,v,width=.18,color=GREEN,label='Assay-matched source ridge' if i==0 else None)
a.set(xticks=[0,1],xticklabels=cohorts,ylabel='Query MAE (years)');a.legend(fontsize=10.5)
save(f,'figure3_robustness_ablation')
print('Generated three figures with 15 panels, each in PNG, SVG and PDF.')
