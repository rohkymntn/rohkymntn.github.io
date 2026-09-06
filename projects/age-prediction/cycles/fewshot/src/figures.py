"""Journal-scale vector plates; all values read from frozen experiment outputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import journal_style
from matplotlib.patches import Rectangle, FancyArrowPatch
R=Path(__file__).resolve().parents[1]; F=R/'figures'
B='#174E78';T='#368C8A';G='#959B9F';K='#242A30';L='#E8ECEF'
plt.rcParams.update({'font.family':'Helvetica','font.size':10,'axes.labelsize':10,'axes.titlesize':11,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'xtick.major.width':.65,'ytick.major.width':.65,'legend.frameon':False,'legend.fontsize':9,'pdf.fonttype':42,'svg.fonttype':'none','text.color':K,'axes.labelcolor':K,'xtick.color':K,'ytick.color':K})
def label(a,l,t):
 a.set_title(t,loc='left',pad=14,fontweight='normal');a.annotate(l,xy=(0,1),xycoords='axes fraction',xytext=(-28,14),textcoords='offset points',fontweight='bold',fontsize=15,va='bottom')
def save(f,n):
 for ext in ['png','pdf','svg']:f.savefig(F/f'{n}.{ext}',dpi=320,bbox_inches='tight',facecolor='white')
 plt.close(f)
C=['GSE40279','GSE37008'];N=['Whole blood','PBMCs'];cols=[B,T]
curves=pd.read_csv(R/'results/learning_curves.csv');tests=pd.read_csv(R/'results/primary_and_secondary_tests.csv');draws=pd.read_csv(R/'results/support_draw_metrics.csv')
f=plt.figure(figsize=(11,7.8));gs=f.add_gridspec(2,2,height_ratios=[1,1.25],hspace=.62,wspace=.5,left=.10,right=.97,top=.93,bottom=.07)
a=f.add_subplot(gs[0,0]);label(a,'a','Age composition')
s=pd.read_csv(R/'results/source_partitions.csv');sets=[('Source training',s[s.partition=='source_train'].age,G),('Source validation',s[s.partition=='source_validation'].age,'#B9BFC3')]
for c,n,col in zip(C,N,cols):
 d=np.load(R/f'data/{c}.npz');sets.append((n+' query',d['y'][~d['support']],col))
for n,y,col in sets:
 y=np.sort(y);a.step(y,np.arange(1,len(y)+1)/len(y),where='post',color=col,lw=1.7,label=f'{n}  (n = {len(y)})')
a.set(xlabel='Chronological age (years)',ylabel='Cumulative proportion',ylim=(0,1.04),yticks=[0,.5,1]);a.legend(loc='lower right',fontsize=8)
a=f.add_subplot(gs[0,1]);label(a,'b','External evaluation partitions')
for i,(c,n,col) in enumerate(zip(C,N,cols)):
 d=np.load(R/f'data/{c}.npz');ns=int(d['support'].sum());nq=int((~d['support']).sum());p=ns/(ns+nq)
 a.barh(i,p,color=L,height=.40);a.barh(i,1-p,left=p,color=col,height=.40)
 a.text(p/2,i,str(ns),ha='center',va='center',fontsize=10);a.text((1+p)/2,i,str(nq),color='white',ha='center',va='center');a.text(0,i-.30,f'{n} / {c}',fontsize=10)
a.set(xlim=(0,1),ylim=(1.55,-.75));a.axis('off');a.text(0,-.02,'Support pool',transform=a.transAxes,color=G);a.text(.48,-.02,'Fixed query set',transform=a.transAxes,color=B)
a=f.add_subplot(gs[1,:]);label(a,'c','Episodic residual adaptation');a.set(xlim=(0,10),ylim=(0,3.7));a.axis('off')
def node(x,y,w,h,title,sub,color=L):
 a.add_patch(Rectangle((x,y),w,h,facecolor=color,edgecolor='none'));a.text(x+w/2,y+h*.65,title,ha='center',va='center',weight='bold',fontsize=10);a.text(x+w/2,y+h*.28,sub,ha='center',va='center',fontsize=9,color='#59636B')
def ar(x,y,u,v):a.add_patch(FancyArrowPatch((x,y),(u,v),arrowstyle='-|>',mutation_scale=11,lw=1,color=K,connectionstyle='arc3'))
a.text(0,3.4,'SOURCE TRAINING',fontsize=8,weight='bold',color=G)
node(0,2.15,2,.85,'CpG profile','1,024 source-selected probes');node(2.65,2.15,2.1,.85,'Shared encoder','128 → 32 dimensions','#E4EEF6');node(5.4,2.15,2.05,.85,'Support residuals','Differentiable ridge','#E1F0ED');node(8.1,2.15,1.9,.85,'Query loss','Update encoder')
for x,u in [(2,2.65),(4.75,5.4),(7.45,8.1)]:ar(x,2.575,u,2.575)
a.text(0,1.64,'EXTERNAL ADAPTATION',fontsize=8,weight='bold',color=G)
node(0,.4,2,.85,'Target samples','Source preprocessing');node(2.65,.4,2.1,.85,'Frozen encoder','No query-label access','#E4EEF6');node(5.4,.4,2.05,.85,'Support-only fit','8, 16 or 32 labelled ages','#E1F0ED');node(8.1,.4,1.9,.85,'Predicted age','Source clock + residual')
for x,u in [(2,2.65),(4.75,5.4),(7.45,8.1)]:ar(x,.825,u,.825)
ar(3.7,2.15,3.7,1.27)
save(f,'figure1_cohorts_architecture')
# Shared-label comparator plate: a single central model key avoids repeated long labels.
f=plt.figure(figsize=(11,9));gs=f.add_gridspec(2,3,height_ratios=[1,1.3],width_ratios=[1,.88,1],hspace=.52,wspace=.10,left=.09,right=.97,top=.92,bottom=.08)
for j,(c,n) in enumerate(zip(C,N)):
 a=f.add_subplot(gs[0,0 if j==0 else 2]);label(a,'ab'[j],n)
 for m,col,ls in [('meta_augmented',B,'-'),('selected_baseline',T,'-'),('source',G,'--'),('affine','#B17B69',':')]:
  z=curves[(curves.cohort==c)&(curves.model==m)].sort_values('shots');a.plot(np.arange(3),z.mae,ls+'o',color=col,lw=1.7,ms=4)
 a.set(xticks=[0,1,2],xticklabels=[8,16,32],xlabel='Labelled participants',ylabel='MAE (years)');a.grid(axis='y',alpha=.15)
a=f.add_subplot(gs[0,1]);a.axis('off')
for i,(t,col,ls) in enumerate([('Neural adapter',B,'-'),('Selected baseline',T,'-'),('Frozen source',G,'--'),('Affine calibration','#B17B69',':')]):
 a.plot([.12,.28],[.72-i*.14]*2,ls,color=col,lw=2,transform=a.transAxes);a.text(.34,.72-i*.14,t,transform=a.transAxes,va='center',fontsize=9)
models=['source','median','affine','target_ridge','residual_linear','residual_rbf','selected_baseline','meta_no_augmentation','meta_augmented'];names=['Frozen source ridge','Support median','Affine calibration','Target-only ridge','Linear residual kernel','RBF residual kernel','CV-selected baseline','Neural, no augmentation','Neural adapter']
for j,c in enumerate(C):
 a=f.add_subplot(gs[1,0 if j==0 else 2]);label(a,'cd'[j],N[j]+' / 16 labels')
 for i,m in enumerate(models):
  r=curves[(curves.cohort==c)&(curves.shots==16)&(curves.model==m)].iloc[0];col=B if 'meta_' in m else T if m=='selected_baseline' else G
  a.axhspan(i-.43,i+.43,color='#F3F6F8' if i>=6 else 'white',zorder=0);a.errorbar(r.mae,i,xerr=[[r.mae-r.low],[r.high-r.mae]],fmt='o',color=col,ms=5,lw=1.25,capsize=2)
 a.set(ylim=(8.65,-.65),yticks=[],xlabel='MAE (years)');a.grid(axis='x',alpha=.15)
a=f.add_subplot(gs[1,1]);a.set(ylim=(8.65,-.65));a.axis('off')
for i,n in enumerate(names):a.text(.5,i,n,ha='center',va='center',fontsize=9,weight='bold' if i>=6 else 'normal')
save(f,'figure2_external_performance')
# Four panels, one inference per panel; training and probe diagnostics retained in extended data.
f,axs=plt.subplots(2,2,figsize=(11,7.5));f.subplots_adjust(left=.13,right=.97,bottom=.10,top=.92,wspace=.43,hspace=.6)
a=axs[0,0];label(a,'a','Primary paired contrasts')
for i,c in enumerate(C):
 r=tests[(tests.cohort==c)&tests.primary].iloc[0];a.errorbar(r.mae_reduction,i,xerr=[[r.mae_reduction-r.low],[r.high-r.mae_reduction]],fmt='o',color=cols[i],capsize=3,ms=6);a.text(.97,.9-i*.17,f'Holm P = {r.holm_p:.4f}',transform=a.transAxes,ha='right',fontsize=9,color=cols[i])
a.axvline(0,color=G,lw=.8,ls='--');a.set(yticks=[0,1],yticklabels=N,ylim=(1.65,-.75),xlabel='Selected baseline − neural MAE (years)',xlim=(-.08,.65))
a=axs[0,1];label(a,'b','Variation across support draws')
for i,c in enumerate(C):
 d=draws[(draws.cohort==c)&(draws.shots==16)].pivot(index='repeat',columns='model',values='absolute_error');v=d.selected_baseline-d.meta_augmented
 a.scatter(i+np.linspace(-.14,.14,len(v)),np.sort(v),s=24,edgecolor='white',linewidth=.5,color=cols[i],alpha=.8);a.plot([i-.24,i+.24],[v.mean()]*2,color=K,lw=1.5)
a.axhline(0,color=G,lw=.8,ls='--');a.set(xticks=[0,1],xticklabels=N,ylabel='MAE reduction (years)',xlim=(-.6,1.6))
a=axs[1,0];label(a,'c','Augmentation ablation')
for i,c in enumerate(C):
 r=tests[(tests.cohort==c)&(tests.baseline=='meta_no_augmentation')].iloc[0];a.errorbar(r.mae_reduction,i,xerr=[[r.mae_reduction-r.low],[r.high-r.mae_reduction]],fmt='o',color=cols[i],capsize=3,ms=6)
a.axvline(0,color=G,lw=.8,ls='--');a.set(yticks=[0,1],yticklabels=N,ylim=(1.6,-.6),xlabel='Unaugmented − augmented MAE (years)')
a=axs[1,1];label(a,'d','Assay-matched refitting / post hoc');diag=pd.read_csv(R/'results/assay_matched_source_predictions.csv')
for i,c in enumerate(C):
 vals=[curves[(curves.cohort==c)&(curves.shots==16)&(curves.model==m)].iloc[0].mae for m in ['source','meta_augmented']]+[diag[diag.cohort==c].absolute_error.mean()]
 a.plot(vals,[i]*3,color='#CDD3D7',lw=1,zorder=0)
 for off,(v,col,mark) in enumerate(zip(vals,[G,B,T],['s','o','D'])):a.scatter(v,i,s=40,color=col,marker=mark,label=['Frozen source','Neural adapter','Assay-matched'][off] if i==0 else None)
a.set(yticks=[0,1],yticklabels=N,ylim=(1.65,-.65),xlabel='Query MAE (years)');a.legend(loc='lower left',fontsize=8,ncol=2)
save(f,'figure3_robustness_ablation')
print('Rebuilt three journal plates; original complete plates retained as previous PDFs.')
