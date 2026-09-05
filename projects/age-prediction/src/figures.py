"""Publication-style, data-derived figures. Vector PDF/SVG and 400-dpi PNG."""
from pathlib import Path
import gzip,csv,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results';F=ROOT/'figures';D=ROOT/'data'
BLUE='#2166AC';TEAL='#008577';ORANGE='#D97923';GRAY='#8B9096';RED='#B84A62'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':9,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'xtick.major.width':.6,'ytick.major.width':.6,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'legend.frameon':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})

def panel(ax,label,title):
    ax.text(-.14,1.13,label,transform=ax.transAxes,fontweight='bold',fontsize=11,va='top')
    ax.set_title(title,loc='left',pad=10,fontweight='medium')

def save(fig,name):
    for fmt in ['pdf','svg','png']:fig.savefig(F/f'{name}.{fmt}',dpi=400,bbox_inches='tight')
    plt.close(fig)

def fig1():
    from scipy.stats import gaussian_kde
    b=pd.read_csv(D/'blood_metadata.csv');e=pd.read_csv(D/'external_metadata.csv');s=pd.read_csv(D/'skin_metadata.csv')
    cohorts=[('Blood discovery · GSE41037',b.age.to_numpy(),BLUE),('Blood validation · GSE19711',e.age.to_numpy(),ORANGE)]
    for study,col in [(1841,'#006D77'),(2010,'#4F9085'),(10317,'#86ACA1')]:
        v=s[s.study==study].groupby('group').age.mean().to_numpy()
        cohorts.append((f'Skin · Qiita {study}',v,col))
    fig=plt.figure(figsize=(7.2,6.4),layout='constrained')
    top,bottom=fig.subfigures(2,1,height_ratios=[1.6,1.15],hspace=.10)
    a=top.subplots();panel(a,'a','Age distribution by modality and study')
    rng=np.random.default_rng(20260905)
    for i,(label,v,c) in enumerate(cohorts):
        xx=np.linspace(v.min(),v.max(),250);density=gaussian_kde(v,bw_method=.28)(xx)
        a.fill_between(xx,i-.05,i-.05-.32*density/density.max(),color=c,alpha=.32,lw=.6,edgecolor=c)
        a.scatter(v,i+.12+rng.uniform(-.035,.035,len(v)),s=3.2,color=c,alpha=.32,edgecolors='none',rasterized=True)
        lo,med,hi=np.quantile(v,[.25,.5,.75]);a.plot([lo,hi],[i-.05]*2,color=c,lw=3,solid_capstyle='round')
        a.scatter([med],[i-.05],s=16,facecolors='white',edgecolors=c,lw=.8,zorder=4)
    a.set(yticks=range(5),yticklabels=[f'{label}  (n = {len(v)})' for label,v,c in cohorts],xlabel='Chronological age (years)',xlim=(15,95),ylim=(4.42,-.55),xticks=np.arange(20,100,10))
    a.tick_params(axis='y',length=0,pad=8);a.spines['left'].set_visible(False)
    a.xaxis.grid(True,color='#E4E6E8',lw=.5);a.set_axisbelow(True)
    a.text(1,1.02,'Median / interquartile range · individual observations',ha='right',transform=a.transAxes,fontsize=6.5,color='#666666')
    lower=bottom.subplots(1,2,gridspec_kw={'wspace':.20})
    a=lower[0];panel(a,'b','Skin sampling structure')
    audit=pd.read_csv(R/'skin_study_audit.csv');ys=np.arange(len(audit))
    a.hlines(ys,audit.participants,audit.samples,color='#C4C9CA',lw=1.8)
    a.scatter(audit.samples,ys,color=GRAY,s=23,label='Samples',zorder=3)
    a.scatter(audit.participants,ys,color=TEAL,s=23,label='Participant IDs',zorder=4)
    a.set(yticks=ys,yticklabels=[str(x)+(' *' if x==11052 else '') for x in audit.study],xscale='log',xlabel='Count (log scale)',ylabel='Qiita study',xlim=(.6,3000))
    a.invert_yaxis();a.legend(loc='upper left',bbox_to_anchor=(0,-.40),fontsize=6,ncol=2)
    a.text(0,-.32,'* Study 11052 excluded: ambiguous participant identifiers.',transform=a.transAxes,fontsize=6.2,color='#666666')
    a=lower[1];panel(a,'c','Participant overlap')
    t=pd.read_csv(R/'skin_split_overlap.csv');labels=['Random sample split','Participant split','Leave study out'];vals=[]
    for l in labels:
        q=t[t.validation==l];vals.append(100*np.average(q.test_participant_overlap_fraction,weights=q.test_rows))
    a.hlines(range(3),0,vals,color='#D4D8DC',lw=2)
    a.scatter(vals,range(3),color=[ORANGE,TEAL,BLUE],s=30,zorder=3)
    for i,v in enumerate(vals):a.annotate(f'{v:.1f}%',(v,i),xytext=(5,0),textcoords='offset points',va='center',fontsize=7)
    a.set(yticks=range(3),yticklabels=['Random samples','Participant holdout','Study holdout'],xlabel='Test samples with participant\nin training (%)',xlim=(-3,112),ylim=(2.55,-.55),xticks=[0,25,50,75,100])
    a.xaxis.grid(True,color='#E4E6E8',lw=.5);a.set_axisbelow(True)
    save(fig,'fig1_data_and_design')

def scatter(a,p,c,title,label):
    panel(a,label,title);a.scatter(p.age,p.prediction,s=9,alpha=.42,color=c,edgecolors='none',rasterized=True)
    low=min(p.age.min(),p.prediction.min(),15)-3;high=max(p.age.max(),p.prediction.max(),85)+3
    a.plot([low,high],[low,high],color=GRAY,ls='--',lw=.8);a.set(xlabel='Recorded age (years)',ylabel='Predicted age (years)',xlim=(15,95),ylim=(low,high));a.xaxis.set_major_locator(MaxNLocator(4));a.yaxis.set_major_locator(MaxNLocator(5))

def fig2():
    met=pd.read_csv(R/'metrics.csv').set_index('analysis');b=pd.read_csv(R/'blood_cv_predictions.csv');e=pd.read_csv(R/'blood_external_predictions.csv');s=pd.read_csv(R/'skin_cv_predictions.csv')
    fig,ax=plt.subplots(2,3,figsize=(7.2,5.7),layout='constrained')
    for a,p,c,title,l,key in [(ax[0,0],b,BLUE,'Blood: plate holdout','a','Blood plate CV'),(ax[0,1],e,ORANGE,'Blood: external cohort','b','Blood external'),(ax[0,2],s,TEAL,'Skin: participant holdout','c','Skin participant CV')]:
        scatter(a,p,c,title,l);z=met.loc[key];a.text(.04,.96,f'MAE {z.mae:.2f} y\nR² {z.r2:.2f}',va='top',transform=a.transAxes,fontsize=7)
    a=ax[1,0];panel(a,'d','Model versus baseline')
    keys=['Blood plate CV','Blood external','Skin participant CV'];labels=['Blood CV','Blood external','Skin CV']
    for i,k in enumerate(keys):
        for offset,key,c in [(-.16,k,BLUE),(.16,k+' median baseline',GRAY)]:
            z=met.loc[key];a.errorbar(z.mae,i+offset,xerr=[[z.mae-z.mae_low],[z.mae_high-z.mae]],fmt='o',color=c,ms=4,capsize=2,label=('Model' if c==BLUE else 'Median') if i==0 else None)
    a.set(yticks=range(3),yticklabels=labels,xlabel='Mean absolute error (years)',xlim=(0,32));a.invert_yaxis();a.legend(loc='upper left',bbox_to_anchor=(0,-.40),fontsize=6,ncol=2)
    a=ax[1,1];panel(a,'e','Skin age-related bias')
    q=s.assign(band=pd.cut(s.age,[18,30,40,50,60,70,100],right=False),error=s.prediction-s.age)
    g=q.groupby(['group','band'],observed=True).error.mean().reset_index();v=g.groupby('band',observed=True).error.agg(['mean','count','std']);xs=np.arange(len(v));a.errorbar(xs,v['mean'],yerr=1.96*v['std']/np.sqrt(v['count']),fmt='o-',color=TEAL,ms=4,capsize=2,lw=1);a.axhline(0,color=GRAY,ls='--',lw=.8)
    a.set(xticks=xs,xticklabels=['18–29','30–39','40–49','50–59','60–69','70+'],xlabel='Recorded age (years)',ylabel='Prediction − age (years)');a.tick_params(axis='x',rotation=40)
    a=ax[1,2];panel(a,'f','Skin study transfer')
    q=pd.read_csv(R/'skin_study_performance.csv');studies=sorted(q.study.unique())
    for offset,val,c in [(-.15,'Nested participant CV',TEAL),(.15,'Leave study out',ORANGE)]:
        z=q[q.validation==val].set_index('study').loc[studies];a.errorbar(range(len(studies)),z.mae,yerr=[z.mae-z.mae_low,z.mae_high-z.mae],fmt='o-',color=c,label='Participant CV' if c==TEAL else 'Study holdout',ms=4,lw=1,capsize=2)
    a.set(xticks=range(len(studies)),xticklabels=[str(v) for v in studies],xlabel='Held-out study',ylabel='Participant-weighted MAE (years)');a.legend(fontsize=6)
    save(fig,'fig2_prediction_and_generalization')

def annotations():
    path=D/'raw/GPL8490_platform.soft';rows=[]
    with open(path) as f:
        for line in f:
            if line.startswith('!platform_table_begin'):break
        reader=csv.DictReader(f,delimiter='\t')
        for row in reader:
            if str(row.get('ID','')).startswith('!platform_table_end'):break
            if row.get('ID','').startswith('cg'):rows.append({'feature':row['ID'],'gene':row.get('Symbol','')})
    return pd.DataFrame(rows).drop_duplicates('feature')

def fig3():
    fig,ax=plt.subplots(2,2,figsize=(7.2,5.7),layout='constrained')
    a=ax[0,0];panel(a,'a','External missingness')
    q=pd.read_csv(R/'external_quality_audit.csv');a.scatter(q.missing_fraction*100,q.error,s=13,color=ORANGE,alpha=.65,edgecolors='none');a.axhline(0,color=GRAY,ls='--',lw=.8);a.axvline(1,color=RED,ls=':',lw=.8)
    a.set(xlabel='Missing CpGs (%)',ylabel='Prediction − age (years)');a.text(.97,.06,'All 274 controls retained in primary analysis.\nDotted line: exploratory 1% missingness flag.',ha='right',transform=a.transAxes,fontsize=6.5)
    a=ax[0,1];panel(a,'b','Selected methylation markers')
    co=pd.read_csv(R/'blood_coefficients.csv').merge(annotations(),on='feature',how='left');co.to_csv(R/'blood_coefficients_annotated.csv',index=False);z=co.head(8).iloc[::-1]
    a.barh(range(len(z)),z.coefficient_per_training_sd,color=[BLUE if v>0 else ORANGE for v in z.coefficient_per_training_sd]);a.set(yticks=range(len(z)),yticklabels=[f'{g if isinstance(g,str) and g else "unannotated"} · {c}' for g,c in zip(z.gene,z.feature)],xlabel='Ridge coefficient (years per training SD)');a.axvline(0,color=GRAY,lw=.6);a.tick_params(axis='y',labelsize=6)
    a=ax[1,0];panel(a,'c','Exploratory microbial ranks')
    imp=pd.read_csv(R/'skin_feature_importance.csv');tax=pd.read_csv(D/'raw/skin_taxonomy.tsv',sep='\t').rename(columns={'Feature.ID':'feature'});imp=imp.merge(tax,on='feature',how='left');imp.to_csv(R/'skin_features_annotated.csv',index=False);z=imp.head(8).iloc[::-1]
    def label(t):
        if not isinstance(t,str):return 'Unclassified ASV'
        parts=[v.strip() for v in t.split(';') if len(v.strip())>3 and not v.strip().startswith('s__')]
        return parts[-1].split('__',1)[-1] if parts else 'Unclassified ASV'
    a.barh(range(len(z)),z.mean_training_impurity_importance,color=TEAL);a.set(yticks=range(len(z)),yticklabels=[f'{label(t)} [ASV {8-i}]' for i,t in enumerate(z.Taxon)],xlabel='Mean training impurity importance');a.tick_params(axis='y',labelsize=6)
    a=ax[1,1];panel(a,'d','Label-shuffle controls')
    observed=pd.read_csv(R/'permutation_observed_matched.csv').set_index('modality');rng=np.random.default_rng(17)
    for i,(name,c) in enumerate([('blood',BLUE),('skin',TEAL)]):
        vals=pd.read_csv(R/f'{name}_permutation.csv').mae;a.scatter(i+rng.uniform(-.08,.08,len(vals)),vals,color=GRAY,s=14,alpha=.65,label='10 shuffled-label runs' if i==0 else None);a.scatter(i,observed.loc[name,'mae'],color=c,s=38,marker='D',label='Matched unshuffled model' if i==0 else None)
    a.set(xticks=[0,1],xticklabels=['Blood','Skin'],xlim=(-.5,1.5),ylabel='Mean absolute error (years)');a.legend(fontsize=6,loc='upper right')
    save(fig,'fig3_quality_and_interpretation')

if __name__=='__main__':
    F.mkdir(exist_ok=True);fig1();fig2();fig3()
