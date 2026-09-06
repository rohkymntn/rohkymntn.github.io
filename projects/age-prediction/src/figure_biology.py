"""Exploratory feature and pathway summaries; original plotting code."""
from pathlib import Path
import re
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import journal_style
R=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'Helvetica','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
fig=plt.figure(figsize=(13,9),layout='constrained')
g=fig.add_gridspec(2,2,height_ratios=[1,1.1])
b=pd.read_csv(R/'results/blood_coefficients_annotated.csv').head(8).iloc[::-1]
a=fig.add_subplot(g[0,0]); a.barh(range(len(b)),b.coefficient_per_training_sd,color='#0F4D92');a.set_yticks(range(len(b)),b.gene.fillna(b.feature));a.set_xlabel('Ridge coefficient (years per training SD)');a.set_title('a  Methylation features',loc='left',fontweight='bold');a.axvline(0,color='#CFCECE',lw=.8)
s=pd.read_csv(R/'results/skin_features_annotated.csv').head(8).iloc[::-1]
labels=[]
for idx,r in s.iterrows():
 taxa=[x.strip() for x in str(r.Taxon).split(';') if len(x.strip())>3]
 labels.append(f'{taxa[-1][3:] if taxa else "Unassigned"} / ASV {idx+1}')
a=fig.add_subplot(g[0,1]);a.barh(range(len(s)),s.mean_training_impurity_importance,color='#42949E');a.set_yticks(range(len(s)),labels);a.set_xlabel('Mean training impurity importance');a.set_title('b  Microbial features',loc='left',fontweight='bold')
p=pd.read_csv(R/'results/reactome_probe_aware_enrichment.csv').head(6).iloc[::-1]
a=fig.add_subplot(g[1,:]);labels=[re.sub(r' R-HSA-\d+','',v) for v in p.pathway];a.scatter(p.fold_enrichment,range(len(p)),s=65,color='#0F4D92');a.set_yticks(range(len(p)),labels);a.axvline(1,color='#CFCECE',lw=1);a.set_xlim(0,max(p.fold_enrichment)+1.8);a.set_xlabel('Observed / expected genes under random probe sampling');a.set_title('c  Reactome annotation enrichment',loc='left',fontweight='bold',pad=25)
for y,r in enumerate(p.itertuples()):a.text(r.fold_enrichment+.14,y,f'p = {r.empirical_p:.3f}; q = {r.bh_q:.2f}',va='center',fontsize=9)
for a in fig.axes:a.tick_params(axis='y',length=0)
for ext in ['png','pdf','svg']:fig.savefig(R/f'figures/fig3_biological_interpretation.{ext}',dpi=300,bbox_inches='tight')
