"""Exploratory Reactome annotation enrichment with probe-aware random-set control.
Null: 10,000 random sets of 500 measured CpGs; unique genes counted per set.
This controls unequal probe representation approximately, not correlated CpGs,
feature-selection behavior, or population-level biological causality.
"""
from pathlib import Path
import json,re,hashlib
import numpy as np,pandas as pd
from scipy import sparse
from scipy.stats import false_discovery_control
from figures import annotations
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results';D=ROOT/'data'
f=np.load(D/'blood.npz');probes=f['features'];an=annotations().set_index('feature').reindex(probes)
sets=[]
for value in an.gene:
 sets.append(set(t.strip().upper() for t in re.split(r';|,|///',value if isinstance(value,str) else '') if t.strip()))
allgenes=sorted(set.union(*sets));gix={g:i for i,g in enumerate(allgenes)};probe_genes=[np.array([gix[g] for g in genes],int) for genes in sets]
pathways=[]
for line in (D/'reactome/Reactome_2022.gmt').read_text().splitlines():
 parts=line.split('\t');genes=set(parts[2:])&set(gix)
 if 10<=len(genes)<=500:pathways.append((parts[0],genes))
pr=[];pc=[]
for i,(_,genes) in enumerate(pathways):
 for g in genes:pr.append(i);pc.append(gix[g])
P=sparse.csr_matrix((np.ones(len(pr),dtype=np.int16),(pr,pc)),shape=(len(pathways),len(allgenes)))
selected=pd.read_csv(R/'blood_coefficients.csv').feature.tolist();ix={p:i for i,p in enumerate(probes)}
selectedgenes=np.unique(np.concatenate([probe_genes[ix[p]] for p in selected]));v=np.zeros(len(allgenes),dtype=np.int16);v[selectedgenes]=1;observed=np.asarray(P@v).ravel()
rng=np.random.default_rng(20260910);nperm=10000;greater=np.zeros(len(pathways));total=np.zeros(len(pathways));total2=np.zeros(len(pathways))
for start in range(0,nperm,500):
 rows=[];cols=[]
 for k in range(500):
  genes=np.unique(np.concatenate([probe_genes[j] for j in rng.choice(len(probes),len(selected),replace=False)]))
  rows.extend([k]*len(genes));cols.extend(genes)
 S=sparse.csr_matrix((np.ones(len(rows),dtype=np.int16),(rows,cols)),shape=(500,len(allgenes)))
 counts=(S@P.T).toarray().astype(float)
 greater+=(counts>=observed).sum(axis=0);total+=counts.sum(axis=0);total2+=(counts**2).sum(axis=0)
p=(greater+1)/(nperm+1);q=false_discovery_control(p,method='bh');expect=total/nperm
result=pd.DataFrame({'pathway':[x[0] for x in pathways],'measured_pathway_genes':[len(x[1]) for x in pathways],'selected_genes':observed,'random_probe_expected_genes':expect,'fold_enrichment':np.divide(observed,expect,out=np.zeros_like(expect),where=expect>0),'empirical_p':p,'bh_q':q,'overlap_genes':[';'.join(sorted(genes&set(np.array(allgenes)[selectedgenes]))) for _,genes in pathways]})
result.sort_values(['bh_q','empirical_p','fold_enrichment'],ascending=[True,True,False]).to_csv(R/'reactome_probe_aware_enrichment.csv',index=False)
summary={'measured_probes':len(probes),'mapped_genes':len(allgenes),'selected_probes':len(selected),'selected_unique_genes':len(selectedgenes),'tested_pathways':len(pathways),'random_sets':nperm,'q_below_005':int((q<.05).sum()),'library':'Reactome_2022, Enrichr distribution','limitations':'Exploratory annotation enrichment; random CpG sets approximate probe-coverage bias but do not preserve CpG correlation or reproduce supervised selection. Not a mechanistic discovery.','library_sha256':hashlib.sha256((D/'reactome/Reactome_2022.gmt').read_bytes()).hexdigest()}
(R/'reactome_analysis_manifest.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2));print(result.sort_values('empirical_p').head(8).to_string(index=False))
