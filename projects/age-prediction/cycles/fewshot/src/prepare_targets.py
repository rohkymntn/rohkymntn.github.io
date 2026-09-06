"""Stream only source-selected probes; do not use external labels for feature selection."""
from pathlib import Path
import gzip,csv,json,hashlib
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[1]
s=np.load(R/'data/source.npz');features=s['features'];wanted=set(features)
def parse(acc):
 path=R/f'data/{acc}_series_matrix.txt.gz';meta={};rows={}
 with gzip.open(path,'rt') as f:
  for row in csv.reader(f,delimiter='\t'):
   if not row:continue
   if row[0]=='!Sample_geo_accession':ids=row[1:]
   if row[0]=='!Sample_title':meta['title']=row[1:]
   if row[0].startswith('!Sample_characteristics'):
    key=row[1].split(':',1)[0];meta[key]=[v.split(':',1)[1].strip() if ':' in v else v for v in row[1:]]
   if row[0]=='!series_matrix_table_begin':break
  header=next(f)
  for line in f:
   key=line.split('\t',1)[0].strip('"')
   if key in wanted:
    vals=next(csv.reader([line],delimiter='\t'))[1:];rows[key]=[float(v) if v not in ['null','NA','NaN',''] else np.nan for v in vals]
 m=pd.DataFrame(meta,index=ids);m['age']=pd.to_numeric(m['age (y)'],errors='coerce');m['sample_id']=m.index
 raw=np.array([rows.get(k,[np.nan]*len(m)) for k in features],dtype=np.float32).T
 raw_range=[float(np.nanmin(raw)),float(np.nanmax(raw))]
 if acc=='GSE37008':raw=1/(1+np.exp2(-raw))
 assert np.nanmin(raw)>=0 and np.nanmax(raw)<=1, 'Unexpected non-beta scale'
 present=np.any(np.isfinite(raw),axis=0);missing=np.isnan(raw[:,present]).mean(1)
 good=m.age.ge(18).to_numpy()&(missing<=.01)
 m=m.loc[good].copy();raw=raw[good];m['donor']=m.title.str.replace(r'\.Rep$','',regex=True) if acc=='GSE37008' else m.index
 conflicts=m.groupby('donor').age.nunique();assert (conflicts<=1).all()
 filled=np.where(np.isfinite(raw),raw,s['impute']);X=np.clip((filled-s['mean'])/s['scale'],-8,8)
 # Average technical replicate measurements in source-standardized space.
 frame=pd.DataFrame(X,index=m.donor);grouped=frame.groupby(level=0,sort=True).mean();meta=m.groupby('donor',sort=True).first().loc[grouped.index]
 X=grouped.to_numpy().astype('float32');y=meta.age.to_numpy();base=X@s['coef']+s['intercept']
 rng=np.random.default_rng(99127 if acc=='GSE40279' else 99128);order=rng.permutation(len(y));pool_n=128 if acc=='GSE40279' else 32
 print('Eligibility audit',acc,'n=',len(y),'observed probes=',present.sum(),'missing quantiles=',np.quantile(missing,[0,.25,.5,.75,1]),flush=True)
 assert len(y)>pool_n+20
 support=np.zeros(len(y),bool);support[order[:pool_n]]=True
 np.savez_compressed(R/f'data/{acc}.npz',X=X,y=y,base=base,ids=grouped.index.to_numpy(str),support=support)
 meta['partition']=np.where(support,'support_pool','query');meta[['sample_id','age','partition','gender']].to_csv(R/f'results/{acc}_partitions.csv')
 audit={'cohort':acc,'deposited_samples':len(ids),'retained_samples':int(good.sum()),'unique_participants':len(y),'technical_replicates_collapsed':int(good.sum()-len(y)),'source_selected_features':len(features),'observed_selected_features':int(present.sum()),'structurally_absent_features':int((~present).sum()),'support_pool':int(support.sum()),'query_participants':int((~support).sum()),'age_range':[float(y.min()),float(y.max())],'input_scale':'M-value' if acc=='GSE37008' else 'beta','raw_value_range':raw_range,'scale_conversion':'beta = 1/(1+2**(-M))' if acc=='GSE37008' else 'none','standardized_clip_fraction':float(np.mean(abs(X)>=7.99)),'raw_sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest(),'eligibility':'Adult with numeric age and <=1% missingness among observed selected probes; structural platform absence is source-median imputed. Fixed random split does not stratify on age.'}
 (R/f'results/{acc}_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2),flush=True)
if __name__=='__main__':
 import sys
 for acc in sys.argv[1:] or ['GSE37008','GSE40279']:parse(acc)
