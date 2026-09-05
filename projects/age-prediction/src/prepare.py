"""Load public processed matrices; audit identity, eligibility and missingness."""
from pathlib import Path
import csv, gzip, json, hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw'; DATA=ROOT/'data'; RES=ROOT/'results'

def geo(accession):
    path=RAW/f'{accession}_series_matrix.txt.gz'
    meta={}
    with gzip.open(path,'rt') as f:
        for row in csv.reader(f,delimiter='\t'):
            if not row: continue
            if row[0]=='!Sample_geo_accession': ids=row[1:]
            if row[0]=='!Sample_title': meta['title']=row[1:]
            if row[0].startswith('!Sample_characteristics'):
                key=row[1].split(':',1)[0]
                meta[key]=[x.split(':',1)[1].strip() if ':' in x else x for x in row[1:]]
            if row[0]=='!series_matrix_table_begin': break
        matrix=pd.read_csv(f,sep='\t',index_col=0,comment='!',na_values=['null','NA']).T
    meta=pd.DataFrame(meta,index=ids).loc[matrix.index]
    matrix=matrix.astype(np.float32)
    assert matrix.index.is_unique and matrix.columns.is_unique
    return matrix,meta

def main():
    audit={}
    x,m=geo('GSE41037')
    m['age']=pd.to_numeric(m.age,errors='coerce')
    eligible=(m.diseasestatus=='1; control')&(m.used_in_analysis=='yes')&m.age.ge(18)
    audit['blood_discovery']={'accession':'GSE41037','total_samples':len(m),'eligible_controls':int(eligible.sum()),'excluded':int((~eligible).sum()),'rules':'control, source QC yes, numeric age >=18; no outcome-based exclusions'}
    x=x.loc[eligible];m=m.loc[eligible];m['group']=m.plate;m['sample_id']=m.index;m['sex']=m.gender
    z,n=geo('GSE19711');n['age']=pd.to_numeric(n.ageatrecruitment,errors='coerce')
    eligible=n['sample type'].str.endswith('from Control')&n.age.ge(18)
    audit['blood_external']={'accession':'GSE19711','total_samples':len(n),'eligible_controls':int(eligible.sum()),'excluded':int((~eligible).sum()),'age_definition':'age at recruitment (not exact age at phlebotomy); external cohort of women'}
    z=z.loc[eligible];n=n.loc[eligible];n['group']=n.index;n['sex']='female';n['sample_id']=n.index
    common=x.columns.intersection(z.columns,sort=False)
    x=x[common];z=z[common]
    for name,a,b in [('blood',x,m),('external',z,n)]:
        np.savez_compressed(DATA/f'{name}.npz',X=a.to_numpy(),features=a.columns.to_numpy(dtype=str))
        b.to_csv(DATA/f'{name}_metadata.csv',index=False)
        audit['blood_discovery' if name=='blood' else 'blood_external'].update({'cpgs':len(common),'missing_fraction':float(a.isna().to_numpy().mean()),'age_min':float(b.age.min()),'age_max':float(b.age.max()),'unique_array_profiles':int(a.drop_duplicates().shape[0])})
    a=pd.read_csv(RAW/'skin_abundance.tsv',sep='\t',index_col=0).astype(np.float32)
    meta=pd.read_csv(RAW/'skin_metadata.tsv',sep='\t',low_memory=False).set_index('#SampleID').loc[a.index]
    meta['age']=pd.to_numeric(meta.qiita_host_age,errors='coerce')
    meta['study']=meta.qiita_study_id.astype(str)
    meta['group']=meta.study+':'+meta.host_subject_id.astype(str)
    meta['sex']=meta.qiita_host_sex.fillna('unknown');meta['site']=meta.body_site;meta['sample_id']=meta.index
    summary=meta.groupby('study').agg(samples=('group','size'),participants=('group','nunique'),age_min=('age','min'),age_max=('age','max'))
    summary.to_csv(RES/'skin_study_audit.csv')
    conflicts=meta.groupby('group').age.agg(['min','max','nunique','size'])
    conflicts[conflicts['nunique'].gt(1)].to_csv(RES/'skin_age_discrepancies.csv')
    # 11052: all 177 rows share one subject identifier, with incompatible ages 31/36.
    # Exclude this identity-ambiguous cohort from all primary predictive comparisons.
    eligible=meta.age.ge(18)&meta.host_subject_id.notna()&meta.study.ne('11052')&(a.sum(axis=1)>0)
    audit['skin']={'total_samples':len(meta),'total_subject_labels':meta.group.nunique(),'excluded_identity_ambiguous_study':'11052','excluded_rows':int((~eligible).sum()),'eligible_rows':int(eligible.sum()),'eligible_subjects':int(meta.loc[eligible,'group'].nunique()),'input_asvs':a.shape[1],'source_filter':'Published processed abundance table was already prevalence-filtered without age labels; additional prevalence filtering fits inside every fold.'}
    a=a.loc[eligible];meta=meta.loc[eligible,['sample_id','age','study','group','sex','site']]
    audit['skin']['row_sum_range']=[float(a.sum(axis=1).min()),float(a.sum(axis=1).max())]
    audit['skin']['subjects_with_age_discrepancy']=int(meta.groupby('group').age.nunique().gt(1).sum())
    np.savez_compressed(DATA/'skin.npz',X=a.to_numpy(),features=a.columns.to_numpy(dtype=str))
    meta.to_csv(DATA/'skin_metadata.csv',index=False)
    audit['pairing']={'verified_paired_blood_skin_participants':0,'action':'No cross-person concatenation, age matching, or empirical fusion accuracy. Separate clocks and a mathematical fusion sensitivity analysis only.'}
    audit['files']={f.name:{'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in RAW.iterdir() if f.is_file()}
    (RES/'audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2),flush=True)
if __name__=='__main__':main()
