"""Download public GEO matrices. Files may exceed 1 GB; skip existing complete files."""
from pathlib import Path
import urllib.request,hashlib,json
R=Path(__file__).resolve().parents[1]
entries=[]
for acc in ['GSE41037','GSE19711','GSE40279','GSE37008']:
 prefix=acc[:-3]+'nnn';url=f'https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{acc}/matrix/{acc}_series_matrix.txt.gz';p=R/f'data/{acc}_series_matrix.txt.gz'
 if not p.exists():
  tmp=p.with_suffix('.part');urllib.request.urlretrieve(url,tmp);tmp.rename(p)
 entries.append({'accession':acc,'url':url,'bytes':p.stat().st_size,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()})
(R/'results/download_manifest.json').write_text(json.dumps(entries,indent=2))
