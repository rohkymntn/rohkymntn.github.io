"""Retrieve public follow-up data; no access credentials required."""
from pathlib import Path
import urllib.request,hashlib,json
R=Path(__file__).resolve().parents[1]
files=[('data/reactome/Reactome_2022.gmt','https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=Reactome_2022'),('data/candidates/Kunihiro2026_GenusOtu.csv','https://ndownloader.figshare.com/files/64073650'),('data/candidates/Kunihiro2026_Metadata.csv','https://ndownloader.figshare.com/files/64073653'),('data/candidates/Kunihiro2026_SpeciesOtu.Csv','https://ndownloader.figshare.com/files/64073656')]
manifest=[]
for name,url in files:
 p=R/name;p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists():urllib.request.urlretrieve(url,p)
 manifest.append({'path':name,'url':url,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(R/'results/expansion_download_manifest.json').write_text(json.dumps(manifest,indent=2))
