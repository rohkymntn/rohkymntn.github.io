from pathlib import Path
import zipfile,json,hashlib
R=Path(__file__).resolve().parents[1]
files=[p for p in R.rglob('*') if p.is_file() and '__pycache__' not in str(p) and 'invalidated_mvalue_run' not in str(p) and p.suffix in ['.py','.csv','.json','.pt','.npz','.png','.svg','.pdf','.html','.md','.txt','.css'] and not p.name.endswith('.gz') and p.name!='artifact_manifest.json']
manifest={str(p.relative_to(R)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}
(R/'results/artifact_manifest.json').write_text(json.dumps(manifest,indent=2));files.append(R/'results/artifact_manifest.json')
with zipfile.ZipFile(R/'fewshot-reproducibility.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
 for p in files:z.write(p,'fewshot/'+str(p.relative_to(R)))
print('Archive bytes:',(R/'fewshot-reproducibility.zip').stat().st_size)
