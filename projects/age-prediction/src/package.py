"""Package reproducible code and derived outputs, excluding third-party raw matrices."""
from pathlib import Path
import json,shutil,zipfile
ROOT=Path(__file__).resolve().parents[1]
with zipfile.ZipFile(ROOT/'age-prediction-reproducibility.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
    for name in ['RESEARCH_PLAN.md','index.html','README.md','PLAN.md','requirements.txt','meta.json','report.html']:
        z.write(ROOT/name,f'age-prediction/{name}')
    for folder in ['src','figures','results']:
        for f in sorted((ROOT/folder).glob('*')):
            if f.is_file() and f.suffix in ['.py','.csv','.json','.png','.pdf','.svg']:
                z.write(f,f'age-prediction/{folder}/{f.name}')
    z.write(ROOT/'data/sources.json','age-prediction/data/sources.json')
print(ROOT/'age-prediction-reproducibility.zip')
