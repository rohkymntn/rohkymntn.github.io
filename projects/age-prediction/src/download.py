"""Retrieve only documented public sources, then verify the recorded SHA-256."""
from pathlib import Path
import json,hashlib,urllib.request
ROOT=Path(__file__).resolve().parents[1]
sources=json.loads((ROOT/'data/sources.json').read_text())
expected=json.loads((ROOT/'results/audit.json').read_text()).get('files',{})
raw=ROOT/'data/raw';raw.mkdir(parents=True,exist_ok=True)
for item in sources['sources']:
    dest=raw/item['file']
    if not dest.exists():
        print('Downloading',item['file'],flush=True)
        urllib.request.urlretrieve(item['url'],dest)
    got=hashlib.sha256(dest.read_bytes()).hexdigest()
    if item['file'] in expected and got!=expected[item['file']]['sha256']:
        raise ValueError(f'Checksum mismatch: {dest.name}; source changed or download incomplete')
    print(dest.name,got)
