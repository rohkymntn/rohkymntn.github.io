from pathlib import Path
import json,modal
ROOT=Path(__file__).resolve().parents[1]
app=modal.App('paired-clock-gut-pilot')
image=(modal.Image.debian_slim(python_version='3.12').pip_install('numpy==2.5.2','pandas==3.0.5','scikit-learn==1.9.0','scipy==1.18.1','joblib==1.6.0','threadpoolctl==3.6.0').add_local_file(ROOT/'src/hiseed_pilot.py','/project/src/hiseed_pilot.py').add_local_dir(ROOT/'data/candidates','/project/data/candidates'))
@app.function(image=image,cpu=2,memory=4096,timeout=600,max_containers=3)
def train(seed):
 import sys
 sys.path.insert(0,'/project/src')
 from hiseed_pilot import run
 return run(seed)
@app.local_entrypoint()
def main():
 seeds=[20260911,20260912,20260913]
 for seed,(csv,choices) in zip(seeds,train.map(seeds)):
  (ROOT/f'results/hiseed_predictions_{seed}.csv').write_text(csv)
  (ROOT/f'results/hiseed_selection_{seed}.json').write_text(json.dumps(choices,indent=2))
  print('Completed',seed)
