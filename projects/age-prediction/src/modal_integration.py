from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
app=modal.App('age-microbiome-integration-benchmark')
image=(modal.Image.debian_slim(python_version='3.12').pip_install('numpy==2.5.2','pandas==3.0.5','scikit-learn==1.9.0','scipy==1.18.1','joblib==1.6.0','threadpoolctl==3.6.0')
 .add_local_dir(ROOT/'src','/project/src').add_local_dir(ROOT/'data','/project/data',ignore=['raw/**']))
@app.function(image=image,cpu=4,memory=8192,timeout=900,max_containers=3)
def train(seed):
 import sys
 sys.path.insert(0,'/project/src')
 from integration_benchmark import run_repeat
 return run_repeat(seed)
@app.local_entrypoint()
def main():
 import json
 for seed,(csv,selection,seconds) in zip([20260906,20260907,20260908],train.map([20260906,20260907,20260908])):
  (ROOT/f'results/integration_predictions_{seed}.csv').write_text(csv)
  (ROOT/f'results/integration_selection_{seed}.json').write_text(json.dumps({'selection':selection,'runtime_seconds':seconds},indent=2))
  print('Saved repeat',seed,'seconds',seconds)
