from pathlib import Path
import modal,json
R=Path(__file__).resolve().parents[1]
app=modal.App('fewshot-epigenetic-adaptation-cycle')
image=(modal.Image.debian_slim(python_version='3.12').pip_install('numpy==2.5.2','torch==2.8.0').add_local_file(R/'src/meta_model.py','/project/src/meta_model.py').add_local_file(R/'data/source.npz','/project/data/source.npz'))
@app.function(image=image,gpu='T4',cpu=2,memory=8192,timeout=1200,max_containers=2)
def fit(args):
 import sys
 sys.path.insert(0,'/project/src')
 from meta_model import train
 return train(*args)
@app.local_entrypoint()
def main():
 args=[(s,a) for a in [False,True] for s in [41,42,43]]
 for arg,(weights,trace) in zip(args,fit.map(args)):
  tag=f'{arg[0]}_{int(arg[1])}';(R/f'results/meta_{tag}.pt').write_bytes(weights);(R/f'results/meta_{tag}.json').write_text(json.dumps(trace,indent=2));print('Completed',tag,trace['best_validation_mae'])
