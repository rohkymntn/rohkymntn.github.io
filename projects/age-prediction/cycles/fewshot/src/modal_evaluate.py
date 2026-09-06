from pathlib import Path
import modal,json
R=Path(__file__).resolve().parents[1]
app=modal.App('fewshot-locked-query-evaluation')
image=(modal.Image.debian_slim(python_version='3.12').pip_install('numpy==2.5.2','torch==2.8.0').pip_install('pandas==3.0.5','scikit-learn==1.9.0','scipy==1.18.1').add_local_dir(R/'src','/project/src').add_local_dir(R/'results','/project/results').add_local_dir(R/'data','/project/data',ignore=['*.gz']))
@app.function(image=image,cpu=4,memory=8192,timeout=900,max_containers=2)
def run(acc):
 import sys
 sys.path.insert(0,'/project/src')
 from evaluate import evaluate
 return evaluate(acc)
@app.local_entrypoint()
def main():
 cohorts=['GSE40279','GSE37008']
 for acc,(csv,selection,checks) in zip(cohorts,run.map(cohorts)):
  (R/f'results/{acc}_predictions.csv').write_text(csv);(R/f'results/{acc}_support_selection.json').write_text(json.dumps(selection,indent=2));(R/f'results/{acc}_model_checks.json').write_text(json.dumps(checks,indent=2));print('Evaluated',acc)
