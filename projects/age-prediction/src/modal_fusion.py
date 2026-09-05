"""Run: modal run src/modal_fusion.py (after download.py and prepare.py)."""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
app=modal.App('blood-skin-age-late-fusion')
image=(modal.Image.debian_slim(python_version='3.12')
    .pip_install('numpy==2.5.2','pandas==3.0.5','scikit-learn==1.9.0','scipy==1.18.1','joblib==1.6.0','threadpoolctl==3.6.0')
    .add_local_dir(ROOT/'src','/project/src')
    .add_local_dir(ROOT/'data','/project/data',ignore=['raw/**'])
    .add_local_file(ROOT/'results/blood_external_predictions.csv','/project/results/blood_external_predictions.csv'))
@app.function(image=image,cpu=4,memory=8192,timeout=600,max_containers=1)
def train():
    import sys
    sys.path.insert(0,'/project/src')
    from fit_fusion import fit_and_save
    report=fit_and_save()
    return report,Path('/project/models/late_fusion_age_model.joblib').read_bytes()
@app.local_entrypoint()
def main():
    import json
    report,weights=train.remote()
    (ROOT/'models').mkdir(exist_ok=True)
    (ROOT/'models/late_fusion_age_model.joblib').write_bytes(weights)
    (ROOT/'results/fusion_model_manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
