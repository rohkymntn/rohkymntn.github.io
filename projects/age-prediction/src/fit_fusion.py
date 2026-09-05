"""Fit and serialize modality regressors plus a prespecified late-fusion operator.
No synthetic pairing, combined predictions, or fusion accuracy are reported.
"""
from pathlib import Path
import time,json,hashlib,joblib
import numpy as np,pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from threadpoolctl import threadpool_limits
from analyze import HellingerFilter,load,weights,SEED,ROOT
from fusion_model import LateFusionAgeModel

def fit_and_save():
    start=time.time();out=ROOT/'models';out.mkdir(exist_ok=True)
    X,b,bf=load('blood');S,s,sf=load('skin');E,e,ef=load('external')
    X.columns=bf;S.columns=sf;E.columns=ef
    blood=Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('select',SelectKBest(f_regression,k=500)),('scale',StandardScaler()),('ridge',Ridge(alpha=100.,solver='lsqr',tol=1e-5))])
    skin=Pipeline([('composition',HellingerFilter()),('rf',RandomForestRegressor(n_estimators=160,max_features='sqrt',min_samples_leaf=1,n_jobs=4,random_state=SEED))])
    with threadpool_limits(limits=4):
        blood.fit(X,b.age)
        skin.fit(S,s.age,rf__sample_weight=weights(s.group))
    model=LateFusionAgeModel(blood,skin,tuple(bf),tuple(sf),.5)
    path=out/'late_fusion_age_model.joblib';joblib.dump(model,path,compress=3)
    restored=joblib.load(path)
    np.testing.assert_allclose(model.predict_blood(E),restored.predict_blood(E),rtol=0,atol=0)
    np.testing.assert_allclose(model.predict_skin(S.iloc[:8]),restored.predict_skin(S.iloc[:8]),rtol=1e-12,atol=1e-10)
    prior=pd.read_csv(ROOT/'results/blood_external_predictions.csv')
    # Verify the refit component against previously recorded external predictions.
    delta=float(np.max(np.abs(restored.predict_blood(E)-prior.prediction.to_numpy())))
    assert delta<0.01,delta
    report={'architecture':'Independent modality regressors with fixed equal-weight late fusion','blood_training_samples':len(b),'skin_training_samples':len(s),'skin_training_participant_ids':s.group.nunique(),'blood_hyperparameters':{'selected_cpgs':500,'ridge_alpha':100},'skin_hyperparameters':{'trees':160,'min_samples_leaf':1,'max_features':'sqrt'},'hyperparameter_provenance':'Blood: existing discovery-only final selection; skin: common optimum in all five prior outer folds. No additional claim of CV performance for final refit.','blood_weight':.5,'fusion_weight_calibration':'None; equal weight is prespecified','paired_test_participants':0,'fusion_accuracy':None,'external_blood_refit_max_prediction_difference':delta,'runtime_seconds':time.time()-start,'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'execution_backend':'Modal CPU','date':'2026-09-05'}
    (ROOT/'results/fusion_model_manifest.json').write_text(json.dumps(report,indent=2))
    return report
if __name__=='__main__':print(json.dumps(fit_and_save(),indent=2))
