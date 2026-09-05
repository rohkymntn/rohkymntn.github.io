"""Scientific integrity checks for stored outputs and paired-data guardrails."""
import unittest,json
from pathlib import Path
import numpy as np
import pandas as pd
from paired_fusion import align_paired_modalities
from analyze import HellingerFilter,evaluate
ROOT=Path(__file__).resolve().parents[1]
class Integrity(unittest.TestCase):
    def test_disjoint_people_rejected(self):
        a=pd.DataFrame({'CpG':[.1]},index=['blood:1']);b=pd.DataFrame({'ASV':[4]},index=['skin:1'])
        with self.assertRaises(ValueError):align_paired_modalities(a,b)
    def test_duplicate_people_rejected(self):
        a=pd.DataFrame({'CpG':[.1,.2]},index=['person:1','person:1'])
        with self.assertRaises(ValueError):align_paired_modalities(a,a)
    def test_target_not_a_feature(self):
        a=pd.DataFrame({'age':[25]},index=['person:1'])
        with self.assertRaises(ValueError):align_paired_modalities(a,a)
    def test_pair_order_is_identity_not_row_order(self):
        a=pd.DataFrame({'CpG':[.1,.2]},index=['p:1','p:2']);b=pd.DataFrame({'ASV':[3,4]},index=['p:2','p:1'])
        x,y=align_paired_modalities(a,b);self.assertEqual(y.ASV.tolist(),[4,3])
    def test_unseen_test_feature_cannot_enter_filter(self):
        f=HellingerFilter().fit(np.array([[1,0],[2,0]]));x=f.transform(np.array([[0,100]]))
        self.assertEqual(x.shape,(1,1));self.assertEqual(x[0,0],0)
    def test_group_split_integrity(self):
        for name in ['skin_cv_predictions.csv','blood_cv_predictions.csv']:
            p=pd.read_csv(ROOT/'results'/name)
            self.assertEqual(p.groupby('group').fold.nunique().max(),1)
            self.assertTrue(p.sample_id.is_unique)
            self.assertTrue(np.isfinite(p.prediction).all())
    def test_external_population_and_prediction_count(self):
        p=pd.read_csv(ROOT/'results/blood_external_predictions.csv')
        self.assertEqual(len(p),274)
        self.assertTrue(p['sample type'].str.endswith('from Control').all())
    def test_ambiguous_cohort_not_in_primary_analysis(self):
        p=pd.read_csv(ROOT/'results/skin_cv_predictions.csv')
        self.assertFalse((p.study==11052).any())
        self.assertEqual(len(p),1798);self.assertEqual(p.group.nunique(),339)
    def test_reported_metrics_recompute(self):
        metrics=pd.read_csv(ROOT/'results/metrics.csv').set_index('analysis')
        for label,file in [('Blood plate CV','blood_cv_predictions.csv'),('Blood external','blood_external_predictions.csv'),('Skin participant CV','skin_cv_predictions.csv')]:
            p=pd.read_csv(ROOT/'results'/file)
            self.assertAlmostEqual(evaluate(p)['mae'],metrics.loc[label,'mae'],places=6)
    def test_no_verified_cross_modal_pairs(self):
        a=json.loads((ROOT/'results/audit.json').read_text())
        self.assertEqual(a['pairing']['verified_paired_blood_skin_participants'],0)
if __name__=='__main__':unittest.main()
