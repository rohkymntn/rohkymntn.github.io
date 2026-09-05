"""Software checks for fusion arithmetic and input integrity, not biological validation."""
import unittest
import numpy as np,pandas as pd
from fusion_model import LateFusionAgeModel
class LinearFixture:
    def predict(self,x):return x.sum(axis=1).to_numpy()
class FusionTests(unittest.TestCase):
    def setUp(self):
        self.model=LateFusionAgeModel(LinearFixture(),LinearFixture(),('b1','b2'),('s1','s2'))
        self.b=pd.DataFrame([[10,20],[20,30]],columns=['b1','b2'],index=['test:a','test:b'])
        self.s=pd.DataFrame([[30,40],[40,50]],columns=['s1','s2'],index=['test:b','test:a'])
    def predict(self,b=None,s=None,provenance='Synthetic software fixture; not participant data'):
        return self.model.predict_paired(self.b if b is None else b,self.s if s is None else s,pairing_provenance=provenance)
    def test_alignment_and_exact_average(self):
        q=self.predict();np.testing.assert_allclose(q.fused_age,[60,60]);self.assertEqual(list(q.index),['test:a','test:b'])
    def test_feature_reordering(self):
        pd.testing.assert_frame_equal(self.predict(),self.predict(b=self.b[['b2','b1']]))
    def test_disjoint_subjects_rejected(self):
        with self.assertRaises(ValueError):self.predict(s=self.s.set_axis(['other:a','other:b']))
    def test_partial_pairing_rejected(self):
        with self.assertRaises(ValueError):self.predict(s=self.s.iloc[:1])
    def test_duplicate_subjects_rejected(self):
        with self.assertRaises(ValueError):self.predict(b=pd.concat([self.b,self.b.iloc[:1]]))
    def test_missing_features_rejected(self):
        with self.assertRaises(ValueError):self.predict(b=self.b[['b1']])
    def test_age_features_rejected(self):
        with self.assertRaises(ValueError):self.predict(b=self.b.assign(age=20))
    def test_provenance_required(self):
        with self.assertRaises(ValueError):self.predict(provenance='')
    def test_weight_bounds(self):
        for w in [-.1,1.1,float('nan')]:
            with self.assertRaises(ValueError):LateFusionAgeModel(None,None,(),(),w)
if __name__=='__main__':unittest.main(verbosity=2)
