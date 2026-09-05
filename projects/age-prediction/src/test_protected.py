import unittest
import numpy as np
from sklearn.linear_model import LinearRegression
from protected_fusion import ProtectedFusion
class ProtectedTests(unittest.TestCase):
 def setUp(self):
  rng=np.random.default_rng(42);self.X=rng.normal(size=(40,7));self.y=3+2*self.X[:,0]+rng.normal(size=40)
 def test_disabled_correction_is_exact_clock_baseline(self):
  model=ProtectedFusion().fit(self.X,self.y)
  expected=LinearRegression().fit(self.X[:,:1],self.y).predict(self.X[:,:1])
  changed=self.X.copy();changed[:,1:]*=1000
  np.testing.assert_allclose(model.predict(changed),expected)
 def test_training_microbes_are_orthogonal_to_clock(self):
  model=ProtectedFusion(kernel='linear').fit(self.X,self.y)
  residual=self.X[:,1:]-model.projection_.predict(self.X[:,:1])
  np.testing.assert_allclose(residual.mean(0),0,atol=1e-12)
  np.testing.assert_allclose(self.X[:,0]@residual,0,atol=1e-12)
 def test_inference_does_not_refit_projection(self):
  model=ProtectedFusion(kernel='rbf').fit(self.X,self.y);coef=model.projection_.coef_.copy()
  individual=model.predict(self.X[:1]);batch=model.predict(np.vstack([self.X[:1],self.X[1:]+20]))
  np.testing.assert_allclose(individual,batch[:1]);np.testing.assert_array_equal(coef,model.projection_.coef_)
if __name__=='__main__':unittest.main()
