"""Numerical and leakage checks for the implemented support-only solver."""
import ast,unittest
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
# Load the pure NumPy solver without requiring a local GPU/PyTorch install.
p=Path(__file__).with_name('evaluate.py');tree=ast.parse(p.read_text());f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='kernel_predict');ns={'np':np};exec(compile(ast.Module(body=[f],type_ignores=[]),str(p),'exec'),ns);predict=ns['kernel_predict']
class KernelTests(unittest.TestCase):
 def setUp(self):
  r=np.random.default_rng(6);self.X=r.normal(size=(30,12));self.y=r.normal(size=30);self.s=np.arange(10);self.q=np.arange(10,30);self.K=self.X@self.X.T/12
 def test_matches_ridge_with_intercept(self):
  actual=predict(self.K,self.y,self.s,self.q,.1);expected=Ridge(alpha=1.2).fit(self.X[self.s],self.y[self.s]).predict(self.X[self.q]);np.testing.assert_allclose(actual,expected,atol=1e-10)
 def test_query_labels_cannot_change_predictions(self):
  a=predict(self.K,self.y,self.s,self.q,.1);y=self.y.copy();y[self.q]=np.nan;b=predict(self.K,y,self.s,self.q,.1);np.testing.assert_array_equal(a,b)
 def test_residual_offset_and_ridge_match(self):
  base=np.arange(30)/3;actual=predict(self.K,self.y,self.s,self.q,.1,base);expected=Ridge(alpha=1.2).fit(self.X[self.s],self.y[self.s]-base[self.s]).predict(self.X[self.q])+base[self.q];np.testing.assert_allclose(actual,expected,atol=1e-10)
if __name__=='__main__':unittest.main()
