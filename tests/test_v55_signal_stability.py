import unittest
from core.signal_stability import rolling_hit_rates,signal_decay,residual_alpha,classify_stability
class TestV55(unittest.TestCase):
 def rows(self,n=120):
  return [({1:.8,2:.2},1) if i%2==0 else ({1:.2,2:.8},2) for i in range(n)]
 def test_rolling(self):
  r=rolling_hit_rates(self.rows(),windows=(30,),k=1);self.assertEqual(r["30"]["last"],1.0)
 def test_decay(self):
  r=signal_decay(self.rows(),windows=(30,));self.assertAlmostEqual(r["30"]["delta"],0.0)
 def test_residual(self):
  r=residual_alpha([1,2,3,4],[1,2,3,4],[0,0,1,1]);self.assertEqual(r["n"],4)
 def test_labels(self):
  s={"rolling":{"hit_at_1":{"30":{"n_windows":5,"min":.8,"last":.9,"mean":.9}}}}
  self.assertEqual(classify_stability(s)["hit_at_1"]["30"],"stable")
if __name__=="__main__":unittest.main()
