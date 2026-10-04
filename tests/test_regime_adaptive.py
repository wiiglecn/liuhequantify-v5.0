import unittest
from core.regime_detection import RegimeDetector,RegimeConfig,fold_features
from core.adaptive_ensemble import fit_adaptive_weights,predict_adaptive
class V56Tests(unittest.TestCase):
    def setUp(self):
        self.c=[1,2,3,4]; self.s=["a","b"]; self.f=[]
        for i in range(20):
            p={n:{x:.25 for x in self.c} for n in self.s}
            p["a"][1]=.7;p["b"][2]=.7
            self.f.append({"prob":p,"actual":1 if i%2==0 else 2})
    def test_features_ignore_actual(self):
        a=fold_features(self.f[0],self.s,self.c); g=dict(self.f[0]);g["actual"]=4
        self.assertEqual(a.tolist(),fold_features(g,self.s,self.c).tolist())
    def test_detector(self):
        d=RegimeDetector(RegimeConfig(n_regimes=2,min_history=2)).fit(self.f,self.s,self.c)
        self.assertIn(d.assign(self.f[0],self.s,self.c),(0,1))
    def test_weights_sum_one(self):
        h=[{"prob":x["prob"],"actual":x["actual"],"regime":0} for x in self.f]
        w=fit_adaptive_weights(h,self.s,self.c,0); self.assertAlmostEqual(sum(w.values()),1.0,8)
        p=predict_adaptive(self.f[0],self.c,w,self.s); self.assertAlmostEqual(sum(p.values()),1.0,8)
if __name__=="__main__": unittest.main()
