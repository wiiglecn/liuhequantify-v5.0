import unittest
from core.residual_alpha import ResidualConfig,discover,_blend

class V583ResidualAlphaTest(unittest.TestCase):
    def test_blend_normalizes(self):
        p=_blend({1:.8,2:.2},{1:.2,2:.8},[1,2],.25)
        self.assertAlmostEqual(sum(p.values()),1.0)
        self.assertGreater(p[1],p[2])
    def test_discovery_rejects_no_gain(self):
        rows=[{"actual":1,"base_prob":{1:.5,2:.5},"prob":{"a":{1:.5,2:.5},"b":{1:.5,2:.5}}} for _ in range(100)]
        chosen,items=discover(rows,["a","b"],[1,2],3,ResidualConfig(validation_size=20,min_gain=.01))
        self.assertIsNone(chosen); self.assertEqual(len(items),2)
    def test_positive_residual_can_be_selected(self):
        rows=[]
        for i in range(120):
            actual=1 if i%2==0 else 2
            rows.append({"actual":actual,"base_prob":{1:.5,2:.5},"prob":{"a":({1:.9,2:.1} if actual==1 else {1:.1,2:.9}),"b":{1:.5,2:.5}}})
        chosen,items=discover(rows,["a","b"],[1,2],3,ResidualConfig(validation_size=20,min_gain=.01,bh_alpha=.2))
        self.assertIsNotNone(chosen); self.assertEqual(chosen.signal,"a")
if __name__=="__main__": unittest.main()
