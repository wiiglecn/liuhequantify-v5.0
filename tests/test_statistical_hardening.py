import unittest
import numpy as np
from core.statistical_hardening import HardeningConfig,rolling_oos,discover

class V5841Test(unittest.TestCase):
    def _rows(self,n=240):
        rng=np.random.default_rng(7)
        rows=[]
        for i in range(n):
            x=float(rng.normal())
            base=.55
            hit=int(rng.random() < (0.72 if x>0 else 0.38))
            rows.append({"feature":x,"base_prob":base,"hit":hit})
        return rows

    def test_rolling_oos_is_frozen(self):
        rows=self._rows()
        cfg=HardeningConfig(min_train=120,validation_size=20,train_window=120,
                            bootstrap_rounds=50,min_blocks=3)
        gains,origins=rolling_oos(rows,"feature",cfg)
        self.assertEqual(len(origins),6)
        self.assertEqual(len(gains),120)
        self.assertTrue(np.isfinite(gains).all())

    def test_discovery_schema_and_multiple_testing(self):
        rows=self._rows()
        cfg=HardeningConfig(min_train=120,validation_size=20,train_window=120,
                            bootstrap_rounds=50,min_blocks=3)
        selected,items,_=discover(rows,["feature"],6,cfg)
        self.assertEqual(len(items),1)
        x=items[0]
        self.assertGreaterEqual(x.ci_high,x.ci_low)
        self.assertGreaterEqual(x.positive_fraction,0.0)
        self.assertLessEqual(x.positive_fraction,1.0)

if __name__=="__main__":
    unittest.main()
