import unittest
from core.residual_features import extract_signal_features,FEATURE_NAMES
class V584Test(unittest.TestCase):
    def test_feature_shape(self):
        f={"prob":{"a":{1:.7,2:.3},"b":{1:.4,2:.6}}}
        x=extract_signal_features(f,["a","b"],[1,2])
        self.assertEqual(set(x),set(FEATURE_NAMES))
        self.assertTrue(all(isinstance(v,float) for v in x.values()))
if __name__=="__main__":unittest.main()
