import unittest
from core.v57_meta_policy import HorizonConfig,fit_meta_policy,predict_meta,evaluate_meta

class V57MetaPolicyTest(unittest.TestCase):
    def _fold(self,actual,probs,regime=0):
        return {"actual":actual,"prob":probs,"regime":regime}
    def test_policy_uses_only_history(self):
        names=["a","b"]; c=[1,2,3,4]
        rows=[]
        for i in range(80):
            rows.append(self._fold(1 if i%4==0 else 2,{
                "a":{1:.7,2:.1,3:.1,4:.1},
                "b":{1:.1,2:.7,3:.1,4:.1}}))
        p=fit_meta_policy(rows,names,c,0,3,HorizonConfig(windows=(30,60),min_history=60))
        self.assertIn(p.horizon,(30,60))
        self.assertAlmostEqual(sum(p.weights.values()),1.0,places=8)
    def test_dynamic_regime_labels_are_recomputed_per_target(self):
        names=["a","b"]; c=[1,2,3,4]
        folds=[]
        for i in range(90):
            if i < 45:
                pa={1:.7,2:.1,3:.1,4:.1}; pb={1:.1,2:.7,3:.1,4:.1}
            else:
                pa={1:.1,2:.7,3:.1,4:.1}; pb={1:.7,2:.1,3:.1,4:.1}
            folds.append({"actual":1 if i%2==0 else 2,"prob":{"a":pa,"b":pb}})
        res=evaluate_meta(
            folds,c,names,None,(3,),
            HorizonConfig(windows=(30,60),min_history=60),
            initial_train=60,final_holdout=10)
        self.assertTrue(res["holdout_isolated"])
        self.assertEqual(len(res["outer_by_k"][3]),20)
        self.assertEqual(len(res["holdout_by_k"][3]),10)
        self.assertEqual(len(res["regime_path"]),20)

    def test_prediction_normalized(self):
        c=[1,2,3]; names=["a","b"]
        fold={"prob":{"a":{1:.6,2:.3,3:.1},"b":{1:.2,2:.3,3:.5}}}
        p=fit_meta_policy([self._fold(1,fold["prob"]) for _ in range(60)],names,c,0,3,
                          HorizonConfig(windows=(30,),min_history=60))
        out=predict_meta(fold,c,p,names)
        self.assertAlmostEqual(sum(out.values()),1.0,places=8)
        self.assertEqual(set(out),set(c))

if __name__=="__main__":
    unittest.main()
