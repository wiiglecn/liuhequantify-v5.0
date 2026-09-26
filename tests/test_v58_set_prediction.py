import unittest
from core.pair_signal import transition_pair_scores
from core.set_prediction import greedy_set
from core.k_specific_policy import KPolicyConfig,evaluate_k_policies

class V58SetPredictionTest(unittest.TestCase):
    def test_pair_scores_centered(self):
        p=transition_pair_scores([1,2,1,3,2,1],[1,2,3])
        self.assertAlmostEqual(sum(p.values()),0.0,places=8)
        self.assertEqual(set(p),{(1,2),(1,3),(2,3)})
    def test_lambda_zero_is_marginal(self):
        probs={"a":{1:.6,2:.3,3:.1},"b":{1:.5,2:.2,3:.3}}
        self.assertEqual(greedy_set(probs,[1,2,3],2,{},["a","b"],0,0),[1,2])
    def test_holdout_is_frozen(self):
        folds=[]
        for i in range(100):
            folds.append({"actual":1 if i%2==0 else 2,
                          "prob":{"a":{1:.7,2:.1,3:.1,4:.1},"b":{1:.1,2:.7,3:.1,4:.1}}})
        r=evaluate_k_policies(folds,[1,2,3,4],["a","b"],(3,4),
                              KPolicyConfig(windows=(30,60),validation_size=10,min_history=30),
                              30,10)
        self.assertTrue(r["holdout_isolated"])
        self.assertEqual(len(r["holdout_by_k"][3]),10)
        self.assertEqual(len(r["outer_by_k"][4]),60)
if __name__=="__main__": unittest.main()
