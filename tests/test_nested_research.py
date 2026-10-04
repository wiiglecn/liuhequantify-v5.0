import unittest
from core.nested_signal_research import NestedResearchConfig,evaluate_nested_folds

def make_folds(n=130):
    folds=[]
    c=[1,2,3,4]
    for i in range(n):
        actual=c[i%4]
        a={1:.7,2:.1,3:.1,4:.1}
        b={1:.1,2:.7,3:.1,4:.1}
        if actual==1: a={1:.7,2:.1,3:.1,4:.1}
        elif actual==2: b={1:.1,2:.7,3:.1,4:.1}
        folds.append({"prob":{"a":a,"b":b},"actual":actual})
    return folds,c

class TestV542(unittest.TestCase):
    def test_nested_policy_and_holdout_isolation(self):
        folds,c=make_folds()
        cfg=NestedResearchConfig(initial_train=20,final_holdout=10,inner_min_folds=20,
                                 inner_window=40,max_signals=2,top_k=(1,2))
        r1=evaluate_nested_folds(folds,c,["a","b"],cfg)
        self.assertTrue(r1["holdout_isolated"])
        self.assertEqual(r1["holdout_start"],120)
        self.assertEqual(len(r1["policy_path"]),100)
        changed=[dict(x) for x in folds]
        for f in changed[120:]: f["actual"]=4
        r2=evaluate_nested_folds(changed,c,["a","b"],cfg)
        self.assertEqual(r1["final_policy"],r2["final_policy"])
        self.assertEqual(r1["policy_path"],r2["policy_path"])

    def test_weights_are_simplex(self):
        folds,c=make_folds()
        cfg=NestedResearchConfig(initial_train=20,final_holdout=10,inner_min_folds=20)
        r=evaluate_nested_folds(folds,c,["a","b"],cfg)
        w=r["final_policy"]["weights"]
        self.assertAlmostEqual(sum(w.values()),1.0,places=8)
        self.assertTrue(all(v>=0 for v in w.values()))

if __name__=="__main__":
    unittest.main()
