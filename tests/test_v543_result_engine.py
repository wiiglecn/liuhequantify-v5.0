import unittest
from core.research_result_engine import bootstrap_mean_ci,metric_ci,multiple_testing_from_pvalues

class TestV543(unittest.TestCase):
    def test_bootstrap_is_deterministic(self):
        a=bootstrap_mean_ci([0,1,1,1,0],iterations=200)
        b=bootstrap_mean_ci([0,1,1,1,0],iterations=200)
        self.assertEqual(a,b)
        self.assertLessEqual(a["lower"],a["upper"])
    def test_metric_ci(self):
        rows=[({1:.8,2:.2},1),({1:.2,2:.8},2),({1:.8,2:.2},1),({1:.2,2:.8},2)]
        r=metric_ci(rows,top_k=(1,2),iterations=100)
        self.assertEqual(r["n"],4)
        self.assertIn("bootstrap_ci",r)
        self.assertEqual(r["hit_at_1"],1.0)
    def test_multiple_testing(self):
        r=multiple_testing_from_pvalues({"a":.001,"b":.02,"c":.5})
        self.assertLessEqual(r["bh"]["a"],.01)
        self.assertLessEqual(r["bonferroni"]["a"],1.0)

if __name__=="__main__": unittest.main()
