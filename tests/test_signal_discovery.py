import unittest
from core.signal_discovery import scalar_actual_scores,adjacent_decay,build_discovery_report

class TestV551Discovery(unittest.TestCase):
    def setUp(self):
        self.rows=[]
        for i in range(120):
            a=i%2
            p={0:0.7,1:0.3} if a==0 else {0:0.3,1:0.7}
            self.rows.append((p,a))

    def test_scalar_actual_score(self):
        self.assertEqual(scalar_actual_scores(self.rows[:2]),[0.7,0.7])

    def test_adjacent_decay_shape(self):
        d=adjacent_decay(self.rows,windows=(30,60))
        self.assertIn("30",d)
        self.assertGreaterEqual(d["30"]["comparisons"],2)

    def test_discovery_registry(self):
        rows={"good":self.rows,
              "redundant":self.rows}
        report=build_discovery_report("unit",rows,[0,1],["2026"]*120)
        self.assertEqual(report["version"],"V5.5.1")
        self.assertEqual(set(report["signals"]),{"good","redundant"})
        self.assertTrue(report["redundancy_groups"])

if __name__=="__main__":
    unittest.main()
