import unittest

from core.signal_discovery import scalar_actual_scores, adjacent_decay, build_discovery_report


class TestV551Discovery(unittest.TestCase):
    def setUp(self):
        self.rows = []
        for i in range(120):
            actual = i % 2
            confidence = 0.65 + (i % 5) * 0.05
            probabilities = (
                {0: confidence, 1: 1 - confidence}
                if actual == 0
                else {0: 1 - confidence, 1: confidence}
            )
            self.rows.append((probabilities, actual))

    def test_scalar_actual_score(self):
        self.assertAlmostEqual(scalar_actual_scores(self.rows[:2])[0], 0.65)
        self.assertAlmostEqual(scalar_actual_scores(self.rows[:2])[1], 0.70)

    def test_adjacent_decay_shape(self):
        decay = adjacent_decay(self.rows, windows=(30, 60))
        self.assertIn("30", decay)
        self.assertGreaterEqual(decay["30"]["comparisons"], 2)

    def test_discovery_registry(self):
        rows = {"good": self.rows, "redundant": self.rows}
        report = build_discovery_report("unit", rows, [0, 1], ["2026"] * 120)
        self.assertEqual(report["version"], "V5.5.1")
        self.assertEqual(set(report["signals"]), {"good", "redundant"})
        self.assertTrue(report["redundancy_groups"])


if __name__ == "__main__":
    unittest.main()
