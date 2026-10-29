import json
import tempfile
import unittest
from pathlib import Path

from tools.compare_live_k_policy import load_clean_records, paired, summary


class TestKPolicyOOSReport(unittest.TestCase):
    def test_deduplicates_repeated_draw_and_keeps_date_consistent_expect(self):
        row = {
            "expect": "2023004",
            "openTime": "2024-01-04 21:32:32",
            "regular": [39, 36, 28, 49, 12, 24],
            "special": 40,
            "waves": ["绿波", "蓝波"],
            "zodiacs": ["牛", "龍"],
        }
        duplicate = dict(row, expect="2024004")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.json"
            path.write_text(json.dumps([row, duplicate], ensure_ascii=False), encoding="utf-8")
            records, audit = load_clean_records(str(path))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].expect, "2024004")
        self.assertEqual(audit["duplicates_removed"], 1)
        self.assertEqual(audit["duplicate_rows"][0]["duplicate_expect"], "2023004")

    def test_hit_summary_reports_random_baseline_and_interval(self):
        result = summary([1, 0, 1, 0], 0.25)
        self.assertEqual(result["hits"], 2)
        self.assertEqual(result["n"], 4)
        self.assertAlmostEqual(result["hit_rate"], 0.5)
        self.assertEqual(len(result["wilson_95_ci"]), 2)

    def test_paired_comparison_is_deterministic(self):
        first = paired([1, 0, 1, 0], [0, 0, 1, 1])
        second = paired([1, 0, 1, 0], [0, 0, 1, 1])
        self.assertEqual(first, second)
        self.assertEqual(first["new_only_hits"], 1)
        self.assertEqual(first["old_only_hits"], 1)


if __name__ == "__main__":
    unittest.main()
