import unittest

from tools.multi_window_oos_audit import parse_windows, segment_indices, evaluate_segment


class TestMultiWindowOOSAudit(unittest.TestCase):
    def test_parse_windows_sorts_values(self):
        self.assertEqual(parse_windows("300,120,180"), [120, 180, 300])

    def test_parse_windows_rejects_duplicates_and_small_values(self):
        with self.assertRaises(Exception):
            parse_windows("120,120")
        with self.assertRaises(Exception):
            parse_windows("9,30")

    def test_segment_indices_uses_calendar_dates(self):
        dates = [
            "2025-12-31 21:00:00",
            "2026-01-01 21:00:00",
            "2026-04-02 21:00:00",
        ]
        self.assertEqual(segment_indices(dates, "year"), {"2025": [0], "2026": [1, 2]})
        self.assertEqual(segment_indices(dates, "quarter"), {
            "2025-Q4": [0], "2026-Q1": [1], "2026-Q2": [2]
        })

    def test_segment_report_has_random_baseline_and_paired_result(self):
        result = evaluate_segment([1, 0, 1], [0, 0, 1], 0.25)
        self.assertEqual(result["n"], 3)
        self.assertEqual(result["k_specific"]["hits"], 2)
        self.assertIn("paired_difference", result)
        self.assertEqual(result["old_stacking"]["baseline"], 0.25)


if __name__ == "__main__":
    unittest.main()
