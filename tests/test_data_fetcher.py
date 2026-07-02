import unittest
from data_fetcher import parse_record, Record, merge_records, save_cache, load_cache


class TestParseRecord(unittest.TestCase):
    def test_parse_standard_record(self):
        raw = {
            "expect": "2026182",
            "openTime": "2026-07-01 21:32:32",
            "openCode": "15,23,09,45,24,39,41",
            "wave": "blue,red,blue,red,red,green,blue",
            "zodiac": "龍,猴,狗,狗,羊,龍,虎",
        }
        rec = parse_record(raw)
        self.assertIsInstance(rec, Record)
        self.assertEqual(rec.expect, "2026182")
        self.assertEqual(rec.open_time, "2026-07-01 21:32:32")
        self.assertEqual(rec.regular, [15, 23, 9, 45, 24, 39])
        self.assertEqual(rec.special, 41)
        self.assertEqual(len(rec.waves), 7)
        self.assertEqual(len(rec.zodiacs), 7)

    def test_parse_pads_short_opencode(self):
        raw = {"expect": "1", "openTime": "t", "openCode": "5,10",
               "wave": "blue,red", "zodiac": "鼠,牛"}
        rec = parse_record(raw)
        # 不足7个时，末位即特码，且特码不重复计入平码(与7码语义一致)
        self.assertEqual(rec.regular, [5])
        self.assertEqual(rec.special, 10)


class TestMergeAndCache(unittest.TestCase):
    def test_merge_dedup_sort(self):
        a = [Record("2026182", "t1", [1, 2, 3, 4, 5, 6], 7),
             Record("2026180", "t0", [1, 2, 3, 4, 5, 6], 8)]
        b = [Record("2026181", "t1b", [2, 2, 3, 4, 5, 6], 9),
             Record("2026182", "t1new", [3, 3, 3, 4, 5, 6], 10)]  # 重复, 后者覆盖
        merged = merge_records([a, b])
        self.assertEqual([r.expect for r in merged], ["2026180", "2026181", "2026182"])
        self.assertEqual(merged[-1].regular, [3, 3, 3, 4, 5, 6])  # b 覆盖 a

    def test_cache_roundtrip(self):
        import tempfile, os
        d = tempfile.mkdtemp()
        path = os.path.join(d, "c.json")
        recs = [Record("1", "t", [1, 2, 3, 4, 5, 6], 7, ["red"], ["鼠"])]
        save_cache(recs, path)
        loaded = load_cache(path)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].expect, "1")
        self.assertEqual(loaded[0].regular, [1, 2, 3, 4, 5, 6])


if __name__ == "__main__":
    unittest.main()
