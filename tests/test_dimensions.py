import unittest
import random
from data_fetcher import Record
from dimensions import (predict_dimensions, DimensionPrediction,
                        wave_of, zodiac_of, tail_of, big_small_of, odd_even_of,
                        head_of, predict_zodiac_pool, ZodiacPool, special_zodiac_of)


def mk(expect, special, wave="red", zodiac="鼠"):
    return Record(expect, "t", [1, 2, 3, 4, 5, 6], special,
                  [wave] * 7, [zodiac] * 7)


class TestExtractors(unittest.TestCase):
    def test_extractors(self):
        r = mk("1", 41, wave="blue", zodiac="虎")
        self.assertEqual(wave_of(r), "blue")
        self.assertEqual(zodiac_of(r), "虎")
        self.assertEqual(tail_of(r), 1)         # 41 -> 1
        self.assertEqual(big_small_of(r), "大")  # 41>=25
        self.assertEqual(odd_even_of(r), "奇")   # 41 奇
        self.assertEqual(head_of(r), 4)          # 41 -> 4

    def test_tail_of_single_digit(self):
        self.assertEqual(tail_of(mk("1", 7)), 7)
        self.assertEqual(head_of(mk("1", 7)), 0)


class TestPredictDimensions(unittest.TestCase):
    def test_returns_all_six_dimensions(self):
        random.seed(3)
        recs = [mk(str(i), (i * 3 % 49) + 1,
                   wave=["red", "green", "blue"][i % 3],
                   zodiac=["鼠", "牛", "虎", "兔"][i % 4]) for i in range(60)]
        dims = predict_dimensions(recs, llm_result=None, backtest_n=10)
        names = [d.name for d in dims]
        for expected in ["波色", "生肖", "尾数", "大小", "奇偶", "头数"]:
            self.assertIn(expected, names)
        self.assertEqual(len(dims), 6)
        for d in dims:
            self.assertIsInstance(d, DimensionPrediction)
            self.assertTrue(d.value)  # 非空预测
            self.assertGreaterEqual(d.accuracy, 0.0)
            self.assertLessEqual(d.accuracy, 1.0)
            self.assertGreater(d.baseline, 0.0)
            self.assertLessEqual(d.baseline, 1.0)
            self.assertAlmostEqual(d.lift, d.accuracy - d.baseline, places=6)

    def test_random_baseline_correct(self):
        random.seed(3)
        recs = [mk(str(i), (i % 49) + 1) for i in range(60)]
        dims = predict_dimensions(recs, llm_result=None, backtest_n=10)
        by_name = {d.name: d for d in dims}
        self.assertAlmostEqual(by_name["波色"].baseline, 1 / 3)
        self.assertAlmostEqual(by_name["生肖"].baseline, 1 / 12)
        self.assertAlmostEqual(by_name["尾数"].baseline, 1 / 10)
        self.assertAlmostEqual(by_name["大小"].baseline, 1 / 2)
        self.assertAlmostEqual(by_name["奇偶"].baseline, 1 / 2)
        self.assertAlmostEqual(by_name["头数"].baseline, 1 / 5)


class TestZodiacPool(unittest.TestCase):
    def test_returns_three_zodiacs(self):
        random.seed(8)
        zodiacs_cycle = ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪"]
        recs = [mk(str(i), (i % 49) + 1, zodiac=zodiacs_cycle[i % 12]) for i in range(60)]
        zp = predict_zodiac_pool(recs, llm_result=None, backtest_n=10)
        self.assertIsInstance(zp, ZodiacPool)
        self.assertEqual(len(zp.zodiacs), 3)
        self.assertEqual(len(zp.zodiacs), len(set(zp.zodiacs)))  # 唯一
        self.assertAlmostEqual(zp.baseline, 3 / 12)
        self.assertGreaterEqual(zp.hit_rate, 0.0)
        self.assertLessEqual(zp.hit_rate, 1.0)
        self.assertAlmostEqual(zp.lift, zp.hit_rate - zp.baseline, places=6)

    def test_special_zodiac_of(self):
        r = mk("1", 41, zodiac="虎")
        self.assertEqual(special_zodiac_of(r), "虎")


if __name__ == "__main__":
    unittest.main()
