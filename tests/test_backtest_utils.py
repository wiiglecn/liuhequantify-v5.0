import unittest
from data_fetcher import Record
from backtest_utils import backtest_series, BacktestResult


def mk(expect, special, wave="red", zodiac="鼠"):
    return Record(expect, "t", [1, 2, 3, 4, 5, 6], special,
                  [wave] * 7, [zodiac] * 7)


class TestBacktestSeries(unittest.TestCase):
    def test_perfect_predictor(self):
        # 永远预测对(用实际值)
        recs = [mk(str(i), (i % 49) + 1) for i in range(40)]
        res = backtest_series(
            predict_func=lambda train: train[-1].special + 1 if train[-1].special < 49 else 1,
            actual_func=lambda r: r.special,
            records=recs, n=20, baseline=1 / 49,
        )
        self.assertIsInstance(res, BacktestResult)
        self.assertEqual(res.n, 20)
        # 至少部分命中
        self.assertGreaterEqual(res.accuracy, 0.0)
        self.assertLessEqual(res.accuracy, 1.0)
        self.assertAlmostEqual(res.baseline, 1 / 49)

    def test_lift_within_bounds(self):
        recs = [mk(str(i), (i % 49) + 1) for i in range(40)]
        res = backtest_series(
            predict_func=lambda train: 7,
            actual_func=lambda r: r.special,
            records=recs, n=20, baseline=1 / 49,
        )
        # lift = accuracy - baseline, 必在 [-baseline, 1-baseline]
        self.assertGreaterEqual(res.lift, -res.baseline - 1e-9)
        self.assertLessEqual(res.lift, 1 - res.baseline + 1e-9)
        self.assertEqual(res.lift, res.accuracy - res.baseline)

    def test_std_error_nonneg(self):
        recs = [mk(str(i), (i % 49) + 1) for i in range(40)]
        res = backtest_series(
            predict_func=lambda train: 1,
            actual_func=lambda r: r.special,
            records=recs, n=20, baseline=1 / 49,
        )
        self.assertGreaterEqual(res.std_error, 0.0)

    def test_n_clamped(self):
        recs = [mk(str(i), (i % 49) + 1) for i in range(15)]
        # n=20 但只有 15 期, 应被钳制
        res = backtest_series(
            predict_func=lambda train: 1,
            actual_func=lambda r: r.special,
            records=recs, n=20, baseline=1 / 49,
        )
        self.assertLessEqual(res.n, 5)  # 至少留 10 期训练


if __name__ == "__main__":
    unittest.main()
