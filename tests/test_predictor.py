import unittest
import random
from data_fetcher import Record
from analysis import build_report
from predictor import (weighted_sample, sample_with_sum_constraint,
                       predict_group_a, predict_group_b, predict_group_c,
                       backtest, predict_all)
from llm_reasoner import LLMResult


def mkrec(i):
    return Record(str(i), "t", [(i % 49) + 1] * 6, (i * 7 % 49) + 1)


class TestSampling(unittest.TestCase):
    def test_weighted_sample_distinct_six(self):
        random.seed(42)
        weights = {n: 1.0 for n in range(1, 50)}
        weights[7] = 100.0  # 极高权重
        picked = weighted_sample(weights, k=6)
        self.assertEqual(len(picked), 6)
        self.assertEqual(len(set(picked)), 6)
        self.assertIn(7, picked)

    def test_sample_with_sum_constraint(self):
        random.seed(1)
        weights = {n: 1.0 for n in range(1, 50)}
        picked = sample_with_sum_constraint(weights, mean=150, std=30, k=6)
        self.assertEqual(len(set(picked)), 6)
        self.assertTrue(150 - 60 <= sum(picked) <= 150 + 60)


class TestThreeGroups(unittest.TestCase):
    def setUp(self):
        random.seed(123)
        self.recs = [mkrec(i) for i in range(60)]
        self.report = build_report(self.recs)

    def test_group_a_valid(self):
        g = predict_group_a(self.recs, self.report)
        self.assertEqual(len(g.regular), 6)
        self.assertEqual(len(set(g.regular)), 6)
        self.assertIn("频率", g.strategy)

    def test_group_b_valid(self):
        g = predict_group_b(self.recs, self.report)
        self.assertEqual(len(set(g.regular)), 6)
        self.assertIn("马尔可夫", g.strategy)

    def test_group_c_without_llm(self):
        g = predict_group_c(self.recs, self.report, llm_result=None)
        self.assertEqual(len(set(g.regular)), 6)
        self.assertIn("遗漏", g.strategy)

    def test_group_c_with_llm(self):
        llm = LLMResult(inferred_models=["马尔可夫"],
                        predicted_set=[3, 12, 18, 25, 33, 41, 7], reasoning="r")
        g = predict_group_c(self.recs, self.report, llm_result=llm)
        self.assertEqual(len(set(g.regular)), 6)


class TestBacktest(unittest.TestCase):
    def test_backtest_returns_rates(self):
        random.seed(7)
        recs = [mkrec(i) for i in range(60)]
        groups = predict_all(recs, llm_result=None, backtest_n=10)
        self.assertEqual(len(groups), 3)
        for g in groups:
            self.assertGreaterEqual(g.backtest_special_hit, 0.0)
            self.assertLessEqual(g.backtest_special_hit, 1.0)
            self.assertGreaterEqual(g.backtest_regular_hits, 0.0)

    def test_predict_all_sorts_by_hitrate(self):
        random.seed(7)
        recs = [mkrec(i) for i in range(60)]
        groups = predict_all(recs, llm_result=None, backtest_n=10)
        # 推荐组(第1个)综合命中率应 >= 其他
        rates = [g.backtest_special_hit + g.backtest_regular_hits / 10 for g in groups]
        self.assertGreaterEqual(rates[0], rates[-1])


if __name__ == "__main__":
    unittest.main()
