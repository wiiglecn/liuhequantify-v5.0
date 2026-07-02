import unittest
import random
from data_fetcher import Record
from analysis import (number_frequency, chi_square_uniform, gap_stats, sum_stats,
                      hot_cold, markov_transition, autocorrelation, build_report,
                      summarize_for_llm)


def mk(expect, special, regular):
    return Record(expect, "t", regular, special)


class TestFrequency(unittest.TestCase):
    def test_number_frequency_counts_all_seven(self):
        recs = [mk("1", 7, [1, 2, 3, 4, 5, 6]), mk("2", 7, [1, 2, 3, 4, 5, 6])]
        freq = number_frequency(recs)
        # 每期 7 个号, 2 期 = 14 次出现; 1-6 各 2 次, 7 出现 2 次
        self.assertEqual(freq[1], 2)
        self.assertEqual(freq[7], 2)
        self.assertEqual(sum(freq.values()), 14)

    def test_chi_square_uniform_uniform_input(self):
        # 完全均匀: 49 个号各出现 10 次 -> chi2 应为 0
        freq = {n: 10 for n in range(1, 50)}
        chi2, p = chi_square_uniform(freq)
        self.assertAlmostEqual(chi2, 0.0, places=6)
        self.assertAlmostEqual(p, 1.0, places=6)


class TestGapAndSum(unittest.TestCase):
    def test_gap_stats(self):
        recs = [mk("1", 7, [1, 2, 3, 4, 5, 6]),
                mk("2", 8, [1, 2, 3, 4, 5, 6]),
                mk("3", 7, [1, 2, 3, 4, 5, 6])]
        gap = gap_stats(recs)
        # 号码 7 出现在第1、3期, 当前遗漏(自最后一次出现)=0, 最大遗漏=1
        self.assertEqual(gap[7]["current"], 0)
        self.assertEqual(gap[7]["max"], 1)
        # 号码 8 只在第2期, 当前遗漏=1
        self.assertEqual(gap[8]["current"], 1)

    def test_sum_stats(self):
        recs = [mk("1", 7, [1, 2, 3, 4, 5, 6]), mk("2", 49, [10, 20, 30, 40, 44, 48])]
        s = sum_stats(recs)
        # 平码和值: 21 与 192
        self.assertAlmostEqual(s["mean"], (21 + 192) / 2)
        self.assertIn("std", s)

    def test_hot_cold(self):
        recs = [mk("1", 7, [1, 1, 1, 1, 1, 1])]  # 1 出现 6 次, 7 出现 1 次
        hot, cold = hot_cold(number_frequency(recs), topn=3)
        self.assertIn(1, hot)
        self.assertIn(49, cold)  # 未出现的号是冷号


class TestMarkovAndAc(unittest.TestCase):
    def test_markov_transition(self):
        recs = [mk("1", 1, [0] * 6), mk("2", 2, [0] * 6), mk("3", 1, [0] * 6)]
        trans = markov_transition(recs)
        # 特码序列 1->2->1, 从 1 转移到 2 出现 1 次
        self.assertTrue(trans[1][2] > 0)

    def test_autocorrelation_constant_series(self):
        # 常数序列自相关应为 0(去均值后分母为0, 返回 0)
        ac = autocorrelation([5, 5, 5, 5, 5], lag=1)
        self.assertEqual(ac, 0.0)

    def test_build_report_returns_dict(self):
        recs = [mk(str(i), (i % 49) + 1, [1, 2, 3, 4, 5, 6]) for i in range(20)]
        rep = build_report(recs)
        self.assertIn("freq", rep)
        self.assertIn("chi2", rep)
        self.assertIn("sum", rep)
        self.assertIn("markov", rep)
        self.assertIn("models", rep)


class TestSummarize(unittest.TestCase):
    def test_summary_contains_models_and_stats(self):
        recs = [mk(str(i), (i % 49) + 1, [1, 2, 3, 4, 5, 6]) for i in range(20)]
        rep = build_report(recs)
        text = summarize_for_llm(rep, recent_n=5)
        self.assertIn("数学模型", text)
        self.assertIn("卡方", text)
        self.assertIn("特码序列", text)


if __name__ == "__main__":
    unittest.main()
