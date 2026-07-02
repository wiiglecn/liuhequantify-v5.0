import unittest
import random
from data_fetcher import Record
from analysis import build_report
from special_pool import predict_special_pools, SpecialPool
from llm_reasoner import LLMResult


def mkrec(i):
    return Record(str(i), "t", [(i % 49) + 1] * 6, (i * 7 % 49) + 1)


class TestSpecialPools(unittest.TestCase):
    def test_returns_pools_in_range(self):
        random.seed(5)
        recs = [mkrec(i) for i in range(60)]
        pools = predict_special_pools(recs, llm_result=None, backtest_n=10)
        self.assertGreaterEqual(len(pools), 1)
        for p in pools:
            self.assertIsInstance(p, SpecialPool)
            self.assertGreaterEqual(len(p.numbers), 8)
            self.assertLessEqual(len(p.numbers), 10)
            # 号码唯一且在 1-49
            self.assertEqual(len(p.numbers), len(set(p.numbers)))
            for n in p.numbers:
                self.assertIn(n, range(1, 50))
            self.assertGreaterEqual(p.hit_rate, 0.0)
            self.assertLessEqual(p.hit_rate, 1.0)
            self.assertGreater(p.baseline, 0.0)
            self.assertLess(p.baseline, 1.0)

    def test_pools_sorted_desc(self):
        random.seed(5)
        recs = [mkrec(i) for i in range(60)]
        pools = predict_special_pools(recs, llm_result=None, backtest_n=10)
        rates = [p.hit_rate for p in pools]
        self.assertEqual(rates, sorted(rates, reverse=True))

    def test_baseline_proportional_to_size(self):
        random.seed(5)
        recs = [mkrec(i) for i in range(60)]
        pools = predict_special_pools(recs, llm_result=None, backtest_n=10)
        for p in pools:
            self.assertAlmostEqual(p.baseline, len(p.numbers) / 49, places=6)

    def test_with_llm_includes_llm_pool(self):
        random.seed(5)
        recs = [mkrec(i) for i in range(60)]
        llm = LLMResult(inferred_models=["马尔可夫"],
                        predicted_set=[3, 12, 18, 25, 33, 41, 7, 14, 22, 30, 38],
                        reasoning="r")
        pools = predict_special_pools(recs, llm_result=llm, backtest_n=10)
        llm_pool = [p for p in pools if "大模型" in p.strategy]
        self.assertTrue(llm_pool)
        # LLM 号码应在集合中
        self.assertIn(7, llm_pool[0].numbers)


if __name__ == "__main__":
    unittest.main()
