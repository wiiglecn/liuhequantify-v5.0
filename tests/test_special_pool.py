import unittest
import random
from data_fetcher import Record
from analysis import build_report
from special_pool import predict_special_pools, SpecialPool, predict_wide_pool, WidePool
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


class TestWidePool(unittest.TestCase):
    def test_wide_pool_has_exactly_20_numbers(self):
        random.seed(9)
        recs = [mkrec(i) for i in range(60)]
        wp = predict_wide_pool(recs, llm_result=None, backtest_n=10)
        self.assertIsInstance(wp, WidePool)
        self.assertEqual(len(wp.numbers), 20)
        self.assertEqual(len(wp.numbers), len(set(wp.numbers)))  # 唯一
        for n in wp.numbers:
            self.assertIn(n, range(1, 50))
        self.assertAlmostEqual(wp.baseline, 20 / 49, places=6)
        self.assertGreaterEqual(wp.hit_rate, 0.0)
        self.assertLessEqual(wp.hit_rate, 1.0)
        self.assertAlmostEqual(wp.lift, wp.hit_rate - wp.baseline, places=6)

    def test_wide_pool_with_llm_seeds(self):
        # 旧版线性模式(use_stacking=False): LLM 号码作为种子直接加分进入集合
        random.seed(9)
        recs = [mkrec(i) for i in range(60)]
        llm = LLMResult(inferred_models=["马尔可夫"],
                        predicted_set=[3, 12, 18, 25, 33, 41, 7, 14, 22, 30],
                        reasoning="r")
        wp = predict_wide_pool(recs, llm_result=llm, backtest_n=10, use_stacking=False)
        self.assertEqual(len(wp.numbers), 20)
        # LLM 提供的号应作为种子进入集合
        for n in [3, 12, 18]:
            self.assertIn(n, wp.numbers)

    def test_wide_pool_stacking(self):
        # Stacking 模式(use_stacking=True, 默认): GBDT 回归器选号
        random.seed(9)
        recs = [mkrec(i) for i in range(60)]
        wp = predict_wide_pool(recs, backtest_n=10, use_stacking=True)
        self.assertEqual(len(wp.numbers), 20)
        self.assertEqual(len(set(wp.numbers)), 20)  # 唯一
        self.assertTrue(all(1 <= n <= 49 for n in wp.numbers))
        self.assertGreaterEqual(wp.hit_rate, 0.0)
        self.assertLessEqual(wp.hit_rate, 1.0)


if __name__ == "__main__":
    unittest.main()
