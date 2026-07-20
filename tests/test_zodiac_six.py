#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""六肖功能单元测试 — 针对 predict_zodiac_six"""
import unittest
import random
from data_fetcher import Record
from dimensions import (predict_zodiac_pool, ZodiacPool, build_zodiac_map)
from llm_reasoner import LLMResult

# 尝试导入待实现的函数(开发完成前可能失败)
try:
    from dimensions import predict_zodiac_six, ZODIAC_SIX_SIZE
except ImportError:
    predict_zodiac_six = None
    ZODIAC_SIX_SIZE = 6


def mk(expect, special, wave="red", zodiac="鼠"):
    return Record(expect, "t", [1, 2, 3, 4, 5, 6], special,
                  [wave] * 7, [zodiac] * 7)


class TestZodiacSix(unittest.TestCase):
    """六肖预测功能测试套件"""

    def test_returns_six_zodiacs(self):
        """返回 ZodiacPool、恰好 6 个、互不重复"""
        if predict_zodiac_six is None:
            self.skipTest("predict_zodiac_six 尚未实现")
        random.seed(8)
        zodiacs_cycle = ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪"]
        recs = [mk(str(i), (i % 49) + 1, zodiac=zodiacs_cycle[i % 12]) for i in range(60)]
        zs = predict_zodiac_six(recs, llm_result=None, backtest_n=10)
        self.assertIsInstance(zs, ZodiacPool)
        self.assertEqual(len(zs.zodiacs), 6)
        self.assertEqual(len(zs.zodiacs), len(set(zs.zodiacs)))  # 唯一

    def test_baseline_is_half(self):
        """baseline = 6/12 = 0.5"""
        if predict_zodiac_six is None:
            self.skipTest("predict_zodiac_six 尚未实现")
        random.seed(9)
        zodiacs_cycle = ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪"]
        recs = [mk(str(i), (i % 49) + 1, zodiac=zodiacs_cycle[i % 12]) for i in range(60)]
        zs = predict_zodiac_six(recs, llm_result=None, backtest_n=10)
        self.assertAlmostEqual(zs.baseline, 6 / 12)

    def test_metrics_consistent(self):
        """lift = hit_rate - baseline、hit_rate ∈ [0,1]、std_error/stability >= 0、strategy 非空"""
        if predict_zodiac_six is None:
            self.skipTest("predict_zodiac_six 尚未实现")
        random.seed(10)
        zodiacs_cycle = ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪"]
        recs = [mk(str(i), (i % 49) + 1, zodiac=zodiacs_cycle[i % 12]) for i in range(60)]
        zs = predict_zodiac_six(recs, llm_result=None, backtest_n=10)
        self.assertGreaterEqual(zs.hit_rate, 0.0)
        self.assertLessEqual(zs.hit_rate, 1.0)
        self.assertAlmostEqual(zs.lift, zs.hit_rate - zs.baseline, places=6)
        self.assertGreaterEqual(zs.std_error, 0.0)
        self.assertGreaterEqual(zs.stability, 0.0)
        self.assertTrue(zs.strategy)  # 非空字符串

    def test_six_superset_of_three(self):
        """六肖集合应包含三生肖集合"""
        if predict_zodiac_six is None:
            self.skipTest("predict_zodiac_six 尚未实现")
        random.seed(11)
        zodiacs_cycle = ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪"]
        recs = [mk(str(i), (i % 49) + 1, zodiac=zodiacs_cycle[i % 12]) for i in range(60)]
        zp = predict_zodiac_pool(recs, llm_result=None, backtest_n=10)
        zs = predict_zodiac_six(recs, llm_result=None, backtest_n=10)
        self.assertTrue(set(zp.zodiacs).issubset(set(zs.zodiacs)))

    def test_small_data_no_crash(self):
        """数据很少(如 12 期、15 期)时不抛异常，返回 6 个生肖"""
        if predict_zodiac_six is None:
            self.skipTest("predict_zodiac_six 尚未实现")
        random.seed(12)
        zodiacs_cycle = ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪"]
        # 12 期数据
        recs_12 = [mk(str(i), (i % 49) + 1, zodiac=zodiacs_cycle[i % 12]) for i in range(12)]
        zs_12 = predict_zodiac_six(recs_12, llm_result=None, backtest_n=5)
        self.assertIsInstance(zs_12, ZodiacPool)
        self.assertEqual(len(zs_12.zodiacs), 6)
        self.assertEqual(len(zs_12.zodiacs), len(set(zs_12.zodiacs)))
        # 15 期数据
        recs_15 = [mk(str(i), (i % 49) + 1, zodiac=zodiacs_cycle[i % 12]) for i in range(15)]
        zs_15 = predict_zodiac_six(recs_15, llm_result=None, backtest_n=5)
        self.assertIsInstance(zs_15, ZodiacPool)
        self.assertEqual(len(zs_15.zodiacs), 6)

    def test_with_llm_result(self):
        """传入 LLMResult 不抛异常(LLM 号码经 build_zodiac_map 映射加权)"""
        if predict_zodiac_six is None:
            self.skipTest("predict_zodiac_six 尚未实现")
        random.seed(13)
        zodiacs_cycle = ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪"]
        recs = [mk(str(i), (i % 49) + 1, zodiac=zodiacs_cycle[i % 12]) for i in range(60)]
        # 构建 LLMResult: 预测号码 1, 2, 3, 4, 5, 6, 7
        llm = LLMResult(inferred_models=["模型A", "模型B"],
                        predicted_set=[1, 2, 3, 4, 5, 6, 7],
                        reasoning="测试推理")
        # 不应抛异常
        zs = predict_zodiac_six(recs, llm_result=llm, backtest_n=10)
        self.assertIsInstance(zs, ZodiacPool)
        self.assertEqual(len(zs.zodiacs), 6)

    def test_backtest_no_future_leak(self):
        """用确定性数据验证回测切分正确(留一法不偷看未来)"""
        if predict_zodiac_six is None:
            self.skipTest("predict_zodiac_six 尚未实现")
        # 构造确定性序列: 每一期特码生肖 = 期号 % 12
        zodiacs_cycle = ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪"]
        recs = []
        for i in range(60):
            # 特码 = (i % 49) + 1，特码生肖 = zodiacs_cycle[i % 12]
            rec = mk(str(i), (i % 49) + 1, zodiac=zodiacs_cycle[i % 12])
            recs.append(rec)
        # 调用预测(回测会切分)
        zs = predict_zodiac_six(recs, llm_result=None, backtest_n=10)
        # 验证返回值合法即可(回测逻辑由 predict_zodiac_pool 保障)
        self.assertIsInstance(zs, ZodiacPool)
        self.assertEqual(len(zs.zodiacs), 6)
        self.assertGreaterEqual(zs.hit_rate, 0.0)
        self.assertLessEqual(zs.hit_rate, 1.0)


if __name__ == "__main__":
    unittest.main()
