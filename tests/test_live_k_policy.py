# -*- coding: utf-8 -*-
import unittest

from core.live_k_policy import predict_and_backtest_k


class TestLiveKPolicy(unittest.TestCase):
    def test_rejects_insufficient_history(self):
        rows = [
            {"actual": i % 3, "prob": {"s1": {0: .6, 1: .3, 2: .1},
                                       "s2": {0: .2, 1: .5, 2: .3}}}
            for i in range(40)
        ]
        result = predict_and_backtest_k(
            rows, {"s1": {0: .4, 1: .4, 2: .2},
                   "s2": {0: .3, 1: .4, 2: .3}},
            ["s1", "s2"], [0, 1, 2], 1, backtest_n=5
        )
        self.assertIsNone(result)

    def test_rejects_invalid_target_k(self):
        """目标集合大小非法时应安全返回 None，而不是中断预测流程。"""
        rows = [
            {"actual": i % 3, "prob": {"s1": {0: .6, 1: .3, 2: .1}}}
            for i in range(80)
        ]
        result = predict_and_backtest_k(
            rows, {"s1": {0: .4, 1: .4, 2: .2}},
            ["s1"], [0, 1, 2], 4, backtest_n=5
        )
        self.assertIsNone(result)

    def test_rejects_empty_signal_names(self):
        """没有可用信号时应安全返回 None。"""
        rows = [
            {"actual": i % 3, "prob": {"s1": {0: .6, 1: .3, 2: .1}}}
            for i in range(80)
        ]
        result = predict_and_backtest_k(
            rows, {"s1": {0: .4, 1: .4, 2: .2}},
            [], [0, 1, 2], 1, backtest_n=5
        )
        self.assertIsNone(result)

    def test_ignores_rows_missing_a_required_signal(self):
        """历史折缺少必需信号时，不应把它当作有效训练样本。"""
        rows = [
            {"actual": i % 3, "prob": {"s1": {0: .6, 1: .3, 2: .1}}}
            for i in range(80)
        ]
        current = {"s1": {0: .4, 1: .4, 2: .2}, "s2": {0: .3, 1: .4, 2: .3}}
        result = predict_and_backtest_k(
            rows, current, ["s1", "s2"], [0, 1, 2], 1, backtest_n=5
        )
        self.assertIsNone(result)

    def test_returns_valid_top_k_and_oos_hits(self):
        candidates = [0, 1, 2]
        rows = []
        for i in range(85):
            actual = i % 3
            # 概率仅用于构造确定性测试样本，不代表真实开奖存在可预测规律。
            p1 = {c: (0.8 if c == actual else 0.1) for c in candidates}
            p2 = {c: (0.6 if c == (actual + 1) % 3 else 0.2) for c in candidates}
            rows.append({"actual": actual, "prob": {"s1": p1, "s2": p2}})
        current = {"s1": {0: .7, 1: .2, 2: .1},
                   "s2": {0: .2, 1: .6, 2: .2}}
        result = predict_and_backtest_k(
            rows, current, ["s1", "s2"], candidates, 1, backtest_n=2
        )
        self.assertIsNotNone(result)
        self.assertEqual(len(result["ranking"]), 1)
        self.assertEqual(len(set(result["ranking"])), 1)
        self.assertEqual(len(result["hits"]), 2)
        self.assertTrue(all(x in (0, 1) for x in result["hits"]))
        self.assertIn("policy", result)


if __name__ == "__main__":
    unittest.main()
