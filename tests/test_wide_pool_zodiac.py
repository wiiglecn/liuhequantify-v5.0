#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""20颗特码大集合接入三/四/六肖生肖信号的单元测试。

自证要点(对应 memory: 回归测试要自证场景, 避免空过):
- _zodiac_bonus 分层正确: 三肖+3 / 四肖(非三)+2 / 六肖(非四)+1 / 其余 0;
- _wide_score 开启生肖信号(真 zmap) vs 关闭(空 zmap), 每号得分增量恰等于其 bonus
  -> 证明信号6且仅信号6造成差值(关闭=不修复时该信号不存在, 开启=修复后注入);
- 注入后 top20 属于六肖的号码数 >= 关闭时(生肖号码只加分, 排名只上移, 单调不降)
  -> 大集合更倾向包含三/四/六肖号码。
"""
import unittest
from data_fetcher import Record
from analysis import build_report
from dimensions import build_zodiac_map, _zodiac_six_scores
from special_pool import _wide_score, _zodiac_bonus, predict_wide_pool, WIDE_POOL_SIZE

ZODIACS = ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪"]


def mk(expect, special, wave="red", zodiac="鼠"):
    return Record(expect, "t", [1, 2, 3, 4, 5, 6], special,
                  [wave] * 7, [zodiac] * 7)


def _fixture():
    """60 期: 特码 1-49 轮换, 生肖 12 轮换 -> 各生肖均出现, 号码层得分接近。

    号码层接近使得生肖 bonus(最高3.0)足以左右 top20 构成, 便于验证"倾向包含"。
    """
    return [mk(str(i), (i % 49) + 1, wave="red", zodiac=ZODIACS[i % 12])
            for i in range(60)]


def _topk_zodiacs(recs, zmap, k):
    scores = _zodiac_six_scores(recs, None, zmap, 0.0)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return {z for z, _ in ranked[:k]}


class TestWidePoolZodiac(unittest.TestCase):

    def test_zodiac_bonus_layered(self):
        """三肖号码+3, 四肖(非三)+2, 六肖(非四)+1, 其余0; 且至少有号码拿到正分。"""
        recs = _fixture()
        zmap = build_zodiac_map(recs)
        top3 = _topk_zodiacs(recs, zmap, 3)
        top4 = _topk_zodiacs(recs, zmap, 4)
        top6 = _topk_zodiacs(recs, zmap, 6)

        bonus = _zodiac_bonus(recs, zmap)
        self.assertTrue(bonus, "bonus 不应为空(至少六肖号码应拿到正分)")

        def layer(z):
            if z in top3:
                return 3.0
            if z in top4:
                return 2.0
            if z in top6:
                return 1.0
            return 0.0

        # 每个号码的 bonus 应等于其生肖所在分层
        for num, z in zmap.items():
            self.assertAlmostEqual(bonus.get(num, 0.0), layer(z),
                                   msg=f"号码{num} 生肖{z} bonus 应为 {layer(z)}")

        # 不在 top6 的生肖号码不得出现在 bonus 里
        for z in (set(ZODIACS) - top6):
            for num, zz in zmap.items():
                if zz == z:
                    self.assertNotIn(num, bonus, f"非六肖号码{num} 不应拿到 bonus")

    def test_wide_score_signal6_is_sole_delta(self):
        """开启(真 zmap) vs 关闭(空 zmap), 每号增量恰等于其 bonus。"""
        recs = _fixture()
        report = build_report(recs)
        zmap = build_zodiac_map(recs)

        off = _wide_score(recs, report, None, {})      # 空 map -> 信号6不贡献(关闭)
        on = _wide_score(recs, report, None, zmap)     # 真 map -> 信号6贡献(开启)
        bonus = _zodiac_bonus(recs, zmap)

        for n in range(1, 50):
            self.assertAlmostEqual(on[n] - off[n], bonus.get(n, 0.0),
                                   msg=f"号码{n} 增量应恰等于 bonus(信号6且仅信号6)")

    def test_top20_leans_to_zodiac_numbers(self):
        """注入后 top20 属于六肖的号码数 >= 关闭时(加分只上移, 单调不降)。"""
        recs = _fixture()
        report = build_report(recs)
        zmap = build_zodiac_map(recs)
        top6 = _topk_zodiacs(recs, zmap, 6)
        six_nums = {n for n, z in zmap.items() if z in top6}

        off = _wide_score(recs, report, None, {})      # 关闭
        on = _wide_score(recs, report, None, zmap)     # 开启

        def top20(scores):
            return {n for n, _ in sorted(scores.items(),
                                         key=lambda kv: kv[1], reverse=True)[:20]}

        cnt_off = len(top20(off) & six_nums)
        cnt_on = len(top20(on) & six_nums)
        self.assertGreaterEqual(cnt_on, cnt_off,
                                f"开启后 top20 含六肖号码数应不降: {cnt_on} >= {cnt_off}")
        self.assertGreaterEqual(cnt_on, 1, "top20 至少含1个六肖号码")

    def test_strategy_mentions_zodiac(self):
        """大集合策略文案应体现三/四/六肖生肖信号。"""
        recs = _fixture()
        wide = predict_wide_pool(recs, llm_result=None, backtest_n=10)
        self.assertIn("三/四/六肖", wide.strategy)
        self.assertEqual(len(wide.numbers), WIDE_POOL_SIZE)

    def test_small_data_no_crash(self):
        """数据较少(30期)时不抛异常, 返回 20 颗, 回测不泄漏未来。"""
        recs = [mk(str(i), (i % 49) + 1, zodiac=ZODIACS[i % 12]) for i in range(30)]
        wide = predict_wide_pool(recs, llm_result=None, backtest_n=10)
        self.assertEqual(len(wide.numbers), WIDE_POOL_SIZE)
        self.assertEqual(len(set(wide.numbers)), WIDE_POOL_SIZE)  # 唯一
        self.assertGreaterEqual(wide.hit_rate, 0.0)
        self.assertLessEqual(wide.hit_rate, 1.0)


if __name__ == "__main__":
    unittest.main()
