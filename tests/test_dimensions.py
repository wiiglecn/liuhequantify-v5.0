import unittest
import random
from data_fetcher import Record
from dimensions import (predict_dimensions, DimensionPrediction,
                        wave_of, zodiac_of, tail_of, big_small_of, odd_even_of,
                        head_of, predict_zodiac_pool, predict_zodiac_quad,
                        ZodiacPool, special_zodiac_of,
                        wave_of_number, build_zodiac_map, zodiac_of_number,
                        _joint_predict_number, comb_prior, normalize_zodiac,
                        _dim_predict_value, DIM_CONFIG)


def mk(expect, special, wave="red", zodiac="鼠"):
    return Record(expect, "t", [1, 2, 3, 4, 5, 6], special,
                  [wave] * 7, [zodiac] * 7)


def mk_rec(expect, special, zodiac="鼠"):
    """special 的波色字段与其固定波色保持一致, 避免波色维度错配(影响联合打分)。"""
    return mk(expect, special, wave=wave_of_number(special) or "red", zodiac=zodiac)


class TestExtractors(unittest.TestCase):
    def test_extractors(self):
        r = mk("1", 41, wave="blue", zodiac="虎")
        self.assertEqual(wave_of(r), "蓝波")
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

    def test_exact_combinatorial_baseline(self):
        """基线=精确组合概率(非 1/N 近似): 波色17/16/16, 尾0为4, 大25/小24等。"""
        self.assertEqual(comb_prior("波色"), {"红波": 17/49, "蓝波": 16/49, "绿波": 16/49})
        self.assertEqual(comb_prior("尾数")[0], 4/49)
        self.assertEqual(comb_prior("尾数")[5], 5/49)
        self.assertEqual(comb_prior("大小"), {"大": 25/49, "小": 24/49})
        self.assertEqual(comb_prior("奇偶"), {"奇": 25/49, "偶": 24/49})
        self.assertEqual(comb_prior("头数")[0], 9/49)
        self.assertEqual(comb_prior("头数")[3], 10/49)
        self.assertEqual(sum(comb_prior("尾数").values()), 1.0)
        # 生肖先验按当前映射号码数(胖5/瘦4)
        zm = {1: "鼠", 2: "鼠", 3: "鼠", 4: "鼠", 5: "鼠", 6: "牛", 7: "牛", 8: "牛", 9: "牛"}
        self.assertEqual(comb_prior("生肖", zm), {"鼠": 5/49, "牛": 4/49})
        self.assertEqual(comb_prior("生肖", {}), {})

    def test_baseline_matches_predicted_value(self):
        """各维回测基线 = 预测值的精确组合概率(口径一致, 全维度覆盖)。"""
        random.seed(3)
        recs = [mk(str(i), (i % 49) + 1) for i in range(60)]
        dims = predict_dimensions(recs, llm_result=None, backtest_n=10)
        by_name = {d.name: d for d in dims}
        self.assertEqual(by_name["波色"].value, "红波")           # 恒选最大类
        self.assertAlmostEqual(by_name["波色"].baseline, 17 / 49)
        self.assertEqual(by_name["头数"].value, "1")              # 头1-4 等先验, 确定性取头1
        self.assertAlmostEqual(by_name["头数"].baseline, 10 / 49)
        # 动态维: 基线必须落在预测值可能的精确概率集合内
        self.assertIn(by_name["尾数"].value, [str(t) for t in range(10)])
        self.assertGreaterEqual(by_name["尾数"].baseline, 4 / 49 - 1e-9)
        self.assertLessEqual(by_name["尾数"].baseline, 5 / 49 + 1e-9)
        for nm in ("大小", "奇偶"):
            self.assertGreaterEqual(by_name[nm].baseline, 24 / 49 - 1e-9)
            self.assertLessEqual(by_name[nm].baseline, 25 / 49 + 1e-9)
        # 生肖: mk 数据全 "鼠" 且特码轮转覆盖 1-49 -> 映射 {全部49号: 鼠}, 预测=鼠, 基线=49/49=1.0
        self.assertEqual(by_name["生肖"].value, "鼠")
        self.assertAlmostEqual(by_name["生肖"].baseline, 1.0)
        # 全维 lift 与 accuracy-baseline 自洽
        for d in dims:
            self.assertAlmostEqual(d.lift, d.accuracy - d.baseline, places=6)


class TestPredictionRespondsToRecentData(unittest.TestCase):
    """修复"期期预测一样": 先验平局维度(尾数)须随近期走势翻动(自证: 前后段预测不同)。"""

    def test_tail_follows_recent_shift(self):
        nums = [1, 3, 4, 8, 11, 14, 17, 19, 30, 33, 37, 40, 44, 47, 48, 49]
        zods = ["鼠", "牛", "虎", "兔", "龍", "蛇", "猴", "雞",
                "狗", "豬", "鼠", "牛", "虎", "兔", "龍", "蛇"]
        base = [mk_rec(str(i), n, z) for i, (n, z) in enumerate(zip(nums, zods), 1)]
        hot5 = base + [mk_rec(str(100 + i), 25, "馬") for i in range(6)]   # 近段尾5热
        v1 = _dim_predict_value(hot5, tail_of, comb_prior("尾数"), DIM_CONFIG["尾数"])
        hot8 = hot5 + [mk_rec(str(200 + i), n, "猴")
                       for i, n in enumerate([8, 18, 28, 38, 48, 8, 18])]   # 更近段尾8热
        v2 = _dim_predict_value(hot8, tail_of, comb_prior("尾数"), DIM_CONFIG["尾数"])
        self.assertEqual(v1, 5)   # 自证: 前段确实预测尾5
        self.assertEqual(v2, 8)   # 后段翻到尾8 -> 预测随期变动, 不再卡死


class TestZodiacTraditional(unittest.TestCase):
    """生肖输出统一繁体(用户要求)。"""
    def test_simplified_normalized(self):
        self.assertEqual(normalize_zodiac("龙"), "龍")
        self.assertEqual(normalize_zodiac("马"), "馬")
        self.assertEqual(normalize_zodiac("鼠"), "鼠")   # 简繁同形不受影响
        r = mk("1", 5, zodiac="龙")
        self.assertEqual(zodiac_of(r), "龍")
        m = build_zodiac_map([r])
        self.assertTrue(all(z in {"龍"} for z in m.values()))


class TestJointAntiPersistence(unittest.TestCase):
    """反持续性: 参考号对上一期特码软惩罚; 新打分器下该机制仍然成立(自证场景)。"""

    def _spread_fillers(self):
        """16 个分散填充号(均非 25/26, 均不在 20-29 段), 不抢 25/26 的热维度。"""
        nums = [1, 3, 4, 8, 11, 14, 17, 19, 30, 33, 37, 40, 44, 47, 48, 49]
        zods = ["鼠", "牛", "虎", "兔", "龍", "蛇", "猴", "雞",
                "狗", "豬", "鼠", "牛", "虎", "兔", "龍", "蛇"]
        return [mk_rec(str(i), n, z) for i, (n, z) in enumerate(zip(nums, zods), 1)]

    def test_discount_mechanism(self):
        """25 近期极热(末6期全25): 不惩罚->25 胜; 禁绝->非25。证明折扣接线正确。"""
        recs = self._spread_fillers()
        recs += [mk_rec(str(100 + i), 25, "馬") for i in range(6)]  # 末6期全25
        zmap = build_zodiac_map(recs)
        self.assertEqual(_joint_predict_number(recs, zmap, discount_last=1.0), 25)
        self.assertNotEqual(_joint_predict_number(recs, zmap, discount_last=0.0), 25)

    def test_default_not_stuck_on_last_special(self):
        """25 与 26 交替、25 为末期: raw 下 25 胜出(先证场景有效);
        默认半折后翻向次名, 不再 = 上一期特码25(回归"两期一模一样")。"""
        recs = self._spread_fillers()
        recs += [mk_rec("100", 26, "羊"), mk_rec("101", 25, "馬"),
                 mk_rec("102", 26, "羊"), mk_rec("103", 25, "馬")]  # 末4: 26,25,26,25
        zmap = build_zodiac_map(recs)
        # 先证"不惩罚"时 25 确实胜出 -> 场景有效(非平凡), 否则用例空过
        self.assertEqual(_joint_predict_number(recs, zmap, discount_last=1.0), 25)
        # 默认 discount_last=0.5: 25 半折后败给次名, 不再卡在 25
        self.assertNotEqual(_joint_predict_number(recs, zmap), 25)


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

    def test_quad_returns_four_zodiacs(self):
        random.seed(9)
        zodiacs_cycle = ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪"]
        recs = [mk(str(i), (i % 49) + 1, zodiac=zodiacs_cycle[i % 12]) for i in range(60)]
        zq = predict_zodiac_quad(recs, llm_result=None, backtest_n=10)
        self.assertIsInstance(zq, ZodiacPool)
        self.assertEqual(len(zq.zodiacs), 4)
        self.assertEqual(len(zq.zodiacs), len(set(zq.zodiacs)))  # 唯一
        self.assertAlmostEqual(zq.baseline, 4 / 12)
        self.assertGreaterEqual(zq.hit_rate, 0.0)
        self.assertLessEqual(zq.hit_rate, 1.0)
        self.assertAlmostEqual(zq.lift, zq.hit_rate - zq.baseline, places=6)

    def test_quad_superset_of_pool(self):
        """四生肖集合应包含三生肖集合(前 3 个即三生肖, 第 4 个为补充)。"""
        random.seed(11)
        zodiacs_cycle = ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪"]
        recs = [mk(str(i), (i % 49) + 1, zodiac=zodiacs_cycle[i % 12]) for i in range(60)]
        zp = predict_zodiac_pool(recs, llm_result=None, backtest_n=10)
        zq = predict_zodiac_quad(recs, llm_result=None, backtest_n=10)
        self.assertTrue(set(zp.zodiacs).issubset(set(zq.zodiacs)))


class TestWaveOfNumber(unittest.TestCase):
    def test_known_mapping(self):
        self.assertEqual(wave_of_number(1), "红波")
        self.assertEqual(wave_of_number(46), "红波")
        self.assertEqual(wave_of_number(3), "蓝波")
        self.assertEqual(wave_of_number(48), "蓝波")
        self.assertEqual(wave_of_number(5), "绿波")
        self.assertEqual(wave_of_number(49), "绿波")

    def test_all_49_mapped_and_counts(self):
        from collections import Counter
        waves = Counter(wave_of_number(n) for n in range(1, 50))
        self.assertEqual(sum(waves.values()), 49)  # 无空串
        self.assertEqual(waves["红波"], 17)
        self.assertEqual(waves["蓝波"], 16)
        self.assertEqual(wave_of_number(7), "红波")

    def test_out_of_range_returns_empty(self):
        self.assertEqual(wave_of_number(0), "")
        self.assertEqual(wave_of_number(50), "")
        self.assertEqual(wave_of_number("x"), "")


class TestZodiacMap(unittest.TestCase):
    def test_most_recent_wins(self):
        # 号码1: 早期"鼠", 近期"牛" -> 取最近"牛"(模拟年度轮换)
        r_old = Record("2026001", "t", [1, 2, 3, 4, 5, 6], 7, ["red"] * 7,
                       ["鼠", "牛", "虎", "鼠", "牛", "虎", "鼠"])
        r_new = Record("2026002", "t", [1, 2, 3, 4, 5, 6], 7, ["red"] * 7,
                       ["牛", "虎", "鼠", "牛", "虎", "鼠", "牛"])
        m = build_zodiac_map([r_old, r_new])
        self.assertEqual(m[1], "牛")  # 最近一次出现 = 牛
        self.assertEqual(len(m), 7)   # 1-7 全覆盖

    def test_empty_and_missing(self):
        self.assertEqual(build_zodiac_map([]), {})
        self.assertEqual(zodiac_of_number(1, {}), "")
        self.assertEqual(zodiac_of_number(1, {1: "鼠"}), "鼠")

    def test_skips_short_zodiacs(self):
        r = Record("1", "t", [1, 2, 3, 4, 5, 6], 7, ["red"] * 7, ["鼠"])  # zodiacs 不足
        self.assertEqual(build_zodiac_map([r]), {})


if __name__ == "__main__":
    unittest.main()
