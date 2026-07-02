#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""预测与回测层"""
import os
import sys
import random
from dataclasses import dataclass

from analysis import build_report

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        os.system("chcp 65001 >nul 2>&1")


@dataclass
class PredictionGroup:
    name: str
    regular: list          # 6 个平码
    special: int           # 特码
    strategy: str          # 策略描述
    backtest_special_hit: float = 0.0   # 回测特码命中率
    backtest_regular_hits: float = 0.0  # 回测平均平码命中数


def weighted_sample(weights: dict, k: int = 6) -> list:
    """按权重无放回采样 k 个不同号码。"""
    pool = list(weights.keys())
    w = [weights[n] for n in pool]
    picked = []
    pool_copy = list(pool)
    w_copy = list(w)
    for _ in range(k):
        if not pool_copy:
            break
        total = sum(w_copy)
        if total <= 0:
            choice = random.choice(pool_copy)
        else:
            r = random.random() * total
            cum = 0.0
            idx = len(pool_copy) - 1
            for i, wi in enumerate(w_copy):
                cum += wi
                if r <= cum:
                    idx = i
                    break
            choice = pool_copy[idx]
        picked.append(choice)
        j = pool_copy.index(choice)
        pool_copy.pop(j)
        w_copy.pop(j)
    return picked


def sample_with_sum_constraint(weights: dict, mean: float, std: float,
                               k: int = 6, max_tries: int = 200) -> list:
    """采样 k 个, 和值需落在 [mean-2*std, mean+2*std]。"""
    lo, hi = mean - 2 * std, mean + 2 * std
    picked = weighted_sample(weights, k=k)
    for _ in range(max_tries):
        if lo <= sum(picked) <= hi:
            return picked
        picked = weighted_sample(weights, k=k)
    return picked  # 实在不行返回最后一次


def _freq_weights(report, boost_hot=True):
    """由频率构造权重; boost_hot=True 热号权重高, False 则用遗漏值(冷号回归)。"""
    if boost_hot:
        freq = report["freq"]
        return {n: freq.get(n, 0) + 1.0 for n in range(1, 50)}  # +1 平滑
    else:
        gap = report["gap"]
        return {n: gap[n]["current"] + 1.0 for n in range(1, 50)}


def predict_group_a(records, report) -> PredictionGroup:
    """A 组: 频率/冷热加权 + 和值约束抽样; 特码取最热号。"""
    weights = _freq_weights(report, boost_hot=True)
    s = report["sum"]
    regular = sample_with_sum_constraint(weights, s["mean"], s["std"], k=6)
    special = max(range(1, 50), key=lambda n: report["freq"].get(n, 0))
    return PredictionGroup(name="A", regular=regular, special=special,
                           strategy="频率/冷热加权 + 和值约束")


def predict_group_b(records, report) -> PredictionGroup:
    """B 组: 马尔可夫特码转移 + 和值约束抽样。"""
    last_special = records[-1].special if records else 0
    trans = report["markov"].get(last_special, {})
    if trans:
        special = max(trans, key=trans.get)
    else:
        special = random.randint(1, 49)
    weights = _freq_weights(report, boost_hot=True)
    s = report["sum"]
    regular = sample_with_sum_constraint(weights, s["mean"], s["std"], k=6)
    return PredictionGroup(name="B", regular=regular, special=special,
                           strategy="马尔可夫一阶转移 + 和值约束")


def predict_group_c(records, report, llm_result=None) -> PredictionGroup:
    """C 组: 大模型集成; 无 LLM 时退化为遗漏值回归。"""
    s = report["sum"]
    if llm_result and len(llm_result.predicted_set) >= 7:
        nums = [n for n in llm_result.predicted_set if 1 <= n <= 49]
        regular = nums[:6]
        special = nums[6] if len(nums) > 6 else nums[-1]
        while len(regular) < 6:
            cand = random.randint(1, 49)
            if cand not in regular:
                regular.append(cand)
        return PredictionGroup(name="C", regular=regular, special=special,
                               strategy="大模型集成推理")
    # 降级: 遗漏值回归(冷号即将出现)
    weights = _freq_weights(report, boost_hot=False)
    regular = sample_with_sum_constraint(weights, s["mean"], s["std"], k=6)
    special = max(range(1, 50), key=lambda n: report["gap"][n]["current"])
    return PredictionGroup(name="C", regular=regular, special=special,
                           strategy="遗漏值回归(无LLM降级)")


def _evaluate(group: PredictionGroup, actual) -> dict:
    """评估单组预测对实际开奖的命中。actual: Record。"""
    reg_set = set(group.regular)
    actual_reg = set(actual.regular)
    reg_hits = len(reg_set & actual_reg)
    special_hit = 1 if group.special == actual.special else 0
    return {"reg_hits": reg_hits, "special_hit": special_hit}


def backtest(records, group_func, n: int = 30) -> tuple:
    """留一回测: 对最近 n 期, 每期用之前数据预测并评估。
    group_func(records, report) -> PredictionGroup。返回 (特码命中率, 平均平码命中数)。"""
    n = max(1, min(n, len(records) - 20))
    special_hits = 0
    reg_hits_total = 0
    for i in range(n):
        split = len(records) - n + i
        train = records[:split]
        actual = records[split]
        report = build_report(train)
        group = group_func(train, report)
        ev = _evaluate(group, actual)
        special_hits += ev["special_hit"]
        reg_hits_total += ev["reg_hits"]
    return special_hits / n, reg_hits_total / n


def predict_all(records, llm_result=None, backtest_n: int = 30) -> list:
    """生成三组预测并回测, 按综合命中率降序排序。"""
    report = build_report(records)
    raw_groups = [
        predict_group_a(records, report),
        predict_group_b(records, report),
        predict_group_c(records, report, llm_result),
    ]
    funcs = [predict_group_a, predict_group_b,
             lambda r, rep: predict_group_c(r, rep, llm_result)]
    for g, fn in zip(raw_groups, funcs):
        sp, reg = backtest(records, fn, n=backtest_n)
        g.backtest_special_hit = sp
        g.backtest_regular_hits = reg
    # 综合分: 特码命中权重高(6倍)
    raw_groups.sort(key=lambda g: (g.backtest_special_hit * 6 + g.backtest_regular_hits),
                    reverse=True)
    return raw_groups


# ======================== 五组特码预测 ========================

@dataclass
class SpecialPrediction:
    name: str
    special: int          # 预测特码 1-49
    strategy: str         # 策略描述
    backtest_hit: float = 0.0  # 回测特码命中率


def _sp_hot(records, report, llm_result):
    """频率最高热号。"""
    return max(range(1, 50), key=lambda n: report["freq"].get(n, 0))


def _sp_markov1(records, report, llm_result):
    """马尔可夫一阶: 上一期特码之后转移概率最高的号。"""
    last = records[-1].special if records else 0
    trans = report["markov"].get(last, {})
    if trans:
        return max(trans, key=trans.get)
    return _sp_hot(records, report, llm_result)


def _sp_markov2(records, report, llm_result):
    """马尔可夫二阶: 上两期特码之后转移概率最高的号。"""
    if len(records) < 2:
        return _sp_hot(records, report, llm_result)
    key = (records[-2].special, records[-1].special)
    trans = report["markov2"].get(key, {})
    if trans:
        return max(trans, key=trans.get)
    return _sp_markov1(records, report, llm_result)


def _sp_due(records, report, llm_result):
    """遗漏回归: 当前遗漏期数最大的号(冷号到期)。"""
    return max(range(1, 50), key=lambda n: report["gap"][n]["current"])


def _sp_autocorr(records, report, llm_result):
    """周期回归: 近 7 期特码众数(lag-7 周期假设)。"""
    from collections import Counter
    recent = [r.special for r in records[-7:]]
    if recent:
        return Counter(recent).most_common(1)[0][0]
    return _sp_hot(records, report, llm_result)


def _sp_llm(records, report, llm_result):
    """大模型推理特码; 无 LLM 时退化为周期回归。"""
    if llm_result and llm_result.predicted_set:
        nums = [n for n in llm_result.predicted_set if 1 <= n <= 49]
        if len(nums) >= 7:
            return nums[6]
        if nums:
            return nums[-1]
    return _sp_autocorr(records, report, llm_result)


# 五组特码策略: (名称, 策略函数, 描述)
SPECIAL_STRATEGIES = [
    ("A", _sp_hot, "频率热号"),
    ("B", _sp_markov1, "马尔可夫一阶转移"),
    ("C", _sp_markov2, "马尔可夫二阶转移"),
    ("D", _sp_due, "遗漏值回归"),
    ("E", _sp_llm, "大模型/周期回归"),
]


def _backtest_special(records, sp_func, llm_result, n: int = 30) -> float:
    """留一回测特码命中率。回测时 llm_result 传 None(无法逐期调用 LLM)。"""
    n = max(1, min(n, len(records) - 20))
    hits = 0
    for i in range(n):
        split = len(records) - n + i
        train = records[:split]
        actual = records[split]
        report = build_report(train)
        pred = sp_func(train, report, None)
        if pred == actual.special:
            hits += 1
    return hits / n


def predict_special_groups(records, llm_result=None, backtest_n: int = 30) -> list:
    """生成五组特码预测并回测, 按回测命中率降序排序。"""
    report = build_report(records)
    groups = []
    for name, fn, desc in SPECIAL_STRATEGIES:
        special = fn(records, report, llm_result)
        hit = _backtest_special(records, fn, llm_result, n=backtest_n)
        groups.append(SpecialPrediction(name=name, special=special,
                                        strategy=desc, backtest_hit=hit))
    groups.sort(key=lambda g: g.backtest_hit, reverse=True)
    return groups
