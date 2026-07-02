#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""特码号码集合预测层 — 给出 8~10 颗最可能开出特码的号码集合"""
import os
import sys
import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from analysis import build_report
from backtest_utils import backtest_series, rolling_stability

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        os.system("chcp 65001 >nul 2>&1")

POOL_SIZE = 10  # 集合大小(8~10, 默认 10)


@dataclass
class SpecialPool:
    name: str
    numbers: list         # 8~10 个号码(1-49, 唯一)
    strategy: str         # 策略描述
    hit_rate: float = 0.0       # 回测命中率(集合含真实特码即命中)
    baseline: float = 0.0       # 随机基线 = size/49
    lift: float = 0.0           # 提升度
    std_error: float = 0.0      # 标准误
    stability: float = 0.0      # 稳定性


def _topn_by_weight(weights: dict, k: int) -> list:
    """按权重取前 k 个号码。"""
    return [n for n, _ in sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[:k]]


def _pool_freq(records, report, llm_result, k=POOL_SIZE):
    """频率热号: 历史出现最多的 k 个号。"""
    weights = {n: report["freq"].get(n, 0) + 1.0 for n in range(1, 50)}
    return _topn_by_weight(weights, k)


def _pool_markov1(records, report, llm_result, k=POOL_SIZE):
    """马尔可夫一阶: 上一期特码之后转移概率最高的 k 个号。"""
    last = records[-1].special if records else 0
    trans = report["markov"].get(last, {})
    if trans:
        ranked = sorted(trans.items(), key=lambda kv: kv[1], reverse=True)
        nums = [n for n, _ in ranked[:k]]
        # 不足 k 个用热号补齐
        if len(nums) < k:
            fill = [n for n in _pool_freq(records, report, llm_result, k) if n not in nums]
            nums += fill[:k - len(nums)]
        return nums
    return _pool_freq(records, report, llm_result, k)


def _pool_markov2(records, report, llm_result, k=POOL_SIZE):
    """马尔可夫二阶: 上两期特码之后转移概率最高的 k 个号。"""
    if len(records) < 2:
        return _pool_markov1(records, report, llm_result, k)
    key = (records[-2].special, records[-1].special)
    trans = report["markov2"].get(key, {})
    if trans:
        ranked = sorted(trans.items(), key=lambda kv: kv[1], reverse=True)
        nums = [n for n, _ in ranked[:k]]
        if len(nums) < k:
            fill = [n for n in _pool_markov1(records, report, llm_result, k) if n not in nums]
            nums += fill[:k - len(nums)]
        return nums
    return _pool_markov1(records, report, llm_result, k)


def _pool_due(records, report, llm_result, k=POOL_SIZE):
    """遗漏回归: 当前遗漏最大的 k 个号(冷号到期)。"""
    weights = {n: report["gap"][n]["current"] + 1.0 for n in range(1, 50)}
    return _topn_by_weight(weights, k)


def _pool_trend(records, report, llm_result, k=POOL_SIZE):
    """近期趋势: 最近 20 期出现最多的 k 个号。"""
    recent = records[-20:]
    c = Counter()
    for r in recent:
        c[r.special] += 1
    weights = {n: c.get(n, 0) + 0.5 for n in range(1, 50)}
    return _topn_by_weight(weights, k)


def _pool_llm(records, report, llm_result, k=POOL_SIZE):
    """大模型集成: 优先 LLM 号码, 不足用频率热号补齐; 无 LLM 用近期趋势。"""
    if llm_result and llm_result.predicted_set:
        nums = []
        for n in llm_result.predicted_set:
            if isinstance(n, int) and 1 <= n <= 49 and n not in nums:
                nums.append(n)
        if nums:
            if len(nums) < k:
                fill = [n for n in _pool_freq(records, report, llm_result, k) if n not in nums]
                nums += fill[:k - len(nums)]
            return nums[:k]
    return _pool_trend(records, report, llm_result, k)


# 集合策略: (名称, 策略函数, 描述)
POOL_STRATEGIES = [
    ("A", _pool_freq, "频率热号集合"),
    ("B", _pool_markov1, "马尔可夫一阶转移集合"),
    ("C", _pool_markov2, "马尔可夫二阶转移集合"),
    ("D", _pool_due, "遗漏值回归集合"),
    ("E", _pool_trend, "近期趋势集合"),
    ("F", _pool_llm, "大模型/趋势集合"),
]


def _backtest_pool(records, pool_func, llm_result, n: int, k: int):
    """留一回测: 集合含真实特码即命中。返回 (命中率, 基线, lift, 标准误, 稳定性)。"""
    baseline = k / 49
    n_clamped = max(1, min(n, len(records) - 10))
    hits = []
    for i in range(n_clamped):
        split = len(records) - n_clamped + i
        train = records[:split]
        actual = records[split]
        pool = set(pool_func(train, build_report(train), None, k))
        hits.append(1 if actual.special in pool else 0)
    accuracy = sum(hits) / n_clamped if n_clamped else 0.0
    std_error = math.sqrt(accuracy * (1 - accuracy) / n_clamped) if n_clamped else 0.0
    stab = rolling_stability(hits)
    return accuracy, baseline, accuracy - baseline, std_error, stab


def predict_special_pools(records, llm_result=None, backtest_n: int = 30,
                          k: int = POOL_SIZE) -> list:
    """生成多个特码号码集合(8~10颗), 回测并按命中率降序排序。"""
    k = max(8, min(10, k))
    report = build_report(records)
    pools = []
    for name, fn, desc in POOL_STRATEGIES:
        numbers = fn(records, report, llm_result, k)
        # 保证唯一性与大小
        numbers = list(dict.fromkeys(numbers))[:k]
        if len(numbers) < k:
            for n in range(1, 50):
                if n not in numbers:
                    numbers.append(n)
                    if len(numbers) == k:
                        break
        hr, baseline, lift, se, stab = _backtest_pool(records, fn, llm_result, backtest_n, k)
        pools.append(SpecialPool(
            name=name, numbers=sorted(numbers), strategy=desc,
            hit_rate=hr, baseline=baseline, lift=lift,
            std_error=se, stability=stab,
        ))
    pools.sort(key=lambda p: p.hit_rate, reverse=True)
    return pools
