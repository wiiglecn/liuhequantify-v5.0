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
from dimensions import build_zodiac_map, _zodiac_six_scores

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        os.system("chcp 65001 >nul 2>&1")

POOL_SIZE = 10  # 小集合大小(8~10, 默认 10)
WIDE_POOL_SIZE = 20  # 大集合大小(20颗)


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


@dataclass
class WidePool:
    """20 颗号码的大集合(单组)。"""
    numbers: list         # 20 个号码(1-49, 唯一)
    strategy: str         # 策略描述
    hit_rate: float = 0.0
    baseline: float = 0.0       # = 20/49
    lift: float = 0.0
    std_error: float = 0.0
    stability: float = 0.0


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


# ======================== 20 颗大集合(单组) ========================

def _zodiac_bonus(records, zodiac_map):
    """三/四/六肖命中给号码加分(与 dimensions._zodiac_six_scores 同口径, 纯统计可回测)。

    三肖⊆四肖⊆六肖, 命中越窄的集合越该进大集合:
      生肖∈三肖 -> +3.0; ∈四肖(非三) -> +2.0; ∈六肖(非四) -> +1.0; 其余 0。
    一次打分切片得 top3/4/6, 避免重复计算; zodiac_map 为空时返回 {}(信号不贡献)。
    """
    if not records or not zodiac_map:
        return {}
    scores = _zodiac_six_scores(records, None, zodiac_map, 0.0)  # llm_weight=0 纯统计, 可回测
    if not scores:
        return {}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top3 = {z for z, _ in ranked[:3]}
    top4 = {z for z, _ in ranked[:4]}
    top6 = {z for z, _ in ranked[:6]}
    bonus = {}
    for num, z in zodiac_map.items():
        if z in top3:
            bonus[num] = 3.0
        elif z in top4:
            bonus[num] = 2.0
        elif z in top6:
            bonus[num] = 1.0
    return bonus


def _wide_score(records, report, llm_result, zodiac_map=None):
    """融合多信号为每个号码打分, 用于排序取 20 颗。

    信号: 1频率 2遗漏 3马尔可夫一阶 4近期趋势 5大模型种子 6三/四/六肖生肖命中。
    信号6 取自生肖层预测(_zodiac_six_scores, 纯统计), 把高概率生肖的号码倾向性纳入
    号码集合, 使大集合与三肖/四肖/六肖预测保持一致。zodiac_map 未传时按 records 就地
    构建(回测每折用 train 自建, 不泄漏未来)。
    """
    scores = {n: 0.0 for n in range(1, 50)}

    # 1. 频率热号(归一化)
    freq = report["freq"]
    fmax = max(freq.values()) if freq and max(freq.values()) > 0 else 1
    for n in range(1, 50):
        scores[n] += (freq.get(n, 0) / fmax) * 3.0

    # 2. 遗漏回归(遗漏越大越到期, 归一化)
    gap = report["gap"]
    gmax = max((gap[n]["current"] for n in range(1, 50)), default=1) or 1
    for n in range(1, 50):
        scores[n] += (gap[n]["current"] / gmax) * 2.0

    # 3. 马尔可夫一阶: 上一期特码转移概率
    last = records[-1].special if records else 0
    trans = report["markov"].get(last, {})
    tmax = max(trans.values()) if trans else 0
    if tmax > 0:
        for n, p in trans.items():
            scores[n] += (p / tmax) * 2.5

    # 4. 近期趋势(最近 20 期)
    recent = records[-20:]
    rc = Counter(r.special for r in recent)
    rmax = max(rc.values()) if rc else 1
    for n, c in rc.items():
        scores[n] += (c / rmax) * 1.5

    # 5. 大模型种子(LLM 号码加权)
    if llm_result and llm_result.predicted_set:
        llm_nums = [n for n in llm_result.predicted_set if isinstance(n, int) and 1 <= n <= 49]
        for n in llm_nums:
            scores[n] += 2.0

    # 6. 生肖集合信号(三/四/六肖命中; 纯统计可回测, 与 dimensions._zodiac_six_scores 同口径)
    if zodiac_map is None:
        zodiac_map = build_zodiac_map(records) if records else {}
    for n, b in _zodiac_bonus(records, zodiac_map).items():
        scores[n] += b

    return scores


def _build_wide_pool(records, report, llm_result, k=WIDE_POOL_SIZE):
    """按融合得分取前 k 颗(默认 20)。"""
    scores = _wide_score(records, report, llm_result)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    numbers = [n for n, _ in ranked[:k]]
    # 保证大小(分数末位可能有并列, 取前 k 个)
    if len(numbers) < k:
        for n in range(1, 50):
            if n not in numbers:
                numbers.append(n)
                if len(numbers) == k:
                    break
    return numbers[:k]


def _backtest_wide_pool(records, llm_result, n: int, k: int):
    """留一回测大集合: 集合含真实特码即命中。返回 (命中率, 基线, lift, 标准误, 稳定性)。"""
    baseline = k / 49
    n_clamped = max(1, min(n, len(records) - 10))
    hits = []
    for i in range(n_clamped):
        split = len(records) - n_clamped + i
        train = records[:split]
        actual = records[split]
        pool = set(_build_wide_pool(train, build_report(train), None, k))
        hits.append(1 if actual.special in pool else 0)
    accuracy = sum(hits) / n_clamped if n_clamped else 0.0
    std_error = math.sqrt(accuracy * (1 - accuracy) / n_clamped) if n_clamped else 0.0
    stab = rolling_stability(hits)
    return accuracy, baseline, accuracy - baseline, std_error, stab


def predict_wide_pool(records, llm_result=None, backtest_n: int = 30,
                      k: int = WIDE_POOL_SIZE) -> WidePool:
    """生成单组 20 颗号码的特码大集合, 并回测。"""
    k = WIDE_POOL_SIZE
    report = build_report(records)
    numbers = _build_wide_pool(records, report, llm_result, k)
    numbers = sorted(set(numbers))
    if len(numbers) < k:
        for n in range(1, 50):
            if n not in numbers:
                numbers.append(n)
                if len(numbers) == k:
                    break
    hr, baseline, lift, se, stab = _backtest_wide_pool(records, llm_result, backtest_n, k)
    return WidePool(
        numbers=numbers,
        strategy="频率+遗漏+马尔可夫+趋势+大模型+三/四/六肖生肖 多信号融合",
        hit_rate=hr, baseline=baseline, lift=lift,
        std_error=se, stability=stab,
    )
