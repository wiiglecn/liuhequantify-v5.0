# -*- coding: utf-8 -*-
"""V5.8.7：命中率统计验证工具（零假设模拟、p 值校正、时间块 bootstrap）。

本模块只评估统计证据，不预测号码，也不把历史表现解释为未来保证。
"""
from __future__ import annotations

import math
import random
from typing import Iterable, Sequence


def binomial_upper_tail(hits: int, n: int, p0: float) -> float:
    """计算公平独立 Bernoulli 零假设下命中数不少于 hits 的精确上尾概率。"""
    if n < 0 or hits < 0 or hits > n:
        raise ValueError("必须满足 0 <= hits <= n")
    if not 0.0 <= p0 <= 1.0:
        raise ValueError("p0 必须在 [0, 1] 范围内")
    if hits == 0:
        return 1.0
    if p0 == 0.0:
        return 0.0
    if p0 == 1.0:
        return 1.0 if hits <= n else 0.0
    # 以对数形式计算组合数，避免大 n 时直接计算阶乘溢出。
    terms = []
    for k in range(hits, n + 1):
        log_comb = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
        terms.append(math.exp(log_comb + k * math.log(p0) + (n - k) * math.log1p(-p0)))
    return min(1.0, math.fsum(terms))


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down 家族错误率校正，返回与输入顺序一致的校正 p 值。"""
    _validate_p_values(p_values)
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[idx]))
        adjusted[idx] = running
    return adjusted


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg FDR 校正，返回与输入顺序一致的 q 值。"""
    _validate_p_values(p_values)
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 1.0
    for rank0 in range(m - 1, -1, -1):
        idx = order[rank0]
        rank = rank0 + 1
        running = min(running, p_values[idx] * m / rank)
        adjusted[idx] = min(1.0, running)
    return adjusted


def _validate_p_values(p_values: Sequence[float]) -> None:
    if any(not math.isfinite(p) or p < 0.0 or p > 1.0 for p in p_values):
        raise ValueError("所有 p 值必须是 [0, 1] 内的有限数值")


def moving_block_bootstrap_ci(
    new_hits: Sequence[int],
    old_hits: Sequence[int],
    *,
    block_size: int = 5,
    repetitions: int = 5000,
    seed: int = 20261029,
) -> list[float]:
    """估计新旧模型配对命中差（新-旧）的 95% 时间块 bootstrap 区间。

    输入为同一批开奖期上的 0/1 命中序列；返回比例差而非百分点。
    循环块抽样保留短程相邻依赖，不能消除所有非平稳性。
    """
    if len(new_hits) != len(old_hits) or not new_hits:
        raise ValueError("新旧命中序列必须非空且长度一致")
    if block_size < 1 or repetitions < 100:
        raise ValueError("block_size 必须 >= 1，repetitions 必须 >= 100")
    if any(x not in (0, 1, False, True) for x in list(new_hits) + list(old_hits)):
        raise ValueError("命中序列只能包含 0/1")
    diffs = [int(bool(a)) - int(bool(b)) for a, b in zip(new_hits, old_hits)]
    n = len(diffs)
    blocks_needed = math.ceil(n / block_size)
    rng = random.Random(seed)
    samples: list[float] = []
    starts = list(range(n))
    for _ in range(repetitions):
        sampled: list[int] = []
        for _ in range(blocks_needed):
            start = rng.choice(starts)
            sampled.extend(diffs[(start + offset) % n] for offset in range(block_size))
        samples.append(sum(sampled[:n]) / n)
    samples.sort()
    low = samples[max(0, int(0.025 * repetitions))]
    high = samples[min(repetitions - 1, int(0.975 * repetitions))]
    return [low, high]


def simulate_max_stat_null(
    *,
    n: int,
    baselines: Sequence[float],
    trials: int = 1000,
    simulations: int = 10000,
    seed: int = 20261029,
) -> dict:
    """模拟多目标/多模型搜索下的最大标准化命中率提升，校正挑选赢家的偏差。

    每个模拟批次在所有 baselines x trials 候选上取最大 z 分数。
    这是 Bernoulli 近似的零假设；若实际完整策略搜索更复杂，应进一步重跑完整管线。
    """
    if n < 1 or trials < 1 or simulations < 100:
        raise ValueError("n/trials 必须为正数，simulations 至少为 100")
    if not baselines or any(not 0.0 < p < 1.0 for p in baselines):
        raise ValueError("baselines 必须是非空且位于 (0, 1) 的列表")
    rng = random.Random(seed)
    max_stats: list[float] = []
    for _ in range(simulations):
        best = 0.0
        for p0 in baselines:
            scale = math.sqrt(n * p0 * (1.0 - p0))
            for _ in range(trials):
                hits = sum(1 for _ in range(n) if rng.random() < p0)
                z = max(0.0, (hits - n * p0) / scale)
                if z > best:
                    best = z
        max_stats.append(best)
    max_stats.sort()
    observed_threshold = max_stats[min(simulations - 1, int(0.95 * simulations))]
    return {
        "n": n,
        "baselines": list(baselines),
        "trials_per_baseline": trials,
        "simulations": simulations,
        "seed": seed,
        "max_z_95_threshold": observed_threshold,
        "note": "近似 Bernoulli 零假设；要评估整个自适应搜索流程，需在模拟序列上重跑完整搜索管线。",
    }


def max_stat_adjusted_p_value(
    observed_z: float,
    *,
    n: int,
    baselines: Sequence[float],
    trials: int = 1000,
    simulations: int = 10000,
    seed: int = 20261029,
) -> float:
    """估计经最大统计量校正的 Monte Carlo p 值（加一修正避免零 p 值）。"""
    if not math.isfinite(observed_z):
        raise ValueError("observed_z 必须是有限数值")
    if n < 1 or trials < 1 or simulations < 100:
        raise ValueError("n/trials 必须为正数，simulations 至少为 100")
    if not baselines or any(not 0.0 < p < 1.0 for p in baselines):
        raise ValueError("baselines 必须是非空且位于 (0, 1) 的列表")
    rng = random.Random(seed)
    exceed = 0
    for _ in range(simulations):
        best = 0.0
        for p0 in baselines:
            scale = math.sqrt(n * p0 * (1.0 - p0))
            for _ in range(trials):
                hits = sum(1 for _ in range(n) if rng.random() < p0)
                best = max(best, (hits - n * p0) / scale)
        exceed += int(best >= observed_z)
    return (exceed + 1) / (simulations + 1)
