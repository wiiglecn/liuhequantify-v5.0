#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多维度预测层 — 针对特码的波色/生肖/尾数/大小/奇偶/头数预测"""
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass

from backtest_utils import backtest_series, rolling_stability

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        os.system("chcp 65001 >nul 2>&1")


@dataclass
class DimensionPrediction:
    name: str            # 维度名 e.g. "波色"
    value: str           # 预测值 e.g. "red"
    strategy: str        # 策略描述
    accuracy: float = 0.0       # 回测准确率
    baseline: float = 0.0       # 随机基线
    lift: float = 0.0           # 提升度 = accuracy - baseline
    std_error: float = 0.0      # 标准误差
    stability: float = 0.0      # 滚动命中率标准差(越小越稳)


@dataclass
class ZodiacPool:
    """特码三生肖预测(单组, 3 个生肖)。"""
    zodiacs: list        # 3 个生肖
    strategy: str
    hit_rate: float = 0.0
    baseline: float = 0.0       # = 3/12
    lift: float = 0.0
    std_error: float = 0.0
    stability: float = 0.0


# ======================== 维度取值函数 ========================

def wave_of(rec) -> str:
    """特码波色。"""
    return rec.waves[6] if len(rec.waves) > 6 else (rec.waves[-1] if rec.waves else "")


def zodiac_of(rec) -> str:
    """特码生肖。"""
    return rec.zodiacs[6] if len(rec.zodiacs) > 6 else (rec.zodiacs[-1] if rec.zodiacs else "")


# 别名: 特码生肖(zodiac_of 的语义化名称)
def special_zodiac_of(rec) -> str:
    return zodiac_of(rec)


def tail_of(rec) -> int:
    """尾数(个位 0-9)。"""
    return rec.special % 10


def big_small_of(rec) -> str:
    """大小: >=25 为大, 否则小。"""
    return "大" if rec.special >= 25 else "小"


def odd_even_of(rec) -> str:
    """奇偶。"""
    return "奇" if rec.special % 2 == 1 else "偶"


def head_of(rec) -> int:
    """头数(十位 0-4): 0-9->0, 10-19->1, ... 40-49->4。"""
    return rec.special // 10


# ======================== 三策略融合预测 ========================

def _freq_predict(train, extract):
    """频率策略: 历史众数。"""
    c = Counter(extract(r) for r in train)
    if not c:
        return None
    return c.most_common(1)[0][0]


def _markov_predict(train, extract):
    """马尔可夫策略: 上一期值之后转移概率最高的值。"""
    if len(train) < 2:
        return None
    seq = [extract(r) for r in train]
    last = seq[-1]
    trans = defaultdict(Counter)
    for a, b in zip(seq[:-1], seq[1:]):
        trans[a][b] += 1
    if last in trans and trans[last]:
        return trans[last].most_common(1)[0][0]
    return None


def _trend_predict(train, extract, k=10):
    """近期趋势: 最近 k 期众数。"""
    recent = train[-k:]
    c = Counter(extract(r) for r in recent)
    if not c:
        return None
    return c.most_common(1)[0][0]


def _ensemble_predict(train, extract):
    """三策略融合: 频率/马尔可夫/趋势各投一票, 最多票者胜; 平票取频率。"""
    votes = Counter()
    for fn in (_freq_predict, _markov_predict, _trend_predict):
        pred = fn(train, extract)
        if pred is not None:
            votes[pred] += 1
    if not votes:
        return _freq_predict(train, extract)
    # 平票时优先频率策略结果
    top = votes.most_common()
    max_count = top[0][1]
    winners = [v for v, c in top if c == max_count]
    freq_pred = _freq_predict(train, extract)
    if freq_pred in winners:
        return freq_pred
    return winners[0]


# 维度配置: (名称, 取值函数, 随机基线)
DIMENSIONS = [
    ("波色", wave_of, 1 / 3),
    ("生肖", zodiac_of, 1 / 12),
    ("尾数", tail_of, 1 / 10),
    ("大小", big_small_of, 1 / 2),
    ("奇偶", odd_even_of, 1 / 2),
    ("头数", head_of, 1 / 5),
]


def predict_dimensions(records, llm_result=None, backtest_n: int = 30) -> list:
    """对特码进行六维度预测, 每维度用三策略融合, 并回测准确率/基线/lift/稳定性。"""
    results = []
    for name, extract, baseline in DIMENSIONS:
        # 完整数据上的融合预测
        value = _ensemble_predict(records, extract)
        if value is None:
            value = ""
        # 回测(用融合预测函数)
        bt = backtest_series(
            predict_func=lambda train, ex=extract: _ensemble_predict(train, ex),
            actual_func=extract,
            records=records, n=backtest_n, baseline=baseline,
        )
        results.append(DimensionPrediction(
            name=name, value=str(value),
            strategy="频率+马尔可夫+趋势 融合投票",
            accuracy=bt.accuracy, baseline=bt.baseline, lift=bt.lift,
            std_error=bt.std_error,
            stability=rolling_stability(bt.hit_series or []),
        ))
    return results


# ======================== 特码三生肖预测(单组) ========================

ZODIAC_POOL_SIZE = 3


def _zodiac_scores(records):
    """融合多信号为每个生肖打分, 用于排序取前 3。"""
    seq = [special_zodiac_of(r) for r in records if special_zodiac_of(r)]
    if not seq:
        return {}
    all_zodiacs = set(seq) | {special_zodiac_of(r) for r in records}
    scores = {z: 0.0 for z in all_zodiacs}

    # 1. 频率(归一化)
    c = Counter(seq)
    cmax = max(c.values()) or 1
    for z, cnt in c.items():
        scores[z] += (cnt / cmax) * 3.0

    # 2. 马尔可夫一阶: 上一期生肖之后转移概率最高的
    trans = defaultdict(Counter)
    for a, b in zip(seq[:-1], seq[1:]):
        trans[a][b] += 1
    last = seq[-1]
    if last in trans and trans[last]:
        tmax = max(trans[last].values())
        for z, p in trans[last].items():
            scores[z] += (p / tmax) * 2.5

    # 3. 遗漏回归: 当前遗漏最大的生肖
    last_seen = {}
    for i, z in enumerate(seq):
        last_seen[z] = i
    total = len(seq)
    gaps = {z: (total - 1 - last_seen[z]) for z in last_seen}
    gmax = max(gaps.values()) if gaps else 1
    if gmax > 0:
        for z, g in gaps.items():
            scores[z] += (g / gmax) * 2.0

    return scores


def _build_zodiac_pool(records, k=ZODIAC_POOL_SIZE):
    """按融合得分取前 k 个生肖。"""
    scores = _zodiac_scores(records)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    zodiacs = [z for z, _ in ranked[:k]]
    # 不足 k 个补齐(按频率)
    if len(zodiacs) < k:
        seq = [special_zodiac_of(r) for r in records if special_zodiac_of(r)]
        c = Counter(seq)
        for z, _ in c.most_common():
            if z not in zodiacs:
                zodiacs.append(z)
                if len(zodiacs) == k:
                    break
    return zodiacs[:k]


def predict_zodiac_pool(records, llm_result=None, backtest_n: int = 30) -> ZodiacPool:
    """生成单组 3 个最可能开出特码的生肖, 并回测。"""
    import math
    k = ZODIAC_POOL_SIZE
    baseline = k / 12
    zodiacs = _build_zodiac_pool(records, k)

    # 留一回测: 真实特码生肖落在 3 个内即命中
    n_clamped = max(1, min(backtest_n, len(records) - 10))
    hits = []
    for i in range(n_clamped):
        split = len(records) - n_clamped + i
        train = records[:split]
        actual = records[split]
        pool = set(_build_zodiac_pool(train, k))
        hits.append(1 if special_zodiac_of(actual) in pool else 0)
    accuracy = sum(hits) / n_clamped if n_clamped else 0.0
    std_error = math.sqrt(accuracy * (1 - accuracy) / n_clamped) if n_clamped else 0.0
    stab = rolling_stability(hits)

    return ZodiacPool(
        zodiacs=zodiacs,
        strategy="频率+马尔可夫+遗漏 多信号融合",
        hit_rate=accuracy, baseline=baseline, lift=accuracy - baseline,
        std_error=std_error, stability=stab,
    )
