#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""回测统计引擎 — 统一计算准确率/随机基线/lift/标准误差"""
import os
import sys
import math
from dataclasses import dataclass

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        os.system("chcp 65001 >nul 2>&1")


@dataclass
class BacktestResult:
    accuracy: float      # 回测准确率/命中率
    n: int               # 实际回测期数
    baseline: float      # 随机基线(1/类别数)
    lift: float          # 提升度 = accuracy - baseline
    std_error: float     # 标准误差 sqrt(p(1-p)/n)
    hit_series: list = None  # 逐期命中序列(0/1), 供稳定性分析


def backtest_series(predict_func, actual_func, records, n: int,
                    baseline: float) -> BacktestResult:
    """通用回测: 对最近 n 期, 每期用之前数据预测, 比对真实值。
    predict_func(train_records) -> 预测值
    actual_func(record) -> 真实值
    baseline: 随机基线(如 1/49, 1/3 等)
    """
    n = max(1, min(n, len(records) - 10))
    hits = []
    for i in range(n):
        split = len(records) - n + i
        train = records[:split]
        actual = records[split]
        pred = predict_func(train)
        hit = 1 if pred == actual_func(actual) else 0
        hits.append(hit)
    accuracy = sum(hits) / n if n else 0.0
    std_error = math.sqrt(accuracy * (1 - accuracy) / n) if n else 0.0
    return BacktestResult(
        accuracy=accuracy, n=n, baseline=baseline,
        lift=accuracy - baseline, std_error=std_error,
        hit_series=hits,
    )


def rolling_stability(hit_series: list, window: int = 10) -> float:
    """滚动窗口命中率的标准差, 越小越稳定。"""
    if len(hit_series) < window:
        window = max(1, len(hit_series))
    if window < 2:
        return 0.0
    rates = []
    for i in range(len(hit_series) - window + 1):
        seg = hit_series[i:i + window]
        rates.append(sum(seg) / window)
    if len(rates) < 2:
        return 0.0
    mean = sum(rates) / len(rates)
    var = sum((r - mean) ** 2 for r in rates) / len(rates)
    return math.sqrt(var)
