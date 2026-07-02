#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统计分析层 — 模型反推"""
import os
import sys
import math
from collections import Counter, defaultdict

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        os.system("chcp 65001 >nul 2>&1")


def _all_numbers(rec):
    """一期全部 7 个号码(平码+特码)。"""
    return list(rec.regular) + [rec.special]


def number_frequency(records) -> dict:
    """统计 1-49 各号码出现次数(含特码)。"""
    c = Counter()
    for r in records:
        c.update(_all_numbers(r))
    return {n: c.get(n, 0) for n in range(1, 50)}


def _gammainc_lower(s, x):
    """下不完全伽马函数 (级数展开)。"""
    if x <= 0:
        return 0.0
    term = 1.0 / s
    total = term
    for n in range(1, 1000):
        term *= x / (s + n)
        total += term
        if abs(term) < 1e-12:
            break
    return total * (x ** s) * math.exp(-x) / math.gamma(s)


def chi_square_uniform(freq: dict):
    """卡方均匀性检验。返回 (chi2, p_value)。自由度 df=48。
    p_value = 1 - CDF = 1 - P(df/2, chi2/2)。"""
    observed = [freq.get(n, 0) for n in range(1, 50)]
    total = sum(observed)
    if total == 0:
        return 0.0, 1.0
    expected = total / 49.0
    chi2 = sum((o - expected) ** 2 / expected for o in observed)
    df = 48
    try:
        p = 1.0 - _gammainc_lower(df / 2.0, chi2 / 2.0)
    except Exception:
        p = float("nan")
    # 钳制到 [0,1]
    if p != p:  # nan
        p = float("nan")
    p = max(0.0, min(1.0, p))
    return chi2, p


def gap_stats(records) -> dict:
    """每个号码的遗漏统计: current(当前遗漏期数), max(历史最大遗漏期数)。
    遗漏 = 两次出现之间错过的期数。"""
    last_seen = {}
    max_gap = {n: 0 for n in range(1, 50)}
    for i, r in enumerate(records):
        nums = set(_all_numbers(r))
        for n in range(1, 50):
            if n in nums:
                if n in last_seen:
                    gap = i - last_seen[n] - 1  # 错过的期数
                    if gap > max_gap[n]:
                        max_gap[n] = gap
                last_seen[n] = i
    # current: 自最后一次出现以来错过的期数
    total = len(records)
    current_gap = {}
    for n in range(1, 50):
        if n in last_seen:
            current_gap[n] = (total - 1) - last_seen[n]
        else:
            current_gap[n] = total  # 从未出现, 遗漏全部期数
    return {n: {"current": current_gap[n], "max": max_gap[n]} for n in range(1, 50)}


def sum_stats(records) -> dict:
    """6 平码和值的均值/标准差/最小/最大。"""
    sums = [sum(r.regular) for r in records if len(r.regular) == 6]
    if not sums:
        return {"mean": 0, "std": 0, "min": 0, "max": 0}
    mean = sum(sums) / len(sums)
    var = sum((s - mean) ** 2 for s in sums) / len(sums)
    return {"mean": mean, "std": math.sqrt(var), "min": min(sums), "max": max(sums)}


def hot_cold(freq: dict, topn: int = 10):
    """返回热号(频次最高)与冷号(频次最低)各 topn 个。"""
    items = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
    hot = [n for n, _ in items[:topn]]
    cold = [n for n, _ in items[-topn:]]
    return hot, cold


def markov_transition(records):
    """特码一阶转移矩阵(归一化行概率)。返回 {from: {to: prob}}。"""
    specials = [r.special for r in records]
    trans = defaultdict(lambda: defaultdict(int))
    for a, b in zip(specials[:-1], specials[1:]):
        trans[a][b] += 1
    result = {}
    for a, nxt in trans.items():
        total = sum(nxt.values())
        result[a] = {b: c / total for b, c in nxt.items()}
    return result


def markov_transition_2(records):
    """特码二阶转移。返回 {(a,b): {to: prob}}。"""
    specials = [r.special for r in records]
    trans = defaultdict(lambda: defaultdict(int))
    for i in range(len(specials) - 2):
        key = (specials[i], specials[i + 1])
        trans[key][specials[i + 2]] += 1
    result = {}
    for key, nxt in trans.items():
        total = sum(nxt.values())
        result[key] = {b: c / total for b, c in nxt.items()}
    return result


def autocorrelation(series: list, lag: int = 1) -> float:
    """序列滞后 lag 的自相关系数。常数序列返回 0。"""
    n = len(series)
    if n <= lag:
        return 0.0
    mean = sum(series) / n
    num = sum((series[i] - mean) * (series[i + lag] - mean) for i in range(n - lag))
    den = sum((x - mean) ** 2 for x in series)
    if den == 0:
        return 0.0
    return num / den


def _zodiac_wave_freq(records):
    zc = Counter()
    wc = Counter()
    for r in records:
        zc.update(r.zodiacs)
        wc.update(r.waves)
    return dict(zc), dict(wc)


def _odd_even_big_small(records):
    odd = even = big = small = 0
    for r in records:
        for n in _all_numbers(r):
            if n % 2 == 0:
                even += 1
            else:
                odd += 1
            if n >= 25:
                big += 1
            else:
                small += 1
    return {"odd": odd, "even": even, "big": big, "small": small}


def build_report(records) -> dict:
    """聚合所有统计, 并给出反推理候选模型清单。"""
    freq = number_frequency(records)
    chi2, p = chi_square_uniform(freq)
    gap = gap_stats(records)
    s = sum_stats(records)
    mk1 = markov_transition(records)
    mk2 = markov_transition_2(records)
    specials = [r.special for r in records]
    ac1 = autocorrelation(specials, 1)
    ac7 = autocorrelation(specials, 7)
    zodiac, wave = _zodiac_wave_freq(records)
    oebs = _odd_even_big_small(records)

    # 模型反推评分(0-1, 越高越可能"解释"该序列)
    models = []
    # 1. 均匀随机: p 值越大越像均匀随机
    models.append(("均匀分布(纯随机)", min(1.0, max(0.0, p)), "卡方 p=%.4f" % p))
    # 2. 马尔可夫: 转移矩阵熵越低越有依赖
    mk_entropy = 0.0
    for a, nxt in mk1.items():
        for prob in nxt.values():
            if prob > 0:
                mk_entropy -= prob * math.log2(prob)
    max_entropy = math.log2(49)
    mk_score = 1.0 - (mk_entropy / max_entropy) if max_entropy else 0
    models.append(("马尔可夫一阶转移", max(0.0, min(1.0, mk_score)), "转移熵=%.3f" % mk_entropy))
    # 3. 自相关周期: |ac| 越大越有周期性
    models.append(("自相关/周期性", min(1.0, abs(ac1) + abs(ac7) / 2),
                   "ac1=%.3f ac7=%.3f" % (ac1, ac7)))
    # 4. 和值正态: 变异系数越小越像正态
    cv = (s["std"] / s["mean"]) if s["mean"] else 0
    models.append(("和值正态分布", max(0.0, 1.0 - cv),
                   "均值=%.1f std=%.1f" % (s["mean"], s["std"])))
    # 5. 频率偏置: chi2 越大越偏离均匀(越有偏置)
    models.append(("频率/冷热号偏置", min(1.0, chi2 / 200.0), "chi2=%.1f" % chi2))

    return {
        "freq": freq, "chi2": (chi2, p), "gap": gap, "sum": s,
        "markov": mk1, "markov2": mk2, "ac": (ac1, ac7),
        "zodiac": zodiac, "wave": wave, "oebs": oebs,
        "models": models,
    }


def summarize_for_llm(report: dict, recent_n: int = 10) -> str:
    """将 AnalysisReport 压缩为大模型 prompt 摘要文本。"""
    freq = report["freq"]
    chi2, p = report["chi2"]
    s = report["sum"]
    ac1, ac7 = report["ac"]
    lines = []
    lines.append("== 澳门六合彩历史开奖统计分析摘要 ==")
    lines.append("【反推理候选数学模型】")
    for name, score, detail in report["models"]:
        lines.append(f"  - {name}: 拟合度={score:.3f} ({detail})")
    lines.append(f"【均匀性】卡方={chi2:.2f}, p={p:.4f} (p>0.05 表示不拒绝均匀分布)")
    lines.append(f"【和值】均值={s['mean']:.1f}, std={s['std']:.1f}, 区间=[{s['min']},{s['max']}]")
    lines.append(f"【自相关】lag1={ac1:.3f}, lag7={ac7:.3f}")
    # 冷热号 top10
    items = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
    hot = [str(n) for n, _ in items[:10]]
    cold = [str(n) for n, _ in items[-10:]]
    lines.append(f"【热号Top10】{','.join(hot)}")
    lines.append(f"【冷号Top10】{','.join(cold)}")
    oebs = report["oebs"]
    lines.append(f"【奇偶/大小】奇={oebs['odd']} 偶={oebs['even']} 大(>=25)={oebs['big']} 小={oebs['small']}")
    lines.append(f"【生肖分布】{report['zodiac']}")
    lines.append(f"【波色分布】{report['wave']}")
    lines.append("【特码序列(近10期)】见 prompt 中 recent 字段")
    return "\n".join(lines)
