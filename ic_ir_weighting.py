# -*- coding: utf-8 -*-
"""IC/IR 因子加权诊断与组合(量化领域经典因子合成方法)。

IC (Information Coefficient): 每期信号预测排名与实际结果排名的 Spearman 秩相关。
IR (Information ratio): Mean(IC)/Std(IC), 衡量因子预测力的稳定性。
组合逻辑: 按 IR 分配权重(w_i ∝ max(0, IR_i)), 用 IR 加权合成最终排名。

本模块同时是诊断工具: 若所有信号 IR ≈ 0, 说明信号无稳定预测力(噪声);
若有信号 IR 显著为正, 说明存在可利用的稳定结构。
"""
import math
import numpy as np
from scipy.stats import spearmanr
from collections import Counter

# 复用两层预计算的 folds 结构
# zodiac folds: {prob:{sig:{zodiac:prob}}, all_z, actual}
# wide folds:   {prob:{sig:{num:prob}}, actual}


def _per_period_ic(folds, signal_name, candidates):
    """逐期计算一个信号的 IC(Spearman 秩相关)。

    folds: precompute_folds 输出
    signal_name: 信号名
    candidates: 候选集列表(生肖用 all_z, 号码用 ALL_NUMS)
    返回 IC 序列(list[float])。
    """
    ics = []
    for f in folds:
        prob = f["prob"][signal_name]
        actual = f["actual"]
        # 预测排名: 信号概率越高 → 预测名次越靠前(数值越小)
        cand = candidates if not isinstance(candidates, str) else f.get("all_z", candidates)
        scores = [prob.get(c, 0.0) for c in cand]
        pred_rank = np.argsort(np.argsort([-s for s in scores]))  # 高分→小rank
        # 实际排名: 真实结果排第1(数值0), 其余随机或并列
        actual_rank = []
        actual_idx = cand.index(actual) if actual in list(cand) else -1
        for i, c in enumerate(cand):
            actual_rank.append(0 if i == actual_idx else 1)
        if actual_idx < 0:
            ics.append(0.0)
            continue
        r, _ = spearmanr(pred_rank, actual_rank)
        ics.append(0.0 if r != r else float(r))  # NaN->0
    return ics


def compute_ir_table(folds, signal_names, candidates):
    """计算所有信号的 IC 均值/标准差/IR/IC_IR(显著性)。

    返回 dict[signal] = {mean_ic, std_ic, ir, ic_positive_rate, n}
    """
    table = {}
    for name in signal_names:
        ics = _per_period_ic(folds, name, candidates)
        arr = np.array(ics)
        mean_ic = float(arr.mean())
        std_ic = float(arr.std())
        ir = mean_ic / std_ic if std_ic > 1e-9 else 0.0
        pos_rate = float((arr > 0).mean())
        # IR 的 t 统计量近似: IR * sqrt(n), |t|>2 近似显著
        t_stat = ir * math.sqrt(len(arr)) if len(arr) > 0 else 0.0
        table[name] = {
            "mean_ic": mean_ic, "std_ic": std_ic, "ir": ir,
            "ic_positive_rate": pos_rate, "t_stat": t_stat, "n": len(arr),
        }
    return table

def rank_ir_weighted(folds, fold_idx, active_signals, candidates,
                     ir_table, min_ir=0.0):
    """IR 加权合成排名: 权重 w_i = max(0, IR_i - min_ir)。

    对 fold_idx, 用 IR 权重加权各信号概率, 按加权概率排序。
    IR <= min_ir 的信号权重为0(剔除负贡献信号)。
    返回排名后的候选列表(降序)。
    """
    weights = {}
    for name in active_signals:
        ir = ir_table.get(name, {}).get("ir", 0.0)
        weights[name] = max(0.0, ir - min_ir)
    total_w = sum(weights.values())
    if total_w <= 0:
        # 所有信号 IR<=0, 降级均匀(等价于无信息)
        weights = {name: 1.0 for name in active_signals}
        total_w = len(active_signals)
    fold = folds[fold_idx]
    cand = candidates if not isinstance(candidates, str) else fold.get("all_z", candidates)
    combined = {c: 0.0 for c in cand}
    for name in active_signals:
        w = weights[name] / total_w
        prob = fold["prob"][name]
        for c in cand:
            combined[c] += w * prob.get(c, 0.0)
    return sorted(cand, key=lambda x: combined[x], reverse=True)


def walk_forward_ir_weighted(folds, active_signals, candidates, k,
                              rolling_window=60, min_ir=0.0):
    """滚动 IR walk-forward 回测。

    预测第 i 期时, 只用 [i-rolling_window, i-1] 的 IC 历史
    估算每个信号 IR, 再 IR 加权排名取 top-k。
    返回命中数组。避免用全样本 IR 造成前视。
    """
    n = len(folds)
    hits = []
    for i in range(n):
        start = max(0, i - rolling_window)
        # 用近期窗口估 IR
        window_folds = folds[start:i]
        if len(window_folds) < 10:
            ir_table = {name: {"ir": 1.0} for name in active_signals}
        else:
            ir_table = compute_ir_table(window_folds, active_signals, candidates)
        rank = rank_ir_weighted(folds, i, active_signals, candidates, ir_table, min_ir)
        actual = folds[i]["actual"]
        cand = candidates if not isinstance(candidates, str) else folds[i].get("all_z", candidates)
        hits.append(1 if actual in set(rank[:k]) else 0)
    return hits

def diagnose_zodiac(records):
    """诊断生肖层 6 信号: 输出 IR 表 + IR 加权命中率 vs 基线。"""
    import time
    import zodiac_ensemble as ZE
    print("=" * 65)
    print("生肖层 IC/IR 诊断 (6信号)")
    print("=" * 65)
    n_folds = min(876, len(records) - 10)
    t0 = time.time()
    folds = ZE.precompute_folds(records, n_folds)
    print(f"  {len(folds)} 折, 预计算 {time.time()-t0:.1f}s")
    ir_table = compute_ir_table(folds, ZE.SIGNAL_NAMES, "all_z")
    print(f"\n  {'信号':<12} {'MeanIC':>9} {'StdIC':>9} {'IR':>8} {'IC>0率':>8} {'|t|':>6} {'显著':>5}")
    print("  " + "-" * 60)
    for name in ZE.SIGNAL_NAMES:
        s = ir_table[name]
        sig = "***" if abs(s["t_stat"]) > 2 else ("" if abs(s["t_stat"]) > 1 else "")
        print(f"  {name:<12} {s['mean_ic']:>+9.4f} {s['std_ic']:>9.4f} {s['ir']:>+8.3f} "
              f"{s['ic_positive_rate']*100:>7.1f}% {abs(s['t_stat']):>6.2f} {sig:>5}")
    # IR 加权命中率
    print(f"\n  IR 加权 vs 各 k 命中率 (基线):")
    for k in (3, 4, 6):
        hits = walk_forward_ir_weighted(folds, ZE.SIGNAL_NAMES, "all_z", k, rolling_window=60)
        rate = sum(hits) / len(hits) * 100
        base = k / 12 * 100
        print(f"    {k}肖: {rate:.1f}% (基线{base:.1f}%, lift {rate-base:+.1f}%)")


def diagnose_wide(records):
    """诊断号码层 6 信号: 输出 IR 表 + IR 加权命中率 vs 基线。"""
    import time
    import wide_ensemble as WE
    print("\n" + "=" * 65)
    print("号码层 IC/IR 诊断 (6信号)")
    print("=" * 65)
    n_folds = min(876, len(records) - 10)
    t0 = time.time()
    folds = WE.precompute_folds(records, n_folds)
    print(f"  {len(folds)} 折, 预计算 {time.time()-t0:.1f}s")
    ir_table = compute_ir_table(folds, WE.SIGNAL_NAMES, WE.ALL_NUMS)
    print(f"\n  {'信号':<12} {'MeanIC':>9} {'StdIC':>9} {'IR':>8} {'IC>0率':>8} {'|t|':>6} {'显著':>5}")
    print("  " + "-" * 60)
    for name in WE.SIGNAL_NAMES:
        s = ir_table[name]
        sig = "***" if abs(s["t_stat"]) > 2 else ""
        print(f"  {name:<12} {s['mean_ic']:>+9.4f} {s['std_ic']:>9.4f} {s['ir']:>+8.3f} "
              f"{s['ic_positive_rate']*100:>7.1f}% {abs(s['t_stat']):>6.2f} {sig:>5}")
    print(f"\n  IR 加权 20颗命中率 (基线{20/49*100:.1f}%):")
    hits = walk_forward_ir_weighted(folds, WE.SIGNAL_NAMES, WE.ALL_NUMS, 20, rolling_window=60)
    rate = sum(hits) / len(hits) * 100
    print(f"    20颗: {rate:.1f}% (lift {rate-20/49*100:+.1f}%)")


if __name__ == "__main__":
    import json, sys
    sys.stdout.reconfigure(encoding="utf-8")
    from data_fetcher import Record
    raw = json.load(open("macau_history.json", encoding="utf-8"))
    records = [Record(expect=str(r["expect"]), open_time=str(r.get("openTime", "")),
        regular=r["regular"], special=r["special"],
        waves=r.get("waves", []), zodiacs=r.get("zodiacs", [])) for r in raw]
    diagnose_zodiac(records)
    diagnose_wide(records)
