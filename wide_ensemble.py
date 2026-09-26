# -*- coding: utf-8 -*-
"""20颗特码大集合 Stacking 非线性集成(与三肖/四肖/六肖的 zodiac_ensemble 平行)。

流程: 线性加权 ->(剔除高相关信号)-> BMA动态加权 -> Stacking非线性集成(GBDT)
6 路号码 base 信号对 1~49 每个号码输出归一化概率, 作为 GBDT 特征,
真实特码作为标签, 输出 P(命中), 取 top-20。
"""
import math
import numpy as np
from collections import Counter, defaultdict
from core.evaluation_engine import evaluate_ranked_walk_forward
from core.signal_registry import register_signal
from analysis import build_report
from dimensions import build_zodiac_map, special_zodiac_of

NUM_MIN, NUM_MAX = 1, 49
ALL_NUMS = list(range(NUM_MIN, NUM_MAX + 1))
N_NUMS = len(ALL_NUMS)

# Stacking 超参(与 zodiac_ensemble 对齐)
STACK_WINDOW = 150
STACK_N_EST = 80
STACK_MAX_DEPTH = 3
STACK_LR = 0.1

SIGNAL_NAMES = ["freq", "gap", "markov", "recent", "zodiac", "uniform"]
DEFAULT_W = {"freq": 3.0, "gap": 2.0, "markov": 2.5, "recent": 1.5, "zodiac": 1.5, "uniform": 0.5}


def _norm_prob(d):
    """归一化 {num: score} 为概率分布(和为1)。全0时均匀分布。"""
    vals = [max(0.0, d.get(n, 0.0)) for n in ALL_NUMS]
    total = sum(vals)
    if total <= 0:
        return {n: 1.0 / N_NUMS for n in ALL_NUMS}
    return {n: v / total for n, v in zip(ALL_NUMS, vals)}


def _signal_freq(report):
    """频率热号: 历史出现最多的号码得分最高。"""
    freq = report["freq"]
    fmax = max(freq.values()) if freq and max(freq.values()) > 0 else 1
    return {n: freq.get(n, 0) / fmax for n in ALL_NUMS}


def _signal_gap(report):
    """遗漏回归: 当前遗漏越大越到期(冷号到期)。"""
    gap = report["gap"]
    gmax = max((gap[n]["current"] for n in ALL_NUMS), default=1) or 1
    return {n: gap[n]["current"] / gmax for n in ALL_NUMS}


def _signal_markov(records, report):
    """马尔可夫一阶: 上一期特码转移概率最高的号码。"""
    last = records[-1].special if records else 0
    trans = report["markov"].get(last, {})
    tmax = max(trans.values()) if trans else 0
    if tmax <= 0:
        return {n: 0.0 for n in ALL_NUMS}
    return {n: trans.get(n, 0) / tmax for n in ALL_NUMS}


def _signal_recent(records):
    """近期趋势: 最近 20 期出现最多的号码。"""
    recent = records[-20:]
    rc = Counter(r.special for r in recent)
    rmax = max(rc.values()) if rc else 1
    return {n: rc.get(n, 0) / rmax for n in ALL_NUMS}


def _signal_zodiac(records, report):
    """生肖信号: 三/四/六肖命中号码倾向。用生肖打分映射到号码。"""
    try:
        from dimensions import _zodiac_six_scores
        zmap = build_zodiac_map(records)
        if not zmap:
            return {n: 0.0 for n in ALL_NUMS}
        scores = _zodiac_six_scores(records, None, zmap, 0.0)
        smax = max(scores.values()) if scores and max(scores.values()) > 0 else 1
        out = {n: 0.0 for n in ALL_NUMS}
        for n, z in zmap.items():
            out[n] = scores.get(z, 0.0) / smax
        return out
    except Exception:
        return {n: 0.0 for n in ALL_NUMS}


def _signal_uniform():
    """结构先验: 1/49 均匀(基线参考, 号码层无胖瘦差异)。"""
    return {n: 1.0 for n in ALL_NUMS}


def _compute_fold_probs(records):
    """计算一折的各信号归一化概率。返回 (prob_dict, None)。"""
    report = build_report(records)
    raw = {
        "freq": _signal_freq(report),
        "gap": _signal_gap(report),
        "markov": _signal_markov(records, report),
        "recent": _signal_recent(records),
        "zodiac": _signal_zodiac(records, report),
        "uniform": _signal_uniform(),
    }
    prob = {name: _norm_prob(raw[name]) for name in SIGNAL_NAMES}
    return prob

def precompute_folds(records, n_folds):
    """walk-forward 预计算每折信号概率 + 真实结果。"""
    folds = []
    n_clamped = max(1, min(n_folds, len(records) - 10))
    for i in range(n_clamped):
        split = len(records) - n_clamped + i
        train = records[:split]
        actual = records[split]
        prob = _compute_fold_probs(train)
        folds.append({"prob": prob, "actual": actual.special})
    return folds


def compute_correlation_matrix(folds):
    """信号间 Pearson 相关(各信号在所有折上的号码概率向量拼接)。"""
    sigvecs = {n: [] for n in SIGNAL_NAMES}
    for f in folds:
        for n in SIGNAL_NAMES:
            sigvecs[n].extend([f["prob"][n].get(num, 0.0) for num in ALL_NUMS])
    corr = {}
    for i, ni in enumerate(SIGNAL_NAMES):
        for j, nj in enumerate(SIGNAL_NAMES):
            vi, vj = np.array(sigvecs[ni]), np.array(sigvecs[nj])
            if vi.std() > 1e-12 and vj.std() > 1e-12:
                corr[(ni, nj)] = float(np.corrcoef(vi, vj)[0, 1])
            else:
                corr[(ni, nj)] = 0.0
    return corr


def select_active_signals(folds, threshold=0.85):
    """剔除高相关信号, 保留单信号命中率高的代表。"""
    sigvecs = {n: [] for n in SIGNAL_NAMES}
    for f in folds:
        for n in SIGNAL_NAMES:
            sigvecs[n].extend([f["prob"][n].get(num, 0.0) for num in ALL_NUMS])
    hr = {}
    for n in SIGNAL_NAMES:
        h = sum(1 for f in folds if f["actual"] in
                set(sorted(ALL_NUMS, key=lambda x: f["prob"][n].get(x, 0.0), reverse=True)[:20]))
        hr[n] = h / len(folds) if folds else 0.0
    order = sorted(SIGNAL_NAMES, key=lambda n: hr[n], reverse=True)
    kept, removed = [], set()
    for n in order:
        if n in removed:
            continue
        kept.append(n)
        for o in SIGNAL_NAMES:
            if o == n or o in removed:
                continue
            vi, vo = np.array(sigvecs[n]), np.array(sigvecs[o])
            if vi.std() > 1e-9 and vo.std() > 1e-9:
                cc = float(np.corrcoef(vi, vo)[0, 1])
                if abs(cc) > threshold:
                    removed.add(o)
    return kept


def _rank_stacking(folds, fold_idx, active_signals=None,
                   window=STACK_WINDOW, ne=STACK_N_EST, md=STACK_MAX_DEPTH, lr=STACK_LR):
    """对 fold_idx 训练 GBDT 回归器, 返回号码排名(按 P(命中) 降序)。

    号码层 49 类别/每折仅 1 正样本(2%正样本率), 分类器严重偏向"全不中"。
    故采用回归器+软标签: 真实号码 target=1.0, 其余 target=信号均值*0.3
    (让回归器学习排序而非硬分类)。实测优于分类器变体。数据不足降级线性。
    """
    if active_signals is None:
        active_signals = SIGNAL_NAMES
    start = max(0, fold_idx - window)
    feats, targets = [], []
    for t in range(start, fold_idx):
        f = folds[t]
        for num in ALL_NUMS:
            feats.append([f["prob"][name].get(num, 0.0) for name in active_signals])
            is_actual = 1.0 if num == f["actual"] else 0.0
            sig_mean = sum(f["prob"][name].get(num, 0.0) for name in active_signals) / len(active_signals)
            targets.append(max(is_actual, sig_mean * 0.3))
    fold = folds[fold_idx]
    if len(feats) < 200 or sum(t > 0.5 for t in targets) < 3:
        return _rank_linear(folds, fold_idx, active_signals)
    try:
        from sklearn.ensemble import GradientBoostingRegressor
    except ImportError:
        return _rank_linear(folds, fold_idx, active_signals)
    reg = GradientBoostingRegressor(n_estimators=ne, max_depth=md,
        learning_rate=lr, subsample=0.8, random_state=42)
    reg.fit(feats, targets)
    test_X = [[fold["prob"][name].get(num, 0.0) for name in active_signals] for num in ALL_NUMS]
    p_pos = reg.predict(test_X)
    return [num for _, num in sorted(zip(p_pos, ALL_NUMS), reverse=True)]


def _rank_linear(folds, fold_idx, active_signals=None, weights=None):
    if active_signals is None:
        active_signals = SIGNAL_NAMES
    if weights is None:
        weights = DEFAULT_W
    fold = folds[fold_idx]
    combined = {num: 0.0 for num in ALL_NUMS}
    for name in active_signals:
        w = weights.get(name, 1.0)
        for num, p in fold["prob"][name].items():
            combined[num] += w * p
    return sorted(ALL_NUMS, key=lambda x: combined[x], reverse=True)

# ==================== 公开 API ====================

_BT_CACHE = {}


def predict_stacking_numbers(records, k=20, active_signals=None,
                              window=STACK_WINDOW, ne=STACK_N_EST,
                              md=STACK_MAX_DEPTH, lr=STACK_LR):
    """预测下一期 top-k 号码: 用全部历史训练 GBDT, 预测当前信号特征。

    数据不足(历史 < 60 期)降级为线性加权。返回号码列表(未排序)。
    """
    if active_signals is None:
        active_signals = SIGNAL_NAMES
    if len(records) < 60:
        report = build_report(records)
        scores = _wide_linear_scores(records, report, active_signals)
        ranked = sorted(ALL_NUMS, key=lambda x: scores[x], reverse=True)
        return ranked[:k]
    n_folds = max(1, len(records) - 10)
    folds = precompute_folds(records, n_folds)
    feats, targets = [], []
    for f in folds:
        for num in ALL_NUMS:
            feats.append([f["prob"][name].get(num, 0.0) for name in active_signals])
            is_actual = 1.0 if num == f["actual"] else 0.0
            sig_mean = sum(f["prob"][name].get(num, 0.0) for name in active_signals) / len(active_signals)
            targets.append(max(is_actual, sig_mean * 0.3))
    if len(feats) < 200 or sum(t > 0.5 for t in targets) < 3:
        rank = _rank_linear(folds, len(folds) - 1, active_signals)
        return rank[:k]
    try:
        from sklearn.ensemble import GradientBoostingRegressor
    except ImportError:
        rank = _rank_linear(folds, len(folds) - 1, active_signals)
        return rank[:k]
    reg = GradientBoostingRegressor(n_estimators=ne, max_depth=md,
        learning_rate=lr, subsample=0.8, random_state=42)
    reg.fit(feats, targets)
    prob = _compute_fold_probs(records)
    test_X = [[prob[name].get(num, 0.0) for name in active_signals] for num in ALL_NUMS]
    p_pos = reg.predict(test_X)
    ranked = [num for _, num in sorted(zip(p_pos, ALL_NUMS), reverse=True)]
    return ranked[:k]


def _wide_linear_scores(records, report, active_signals=None, weights=None):
    if active_signals is None:
        active_signals = SIGNAL_NAMES
    if weights is None:
        weights = DEFAULT_W
    raw = {
        "freq": _signal_freq(report),
        "gap": _signal_gap(report),
        "markov": _signal_markov(records, report),
        "recent": _signal_recent(records),
        "zodiac": _signal_zodiac(records, report),
        "uniform": _signal_uniform(),
    }
    prob = {name: _norm_prob(raw[name]) for name in SIGNAL_NAMES}
    combined = {num: 0.0 for num in ALL_NUMS}
    for name in active_signals:
        w = weights.get(name, 1.0)
        for num, p in prob[name].items():
            combined[num] += w * p
    return combined


def backtest_stacking(records, backtest_n, k=20, active_signals=None, window=STACK_WINDOW):
    """V5.2 strict OOS evaluation; the ranking predictor sees only each fold's prefix."""
    if active_signals is None: active_signals = SIGNAL_NAMES
    key = (len(records), backtest_n, k, tuple(active_signals), window, "oos-v52")
    if key in _BT_CACHE: return _BT_CACHE[key]
    n_test = max(1, min(backtest_n, len(records)-10))
    def ranker(train, cands):
        return predict_stacking_numbers(train, k=k, active_signals=active_signals, window=window)
    report = evaluate_ranked_walk_forward(records, ALL_NUMS, lambda r:r.special, ranker, initial_train=len(records)-n_test, test_size=n_test, top_k=(k,))
    hits = [f.hit_at_k[k] for f in report.folds]
    _BT_CACHE[key] = hits
    return hits

def clear_cache():
    _BT_CACHE.clear()


if __name__ == "__main__":
    import json, sys, time
    sys.stdout.reconfigure(encoding="utf-8")
    from data_fetcher import Record
    raw = json.load(open("macau_history.json", encoding="utf-8"))
    records = [Record(expect=str(r["expect"]), open_time=str(r.get("openTime", "")),
        regular=r["regular"], special=r["special"],
        waves=r.get("waves", []), zodiacs=r.get("zodiacs", [])) for r in raw]
    t0 = time.time()
    hits = backtest_stacking(records, 30, 20)
    n = len(hits)
    h = sum(hits)
    base = 20 / 49 * 100
    print("=" * 55)
    print(f"20颗大集合 Stacking: 近{n}期命中 {h/n*100:.1f}% (基线{base:.1f}% lift{h/n*100-base:+.1f}%)")
    nums = predict_stacking_numbers(records, 20)
    print(f"预测20颗: {sorted(nums)}")
    print(f"耗时 {time.time()-t0:.1f}s")\n\n# V5.2 registry registration occurs after all signal functions are defined.\nfor _n, _fn in {\n    "freq": _signal_freq, "gap": _signal_gap, "markov": _signal_markov,\n    "recent": _signal_recent, "zodiac": _signal_zodiac, "uniform": _signal_uniform,\n}.items():\n    register_signal(f"wide.{_n}.v1", "v1", "wide", f"wide base signal: {_n}", _fn)\n
