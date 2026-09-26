# -*- coding: utf-8 -*-
"""六维度特码属性 Stacking 非线性集成(与生肖层/号码层平行)。

对每个维度(波色/生肖/尾数/大小/奇偶/头数)独立训练一个 GBDT 分类器:
  特征 = 各 base 信号对每个候选值的归一化概率
  标签 = 真实取值
  预测 = GBDT 输出 P(命中), 取 argmax

解决旧版"衰减贝叶斯+马尔可夫混合"在结构性先验强的维度(波色/生肖/奇偶)
预测值期期不变的缺陷; Stacking 能学到信号的非线性交互, 使预测随近期翻动。
"""
import math
import numpy as np
from collections import Counter, defaultdict
from core.evaluation_engine import evaluate_ranked_walk_forward
from core.signal_registry import register_signal
from dimensions import (
    DIMENSIONS, comb_prior, build_zodiac_map, _freq_predict,
)

STACK_WINDOW = 150
STACK_N_EST = 80
STACK_MAX_DEPTH = 3
STACK_LR = 0.1

SIGNAL_NAMES = ["freq", "decay", "markov", "recent", "gap", "prior"]


def _extract_seq(records, extract):
    return [extract(r) for r in records if extract(r) is not None]


def _signal_freq(seq, vals):
    c = Counter(seq)
    total = len(seq) or 1
    return {v: c.get(v, 0) / total for v in vals}


def _signal_decay(seq, vals, half_life=24):
    n = len(seq)
    if n == 0:
        return {v: 1.0 / len(vals) for v in vals}
    decay = 0.5 ** (1.0 / half_life) if half_life > 0 else 1.0
    df = defaultdict(float)
    for i, v in enumerate(seq):
        df[v] += decay ** (n - 1 - i)
    tot = sum(df.values()) or 1.0
    return {v: df.get(v, 0.0) / tot for v in vals}


def _signal_markov(seq, vals):
    if len(seq) < 2:
        return {v: 1.0 / len(vals) for v in vals}
    last = seq[-1]
    trans = defaultdict(Counter)
    for a, b in zip(seq[:-1], seq[1:]):
        trans[a][b] += 1
    row = trans.get(last, Counter())
    s = sum(row.values())
    uni = Counter(seq)
    total = len(seq)
    out = {}
    for v in vals:
        if s > 0:
            out[v] = 0.6 * (row.get(v, 0) / s) + 0.4 * (uni.get(v, 0) / total)
        else:
            out[v] = uni.get(v, 0) / total if total else 1.0 / len(vals)
    return out


def _signal_recent(seq, vals, window=20):
    recent = seq[-window:]
    c = Counter(recent)
    total = len(recent) or 1
    return {v: c.get(v, 0) / total for v in vals}


def _signal_gap(seq, vals):
    """遗漏回归: 越久没出越到期。"""
    n = len(seq)
    if n == 0:
        return {v: 1.0 / len(vals) for v in vals}
    last_seen = {}
    for i, v in enumerate(seq):
        last_seen[v] = i
    out = {}
    for v in vals:
        gap = n - 1 - last_seen.get(v, -1)
        out[v] = gap / n if n > 0 else 0.0
    tot = sum(out.values()) or 1.0
    return {v: out[v] / tot for v in vals}


def _signal_prior(prior, vals):
    return {v: prior.get(v, 0.0) for v in vals}
\n\nfor _n, _fn in {"freq":_signal_freq,"decay":_signal_decay,"markov":_signal_markov,"recent":_signal_recent,"gap":_signal_gap,"prior":_signal_prior}.items():
    register_signal(f"dimension.{_n}.v1","v1","dimension",f"dimension base signal: {_n}",_fn)

def _compute_fold_probs(train, extract, prior, zmap):
    """计算一折的各信号概率。返回 {signal: {val: prob}}。"""
    vals = list(prior.keys()) if prior else []
    if not vals:
        return {}
    seq = _extract_seq(train, extract)
    raw = {
        "freq": _signal_freq(seq, vals),
        "decay": _signal_decay(seq, vals),
        "markov": _signal_markov(seq, vals),
        "recent": _signal_recent(seq, vals),
        "gap": _signal_gap(seq, vals),
        "prior": _signal_prior(prior, vals),
    }
    # 归一化每个信号为概率分布
    prob = {}
    for name in SIGNAL_NAMES:
        d = raw[name]
        tot = sum(d.values())
        if tot > 0:
            prob[name] = {v: d[v] / tot for v in vals}
        else:
            prob[name] = {v: 1.0 / len(vals) for v in vals}
    return prob


def precompute_folds(records, extract, prior_fn, n_folds):
    """walk-forward 预计算每折信号概率 + 真实结果。

    prior_fn(zmap) 返回该维度的候选值->先验概率 dict。
    返回 list[dict]: {prob, vals, actual}
    """
    folds = []
    n_clamped = max(1, min(n_folds, len(records) - 10))
    for i in range(n_clamped):
        split = len(records) - n_clamped + i
        train = records[:split]
        actual = records[split]
        train_zmap = build_zodiac_map(train)
        prior = prior_fn(train_zmap)
        if not prior:
            continue
        prob = _compute_fold_probs(train, extract, prior, train_zmap)
        folds.append({
            "prob": prob,
            "vals": list(prior.keys()),
            "actual": extract(actual),
        })
    return folds

def _rank_stacking(folds, fold_idx, window=STACK_WINDOW,
                   ne=STACK_N_EST, md=STACK_MAX_DEPTH, lr=STACK_LR):
    """对 fold_idx 训练 GBDT 分类器, 返回候选值排名(按 P(命中) 降序)。"""
    start = max(0, fold_idx - window)
    feats, labels = [], []
    for t in range(start, fold_idx):
        f = folds[t]
        for v in f["vals"]:
            feats.append([f["prob"][name].get(v, 0.0) for name in SIGNAL_NAMES])
            labels.append(v)
    fold = folds[fold_idx]
    vals = fold["vals"]
    if len(feats) < 30 or len(set(labels)) < 2:
        return _rank_linear(folds, fold_idx)
    try:
        from sklearn.ensemble import GradientBoostingClassifier
    except ImportError:
        return _rank_linear(folds, fold_idx)
    clf = GradientBoostingClassifier(n_estimators=ne, max_depth=md,
        learning_rate=lr, subsample=0.8, random_state=42)
    clf.fit(feats, labels)
    test_X = [[fold["prob"][name].get(v, 0.0) for name in SIGNAL_NAMES] for v in vals]
    proba = clf.predict_proba(test_X)
    classes = list(clf.classes_)
    p_actual = np.zeros(len(vals))
    for j, v in enumerate(vals):
        if v in classes:
            p_actual[j] = proba[j, classes.index(v)]
        else:
            p_actual[j] = 0.0
    return [v for _, v in sorted(zip(p_actual, vals), reverse=True)]


def _rank_linear(folds, fold_idx):
    """线性加权(降级用): 各信号概率求和。"""
    fold = folds[fold_idx]
    vals = fold["vals"]
    combined = {v: 0.0 for v in vals}
    for name in SIGNAL_NAMES:
        for v in vals:
            combined[v] += fold["prob"][name].get(v, 0.0)
    return sorted(vals, key=lambda x: combined[x], reverse=True)


_BT_CACHE = {}


def predict_stacking_value(records, extract, prior_fn, zmap):
    """预测下一期该维度的取值: 用全部历史训练 GBDT, 预测当前信号。

    数据不足(历史 < 30 期)降级为全历史众数。返回预测值。
    """
    prior = prior_fn(zmap)
    if not prior:
        return _freq_predict(records, extract)
    if len(records) < 30:
        return _freq_predict(records, extract)
    n_folds = max(1, len(records) - 10)
    folds = precompute_folds(records, extract, prior_fn, n_folds)
    feats, labels = [], []
    for f in folds:
        for v in f["vals"]:
            feats.append([f["prob"][name].get(v, 0.0) for name in SIGNAL_NAMES])
            labels.append(v)
    if len(feats) < 30 or len(set(labels)) < 2:
        rank = _rank_linear(folds, len(folds) - 1)
        return rank[0] if rank else _freq_predict(records, extract)
    try:
        from sklearn.ensemble import GradientBoostingClassifier
    except ImportError:
        rank = _rank_linear(folds, len(folds) - 1)
        return rank[0] if rank else _freq_predict(records, extract)
    clf = GradientBoostingClassifier(n_estimators=STACK_N_EST, max_depth=STACK_MAX_DEPTH,
        learning_rate=STACK_LR, subsample=0.8, random_state=42)
    clf.fit(feats, labels)
    prob = _compute_fold_probs(records, extract, prior, zmap)
    vals = list(prior.keys())
    test_X = [[prob[name].get(v, 0.0) for name in SIGNAL_NAMES] for v in vals]
    proba = clf.predict_proba(test_X)
    classes = list(clf.classes_)
    p_actual = np.zeros(len(vals))
    for j, v in enumerate(vals):
        if v in classes:
            p_actual[j] = proba[j, classes.index(v)]
        else:
            p_actual[j] = 0.0
    ranked = [v for _, v in sorted(zip(p_actual, vals), reverse=True)]
    return ranked[0] if ranked else _freq_predict(records, extract)


def predict_stacking_topk(records, extract, prior_fn, zmap, k):
    """预测下一期该维度 top-k 取值列表(按 P(命中) 降序)。数据不足降级线性。"""
    prior = prior_fn(zmap)
    if not prior:
        seq = [extract(r) for r in records if extract(r) is not None]
        c = Counter(seq)
        return [v for v, _ in c.most_common(k)]
    if len(records) < 30:
        seq = [extract(r) for r in records if extract(r) is not None]
        c = Counter(seq)
        return [v for v, _ in c.most_common(k)]
    n_folds = max(1, len(records) - 10)
    folds = precompute_folds(records, extract, prior_fn, n_folds)
    feats, labels = [], []
    for f in folds:
        for v in f["vals"]:
            feats.append([f["prob"][name].get(v, 0.0) for name in SIGNAL_NAMES])
            labels.append(v)
    if len(feats) < 30 or len(set(labels)) < 2:
        folds2 = precompute_folds(records, extract, prior_fn, max(1, len(records) - 10))
        rank = _rank_linear(folds2, len(folds2) - 1)
        return rank[:k]
    try:
        from sklearn.ensemble import GradientBoostingClassifier
    except ImportError:
        folds2 = precompute_folds(records, extract, prior_fn, max(1, len(records) - 10))
        rank = _rank_linear(folds2, len(folds2) - 1)
        return rank[:k]
    clf = GradientBoostingClassifier(n_estimators=STACK_N_EST, max_depth=STACK_MAX_DEPTH,
        learning_rate=STACK_LR, subsample=0.8, random_state=42)
    clf.fit(feats, labels)
    prob = _compute_fold_probs(records, extract, prior, zmap)
    vals = list(prior.keys())
    test_X = [[prob[name].get(v, 0.0) for name in SIGNAL_NAMES] for v in vals]
    proba = clf.predict_proba(test_X)
    classes = list(clf.classes_)
    p_actual = np.zeros(len(vals))
    for j, v in enumerate(vals):
        if v in classes:
            p_actual[j] = proba[j, classes.index(v)]
    ranked = [v for _, v in sorted(zip(p_actual, vals), reverse=True)]
    return ranked[:k]


def backtest_stacking_topk(records, extract, prior_fn, backtest_n, k, window=STACK_WINDOW):
    """V5.2 strict OOS top-k evaluation."""
    key=(id(extract),backtest_n,k,window,"oos-v52-topk")
    if key in _BT_CACHE:return _BT_CACHE[key]
    n_test=max(1,min(backtest_n,len(records)-10))
    def ranker(train,cands):
        return predict_stacking_topk(train,extract,prior_fn,build_zodiac_map(train),k)
    candidates=list(prior_fn(build_zodiac_map(records)).keys())
    report=evaluate_ranked_walk_forward(records,candidates,extract,ranker,initial_train=len(records)-n_test,test_size=n_test,top_k=(k,))
    hits=[f.hit_at_k[k] for f in report.folds];_BT_CACHE[key]=hits;return hits


def backtest_stacking(records, extract, prior_fn, backtest_n, window=STACK_WINDOW):
    """V5.2 strict OOS exact-value evaluation."""
    key=(id(extract),backtest_n,window,"oos-v52")
    if key in _BT_CACHE:return _BT_CACHE[key]
    n_test=max(1,min(backtest_n,len(records)-10))
    def ranker(train,cands):
        return predict_stacking_topk(train,extract,prior_fn,build_zodiac_map(train),1)
    candidates=list(prior_fn(build_zodiac_map(records)).keys())
    report=evaluate_ranked_walk_forward(records,candidates,extract,ranker,initial_train=len(records)-n_test,test_size=n_test,top_k=(1,))
    hits=[f.hit_at_k[1] for f in report.folds];_BT_CACHE[key]=hits;return hits


def clear_cache():
    _BT_CACHE.clear()

DIM_PRIOR_FN = {}


def _make_prior_fn(dim_name):
    """为维度 dim_name 创建 prior_fn(zmap) -> {val: prior_prob}。"""
    def fn(zmap):
        return comb_prior(dim_name, zmap)
    return fn


for _name, _extract in DIMENSIONS:
    DIM_PRIOR_FN[_name] = _make_prior_fn(_name)


def predict_all_dimensions(records, backtest_n=30):
    """预测六个维度, 返回 list[(name, value, hit_rate, baseline, lift, uniq_count)]。"""
    import time
    results = []
    for name, extract in DIMENSIONS:
        zmap = build_zodiac_map(records)
        prior_fn = DIM_PRIOR_FN[name]
        value = predict_stacking_value(records, extract, prior_fn, zmap)
        hits = backtest_stacking(records, extract, prior_fn, backtest_n)
        n = len(hits)
        hr = sum(hits) / n if n else 0.0
        # 基线: 该维度精确组合先验的 max(最常见的候选值概率)
        prior = comb_prior(name, zmap)
        baseline = max(prior.values()) if prior else 1.0 / 12
        lift = hr - baseline
        # 预测多样性: 近 backtest_n 期预测值唯一数
        uniq = _count_unique_predictions(records, extract, prior_fn, backtest_n)
        results.append((name, value, hr, baseline, lift, uniq))
    return results


def _count_unique_predictions(records, extract, prior_fn, n):
    """统计近 n 期预测值的唯一值数(衡量是否"期期一样")。"""
    total_folds = max(n + STACK_WINDOW, len(records) - 10)
    folds = precompute_folds(records, extract, prior_fn, total_folds)
    start_idx = len(folds) - n
    preds = set()
    for i in range(start_idx, len(folds)):
        rank = _rank_stacking(folds, i)
        if rank:
            preds.add(rank[0])
    return len(preds)


if __name__ == "__main__":
    import json, sys, time
    sys.stdout.reconfigure(encoding="utf-8")
    from data_fetcher import Record
    raw = json.load(open("macau_history.json", encoding="utf-8"))
    records = [Record(expect=str(r["expect"]), open_time=str(r.get("openTime", "")),
        regular=r["regular"], special=r["special"],
        waves=r.get("waves", []), zodiacs=r.get("zodiacs", [])) for r in raw]
    t0 = time.time()
    print("=" * 65)
    print("六维度 Stacking 回测 (近100期)")
    print("=" * 65)
    clear_cache()
    results = predict_all_dimensions(records, backtest_n=100)
    print(f"{'维度':<8} {'预测值':>6} {'命中率':>8} {'基线':>8} {'lift':>8} {'唯一值数':>8}")
    print("-" * 65)
    for name, value, hr, base, lift, uniq in results:
        print(f"{name:<8} {str(value):>6} {hr*100:>7.1f}% {base*100:>7.1f}% {lift*100:>+7.1f}% {uniq:>8}")
    print("=" * 65)
    print(f"耗时 {time.time()-t0:.1f}s")
