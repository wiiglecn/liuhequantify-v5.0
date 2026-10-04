# -*- coding: utf-8 -*-
"""三阶段生肖集成优化管线(已接入主预测链):

  线性加权 ->(1.剔除高相关信号)-> BMA动态贝叶斯加权 ->(2.阈值过滤)-> Stacking非线性集成

最终采用 Stacking(GBDT 元学习器): 用各 base 信号对每个生肖的归一化概率作为特征,
真实生肖作为标签, GBDT 学习非线性映射, 输出 P(命中), 取 top-k。

历史研究结果以当前数据集的独立 OOS 运行报告为准；本模块不在源码中固化未经当前版本重新验证的命中率。
"""
import math
import numpy as np
from collections import Counter
from core.evaluation_engine import evaluate_ranked_walk_forward
from core.signal_registry import register_signal
from dimensions import (
    build_zodiac_map, special_zodiac_of, _SigCtx,
    _signal_freq, _signal_markov, _signal_cross_dim,
    _signal_recent, _signal_bayes, _signal_numcount,
)

BASE_SIGNALS = [
    ("freq", _signal_freq), ("markov", _signal_markov),
    ("cross_dim", _signal_cross_dim), ("recent", _signal_recent),
    ("bayes", _signal_bayes), ("numcount", _signal_numcount),
]
SIGNAL_NAMES = [n for n, _ in BASE_SIGNALS]

# V5.2 unified registry: metadata only; execution remains in this module.
for _n, _fn in BASE_SIGNALS:
    register_signal(f"zodiac.{_n}.v1", "v1", "zodiac", f"zodiac base signal: {_n}", _fn)
DEFAULT_W = {"freq": 3.0, "markov": 4.0, "cross_dim": 2.0,
             "recent": 2.8, "bayes": 1.5, "numcount": 1.5}

# Stacking 超参(876 期调参定稿)
STACK_WINDOW = 150
STACK_N_EST = 80
STACK_MAX_DEPTH = 3
STACK_LR = 0.1
# 二分类 GBDT 的温度缩放是单调变换，不改变按正类概率排序的名次。\n# 因而不把它作为命中率优化项，避免把“概率校准”误认为“排序提升”。\nSTACK_TEMP = 1.0


def _compute_signals(train, zmap):
    seq = [special_zodiac_of(r) for r in train if special_zodiac_of(r)]
    all_z = sorted(set(seq))
    if not all_z:
        return {}, all_z
    ctx = _SigCtx(seq, all_z, None, zmap, 0.0, train)
    out = {}
    for name, fn in BASE_SIGNALS:
        contrib = fn(ctx)
        out[name] = dict(contrib) if contrib else {}
    return out, all_z


def _normalize_to_prob(scores_dict, all_z):
    vals = [max(0.0, scores_dict.get(z, 0.0)) for z in all_z]
    total = sum(vals)
    if total <= 0:
        u = 1.0 / len(all_z)
        return {z: u for z in all_z}
    return {z: v / total for z, v in zip(all_z, vals)}


def _compute_fold_probs(train, zmap):
    raw, all_z = _compute_signals(train, zmap)
    prob = {name: _normalize_to_prob(raw.get(name, {}), all_z) for name in SIGNAL_NAMES}
    return prob, all_z

def precompute_folds(records, n_folds):
    """walk-forward 预计算每折信号概率 + 真实结果。"""
    folds = []
    n_clamped = max(1, min(n_folds, len(records) - 10))
    for i in range(n_clamped):
        split = len(records) - n_clamped + i
        train = records[:split]
        actual = records[split]
        zmap = build_zodiac_map(train)
        prob, all_z = _compute_fold_probs(train, zmap)
        folds.append({"prob": prob, "all_z": all_z, "actual": special_zodiac_of(actual)})
    return folds


def _rank_stacking(folds, fold_idx, active_signals=None,
                   window=STACK_WINDOW, ne=STACK_N_EST, md=STACK_MAX_DEPTH, lr=STACK_LR):
    """对 fold_idx 训练 GBDT 并返回生肖排名(按 P(命中) 降序)。数据不足降级 BMA。"""
    if active_signals is None:
        active_signals = SIGNAL_NAMES
    start = max(0, fold_idx - window)
    feats, labels = [], []
    for t in range(start, fold_idx):
        f = folds[t]
        for z in f["all_z"]:
            feats.append([f["prob"][name].get(z, 0.0) for name in active_signals])
            labels.append(1 if z == f["actual"] else 0)
    fold = folds[fold_idx]
    all_z = fold["all_z"]
    if len(feats) < 50 or sum(labels) < 3:
        return _rank_bma(folds, fold_idx, active_signals)
    try:
        from sklearn.ensemble import GradientBoostingClassifier
    except ImportError:
        return _rank_bma(folds, fold_idx, active_signals)
    clf = GradientBoostingClassifier(n_estimators=ne, max_depth=md,
        learning_rate=lr, subsample=0.8, random_state=42)
    clf.fit(feats, labels)
    test_X = [[fold["prob"][name].get(z, 0.0) for name in active_signals] for z in all_z]
    proba = clf.predict_proba(test_X)
    pos_col = list(clf.classes_).index(1) if 1 in clf.classes_ else 0
    p_pos = proba[:, pos_col] if proba.ndim == 2 else proba
    return [z for _, z in sorted(zip(p_pos, all_z), reverse=True)]


def _bma_weights(folds, fold_idx, active, decay=0.95, min_w=0.05):
    ns = len(active)
    lw = np.zeros(ns)
    for t in range(fold_idx):
        df = decay ** (fold_idx - 1 - t)
        actual = folds[t]["actual"]
        for j, name in enumerate(active):
            p = folds[t]["prob"][name].get(actual, 1e-6)
            lw[j] += math.log(max(p, 1e-6)) * df
    lw -= lw.max()
    w = np.exp(lw)
    w = w / w.sum()
    w = np.maximum(w, min_w)
    return w / w.sum()


def _rank_bma(folds, fold_idx, active, **kw):
    w = _bma_weights(folds, fold_idx, active)
    fold = folds[fold_idx]
    all_z = fold["all_z"]
    comb = {z: 0.0 for z in all_z}
    for j, name in enumerate(active):
        for z, p in fold["prob"][name].items():
            if z in comb:
                comb[z] += w[j] * p
    return sorted(all_z, key=lambda z: comb[z], reverse=True)


def select_active_signals(folds, threshold=0.85):
    """剔除高相关信号, 保留命中率高的代表。"""
    sigvecs = {n: [] for n in SIGNAL_NAMES}
    for f in folds:
        for n in SIGNAL_NAMES:
            sigvecs[n].extend([f["prob"][n].get(z, 0.0) for z in f["all_z"]])
    hr = {}
    for n in SIGNAL_NAMES:
        h = sum(1 for f in folds if f["actual"] in
                set(sorted(f["all_z"], key=lambda z: f["prob"][n].get(z, 0.0), reverse=True)[:6]))
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

# ==================== 公开 API ====================

_BT_CACHE = {}  # (n_records, backtest_n) -> {k: hits_list}


def _rank_current_stacking_zodiacs(records, k, active_signals=None, window=STACK_WINDOW, ne=STACK_N_EST, md=STACK_MAX_DEPTH, lr=STACK_LR):
    if active_signals is None: active_signals = SIGNAL_NAMES
    if len(records) < 60:
        zmap = build_zodiac_map(records)
        from dimensions import _build_zodiac_six
        return _build_zodiac_six(records, None, zmap)[:k]
    n_folds = max(1, len(records) - 10)
    folds = precompute_folds(records, n_folds)
    train_end = len(folds) - 1
    train_start = max(0, train_end - window)
    train_folds = folds[train_start:train_end]
    feats, labels = [], []
    for f in train_folds:
        for z in f["all_z"]:
            feats.append([f["prob"][name].get(z, 0.0) for name in active_signals])
            labels.append(1 if z == f["actual"] else 0)
    if len(feats) < 50 or sum(labels) < 3:
        return _rank_bma(folds, len(folds)-1, active_signals)[:k]
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(n_estimators=ne, max_depth=md, learning_rate=lr, subsample=0.8, random_state=42)
        clf.fit(feats, labels)
        zmap = build_zodiac_map(records)
        prob, all_z = _compute_fold_probs(records, zmap)
        X = [[prob[name].get(z,0.0) for name in active_signals] for z in all_z]
        proba = clf.predict_proba(X); pos = list(clf.classes_).index(1) if 1 in clf.classes_ else 0
        scores = proba[:,pos] if proba.ndim==2 else proba
        return [z for _,z in sorted(zip(scores,all_z), reverse=True)][:k]
    except Exception:
        return _rank_bma(folds, len(folds)-1, active_signals)[:k]


def predict_stacking_zodiacs(records, k, active_signals=None,
                              window=STACK_WINDOW, ne=STACK_N_EST,
                              md=STACK_MAX_DEPTH, lr=STACK_LR):
    """预测下一期 top-k 生肖: 用最近 window 个历史 OOS folds 训练 GBDT，预测当前信号特征。

    流程:
      1. 构建 walk-forward 折，严格只使用当前时点之前的历史
      2. 元学习器训练集受 window 约束，避免旧 regime 混入\n      3. 当前信号特征 = 全部 records 的各信号归一化概率
      4. GBDT 输出每个生肖 P(命中), 取 top-k
    数据不足(历史 < 60 期)降级为号码数先验(_zodiac_six_scores)。
    """
    if active_signals is None:
        active_signals = SIGNAL_NAMES
    if len(records) < 60:
        from dimensions import _build_zodiac_six
        return _build_zodiac_six(records, None, build_zodiac_map(records))[:k]
    n_folds = max(1, len(records) - 10)
    folds = precompute_folds(records, n_folds)
    # 训练数据只取最近 window 个历史 OOS folds，避免旧 regime 淹没近期信号。
    train_end = len(folds) - 1
    train_start = max(0, train_end - window)
    train_folds = folds[train_start:train_end]
    feats, labels = [], []
    for f in train_folds:
        for z in f["all_z"]:
            feats.append([f["prob"][name].get(z, 0.0) for name in active_signals])
            labels.append(1 if z == f["actual"] else 0)
    if len(feats) < 50 or sum(labels) < 3:
        rank = _rank_bma(folds, len(folds) - 1, active_signals)
        return rank[:k]
    try:
        from sklearn.ensemble import GradientBoostingClassifier
    except ImportError:
        rank = _rank_bma(folds, len(folds) - 1, active_signals)
        return rank[:k]
    clf = GradientBoostingClassifier(n_estimators=ne, max_depth=md,
        learning_rate=lr, subsample=0.8, random_state=42)
    clf.fit(feats, labels)
    # 当前特征(全部历史)
    zmap = build_zodiac_map(records)
    prob, all_z = _compute_fold_probs(records, zmap)
    test_X = [[prob[name].get(z, 0.0) for name in active_signals] for z in all_z]
    proba = clf.predict_proba(test_X)
    if STACK_TEMP != 1.0:
        logit = np.log(np.clip(proba, 1e-9, 1 - 1e-9))
        logit = logit / STACK_TEMP
        logit = logit - logit.max(axis=1, keepdims=True)
        proba = np.exp(logit)
        proba = proba / proba.sum(axis=1, keepdims=True)
    pos_col = list(clf.classes_).index(1) if 1 in clf.classes_ else 0
    p_pos = proba[:, pos_col] if proba.ndim == 2 else proba
    ranked = [z for _, z in sorted(zip(p_pos, all_z), reverse=True)]
    if len(ranked) < k:
        from dimensions import _build_zodiac_six
        extra = [z for z in _build_zodiac_six(records, None, zmap) if z not in ranked]
        ranked.extend(extra)
    return ranked[:k]


def backtest_stacking(records, backtest_n, active_signals=None, window=STACK_WINDOW):
    """V5.2 strict OOS evaluation; every fold rebuilds the predictor from its training prefix."""
    if active_signals is None: active_signals = SIGNAL_NAMES
    key = (len(records), backtest_n, tuple(active_signals), window, "oos-v52")
    if key in _BT_CACHE: return _BT_CACHE[key]
    candidates = sorted(set(build_zodiac_map(records).values()))
    n_test = max(1, min(backtest_n, len(records)-10))
    def ranker(train, cands):
        return _rank_current_stacking_zodiacs(train, 6, active_signals, window)
    report = evaluate_ranked_walk_forward(records, candidates, special_zodiac_of, ranker, initial_train=len(records)-n_test, test_size=n_test, top_k=(3,4,6))
    hits = {k: [f.hit_at_k[k] for f in report.folds] for k in (3,4,6)}
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
    hits = backtest_stacking(records, 876)
    n = len(hits[6])
    print("=" * 55)
    for k in (3, 4, 6):
        h = sum(hits[k])
        rate = h / n * 100
        base = k / 12 * 100
        print(f"  {k}肖: {rate:.1f}%  (基线 {base:.1f}%, lift {rate-base:+.1f})")
    print("=" * 55)
    z6 = predict_stacking_zodiacs(records, 6)
    print(f"预测六肖: {z6}")
    print(f"耗时 {time.time()-t0:.1f}s")
