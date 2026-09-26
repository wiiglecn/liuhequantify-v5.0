# -*- coding: utf-8 -*-
"""测试温度缩放修复过度自信, 提升四肖命中率"""
import json, sys, time
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from data_fetcher import Record
import zodiac_ensemble as ZE
from sklearn.ensemble import GradientBoostingClassifier

raw = json.load(open("macau_history.json", encoding="utf-8"))
records = [Record(expect=str(r["expect"]), open_time=str(r.get("openTime","")),
    regular=r["regular"], special=r["special"],
    waves=r.get("waves",[]), zodiacs=r.get("zodiacs",[])) for r in raw]

folds = ZE.precompute_folds(records, 180)
start = 150

def rank_with_temp(folds, i, temp=1.0):
    s = max(0, i-150)
    feats, labels = [], []
    for t in range(s, i):
        f = folds[t]
        for v in f["all_z"]:
            feats.append([f["prob"][n].get(v,0.0) for n in ZE.SIGNAL_NAMES])
            labels.append(v)
    fold = folds[i]
    clf = GradientBoostingClassifier(n_estimators=80, max_depth=3, learning_rate=0.1, subsample=0.8, random_state=42)
    clf.fit(feats, labels)
    test_X = [[fold["prob"][n].get(v,0.0) for n in ZE.SIGNAL_NAMES] for v in fold["all_z"]]
    proba = clf.predict_proba(test_X)
    if temp != 1.0:
        # 温度缩放: logit/T 再归一化
        logit = np.log(np.clip(proba, 1e-9, 1-1e-9))
        logit = logit / temp
        logit = logit - logit.max(axis=1, keepdims=True)
        proba = np.exp(logit)
        proba = proba / proba.sum(axis=1, keepdims=True)
    classes = list(clf.classes_)
    p_actual = np.zeros(len(fold["all_z"]))
    for j, v in enumerate(fold["all_z"]):
        if v in classes:
            p_actual[j] = proba[j, classes.index(v)]
    return [v for _, v in sorted(zip(p_actual, fold["all_z"]), reverse=True)]

print("温度缩放对三肖/四肖命中率的影响 (近30期):")
print("="*60)
print(f"{'温度':<8} {'三肖':>8} {'四肖':>8} {'六肖':>8} {'3→4增益':>10}")
print("-"*60)
for temp in [1.0, 1.5, 2.0, 3.0, 5.0]:
    h3=h4=h6=0
    for i in range(start, len(folds)):
        rank = rank_with_temp(folds, i, temp)
        actual = folds[i]["actual"]
        h3 += actual in set(rank[:3])
        h4 += actual in set(rank[:4])
        h6 += actual in set(rank[:6])
    n = len(folds) - start
    print(f"  T={temp:<5} {h3/n*100:>6.1f}%  {h4/n*100:>6.1f}%  {h6/n*100:>6.1f}%  {h4-h3:>+8d}期")
print("="*60)
