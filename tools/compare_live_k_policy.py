#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对比旧 Stacking 与 K 专项策略在同一历史序列上的严格滚动 OOS 表现。"""
from __future__ import annotations
import argparse, json, math, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_fetcher import Record
import zodiac_ensemble as zodiac
import wide_ensemble as wide
from dimensions import build_zodiac_map
from core.live_k_policy import predict_and_backtest_k

def load_clean_records(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    seen, kept, duplicates = {}, [], []
    for index, row in enumerate(raw):
        key = (str(row.get("openTime", "")), tuple(row.get("regular", [])), row.get("special"),
               tuple(row.get("waves", [])), tuple(row.get("zodiacs", [])))
        prior = seen.get(key)
        if prior is None:
            seen[key] = row
            kept.append(row)
            continue
        date_year = str(row.get("openTime", ""))[:4]
        prior_date_year = str(prior.get("openTime", ""))[:4]
        current_matches = str(row.get("expect", ""))[:4] == date_year
        prior_matches = str(prior.get("expect", ""))[:4] == prior_date_year
        if current_matches and not prior_matches:
            kept.remove(prior)
            kept.append(row)
            seen[key] = row
            kept_expect, duplicate_expect = row.get("expect"), prior.get("expect")
        else:
            kept_expect, duplicate_expect = prior.get("expect"), row.get("expect")
        duplicates.append({"index": index, "kept_expect": kept_expect,
                           "duplicate_expect": duplicate_expect, "openTime": row.get("openTime")})
    kept.sort(key=lambda row: (str(row.get("openTime", "")), str(row.get("expect", ""))))
    records = [Record(expect=str(x["expect"]), open_time=str(x.get("openTime", "")),
                      regular=x["regular"], special=x["special"], waves=x.get("waves", []),
                      zodiacs=x.get("zodiacs", [])) for x in kept]
    dates = [str(x.get("openTime", "")) for x in kept if x.get("openTime")]
    audit = {"raw_count": len(raw), "clean_count": len(records), "duplicates_removed": len(raw)-len(records),
             "duplicate_rows": duplicates, "first_open_time": min(dates) if dates else None,
             "last_open_time": max(dates) if dates else None}
    return records, audit

def wilson(hits, n, z=1.96):
    if n == 0: return None
    p = hits/n
    d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return [max(0,c-h), min(1,c+h)]

def summary(hits, baseline):
    n, wins = len(hits), sum(bool(x) for x in hits)
    rate = wins/n if n else 0
    return {"hits":wins,"n":n,"hit_rate":rate,"hit_rate_percent":round(rate*100,3),
            "baseline":baseline,"baseline_percent":round(baseline*100,3),
            "lift_percentage_points":round((rate-baseline)*100,3),"wilson_95_ci":wilson(wins,n)}

def paired(new, old, seed=20261029):
    if not new or len(new) != len(old):
        return {"available":False,"reason":"OOS 期数不一致或为空"}
    diff = [int(bool(a))-int(bool(b)) for a,b in zip(new,old)]
    rng, n = random.Random(seed), len(diff)
    samples = sorted(sum(diff[rng.randrange(n)] for _ in range(n))/n for _ in range(10000))
    return {"available":True,"new_only_hits":sum(a and not b for a,b in zip(new,old)),
            "old_only_hits":sum(b and not a for a,b in zip(new,old)),
            "ties":sum(bool(a)==bool(b) for a,b in zip(new,old)),
            "difference_percentage_points":round(sum(diff)/n*100,3),
            "paired_bootstrap_95_ci":[samples[int(.025*len(samples))],samples[int(.975*len(samples))]],
            "note":"配对短窗口描述性结果，不构成未来预测优势证明"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",default="macau_history.json")
    ap.add_argument("--output",default="reports/k_policy_oos_comparison.json")
    ap.add_argument("--backtest-n",type=int,default=30)
    args = ap.parse_args()
    if args.backtest_n < 10: raise SystemExit("--backtest-n 至少为 10")
    records,audit = load_clean_records(args.input)
    if len(records) < 100: raise SystemExit(f"历史样本不足：{len(records)}")
    old_z = zodiac.backtest_stacking(records,args.backtest_n).get(3,[])
    old_w = wide.backtest_stacking(records,args.backtest_n,20)
    n_folds = min(max(1,len(records)-10), max(150,args.backtest_n+120))
    z_folds = zodiac.precompute_folds(records,n_folds)
    zmap = build_zodiac_map(records)
    current_z,_ = zodiac._compute_fold_probs(records,zmap)
    z_candidates = sorted(set(zmap.values()))
    new_z = predict_and_backtest_k(z_folds,current_z,zodiac.SIGNAL_NAMES,z_candidates,3,args.backtest_n)
    if new_z is None: raise SystemExit("三肖 K 策略未产生有效 OOS 结果")
    w_folds = wide.precompute_folds(records,n_folds)
    current_w = wide._compute_fold_probs(records)
    new_w = predict_and_backtest_k(w_folds,current_w,wide.SIGNAL_NAMES,wide.ALL_NUMS,20,args.backtest_n)
    if new_w is None: raise SystemExit("20码 K 策略未产生有效 OOS 结果")
    if len(old_z) != len(new_z["hits"]) or len(old_w) != len(new_w["hits"]):
        raise SystemExit("新旧策略 OOS 期数不一致，拒绝进行不公平比较")
    report = {
      "version":"K_POLICY_OOS_COMPARISON_V1","history":audit,"backtest_n_requested":args.backtest_n,
      "method":{"evaluation":"最后 N 期严格 walk-forward，策略只使用当期之前的 OOS folds",
                "random_baselines":{"three_zodiac":3/12,"twenty_code":20/49},
                "caveat":"历史命中率不能保证未来结果；30 期窗口的置信区间通常很宽"},
      "three_zodiac":{"old_stacking":summary(old_z,3/12),"k_specific":summary(new_z["hits"],3/12),
                      "paired_difference":paired(new_z["hits"],old_z),"current_prediction":new_z["ranking"],
                      "policy":new_z["policy"]},
      "twenty_code":{"old_stacking":summary(old_w,20/49),"k_specific":summary(new_w["hits"],20/49),
                     "paired_difference":paired(new_w["hits"],old_w),"current_prediction":sorted(new_w["ranking"]),
                     "policy":new_w["policy"]}
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    print(json.dumps({"output":str(out),"history":audit,
      "three_zodiac":{k:report["three_zodiac"][k] for k in ("old_stacking","k_specific","paired_difference")},
      "twenty_code":{k:report["twenty_code"][k] for k in ("old_stacking","k_specific","paired_difference")}},
      ensure_ascii=False,indent=2))

if __name__ == "__main__":
    main()
