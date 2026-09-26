#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V5.4.1 full historical research runner.

Run locally from repository root:
    python v54_research_run.py --years 2020 2021 2022 2023 2024 2025 2026

The runner creates a machine-readable JSON report and never tunes on the final
holdout. It is intentionally separate from GUI prediction code.
"""
import argparse,json,os
from dataclasses import asdict
from data_fetcher import load_history
from core.nested_evaluation import NestedConfig,evaluate_nested
from core.signal_metrics import metric_summary
from core.signal_analysis import signal_correlation,redundancy_groups,effective_signal_count
from core.multiple_testing import benjamini_hochberg

def zodiac_predictions(records):
    import zodiac_ensemble as z
    n=max(1,len(records)-10)
    folds=z.precompute_folds(records,n)
    return folds

def wide_predictions(records):
    import wide_ensemble as w
    n=max(1,len(records)-10)
    return w.precompute_folds(records,n)

def fold_rows(folds,actual_key,candidates):
    rows={name:[] for name in folds[0]["prob"]} if folds else {}
    for f in folds:
        actual=f[actual_key]
        for name,p in f["prob"].items():
            rows[name].append((p,actual))
    return rows

def summarize_layer(folds,actual_key,candidates,top_k):
    rows=fold_rows(folds,actual_key,candidates)
    metrics={name:metric_summary(v,top_k=top_k) for name,v in rows.items()}
    vectors={name:[p.get(c,0.0) for p,a in vals for c in candidates]
             for name,vals in rows.items()}
    corr=signal_correlation(vectors)
    return {
        "folds":len(folds),
        "metrics":metrics,
        "correlation":corr,
        "redundancy_groups":redundancy_groups(vectors),
        "effective_signal_count":effective_signal_count(corr),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--years",nargs="+",type=int,default=list(range(2020,2027)))
    ap.add_argument("--refresh",action="store_true")
    ap.add_argument("--output",default="reports/v5.4.1_full_research.json")
    args=ap.parse_args()
    records=load_history(args.years,refresh=args.refresh)
    if len(records)<300: raise SystemExit(f"insufficient history: {len(records)}")
    zc=sorted(set(__import__("dimensions").special_zodiac_of(r) for r in records if __import__("dimensions").special_zodiac_of(r))
    wc=list(range(1,50))
    zf=zodiac_predictions(records); wf=wide_predictions(records)
    report={
      "version":"V5.4.1","history_count":len(records),
      "first_expect":records[0].expect,"last_expect":records[-1].expect,
      "zodiac":summarize_layer(zf,"actual",zc,(3,4,6)),
      "wide":summarize_layer(wf,"actual",wc,(10,20)),
      "multiple_testing":"BH correction available via core.multiple_testing.benjamini_hochberg",
      "note":"These are historical/OOS diagnostics, not evidence of guaranteed future predictability."
    }
    os.makedirs(os.path.dirname(args.output) or ".",exist_ok=True)
    with open(args.output,"w",encoding="utf-8") as f: json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps({"output":args.output,"history_count":len(records),
                      "first_expect":records[0].expect,"last_expect":records[-1].expect,
                      "zodiac_folds":len(zf),"wide_folds":len(wf)},ensure_ascii=False))
if __name__=="__main__": main()
