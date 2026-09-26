#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V5.4.2 real-data nested research runner.

Input is the committed/local macau_history.json. No network refresh is used.
All selection, redundancy filtering, weights and calibration happen inside
inner OOS folds; the final holdout is scored with one frozen policy.
"""
import argparse,json,os
from data_fetcher import Record
from core.nested_signal_research import NestedResearchConfig,evaluate_nested_folds\nfrom core.research_result_engine import build_result_report

def load_json(path):
    with open(path,encoding="utf-8") as f: raw=json.load(f)
    return [Record(expect=str(x["expect"]),open_time=str(x.get("openTime","")),
                   regular=x["regular"],special=x["special"],
                   waves=x.get("waves",[]),zodiacs=x.get("zodiacs",[])) for x in raw]

def zodiac_layer(records):
    import zodiac_ensemble as z
    folds=z.precompute_folds(records,max(1,len(records)-10))
    return folds,list(z.SIGNAL_NAMES),sorted({f["actual"] for f in folds if f["actual"] is not None})

def wide_layer(records):
    import wide_ensemble as w
    folds=w.precompute_folds(records,max(1,len(records)-10))
    return folds,list(w.SIGNAL_NAMES),list(range(1,50))

def dimension_layers(records):
    import dim_ensemble as d
    from dimensions import DIMENSIONS
    out={}
    for name,extract in DIMENSIONS:
        prior_fn=d.DIM_PRIOR_FN[name]
        folds=d.precompute_folds(records,extract,prior_fn,max(1,len(records)-10))
        candidates=sorted({f["actual"] for f in folds if f["actual"] is not None})
        out[name]=(folds,list(d.SIGNAL_NAMES),candidates)
    return out

def run(name,folds,signals,candidates,cfg):
    # Keep only folds with a valid actual and complete candidate probability maps.
    usable=[f for f in folds if f.get("actual") is not None and all(s in f.get("prob",{}) for s in signals)]
    result=evaluate_nested_folds(usable,candidates,signals,cfg)
    result["layer"]=name
    result["signal_names"]=signals
    result["candidate_count"]=len(candidates)
    result["raw_folds"]=len(folds)
    result["usable_folds"]=len(usable)
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",default="macau_history.json")
    ap.add_argument("--output",default="reports/v5.4.2_nested_research.json")
    ap.add_argument("--holdout",type=int,default=60)
    ap.add_argument("--initial-train",type=int,default=180)
    ap.add_argument("--inner-min-folds",type=int,default=90)
    ap.add_argument("--inner-window",type=int,default=180)
    ap.add_argument("--max-signals",type=int,default=4)
    ap.add_argument("--redundancy-threshold",type=float,default=.85)
    args=ap.parse_args()
    records=load_json(args.input)
    if len(records) < args.initial_train + args.holdout + 1:
        raise SystemExit("insufficient history for configured holdout")
    cfg=NestedResearchConfig(initial_train=args.initial_train,final_holdout=args.holdout,
        inner_min_folds=args.inner_min_folds,inner_window=args.inner_window,
        max_signals=args.max_signals,redundancy_threshold=args.redundancy_threshold)
    report={"version":"V5.4.2","history_count":len(records),
            "first_expect":records[0].expect,"last_expect":records[-1].expect,
            "config":cfg.__dict__,"layers":{}}
    zf,zs,zc=zodiac_layer(records); report["layers"]["zodiac"]=run("zodiac",zf,zs,zc,cfg)
    wf,ws,wc=wide_layer(records); report["layers"]["wide"]=run("wide",wf,ws,wc,cfg)
    for name,(folds,sigs,cands) in dimension_layers(records).items():
        report["layers"]["dimension."+name]=run("dimension."+name,folds,sigs,cands,cfg)
    os.makedirs(os.path.dirname(args.output) or ".",exist_ok=True)
    with open(args.output,"w",encoding="utf-8") as f: json.dump(report,f,ensure_ascii=False,indent=2,default=str)
    print(json.dumps({"output":args.output,"history_count":len(records),
                      "layers":list(report["layers"])},ensure_ascii=False))
if __name__=="__main__": main()
