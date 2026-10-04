#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V5.6 real historical Regime Detection / Adaptive Ensemble runner."""
import argparse,json,os
from data_fetcher import Record
from core.regime_detection import RegimeDetector,RegimeConfig
from core.adaptive_ensemble import AdaptiveConfig,evaluate_adaptive
from core.signal_metrics import metric_summary,bootstrap_metric_ci,binomial_two_sided_pvalue

def load_json(path):
    raw=json.load(open(path,encoding="utf-8"))
    return [Record(expect=str(x["expect"]),open_time=str(x.get("openTime","")),regular=x["regular"],special=x["special"],waves=x.get("waves",[]),zodiacs=x.get("zodiacs",[])) for x in raw]

def build_layers(records):
    import zodiac_ensemble as z,wide_ensemble as w,dim_ensemble as d
    from dimensions import DIMENSIONS
    out=[]
    n=max(1,len(records)-10)
    zf=z.precompute_folds(records,n);out.append(("zodiac",zf,list(z.SIGNAL_NAMES),sorted({f["actual"] for f in zf if f["actual"] is not None})))
    wf=w.precompute_folds(records,n);out.append(("wide",wf,list(w.SIGNAL_NAMES),list(w.ALL_NUMS)))
    for name,extract in DIMENSIONS:
        pf=d.DIM_PRIOR_FN[name];fs=d.precompute_folds(records,extract,pf,n)
        c=sorted({f["actual"] for f in fs if f["actual"] is not None})
        out.append(("dimension."+name,fs,list(d.SIGNAL_NAMES),c))
    return out

def evaluate_layer(name,folds,signals,candidates,args):
    usable=[f for f in folds if f.get("actual") is not None and all(s in f.get("prob",{}) for s in signals)]
    need=args.min_history+args.final_holdout+1
    if len(usable)<need: raise ValueError(name+" insufficient usable folds: "+str(len(usable)))
    cfg=RegimeConfig(n_regimes=args.regimes,context_window=args.context_window,min_history=args.min_history)
    regimes=[];details=[]
    for j in range(len(usable)):
        if j<args.min_history: regimes.append(0);details.append({"status":"insufficient_history","n_history":j})
        else:
            h=usable[:j]
            d=RegimeDetector(cfg).fit(h[-args.context_window*4:],signals,candidates)
            regimes.append(d.assign(usable[j],signals,candidates))
            details.append({"status":"ok","n_history":min(j,args.context_window*4),"regime":regimes[-1]})
    res=evaluate_adaptive(usable,candidates,signals,regimes,AdaptiveConfig(decay=args.decay,shrinkage=args.shrinkage,min_weight=args.min_weight),args.min_history,args.final_holdout)
    def summarize(rows):
        return metric_summary(rows,top_k=(1,3,4,6,20))
    outer=res["outer_rows"];hold=res["holdout_rows"];om=summarize(outer);hm=summarize(hold)
    base={str(k):min(1.0,k/max(1,len(candidates))) for k in (1,3,4,6,20)}
    pv={str(k):binomial_two_sided_pvalue(int(round(om.get("hit_at_"+str(k),0)*len(outer))),len(outer),base[str(k)]) for k in (1,3,6)}
    mets=("hit_at_1","hit_at_3","hit_at_4","hit_at_6","hit_at_20","logloss","brier","ece","information_gain")
    return {"layer":name,"folds":len(usable),"regime_count":len(set(regimes)),"regimes":regimes,"regime_details":details,
            "outer":om,"outer_n":len(outer),"outer_p_values":pv,
            "outer_ci":{m:bootstrap_metric_ci(outer,m,rounds=800) for m in mets},
            "final_holdout":hm,"holdout_n":len(hold),
            "final_holdout_ci":{m:bootstrap_metric_ci(hold,m,rounds=800) for m in mets},
            "adaptive":{"final_weights":res["final_weights"],"policy_path":res["policy_path"]},
            "holdout_isolated":True}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",default="macau_history.json");ap.add_argument("--output-dir",default="reports/v5.6")
    ap.add_argument("--regimes",type=int,default=3);ap.add_argument("--context-window",type=int,default=60)
    ap.add_argument("--min-history",type=int,default=60);ap.add_argument("--final-holdout",type=int,default=60)
    ap.add_argument("--decay",type=float,default=.97);ap.add_argument("--shrinkage",type=float,default=.35);ap.add_argument("--min-weight",type=float,default=.03)
    a=ap.parse_args();records=load_json(a.input);layers={}
    for name,folds,signals,cands in build_layers(records): layers[name]=evaluate_layer(name,folds,signals,cands,a)
    report={"version":"V5.6","history_count":len(records),"first_expect":records[0].expect,"last_expect":records[-1].expect,"layers":layers,
            "research_boundary":"historical OOS diagnostics only; no future predictability claim"}
    os.makedirs(a.output_dir,exist_ok=True)
    json.dump(report,open(os.path.join(a.output_dir,"regime_adaptive_registry.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=2)
    with open(os.path.join(a.output_dir,"report.md"),"w",encoding="utf-8") as f:
        f.write("# V5.6 Regime Detection / Adaptive Ensemble\n\n")
        f.write("Leakage-safe historical OOS diagnostics. Final holdout is frozen.\n\n")
        for name,r in layers.items():
            o=r["outer"];h=r["final_holdout"]
            f.write("## "+name+"\n")
            f.write("folds: "+str(r["folds"])+"; regimes: "+str(r["regime_count"])+"\n")
            f.write("outer Hit@1/3/4/6/20: %.4f / %.4f / %.4f / %.4f / %.4f\n\n"%(o.get("hit_at_1",0),o.get("hit_at_3",0),o.get("hit_at_4",0),o.get("hit_at_6",0),o.get("hit_at_20",0)))
            f.write("holdout Hit@1/3/4/6/20: %.4f / %.4f / %.4f / %.4f / %.4f\n\n"%(h.get("hit_at_1",0),h.get("hit_at_3",0),h.get("hit_at_4",0),h.get("hit_at_6",0),h.get("hit_at_20",0)))
    print(json.dumps({"version":"V5.6","history_count":len(records),"layers":list(layers),"output_dir":a.output_dir},ensure_ascii=False))
if __name__=="__main__":main()
