#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V5.7 real historical research runner: Regime x Signal x Horizon."""
import argparse,json,os
from data_fetcher import Record
from core.regime_detection import RegimeDetector,RegimeConfig
from core.v57_meta_policy import HorizonConfig,evaluate_meta
from core.signal_metrics import metric_summary,bootstrap_metric_ci,binomial_two_sided_pvalue

def load_json(path):
    raw=json.load(open(path,encoding="utf-8"))
    return [Record(expect=str(x["expect"]),open_time=str(x.get("openTime","")),
                   regular=x["regular"],special=x["special"],waves=x.get("waves",[]),
                   zodiacs=x.get("zodiacs",[])) for x in raw]

def build_layers(records):
    import zodiac_ensemble as z,wide_ensemble as w
    n=max(1,len(records)-10)
    zf=z.precompute_folds(records,n)
    wf=w.precompute_folds(records,n)
    return [
        ("zodiac",zf,list(z.SIGNAL_NAMES),
         sorted({f["actual"] for f in zf if f["actual"] is not None}),(3,4,6)),
        ("wide",wf,list(w.SIGNAL_NAMES),list(w.ALL_NUMS),(20,))
    ]

def transition_stats(regimes):
    if len(regimes)<2:
        return {"matrix":[],"persistence":0.0}
    k=max(regimes)+1
    m=[[0]*k for _ in range(k)]
    for a,b in zip(regimes,regimes[1:]):
        m[a][b]+=1
    for i,row in enumerate(m):
        s=sum(row)
        if s: m[i]=[round(v/s,6) for v in row]
    return {"matrix":m,"persistence":sum(a==b for a,b in zip(regimes,regimes[1:]))/(len(regimes)-1)}

def evaluate_layer(name,folds,signals,candidates,target_ks,args):
    usable=[f for f in folds if f.get("actual") is not None and
            all(s in f.get("prob",{}) for s in signals)]
    if len(usable)<args.min_history+args.final_holdout+1:
        raise ValueError(name+" insufficient usable folds: "+str(len(usable)))
    rc=RegimeConfig(n_regimes=args.regimes,context_window=args.context_window,
                    min_history=args.min_history)
    regimes=[]; details=[]
    for j in range(len(usable)):
        if j<args.min_history:
            regimes.append(0); details.append({"status":"insufficient_history","n_history":j})
        else:
            h=usable[max(0,j-args.context_window*4):j]
            d=RegimeDetector(rc).fit(h,signals,candidates)
            r=d.assign(usable[j],signals,candidates)
            regimes.append(r)
            details.append({"status":"ok","n_history":len(h),"regime":r})
    cfg=HorizonConfig(windows=tuple(args.horizons),min_history=args.min_history,
                      shrinkage=args.shrinkage,min_weight=args.min_weight,
                      temperature=args.temperature,regime_weight=args.regime_weight)
    res=evaluate_meta(usable,candidates,signals,regimes,target_ks,cfg,
                      args.min_history,args.final_holdout)
    out={"layer":name,"folds":len(usable),"signals":signals,
         "candidates":len(candidates),"regime_count":len(set(regimes)),
         "regimes":regimes,"regime_details":details,
         "regime_transitions":transition_stats(regimes),
         "targets":{},"holdout_isolated":True}
    for k in target_ks:
        outer=res["outer_by_k"][k]; hold=res["holdout_by_k"][k]
        om=metric_summary(outer,top_k=(k,))
        hm=metric_summary(hold,top_k=(k,))
        base=min(1.0,k/max(1,len(candidates)))
        hits=int(round(om.get("hit_at_"+str(k),0)*len(outer)))
        out["targets"][str(k)]={
            "outer":om,"outer_n":len(outer),
            "outer_baseline":base,
            "outer_p_value":binomial_two_sided_pvalue(hits,len(outer),base),
            "outer_ci":{m:bootstrap_metric_ci(outer,m,rounds=args.bootstrap_rounds)
                        for m in ("hit_at_"+str(k),"logloss","brier","ece","information_gain")},
            "final_holdout":hm,"holdout_n":len(hold),
            "final_holdout_ci":{m:bootstrap_metric_ci(hold,m,rounds=args.bootstrap_rounds)
                                for m in ("hit_at_"+str(k),"logloss","brier","ece","information_gain")},
            "final_policy":res["final_policy"][k],
            "policy_path":res["policy_path"][k]
        }
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",default="macau_history.json")
    ap.add_argument("--output-dir",default="reports/v5.7")
    ap.add_argument("--regimes",type=int,default=3)
    ap.add_argument("--context-window",type=int,default=60)
    ap.add_argument("--min-history",type=int,default=60)
    ap.add_argument("--final-holdout",type=int,default=60)
    ap.add_argument("--horizons",default="30,60,120,240")
    ap.add_argument("--shrinkage",type=float,default=.35)
    ap.add_argument("--min-weight",type=float,default=.03)
    ap.add_argument("--temperature",type=float,default=.20)
    ap.add_argument("--regime-weight",type=float,default=.65)
    ap.add_argument("--bootstrap-rounds",type=int,default=800)
    a=ap.parse_args(); a.horizons=[int(x) for x in a.horizons.split(",") if int(x)>0]
    records=load_json(a.input); layers={}
    for x in build_layers(records):
        layers[x[0]]=evaluate_layer(*x,a)
    report={"version":"V5.7","history_count":len(records),
            "first_expect":records[0].expect,"last_expect":records[-1].expect,
            "design":"Regime x Signal x Horizon; target-specific Top-K policies",
            "layers":layers,
            "research_boundary":"historical OOS diagnostics only; no future predictability claim"}
    os.makedirs(a.output_dir,exist_ok=True)
    with open(os.path.join(a.output_dir,"meta_policy_registry.json"),"w",encoding="utf-8") as f:
        json.dump(report,f,ensure_ascii=False,indent=2)
    with open(os.path.join(a.output_dir,"report.md"),"w",encoding="utf-8") as f:
        f.write("# V5.7 Regime × Signal × Horizon\n\n")
        f.write("Leakage-safe historical OOS research. Each target K has its own policy; final holdout is frozen.\n\n")
        for name,r in layers.items():
            f.write("## "+name+"\n\n")
            f.write("folds=%d; signals=%d; regimes=%d; regime persistence=%.4f\n\n" %
                    (r["folds"],len(r["signals"]),r["regime_count"],
                     r["regime_transitions"]["persistence"]))
            f.write("| Target | Outer Hit@K | Holdout Hit@K | Random baseline | Outer p-value | Selected horizon |\n|---:|---:|---:|---:|---:|---:|\n")
            for ks,t in r["targets"].items():
                f.write("| %s | %.4f | %.4f | %.4f | %.4f | %s |\n" %
                        (ks,t["outer"].get("hit_at_"+ks,0),t["final_holdout"].get("hit_at_"+ks,0),
                         t["outer_baseline"],t["outer_p_value"],t["final_policy"]["horizon"]))
            f.write("\n")
    print(json.dumps({"version":"V5.7","history_count":len(records),"layers":list(layers),
                      "output_dir":a.output_dir},ensure_ascii=False))
if __name__=="__main__":main()
