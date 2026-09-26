#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V5.8.2 K-specific Meta Policy historical research runner."""
import argparse,json,os
from data_fetcher import Record
from core.k_specific_policy import KPolicyConfig,evaluate_k_policies
from core.signal_metrics import metric_summary,bootstrap_metric_ci,binomial_two_sided_pvalue

def load_json(path):
    raw=json.load(open(path,encoding="utf-8"))
    return [Record(expect=str(x["expect"]),open_time=str(x.get("openTime","")),
                   regular=x["regular"],special=x["special"],waves=x.get("waves",[]),
                   zodiacs=x.get("zodiacs",[])) for x in raw]

def build_layers(records):
    import zodiac_ensemble as z,wide_ensemble as w
    n=max(1,len(records)-10); zf=z.precompute_folds(records,n); wf=w.precompute_folds(records,n)
    return [
        ("zodiac",zf,list(z.SIGNAL_NAMES),sorted({f["actual"] for f in zf if f["actual"] is not None}),(3,4,6)),
        ("wide",wf,list(w.SIGNAL_NAMES),list(w.ALL_NUMS),(20,))
    ]

def evaluate_layer(name,folds,signals,candidates,target_ks,args):
    usable=[f for f in folds if f.get("actual") is not None and all(s in f.get("prob",{}) for s in signals)]
    if len(usable)<args.min_history+args.final_holdout+1: raise ValueError(name+" insufficient usable folds: "+str(len(usable)))
    cfg=KPolicyConfig(windows=tuple(args.horizons),pair_lambdas=tuple(args.pair_lambdas),
        diversity_lambdas=tuple(args.diversity_lambdas),min_history=args.min_history,
        validation_size=args.validation_size,shrinkage=args.shrinkage,min_weight=args.min_weight,
        temperature=args.temperature,complexity_penalty=args.complexity_penalty,min_gain=args.min_gain)
    res=evaluate_k_policies(usable,candidates,signals,target_ks,cfg,args.min_history,args.final_holdout)
    out={"layer":name,"folds":len(usable),"signals":signals,"candidates":len(candidates),"targets":{},"holdout_isolated":True}
    for k in target_ks:
        outer=res["outer_by_k"][k]; hold=res["holdout_by_k"][k]
        om=metric_summary(outer,top_k=(k,)); hm=metric_summary(hold,top_k=(k,))
        base=min(1.0,k/max(1,len(candidates))); hits=sum(1 for p,a in outer if a in p)
        out["targets"][str(k)]={
            "outer":om,"outer_n":len(outer),"outer_baseline":base,
            "outer_p_value":binomial_two_sided_pvalue(hits,len(outer),base),
            "outer_ci":{m:bootstrap_metric_ci(outer,m,rounds=args.bootstrap_rounds,top_k=(k,))
                        for m in ("hit_at_"+str(k),"logloss","brier","ece","information_gain")},
            "final_holdout":hm,"holdout_n":len(hold),
            "final_holdout_ci":{m:bootstrap_metric_ci(hold,m,rounds=args.bootstrap_rounds,top_k=(k,))
                                for m in ("hit_at_"+str(k),"logloss","brier","ece","information_gain")},
            "final_policy":res["final_policy"][k],"policy_path":res["policy_path"][k]}
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",default="macau_history.json"); ap.add_argument("--output-dir",default="reports/v5.8")
    ap.add_argument("--min-history",type=int,default=60); ap.add_argument("--final-holdout",type=int,default=60)
    ap.add_argument("--validation-size",type=int,default=30); ap.add_argument("--horizons",default="30,60,120,240")
    ap.add_argument("--pair-lambdas",default="0,0.02,0.05,0.1"); ap.add_argument("--diversity-lambdas",default="0,0.02,0.05")
    ap.add_argument("--shrinkage",type=float,default=.35); ap.add_argument("--min-weight",type=float,default=.03)
    ap.add_argument("--temperature",type=float,default=.20); ap.add_argument("--complexity-penalty",type=float,default=.002); ap.add_argument("--min-gain",type=float,default=.005); ap.add_argument("--bootstrap-rounds",type=int,default=800)
    a=ap.parse_args()
    a.horizons=[int(x) for x in a.horizons.split(",") if int(x)>0]
    a.pair_lambdas=[float(x) for x in a.pair_lambdas.split(",")]
    a.diversity_lambdas=[float(x) for x in a.diversity_lambdas.split(",")]
    records=load_json(a.input); layers={}
    for x in build_layers(records): layers[x[0]]=evaluate_layer(*x,a)
    report={"version":"V5.8.2","history_count":len(records),"first_expect":records[0].expect,
            "last_expect":records[-1].expect,"design":"Direct Set Prediction + Pair Structure + True K-specific Meta Policy",
            "layers":layers,"research_boundary":"historical OOS diagnostics only; no future predictability claim"}
    os.makedirs(a.output_dir,exist_ok=True)
    with open(os.path.join(a.output_dir,"set_prediction_registry.json"),"w",encoding="utf-8") as f: json.dump(report,f,ensure_ascii=False,indent=2)
    with open(os.path.join(a.output_dir,"report.md"),"w",encoding="utf-8") as f:
        f.write("# V5.8.1 Direct Set Prediction\n\nLeakage-safe historical OOS research. K-specific policy; final holdout is frozen.\n\n")
        for name,r in layers.items():
            f.write("## "+name+"\n\n| Target | Outer Hit@K | Holdout Hit@K | Random baseline | p-value | Horizon | Pair | Diversity | Gain | Structure |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for ks,t in r["targets"].items():
                p=t["final_policy"]
                f.write("| %s | %.4f | %.4f | %.4f | %.4f | %s | %.3f | %.3f |\n" %
                    (ks,t["outer"].get("hit_at_"+ks,0),t["final_holdout"].get("hit_at_"+ks,0),
                     t["outer_baseline"],t["outer_p_value"],p["horizon"],p["lambda_pair"],p["lambda_diversity"],p["gain"],p["structure_enabled"]))
            f.write("\n")
    print(json.dumps({"version":"V5.8.1","history_count":len(records),"layers":list(layers),"output_dir":a.output_dir},ensure_ascii=False))
if __name__=="__main__": main()
