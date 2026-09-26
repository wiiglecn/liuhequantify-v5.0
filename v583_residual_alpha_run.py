#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V5.8.3 Residual Alpha / signal residual discovery historical runner."""
import argparse,json,os
from data_fetcher import Record
from core.k_specific_policy import KPolicyConfig
from core.residual_alpha import ResidualConfig,evaluate_residual
from core.signal_metrics import metric_summary

def load_json(path):
    raw=json.load(open(path,encoding="utf-8"))
    return [Record(expect=str(x["expect"]),open_time=str(x.get("openTime","")),regular=x["regular"],special=x["special"],waves=x.get("waves",[]),zodiacs=x.get("zodiacs",[])) for x in raw]

def build_layers(records):
    import zodiac_ensemble as z,wide_ensemble as w
    n=max(1,len(records)-10); zf=z.precompute_folds(records,n); wf=w.precompute_folds(records,n)
    return [("zodiac",zf,list(z.SIGNAL_NAMES),sorted({f["actual"] for f in zf if f["actual"] is not None}),(3,4,6)),
            ("wide",wf,list(w.SIGNAL_NAMES),list(w.ALL_NUMS),(20,))]

def run_layer(name,folds,signals,candidates,targets,args):
    cfg=KPolicyConfig(windows=tuple(args.horizons),pair_lambdas=(0.0,),diversity_lambdas=(0.0,),min_history=args.min_history,validation_size=args.validation_size,shrinkage=args.shrinkage,min_weight=args.min_weight,temperature=args.temperature,complexity_penalty=args.complexity_penalty,min_gain=args.min_gain)
    rcfg=ResidualConfig(residual_lambdas=tuple(args.residual_lambdas),min_history=args.residual_min_history,validation_size=args.residual_validation,min_gain=args.residual_min_gain,alpha=args.alpha,bh_alpha=args.bh_alpha,bootstrap_rounds=args.bootstrap_rounds)
    usable=[f for f in folds if f.get("actual") is not None and all(s in f.get("prob",{}) for s in signals)]
    out={"layer":name,"folds":len(usable),"signals":signals,"candidates":len(candidates),"targets":{}}
    for k in targets: out["targets"][str(k)]=evaluate_residual(usable,signals,candidates,k,cfg,rcfg,args.min_history,args.final_holdout)
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",default="macau_history.json"); ap.add_argument("--output-dir",default="reports/v5.8.3")
    ap.add_argument("--min-history",type=int,default=60); ap.add_argument("--final-holdout",type=int,default=60); ap.add_argument("--validation-size",type=int,default=30); ap.add_argument("--horizons",default="30,60,120,240")
    ap.add_argument("--shrinkage",type=float,default=.35); ap.add_argument("--min-weight",type=float,default=.03); ap.add_argument("--temperature",type=float,default=.20); ap.add_argument("--complexity-penalty",type=float,default=.002); ap.add_argument("--min-gain",type=float,default=.005)
    ap.add_argument("--residual-lambdas",default="0.05,0.10,0.20,0.30,0.50"); ap.add_argument("--residual-min-history",type=int,default=90); ap.add_argument("--residual-validation",type=int,default=30); ap.add_argument("--residual-min-gain",type=float,default=.005)
    ap.add_argument("--alpha",type=float,default=.05); ap.add_argument("--bh-alpha",type=float,default=.10); ap.add_argument("--bootstrap-rounds",type=int,default=800)
    a=ap.parse_args(); a.horizons=[int(x) for x in a.horizons.split(",") if int(x)>0]; a.residual_lambdas=[float(x) for x in a.residual_lambdas.split(",")]
    records=load_json(a.input); layers={}
    for x in build_layers(records): layers[x[0]]=run_layer(*x,a)
    report={"version":"V5.8.3","history_count":len(records),"first_expect":records[0].expect,"last_expect":records[-1].expect,"design":"V5.8.2 K-specific Set Prediction + leakage-safe Residual Alpha Discovery","layers":layers,"research_boundary":"historical OOS diagnostics only; no future predictability claim"}
    os.makedirs(a.output_dir,exist_ok=True)
    with open(os.path.join(a.output_dir,"residual_alpha_registry.json"),"w",encoding="utf-8") as f: json.dump(report,f,ensure_ascii=False,indent=2)
    with open(os.path.join(a.output_dir,"report.md"),"w",encoding="utf-8") as f:
        f.write("# V5.8.3 Residual Alpha / Signal Residual Discovery\n\nResidual candidates are selected from prior OOS rows only. BH/Bonferroni, bootstrap CI and two-period stability gates are applied before acceptance. Final holdout remains frozen.\n\n")
        for name,r in layers.items():
            f.write("## "+name+"\n\n| K | Selected residual | Lambda | Mean delta LogLoss | p | BH q | CI low | Stable | Accepted | Holdout LogLoss |\n|---:|---|---:|---:|---:|---:|---:|:---:|:---:|---:|\n")
            for ks,t in r["targets"].items():
                p=t["selected_residual"] or {}; hold=t["holdout_rows"]; hm=metric_summary(hold,top_k=(int(ks),)) if hold else {}
                f.write("| %s | %s | %.3f | %.6f | %.4f | %.4f | %.6f | %s | %s | %.6f |\n" % (ks,p.get("signal","NONE"),p.get("residual_lambda",0),p.get("mean_gain",0),p.get("p_value",1),p.get("bh_q_value",1),p.get("ci_low",0),p.get("stable",False),p.get("accepted",False),hm.get("logloss",0)))
            f.write("\n")
    print(json.dumps({"version":"V5.8.3","history_count":len(records),"layers":list(layers),"output_dir":a.output_dir},ensure_ascii=False))
if __name__=="__main__": main()
