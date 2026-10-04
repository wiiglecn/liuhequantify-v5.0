#!/usr/bin/env python3
"""Residual feature discovery runner."""
import argparse,json,os
from dataclasses import asdict
from data_fetcher import Record
from core.residual_features import build_feature_rows,FEATURE_NAMES
from core.residual_feature_discovery import FeatureDiscoveryConfig,discover
from core.k_specific_policy import KPolicyConfig,fit_k_policy
from core.ensemble_weighting import combine_signal_probabilities

def load(path):
    raw=json.load(open(path,encoding="utf-8"))
    return [Record(expect=str(x["expect"]),open_time=str(x.get("openTime","")),regular=x["regular"],special=x["special"],waves=x.get("waves",[]),zodiacs=x.get("zodiacs",[])) for x in raw]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",default="macau_history.json"); ap.add_argument("--output-dir",default="reports/v5.8.4"); ap.add_argument("--final-holdout",type=int,default=60)
    a=ap.parse_args(); records=load(a.input)
    import zodiac_ensemble as z,wide_ensemble as w
    results={}
    for layer,folds,signals,candidates,ks in [
        ("zodiac",z.precompute_folds(records,max(1,len(records)-10)),list(z.SIGNAL_NAMES),sorted({f["actual"] for f in z.precompute_folds(records,max(1,len(records)-10)) if f["actual"] is not None}),(3,4,6)),
        ("wide",w.precompute_folds(records,max(1,len(records)-10)),list(w.SIGNAL_NAMES),list(w.ALL_NUMS),(20,))]:
        usable=[f for f in folds if f.get("actual") is not None and all(s in f.get("prob",{}) for s in signals)]
        feat=build_feature_rows(usable,signals,candidates)
        results[layer]={}
        for k in ks:
            cfg=KPolicyConfig(windows=(60,120),pair_lambdas=(0.0,),diversity_lambdas=(0.0,))
            rows=[]
            for i in range(60,len(usable)-a.final_holdout):
                p=fit_k_policy(usable[:i],signals,candidates,k,cfg)
                base=combine_signal_probabilities({s:usable[i]["prob"][s] for s in signals},candidates,p.weights)
                top=sorted(base,key=base.get,reverse=True)[:k]
                rows.append({**feat[i],"hit":1 if usable[i]["actual"] in top else 0})
            chosen,items=discover(rows,list(FEATURE_NAMES),k,FeatureDiscoveryConfig())
            results[layer][str(k)]={"selected":asdict(chosen) if chosen else None,"candidates":[asdict(x) for x in items],"n":len(rows)}
    os.makedirs(a.output_dir,exist_ok=True)
    with open(os.path.join(a.output_dir,"feature_registry.json"),"w",encoding="utf-8") as f:json.dump({"version":"V5.8.4","results":results},f,ensure_ascii=False,indent=2)
    with open(os.path.join(a.output_dir,"report.md"),"w",encoding="utf-8") as f:
        f.write("# Residual Feature Discovery\n\n")
        for layer,rr in results.items():
            f.write("## "+layer+"\n\n| K | Feature | Validation gain | p | BH q | Accepted |\n|---:|---|---:|---:|---:|:---:|\n")
            for k,x in rr.items():
                s=x["selected"] or {}; f.write("| %s | %s | %.6f | %.4f | %.4f | %s |\n"%(k,s.get("feature","NONE"),s.get("mean_gain",0),s.get("p_value",1),s.get("bh_q_value",1),s.get("accepted",False)))
    print(json.dumps({"version":"V5.8.4","layers":list(results),"output_dir":a.output_dir}))
if __name__=="__main__":main()
