#!/usr/bin/env python3
"""Strict historical statistical hardening runner."""
import argparse,json,os
from dataclasses import asdict
from data_fetcher import Record
from core.residual_features import build_feature_rows,FEATURE_NAMES
from core.statistical_hardening import HardeningConfig,discover
from core.k_specific_policy import KPolicyConfig,fit_k_policy
from core.ensemble_weighting import combine_signal_probabilities

def load(path):
    raw=json.load(open(path,encoding="utf-8"))
    return [Record(expect=str(x["expect"]),open_time=str(x.get("openTime","")),
                   regular=x["regular"],special=x["special"],waves=x.get("waves",[]),
                   zodiacs=x.get("zodiacs",[])) for x in raw]

def build_rows(usable,features,signals,candidates,k,final_holdout):
    cfg=KPolicyConfig(windows=(60,120),pair_lambdas=(0.0,),diversity_lambdas=(0.0,))
    fr=build_feature_rows(usable,signals,candidates)
    rows=[]
    stop=len(usable)-final_holdout
    for i in range(60,stop):
        policy=fit_k_policy(usable[:i],signals,candidates,k,cfg)
        base=combine_signal_probabilities({s:usable[i]["prob"][s] for s in signals},candidates,policy.weights)
        top=sorted(base,key=base.get,reverse=True)[:k]
        bp=float(sum(base.get(x,0.0) for x in top))
        rows.append({**fr[i],"hit":1 if usable[i]["actual"] in top else 0,
                     "base_prob":min(.999999,max(.000001,bp))})
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",default="macau_history.json")
    ap.add_argument("--output-dir",default="reports/v5.8.4.1")
    ap.add_argument("--final-holdout",type=int,default=60)
    a=ap.parse_args()
    records=load(a.input)
    import zodiac_ensemble as z,wide_ensemble as w
    zfold=z.precompute_folds(records,max(1,len(records)-10))
    wfold=w.precompute_folds(records,max(1,len(records)-10))
    cases=[
      ("zodiac",zfold,list(z.SIGNAL_NAMES),sorted({f["actual"] for f in zfold if f.get("actual") is not None}),(3,4,6)),
      ("wide",wfold,list(w.SIGNAL_NAMES),list(w.ALL_NUMS),(20,))
    ]
    results={}
    hcfg=HardeningConfig()
    for layer,folds,signals,candidates,ks in cases:
        usable=[f for f in folds if f.get("actual") is not None and all(s in f.get("prob",{}) for s in signals)]
        results[layer]={}
        for k in ks:
            rows=build_rows(usable,FEATURE_NAMES,signals,candidates,k,a.final_holdout)
            selected,items,raw=discover(rows,FEATURE_NAMES,k,hcfg)
            results[layer][str(k)]={
                "selected":asdict(selected) if selected else None,
                "candidates":[asdict(x) for x in items],
                "n_rows":len(rows),
                "oos_origins":len(next(iter(raw.values()))[1]) if raw else 0,
                "protocol":"train_fit_validation_freeze_rolling_oos",
                "config":asdict(hcfg)
            }
    os.makedirs(a.output_dir,exist_ok=True)
    payload={"version":"V5.8.4.1","results":results}
    with open(os.path.join(a.output_dir,"statistical_hardening.json"),"w",encoding="utf-8") as f:
        json.dump(payload,f,ensure_ascii=False,indent=2)
    with open(os.path.join(a.output_dir,"report.md"),"w",encoding="utf-8") as f:
        f.write("# Statistical Hardening\n\n")
        f.write("Protocol: TRAIN-only fit -> frozen VALIDATION -> rolling OOS aggregation -> bootstrap CI + exact sign test + BH/Bonferroni.\n\n")
        for layer,rr in results.items():
            f.write("## "+layer+"\n\n")
            f.write("| K | Feature | OOS gain | 95% CI | p | BH q | Bonferroni | +fraction | Blocks | Accepted |\n|---:|---|---:|---|---:|---:|---:|---:|---:|:---:|\n")
            for k,x in rr.items():
                s=x["selected"]
                if s:
                    f.write("| %s | %s | %.6f | [%.6f, %.6f] | %.4f | %.4f | %.4f | %.3f | %d/%d | %s |\n" %
                            (k,s["feature"],s["mean_gain"],s["ci_low"],s["ci_high"],s["p_value"],
                             s["bh_q_value"],s["bonferroni_p_value"],s["positive_fraction"],
                             s["positive_blocks"],s["blocks"],s["accepted"]))
                else:
                    f.write("| %s | NONE | — | — | — | — | — | — | — | false |\n"%k)
    print(json.dumps({"version":"V5.8.4.1","layers":list(results),"output_dir":a.output_dir}))
if __name__=="__main__":main()
