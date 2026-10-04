#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V5.5.1 real historical signal discovery runner."""
import argparse,csv,json,os
from data_fetcher import Record
from core.signal_discovery import build_discovery_report

def load_json(path):
    with open(path,encoding="utf-8") as f: raw=json.load(f)
    return [Record(expect=str(x["expect"]),open_time=str(x.get("openTime","")),
                   regular=x["regular"],special=x["special"],
                   waves=x.get("waves",[]),zodiacs=x.get("zodiacs",[])) for x in raw]

def zodiac_layer(records):
    import zodiac_ensemble as z
    n=max(1,len(records)-10); folds=z.precompute_folds(records,n)
    return folds,list(z.SIGNAL_NAMES),sorted({f["actual"] for f in folds if f["actual"] is not None})

def wide_layer(records):
    import wide_ensemble as w
    n=max(1,len(records)-10); folds=w.precompute_folds(records,n)
    return folds,list(w.SIGNAL_NAMES),list(range(1,50))

def dimension_layers(records):
    import dim_ensemble as d
    from dimensions import DIMENSIONS
    out={}
    for name,extract in DIMENSIONS:
        prior_fn=d.DIM_PRIOR_FN[name]; n=max(1,len(records)-10)
        folds=d.precompute_folds(records,extract,prior_fn,n)
        candidates=sorted({f["actual"] for f in folds if f["actual"] is not None})
        out[name]=(folds,list(d.SIGNAL_NAMES),candidates)
    return out

def usable_signal_rows(folds,signals):
    usable=[f for f in folds if f.get("actual") is not None and all(s in f.get("prob",{}) for s in signals)]
    return {s:[(f["prob"].get(s,{}),f["actual"]) for f in usable] for s in signals},usable

def periods_for(records,folds):
    n=len(folds); start=len(records)-n
    return [str(records[start+i].open_time[:4]) if records[start+i].open_time else "unknown" for i in range(n)]

def analyze_layer(name,folds,signals,candidates,records,threshold):
    rows_by_signal,usable=usable_signal_rows(folds,signals)
    if not usable: raise ValueError("no usable OOS folds for "+name)
    all_periods=periods_for(records,folds)
    usable_idx=[i for i,f in enumerate(folds) if f.get("actual") is not None and all(s in f.get("prob",{}) for s in signals)]
    periods=[all_periods[i] for i in usable_idx]
    report=build_discovery_report(name,rows_by_signal,candidates,periods,threshold)
    report.update({"raw_folds":len(folds),"usable_folds":len(usable),
                   "first_period":periods[0],"last_period":periods[-1]})
    return report

def write_csv(reports,path):
    rows=[]
    for layer,r in reports.items():
        for name,s in r["signals"].items():
            rows.append({"layer":layer,"signal":name,"n":s["n"],
                "classification":s["classification"],
                "hit_at_1":s["metrics"].get("hit_at_1",0),
                "hit_at_3":s["metrics"].get("hit_at_3",0),
                "hit_at_6":s["metrics"].get("hit_at_6",0),
                "logloss":s["metrics"].get("logloss",0),
                "brier":s["metrics"].get("brier",0),
                "ece":s["metrics"].get("ece",0),
                "information_gain":s["metrics"].get("information_gain",0),
                "p_hit1":s["p_values_vs_baseline"].get("1",1),
                "ig_ci_low":s["bootstrap_ci"]["information_gain"]["low"],
                "ig_ci_high":s["bootstrap_ci"]["information_gain"]["high"],
                "max_abs_scalar_correlation":s.get("max_abs_scalar_correlation",0)})
    os.makedirs(os.path.dirname(path) or ".",exist_ok=True)
    with open(path,"w",newline="",encoding="utf-8-sig") as f:
        fields=list(rows[0]) if rows else ["layer"]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def write_md(reports,path):
    lines=["# V5.5.1 Signal Discovery Report","",
           "> Historical point-in-time OOS diagnostics only. Labels do not establish future predictability.",""]
    for layer,r in reports.items():
        lines += [f"## {layer}",f"- OOS folds: {r['usable_folds']}/{r['raw_folds']}",
                  f"- Candidates: {r['candidate_count']}","",
                  "| Signal | Hit@1 | Hit@3 | Hit@6 | LogLoss | IG | IG 95% CI | Class |",
                  "|---|---:|---:|---:|---:|---:|---|---|"]
        for n,s in r["signals"].items():
            ci=s["bootstrap_ci"]["information_gain"]
            lines.append(f"| {n} | {s['metrics'].get('hit_at_1',0):.4f} | {s['metrics'].get('hit_at_3',0):.4f} | {s['metrics'].get('hit_at_6',0):.4f} | {s['metrics'].get('logloss',0):.4f} | {s['metrics'].get('information_gain',0):.4f} | [{ci['low']:.4f}, {ci['high']:.4f}] | {s['classification']} |")
        lines += ["",f"Redundancy groups: {json.dumps(r['redundancy_groups'],ensure_ascii=False)}",""]
    os.makedirs(os.path.dirname(path) or ".",exist_ok=True)
    with open(path,"w",encoding="utf-8") as f:f.write("\n".join(lines))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",default="macau_history.json")
    ap.add_argument("--output-dir",default="reports/v5.5.1")
    ap.add_argument("--redundancy-threshold",type=float,default=.85)
    args=ap.parse_args()
    records=load_json(args.input)
    if len(records)<240: raise SystemExit("insufficient history for V5.5.1")
    layers={}
    zf,zs,zc=zodiac_layer(records); layers["zodiac"]=analyze_layer("zodiac",zf,zs,zc,records,args.redundancy_threshold)
    wf,ws,wc=wide_layer(records); layers["wide"]=analyze_layer("wide",wf,ws,wc,records,args.redundancy_threshold)
    for name,(folds,sigs,cands) in dimension_layers(records).items():
        layers["dimension."+name]=analyze_layer("dimension."+name,folds,sigs,cands,records,args.redundancy_threshold)
    report={"version":"V5.5.1","history_count":len(records),
            "first_expect":records[0].expect,"last_expect":records[-1].expect,
            "layers":layers,"research_boundary":"OOS diagnostics only; no future claim"}
    os.makedirs(args.output_dir,exist_ok=True)
    with open(os.path.join(args.output_dir,"signal_registry.json"),"w",encoding="utf-8") as f:
        json.dump(report,f,ensure_ascii=False,indent=2)
    write_csv(layers,os.path.join(args.output_dir,"signal_metrics.csv"))
    write_md(layers,os.path.join(args.output_dir,"report.md"))
    print(json.dumps({"version":"V5.5.1","history_count":len(records),
                      "layers":list(layers),"output_dir":args.output_dir},ensure_ascii=False))
if __name__=="__main__":main()
