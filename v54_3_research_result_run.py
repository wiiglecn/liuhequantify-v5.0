#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V5.4.3 Research Result Engine: aggregate nested results into JSON/CSV/Markdown."""
import argparse,csv,json,os
from core.multiple_testing import benjamini_hochberg,bonferroni

def load(path):
    with open(path,encoding="utf-8") as f:return json.load(f)

def flatten(report):
    rows=[]
    for key,r in report["layers"].items():
        for scope in ("outer","final_holdout"):
            m=r.get(scope,{})
            if not m or "n" not in m:continue
            row={"layer":key,"scope":scope,"n":m.get("n")}
            row.update({k:v for k,v in m.items() if k!="n"})
            rows.append(row)
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",default="reports/v5.4.2_nested_research.json")
    ap.add_argument("--output-dir",default="reports/v5.4.3")
    args=ap.parse_args()
    r=load(args.input); rows=flatten(r)
    os.makedirs(args.output_dir,exist_ok=True)
    with open(os.path.join(args.output_dir,"metrics.csv"),"w",newline="",encoding="utf-8") as f:
        keys=sorted({k for x in rows for k in x});w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    # Multiple testing is applied only to the already-defined OOS baseline tests.
    raw={}
    for layer,x in r["layers"].items():
        for k,p in x.get("outer_p_value_vs_baseline",{}).items():raw[f"{layer}.hit_at_{k}"]=p
    summary={"version":"V5.4.3","source":args.input,"history_count":r["history_count"],
             "multiple_testing":{"raw_p":raw,"bh":benjamini_hochberg(raw),"bonferroni":bonferroni(raw)}}
    with open(os.path.join(args.output_dir,"summary.json"),"w",encoding="utf-8") as f:json.dump(summary,f,ensure_ascii=False,indent=2)
    lines=[f"# V5.4.3 Research Result Engine","","- History: %d"%r["history_count"],"- Final holdout is sealed; selection/weights/calibration are inner-OOS only.",""]
    lines+=["| Layer | Scope | N | Hit@1 | Hit@3 | Hit@6 | LogLoss | Brier | ECE | IG |","|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for x in rows:
        lines.append("| {layer} | {scope} | {n} | {hit_at_1:.4f} | {hit_at_3:.4f} | {hit_at_6:.4f} | {logloss:.4f} | {brier:.4f} | {ece:.4f} | {information_gain:.4f} |".format(**{k:(0 if v is None else v) for k,v in x.items()}))
    with open(os.path.join(args.output_dir,"report.md"),"w",encoding="utf-8") as f:f.write("
".join(lines))
    print(json.dumps({"output_dir":args.output_dir,"rows":len(rows),"tests":len(raw)},ensure_ascii=False))
if __name__=="__main__":main()
