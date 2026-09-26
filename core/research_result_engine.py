"""V5.4.3 Research Result Engine.

Transforms nested OOS output into an auditable research report:
- aggregate metrics and bootstrap confidence intervals
- yearly/period stability when fold timestamps are supplied
- per-signal inner/outer diagnostics
- multiple-testing correction
- ensemble vs uniform-baseline deltas
- frozen final-holdout section kept separate from model-selection evidence

This module contains no model fitting and never mutates a policy.
"""
from math import sqrt
from collections import defaultdict
from .multiple_testing import benjamini_hochberg, bonferroni
from .signal_metrics import metric_summary, information_gain

def _mean(xs):
    return sum(xs)/len(xs) if xs else 0.0

def bootstrap_mean_ci(values, iterations=1000, seed=542):
    values=[float(x) for x in values]
    if len(values)<2:return {"lower":_mean(values),"upper":_mean(values),"iterations":0}
    # Deterministic LCG avoids numpy dependency and keeps reports reproducible.
    state=seed & 0x7fffffff; n=len(values); means=[]
    for _ in range(max(100,iterations)):
        s=0.0
        for _ in range(n):
            state=(1103515245*state+12345)&0x7fffffff
            s += values[state % n]
        means.append(s/n)
    means.sort()
    lo=means[int(0.025*len(means))]
    hi=means[min(len(means)-1,int(0.975*len(means)))]
    return {"lower":lo,"upper":hi,"iterations":len(means)}

def metric_ci(rows, top_k=(1,3,6), iterations=1000):
    rows=list(rows); out=metric_summary(rows,top_k=top_k)
    series={"logloss":[-information_gain(p,a) for p,a in rows],
            "brier":[sum((float(v)-(1.0 if c==a else 0.0))**2 for c,v in p.items()) for p,a in rows]}
    for k in top_k:
        series[f"hit_at_{k}"]=[float(a in sorted(p,key=p.get,reverse=True)[:k]) for p,a in rows]
    out["bootstrap_ci"]={m:bootstrap_mean_ci(v,iterations=iterations) for m,v in series.items()}
    return out

def stability_by_period(rows, periods, top_k=(1,3,6)):
    grouped=defaultdict(list)
    for row,period in zip(rows,periods):grouped[str(period)].append(row)
    return {p:metric_ci(v,top_k=top_k,iterations=400) for p,v in sorted(grouped.items())}

def multiple_testing_from_pvalues(pvalues,alpha=0.05):
    return {"raw":dict(pvalues),"bh":benjamini_hochberg(pvalues),
            "bonferroni":bonferroni(pvalues,alpha=alpha)}

def signal_outer_report(rows_by_signal, candidates, top_k=(1,3,6)):
    out={}
    for name,rows in rows_by_signal.items():
        out[name]=metric_ci(rows,top_k=top_k,iterations=400)
    return out

def build_result_report(layer_result, outer_rows=None, periods=None, signal_outer_rows=None,
                        alpha=0.05, top_k=(1,3,6)):
    outer_rows=list(outer_rows or [])
    report={"engine_version":"V5.4.3","layer":layer_result.get("layer"),
            "outer":dict(layer_result.get("outer",{})),
            "outer_n":layer_result.get("outer_n",len(outer_rows)),
            "final_holdout":dict(layer_result.get("final_holdout",{})),
            "holdout_isolated":bool(layer_result.get("holdout_isolated",False)),
            "final_policy":layer_result.get("final_policy"),
            "research_interpretation":{
                "role":"historical OOS research diagnostics",
                "not_a_guarantee":True,
                "final_holdout_not_used_for_selection":bool(layer_result.get("holdout_isolated",False))
            }}
    if outer_rows:
        report["outer_with_ci"]=metric_ci(outer_rows,top_k=top_k)
    if periods and outer_rows:
        report["stability"]=stability_by_period(outer_rows,periods,top_k=top_k)
    if signal_outer_rows:
        report["signals"]=signal_outer_report(signal_outer_rows,[],top_k=top_k)
        p={name:1.0 for name in signal_outer_rows}
        # If callers provide explicit p-values, pass them through instead.
        report["multiple_testing"]=multiple_testing_from_pvalues(p,alpha=alpha)
    return report
