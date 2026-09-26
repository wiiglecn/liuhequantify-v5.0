"""V5.5.1 historical signal discovery utilities.

All inputs are point-in-time OOS rows: (probability map, actual).
The module performs diagnostics only; it never fits a predictive model.
"""
from collections import defaultdict
from .signal_metrics import metric_summary, bootstrap_metric_ci, binomial_two_sided_pvalue
from .signal_analysis import signal_correlation, redundancy_groups
from .signal_stability import rolling_hit_rates, signal_decay, residual_alpha, _mean

def scalar_actual_scores(rows):
    return [float(p.get(a,0.0)) for p,a in rows]

def adjacent_decay(rows, metric_fn=lambda r: sum(float(a in sorted(p,key=p.get,reverse=True)[:1]) for p,a in r)/len(r) if r else 0.0,
                   windows=(30,60,120,180)):
    rows=list(rows); out={}
    for w in windows:
        if len(rows) < 2*w: continue
        pairs=[]
        for end in range(2*w,len(rows)+1,w):
            old=metric_fn(rows[end-2*w:end-w]); new=metric_fn(rows[end-w:end])
            pairs.append({"old":old,"recent":new,"delta":new-old})
        if pairs:
            out[str(w)]={"comparisons":len(pairs),"mean_delta":_mean([x["delta"] for x in pairs]),
                         "last":pairs[-1],"min_delta":min(x["delta"] for x in pairs),
                         "max_delta":max(x["delta"] for x in pairs)}
    return out

def period_rows(rows, periods):
    groups=defaultdict(list)
    for r,p in zip(rows,periods): groups[str(p)].append(r)
    return {k:v for k,v in sorted(groups.items())}

def classify_signal(rows, candidates, top_k=1, redundancy=False, correlation_max=0.85):
    rows=list(rows); n=len(rows)
    if not n: return "NO_EVIDENCE"
    m=metric_summary(rows,top_k=(1,3,6))
    ci=bootstrap_metric_ci(rows,"hit_at_1",rounds=800)
    baseline=min(1.0,1.0/max(1,len(candidates)))
    p=binomial_two_sided_pvalue(int(round(m["hit_at_1"]*n)),n,baseline)
    ig=bootstrap_metric_ci(rows,"information_gain",rounds=800)
    decay=adjacent_decay(rows)
    deltas=[v["last"]["delta"] for v in decay.values()]
    if redundancy and abs(float(redundancy))>=correlation_max:
        return "REDUNDANT"
    if ig["high"] < 0 or ci["high"] < baseline:
        return "NO_EVIDENCE"
    if deltas and sum(d<0 for d in deltas)>=max(1,len(deltas)//2) and m["hit_at_1"] < baseline:
        return "DECAYING"
    if p < 0.05 and ci["low"] >= baseline and ig["low"] >= 0:
        return "STABLE_OOS"
    return "UNSTABLE"

def discover_signal(name, rows, candidates, periods=None, redundancy_corr=0.0):
    rows=list(rows); m=metric_summary(rows,top_k=(1,3,6))
    ci={x:bootstrap_metric_ci(rows,x,rounds=800) for x in ("hit_at_1","hit_at_3","hit_at_6","logloss","brier","ece","information_gain")}
    baseline={str(k):min(1.0,k/max(1,len(candidates))) for k in (1,3,6)}
    pvals={str(k):binomial_two_sided_pvalue(int(round(m.get(f"hit_at_{k}",0)*len(rows))),len(rows),baseline[str(k)]) for k in (1,3,6)}
    result={"signal":name,"n":len(rows),"metrics":m,"bootstrap_ci":ci,"baseline_hit_rate":baseline,
            "p_values_vs_baseline":pvals,"scalar_actual_score_mean":_mean(scalar_actual_scores(rows)),
            "decay":adjacent_decay(rows),"rolling_hit":{str(k):rolling_hit_rates(rows,k=k) for k in (1,3,6)},
            "classification":classify_signal(rows,candidates,redundancy=redundancy_corr)}
    if periods:
        result["periods"]={p:metric_summary(v,top_k=(1,3,6)) for p,v in period_rows(rows,periods).items()}
    return result

def build_discovery_report(layer, signal_rows, candidates, periods=None, redundancy_threshold=.85):
    scalar={n:scalar_actual_scores(r) for n,r in signal_rows.items()}
    corr=signal_correlation(scalar)
    groups=redundancy_groups(scalar,threshold=redundancy_threshold)
    results={}
    for n,rows in signal_rows.items():
        maxcorr=max([abs(corr[n].get(o,0.0)) for o in signal_rows if o!=n] or [0.0])
        results[n]=discover_signal(n,rows,candidates,periods,maxcorr)
        results[n]["max_abs_scalar_correlation"]=maxcorr
    return {"version":"V5.5.1","layer":layer,"candidate_count":len(candidates),
            "signal_count":len(signal_rows),"signals":results,
            "scalar_oos_correlation":corr,"redundancy_groups":groups,
            "method":{"score":"p(actual|signal) per OOS fold","selection":"none; diagnostics only",
                      "classification":"descriptive historical OOS label; not a future predictability claim"}}
