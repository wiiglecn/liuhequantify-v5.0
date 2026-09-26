"""V5.5 Signal Stability & Alpha Discovery Engine.

Pure research layer: consumes point-in-time OOS signal scores/probabilities and
actuals. It never fits a predictor and never uses future observations.
"""
from collections import defaultdict
from math import sqrt
from .signal_metrics import pearson, spearman, metric_summary, information_gain

def _mean(xs): return sum(xs)/len(xs) if xs else 0.0

def rolling_metric(rows, metric_fn, windows=(30,60,120,180,300)):
    rows=list(rows); out={}
    for w in windows:
        if len(rows)<w: continue
        vals=[metric_fn(rows[i-w:i]) for i in range(w,len(rows)+1)]
        out[str(w)]={"n_windows":len(vals),"mean":_mean(vals),"min":min(vals),"max":max(vals),
                     "last":vals[-1]}
    return out

def hit_rate(rows,k):
    return _mean([float(a in sorted(p,key=p.get,reverse=True)[:k]) for p,a in rows])

def rolling_hit_rates(rows,windows=(30,60,120,180,300),k=1):
    return rolling_metric(rows,lambda x:hit_rate(x,k),windows)

def signal_stability(rows,top_k=(1,3,6),windows=(30,60,120,180,300)):
    rows=list(rows)
    return {"n":len(rows),
            "rolling":{f"hit_at_{k}":rolling_hit_rates(rows,windows,k) for k in top_k},
            "overall":metric_summary(rows,top_k=top_k)}

def stability_by_period(rows,periods,top_k=(1,3,6)):
    groups=defaultdict(list)
    for row,p in zip(rows,periods):groups[str(p)].append(row)
    return {p:metric_summary(v,top_k=top_k) for p,v in sorted(groups.items())}

def residual_alpha(base_scores,candidate_scores,actual):
    n=min(len(base_scores),len(candidate_scores),len(actual))
    if n<3:return {"n":n,"base_corr":0.0,"candidate_corr":0.0,"residual_corr":0.0}
    x,z,y=map(lambda q:list(map(float,q[:n])),(base_scores,candidate_scores,actual))
    mx,mz=_mean(x),_mean(z);vx=sum((v-mx)**2 for v in x)
    if vx<=1e-12: residual=z
    else:
        beta=sum((x[i]-mx)*(z[i]-mz) for i in range(n))/vx
        alpha=mz-beta*mx
        residual=[z[i]-(alpha+beta*x[i]) for i in range(n)]
    return {"n":n,"base_corr":pearson(x,y),"candidate_corr":pearson(z,y),
            "residual_corr":pearson(residual,y)}

def signal_alpha_matrix(signal_rows,actual_scores):
    names=list(signal_rows);out={}
    for i,a in enumerate(names):
        out[a]={}
        for b in names:
            out[a][b]=residual_alpha(signal_rows[a],signal_rows[b],actual_scores)
    return out

def signal_decay(rows,metric_fn=lambda r: hit_rate(r,1),windows=(30,60,120,180,300)):
    rows=list(rows); out={}
    for w in windows:
        if len(rows)>=2*w:
            old=metric_fn(rows[-2*w:-w]); new=metric_fn(rows[-w:])
            out[str(w)]={"old":old,"recent":new,"delta":new-old,"ratio":new/max(1e-12,old)}
    return out

def rank_stability(period_scores):
    vals=[float(x) for x in period_scores.values()]
    if len(vals)<2:return {"periods":len(vals),"spearman":0.0}
    return {"periods":len(vals),"spearman":spearman(list(range(len(vals))),vals)}

def classify_stability(stability, min_windows=3):
    labels={}
    for k,v in stability.get("rolling",{}).items():
        labels[k]={}
        for w,x in v.items():
            if x["n_windows"]<min_windows: labels[k][w]="insufficient"
            elif x["min"]>0 and x["last"]>=x["mean"]*0.9: labels[k][w]="stable"
            elif x["last"]<x["mean"]*0.75: labels[k][w]="decaying"
            else: labels[k][w]="unstable"
    return labels

def build_signal_registry(signal_rows,actuals,periods=None,top_k=(1,3,6)):
    registry={}
    for name,rows in signal_rows.items():
        r={"signal":name,**signal_stability(rows,top_k=top_k),
           "decay":signal_decay(rows),
           "stability_class":classify_stability(signal_stability(rows,top_k=top_k))}
        if periods:r["periods"]=stability_by_period(rows,periods,top_k)
        r["mean_information_gain"]=_mean([information_gain(p,a) for p,a in rows])
        registry[name]=r
    return registry
