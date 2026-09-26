"""V5.4 leakage-safe signal metrics."""
from math import log, sqrt
from statistics import NormalDist
from typing import Any, Iterable, Mapping, Sequence
EPS=1e-12

def hit_at_k(rank: Sequence[Any], actual: Any, k: int) -> float:
    return float(actual in set(rank[:max(0,k)]))

def multiclass_logloss(probabilities: Mapping[Any,float], actual: Any) -> float:
    return -log(max(EPS,float(probabilities.get(actual,0.0))))

def multiclass_brier(probabilities: Mapping[Any,float], actual: Any) -> float:
    return sum((float(p)-(1.0 if c==actual else 0.0))**2 for c,p in probabilities.items())

def top1_confidence(probabilities: Mapping[Any,float]) -> float:
    return max((float(v) for v in probabilities.values()),default=0.0)

def expected_calibration_error(rows: Iterable[tuple[Mapping[Any,float],Any]], bins:int=10)->float:
    rows=list(rows)
    if not rows or bins<=0:return 0.0
    buckets=[[] for _ in range(bins)]
    for p,actual in rows:
        conf=top1_confidence(p); pred=max(p,key=p.get) if p else None
        idx=min(bins-1,int(max(0.0,min(1.0,conf))*bins))
        buckets[idx].append((conf,float(pred==actual)))
    n=len(rows)
    return sum(len(b)/n*abs(sum(x for x,_ in b)/len(b)-sum(y for _,y in b)/len(b)) for b in buckets if b)

def reliability_curve(rows:Iterable[tuple[Mapping[Any,float],Any]],bins:int=10)->list[dict]:
    rows=list(rows); buckets=[[] for _ in range(max(1,bins))]
    for p,actual in rows:
        conf=top1_confidence(p); pred=max(p,key=p.get) if p else None
        idx=min(len(buckets)-1,int(max(0.0,min(1.0,conf))*len(buckets)))
        buckets[idx].append((conf,float(pred==actual)))
    return [{"bin":i,"count":len(b),"confidence":sum(x for x,_ in b)/len(b) if b else 0.0,
             "accuracy":sum(y for _,y in b)/len(b) if b else 0.0} for i,b in enumerate(buckets)]

def _rank(values:Sequence[float])->list[float]:
    order=sorted(range(len(values)),key=lambda i:(float(values[i]),i)); ranks=[0.0]*len(values);i=0
    while i<len(order):
        j=i+1
        while j<len(order) and float(values[order[j]])==float(values[order[i]]):j+=1
        r=(i+j-1)/2.0
        for q in range(i,j):ranks[order[q]]=r
        i=j
    return ranks

def pearson(x:Sequence[float],y:Sequence[float])->float:
    n=min(len(x),len(y))
    if n<2:return 0.0
    x,y=list(x[:n]),list(y[:n]);mx,my=sum(x)/n,sum(y)/n
    a=sum((v-mx)**2 for v in x);b=sum((v-my)**2 for v in y)
    if a<=EPS or b<=EPS:return 0.0
    return sum((x[i]-mx)*(y[i]-my) for i in range(n))/sqrt(a*b)

def spearman(x:Sequence[float],y:Sequence[float])->float:
    return pearson(_rank(x),_rank(y))

def rank_ic(scores:Mapping[Any,float],actual:Any,candidates:Sequence[Any])->float:
    return spearman([float(scores.get(c,0.0)) for c in candidates],[1.0 if c==actual else 0.0 for c in candidates])

def information_gain(probabilities:Mapping[Any,float],actual:Any,baseline:float|None=None)->float:
    if baseline is None:baseline=1.0/max(1,len(probabilities))
    return log(max(EPS,float(probabilities.get(actual,0.0)))/max(EPS,float(baseline)))

def metric_summary(rows:Iterable[tuple[Mapping[Any,float],Any]],top_k=(1,3,6))->dict:
    rows=list(rows);n=len(rows)
    if not n:return {"n":0}
    out={"n":n,"logloss":sum(multiclass_logloss(p,a) for p,a in rows)/n,
         "brier":sum(multiclass_brier(p,a) for p,a in rows)/n,
         "ece":expected_calibration_error(rows),
         "information_gain":sum(information_gain(p,a) for p,a in rows)/n}
    for k in top_k:
        out[f"hit_at_{k}"]=sum(hit_at_k(sorted(p,key=p.get,reverse=True),a,k) for p,a in rows)/n
    return out

def bootstrap_metric_ci(rows, metric, rounds=1000, seed=20260926, confidence=.95, top_k=(1,3,6)):
    """Non-parametric bootstrap CI for an OOS metric."""
    rows=list(rows); n=len(rows)
    if not rows:
        return {"low":0.0,"high":0.0,"mean":0.0,"rounds":0}
    import random
    rng=random.Random(seed); vals=[]
    for _ in range(max(1,rounds)):
        sample=[rows[rng.randrange(n)] for _ in range(n)]
        vals.append(metric_summary(sample,top_k=top_k).get(metric,0.0))
    vals.sort()
    alpha=(1.0-confidence)/2.0
    lo=vals[min(len(vals)-1,max(0,int(alpha*len(vals))))]
    hi=vals[min(len(vals)-1,max(0,int((1-alpha)*len(vals))-1))]
    return {"low":lo,"high":hi,"mean":sum(vals)/len(vals),"rounds":len(vals)}

def binomial_two_sided_pvalue(hits:int,n:int,baseline:float)->float:
    if n<=0:return 1.0
    baseline=min(1.0-EPS,max(EPS,float(baseline)));se=sqrt(baseline*(1-baseline)/n)
    z=((hits/n)-baseline)/max(EPS,se)
    return 2.0*(1.0-NormalDist().cdf(abs(z)))
