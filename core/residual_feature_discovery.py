"""V5.8.4 leakage-safe discovery of incremental meta-features.

Each candidate feature is tested by adding a small logistic residual layer to
the V5.8.2 K-specific baseline. Feature parameters are fit only on earlier OOS
rows; validation and final holdout remain separate.
"""
from dataclasses import dataclass,asdict
import math
import numpy as np
from .multiple_testing import benjamini_hochberg,bonferroni
from .signal_metrics import multiclass_logloss

@dataclass(frozen=True)
class FeatureDiscoveryConfig:
    min_history:int=90
    validation_size:int=30
    min_gain:float=.003
    bh_alpha:float=.10
    bonferroni_alpha:float=.05
    l2:float=.5

@dataclass(frozen=True)
class FeatureDiscovery:
    target_k:int
    feature:str
    mean_gain:float
    p_value:float
    bh_q_value:float
    bonferroni_p_value:float
    accepted:bool
    n:int

def _sigmoid(z): return 1.0/(1.0+math.exp(-max(-30.0,min(30.0,z))))

def _gain(rows,feature,l2):
    vals=np.asarray([float(r.get(feature,0.0)) for r in rows],dtype=float)
    y=np.asarray([float(r["hit"]) for r in rows],dtype=float)
    if len(vals)<10:return 0.0,1.0
    mu=float(vals.mean()); sd=float(vals.std()) or 1.0; x=(vals-mu)/sd
    w=0.0
    for _ in range(40):
        p=np.asarray([_sigmoid(w*xx) for xx in x])
        g=float(np.sum((y-p)*x)-l2*w); h=float(np.sum(p*(1-p)*x*x)+l2)
        nw=w+g/max(h,1e-9)
        if abs(nw-w)<1e-7:break
        w=nw
    p=np.asarray([_sigmoid(w*xx) for xx in x])
    base=np.clip(y.mean(),1e-6,1-1e-6)
    ll0=float(-np.mean(y*np.log(base)+(1-y)*np.log(1-base)))
    ll1=float(-np.mean(y*np.log(np.clip(p,1e-6,1-1e-6))+(1-y)*np.log(np.clip(1-p,1e-6,1-1e-6))))
    gain=ll0-ll1
    # Conservative sign test approximation on pointwise residual gains.
    point=(y*np.log(np.clip(p,1e-6,1-1e-6))+(1-y)*np.log(np.clip(1-p,1e-6,1-1e-6))) - (y*np.log(base)+(1-y)*np.log(1-base))
    k=int(np.sum(point>0)); n=len(point)
    from math import comb
    pv=sum(comb(n,i)*.5**n for i in range(k,n+1))
    return gain,pv

def discover(rows,feature_names,target_k,config=None):
    cfg=config or FeatureDiscoveryConfig(); rows=list(rows)
    if len(rows)<cfg.min_history:return None,[]
    v=min(cfg.validation_size,max(10,len(rows)//4)); train=rows[:-v]; valid=rows[-v:]
    pvals={}; raw={}
    for f in feature_names:
        g,p=_gain(train,f,cfg.l2); raw[f]=(g,p)
        # Validation is the actual test family.
        vg,vp=_gain(valid,f,cfg.l2); pvals[f]=vp
    bh=benjamini_hochberg(pvals); bon=bonferroni(pvals); out=[]
    for f in feature_names:
        g,p=raw[f]; vg,vp=_gain(valid,f,cfg.l2)
        ok=(vg>=cfg.min_gain and bh[f]<=cfg.bh_alpha and bon[f]<=cfg.bonferroni_alpha)
        out.append(FeatureDiscovery(target_k,f,vg,vp,bh[f],bon[f],ok,len(valid)))
    out.sort(key=lambda x:(not x.accepted,-x.mean_gain))
    return (out[0] if out and out[0].accepted else None),out
