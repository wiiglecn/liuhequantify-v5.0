"""V5.8.4 point-in-time residual feature factory.

Creates orthogonal meta-features from the signal probability surface without
using future outcomes. Features are deliberately generic: entropy, top-gap,
cross-signal disagreement, rank dispersion, recent surprise and regime-like
concentration. They are candidates for a later residual learner, not claims
of predictive power.
"""
from dataclasses import dataclass,asdict
import math
import numpy as np

@dataclass(frozen=True)
class ResidualFeatureConfig:
    recent_window:int=30
    eps:float=1e-12

FEATURE_NAMES=("entropy_mean","entropy_std","top1_gap_mean","top1_gap_std",
               "signal_disagreement","rank_dispersion","recent_surprise",
               "concentration_mean","concentration_std")

def _norm(p,candidates,eps):
    x=np.asarray([max(0.0,float(p.get(c,0.0))) for c in candidates],dtype=float)
    s=float(x.sum())
    return x/s if s>eps else np.ones(len(candidates),dtype=float)/max(1,len(candidates))

def _entropy(x,eps):
    y=np.clip(x,eps,None); return float(-np.sum(y*np.log(y)))

def extract_signal_features(fold,signal_names,candidates,config=None):
    cfg=config or ResidualFeatureConfig()
    mats=[_norm(fold.get("prob",{}).get(n,{}),candidates,cfg.eps) for n in signal_names]
    if not mats:
        return {k:0.0 for k in FEATURE_NAMES}
    ent=np.asarray([_entropy(x,cfg.eps) for x in mats])
    gaps=np.asarray([x[0]-x[1] if len(x)>1 else x[0] for x in [np.sort(m)[::-1] for m in mats]])
    conc=np.asarray([float(np.sum(np.sort(m)[::-1][:min(3,len(m))])) for m in mats])
    M=np.asarray(mats)
    disagreement=float(np.mean(np.std(M,axis=0))) if len(M)>1 else 0.0
    ranks=np.argsort(np.argsort(-M,axis=1),axis=1) if len(M)>1 else np.zeros((1,len(candidates)))
    rank_disp=float(np.mean(np.std(ranks,axis=0))) if len(M)>1 else 0.0
    return {"entropy_mean":float(ent.mean()),"entropy_std":float(ent.std()),
            "top1_gap_mean":float(gaps.mean()),"top1_gap_std":float(gaps.std()),
            "signal_disagreement":disagreement,"rank_dispersion":rank_disp,
            "recent_surprise":0.0,"concentration_mean":float(conc.mean()),
            "concentration_std":float(conc.std())}

def add_recent_surprise(current,history,signal_names,candidates,config=None):
    cfg=config or ResidualFeatureConfig()
    if not history:return current
    h=history[-cfg.recent_window:] if cfg.recent_window else history
    vals=[]
    cur=current
    for n in signal_names:
        cp=_norm(cur.get("prob",{}).get(n,{}),candidates,cfg.eps)
        hp=[]
        for f in h:
            hp.append(_norm(f.get("prob",{}).get(n,{}),candidates,cfg.eps))
        mean=np.mean(np.asarray(hp),axis=0)
        vals.append(float(np.mean(np.abs(cp-mean))))
    current=dict(current); current["recent_surprise"]=float(np.mean(vals)) if vals else 0.0
    return current

def build_feature_rows(folds,signal_names,candidates,config=None):
    rows=[]; cfg=config or ResidualFeatureConfig()
    for i,f in enumerate(folds):
        x=extract_signal_features(f,signal_names,candidates,cfg)
        x=add_recent_surprise(x,folds[:i],signal_names,candidates,cfg)
        x["fold"]=i
        if f.get("actual") is not None:x["actual"]=f["actual"]
        rows.append(x)
    return rows
