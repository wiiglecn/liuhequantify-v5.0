"""V5.6 leakage-safe regime detection."""
from dataclasses import dataclass,asdict
import numpy as np

@dataclass(frozen=True)
class RegimeConfig:
    n_regimes:int=3
    context_window:int=60
    min_history:int=60
    max_iter:int=40
    random_seed:int=20260926

def _entropy(vals):
    x=np.asarray(vals,dtype=float); x=np.clip(x,1e-12,None); x/=x.sum()
    return float(-np.sum(x*np.log(x)))

def fold_features(fold,signal_names,candidates):
    c=list(candidates); mats=[]; feats=[]
    for n in signal_names:
        v=np.asarray([max(0.0,float(fold.get("prob",{}).get(n,{}).get(x,0.0))) for x in c])
        if v.sum()<=0: v=np.ones(len(c))/max(1,len(c))
        else: v/=v.sum()
        mats.append(v); s=np.sort(v)[::-1]
        feats += [_entropy(v),float(s[0]),float(s[0]-s[min(2,len(s)-1)])]
    m=np.asarray(mats)
    dis=float(np.mean(np.std(m,axis=0))) if len(m) else 0.0
    corr=np.corrcoef(m) if len(m)>1 else np.ones((1,1))
    tri=corr[np.triu_indices_from(corr,k=1)] if len(m)>1 else np.array([])
    feats += [dis,float(np.mean(tri)) if len(tri) else 0.0]
    return np.asarray(feats,dtype=float)

class RegimeDetector:
    def __init__(self,config=None):
        self.config=config or RegimeConfig(); self.centers=None; self.scale=None
    def fit(self,folds,signal_names,candidates):
        X=np.asarray([fold_features(f,signal_names,candidates) for f in folds],dtype=float)
        if not len(X): raise ValueError("no folds")
        self.scale=np.std(X,axis=0); self.scale[self.scale<1e-9]=1.0
        X=X/self.scale; k=min(self.config.n_regimes,len(X))
        if k<=1: self.centers=X.mean(axis=0,keepdims=True); return self
        idx=np.linspace(0,len(X)-1,k,dtype=int); C=X[idx].copy()
        for _ in range(self.config.max_iter):
            d=((X[:,None,:]-C[None,:,:])**2).sum(axis=2); lab=d.argmin(axis=1); N=C.copy()
            for j in range(k):
                if np.any(lab==j): N[j]=X[lab==j].mean(axis=0)
            if np.allclose(C,N): break
            C=N
        self.centers=C; return self
    def assign(self,fold,signal_names,candidates):
        if self.centers is None: raise ValueError("not fitted")
        x=fold_features(fold,signal_names,candidates)/self.scale
        return int(((self.centers-x[None,:])**2).sum(axis=1).argmin())
    def to_dict(self):
        return {"config":asdict(self.config),"centers":self.centers.tolist() if self.centers is not None else None}

def detect_regime_at(folds,target_idx,signal_names,candidates,config=None):
    cfg=config or RegimeConfig()
    h=list(folds[:target_idx])
    if len(h)<cfg.min_history:return 0,{"status":"insufficient_history","n_history":len(h)}
    h=h[-cfg.context_window*4:] if cfg.context_window else h
    d=RegimeDetector(cfg).fit(h,signal_names,candidates)
    return d.assign(folds[target_idx],signal_names,candidates),{"status":"ok","n_history":len(h),"detector":d.to_dict()}
