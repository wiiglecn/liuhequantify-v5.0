"""V5.8.4.1 statistical hardening for residual feature discovery.

Strict protocol:
1. Fit intercept/coefficient on TRAIN only.
2. Freeze parameters and score the following VALIDATION block.
3. Roll the train/validation origin forward and aggregate only frozen OOS rows.
4. Report paired sign-test p-value, bootstrap CI, and directional block stability.
5. Apply BH and Bonferroni across the complete feature family.
"""
from dataclasses import dataclass,asdict
from math import comb,log,exp
import numpy as np
from .multiple_testing import benjamini_hochberg,bonferroni

@dataclass(frozen=True)
class HardeningConfig:
    min_train:int=120
    validation_size:int=30
    train_window:int=180
    min_gain:float=.003
    bh_alpha:float=.10
    bonferroni_alpha:float=.05
    l2:float=.5
    bootstrap_rounds:int=500
    bootstrap_seed:int=20260926
    min_blocks:int=4
    stability_threshold:float=.60

@dataclass(frozen=True)
class HardenedFeature:
    target_k:int
    feature:str
    mean_gain:float
    ci_low:float
    ci_high:float
    p_value:float
    bh_q_value:float
    bonferroni_p_value:float
    positive_fraction:float
    positive_blocks:int
    blocks:int
    accepted:bool
    n_oos:int

def _sigmoid(z):
    z=max(-30.0,min(30.0,float(z)))
    return 1.0/(1.0+exp(-z))

def _fit(train,feature,l2):
    x=np.asarray([float(r.get(feature,0.0)) for r in train],dtype=float)
    base=np.asarray([float(r["base_prob"]) for r in train],dtype=float)
    y=np.asarray([float(r["hit"]) for r in train],dtype=float)
    mu=float(x.mean()) if len(x) else 0.0
    sd=float(x.std()) if len(x) else 1.0
    if sd<1e-9: sd=1.0
    z=(x-mu)/sd
    bp=np.clip(base,1e-6,1-1e-6)
    b0=float(np.log(bp/(1.0-bp)).mean())
    w=0.0
    # Intercept is initialized from the frozen baseline probability surface;
    # only the residual coefficient is regularized and optimized on TRAIN.
    for _ in range(60):
        p=np.asarray([_sigmoid(b0+w*xx) for xx in z])
        g=float(np.sum((y-p)*z)-l2*w)
        h=float(np.sum(p*(1-p)*z*z)+l2)
        nw=w+g/max(h,1e-9)
        if abs(nw-w)<1e-8: break
        w=nw
    return {"mu":mu,"sd":sd,"intercept":b0,"weight":w}

def _score(rows,feature,params):
    x=np.asarray([float(r.get(feature,0.0)) for r in rows],dtype=float)
    y=np.asarray([float(r["hit"]) for r in rows],dtype=float)
    bp=np.clip(np.asarray([float(r["base_prob"]) for r in rows],dtype=float),1e-6,1-1e-6)
    z=(x-params["mu"])/params["sd"]
    p=np.asarray([_sigmoid(params["intercept"]+params["weight"]*xx) for xx in z])
    base_ll=-(y*np.log(bp)+(1-y)*np.log(1-bp))
    model_ll=-(y*np.log(np.clip(p,1e-6,1-1e-6))+(1-y)*np.log(np.clip(1-p,1e-6,1-1e-6)))
    return base_ll-model_ll

def _sign_pvalue(gains):
    g=np.asarray(gains,dtype=float)
    n=len(g); k=int(np.sum(g>0))
    if n==0:return 1.0
    # Two-sided exact sign test, conservative under ties.
    t=min(k,n-k)
    return min(1.0,2.0*sum(comb(n,i)*0.5**n for i in range(t+1)))

def _bootstrap_ci(gains,rounds,seed):
    g=np.asarray(gains,dtype=float)
    if len(g)==0:return 0.0,0.0
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(g),size=(max(1,rounds),len(g)))
    means=g[idx].mean(axis=1)
    return float(np.quantile(means,.025)),float(np.quantile(means,.975))

def _block_stats(gains,validation_size):
    g=np.asarray(gains,dtype=float)
    blocks=[]
    for i in range(0,len(g),validation_size):
        b=g[i:i+validation_size]
        if len(b): blocks.append(float(b.mean()))
    positive=sum(x>0 for x in blocks)
    return positive,len(blocks)

def rolling_oos(rows,feature,config=None):
    cfg=config or HardeningConfig()
    rows=list(rows); all_gains=[]; origins=[]
    start=cfg.min_train
    while start+cfg.validation_size<=len(rows):
        lo=max(0,start-cfg.train_window) if cfg.train_window else 0
        train=rows[lo:start]
        valid=rows[start:start+cfg.validation_size]
        params=_fit(train,feature,cfg.l2)
        gains=_score(valid,feature,params)
        all_gains.extend(gains.tolist())
        origins.append({"train_start":lo,"train_end":start,
                        "valid_start":start,"valid_end":start+len(valid),
                        "mean_gain":float(gains.mean()) if len(gains) else 0.0})
        start+=cfg.validation_size
    return np.asarray(all_gains,dtype=float),origins

def discover(rows,feature_names,target_k,config=None):
    cfg=config or HardeningConfig()
    raw={}
    for f in feature_names:
        gains,origins=rolling_oos(rows,f,cfg)
        pos_blocks,blocks=_block_stats(gains,cfg.validation_size)
        p=_sign_pvalue(gains)
        ci=_bootstrap_ci(gains,cfg.bootstrap_rounds,cfg.bootstrap_seed+sum(ord(c) for c in f))
        raw[f]=(gains,origins,pos_blocks,blocks,p,ci)
    pvals={f:v[4] for f,v in raw.items()}
    bh=benjamini_hochberg(pvals); bon=bonferroni(pvals)
    out=[]
    for f,v in raw.items():
        gains,origins,pos_blocks,blocks,p,ci=v
        mean=float(gains.mean()) if len(gains) else 0.0
        positive=float(np.mean(gains>0)) if len(gains) else 0.0
        ok=(mean>=cfg.min_gain and ci[0]>0 and
            bh[f]<=cfg.bh_alpha and bon[f]<=cfg.bonferroni_alpha and
            blocks>=cfg.min_blocks and
            pos_blocks/max(1,blocks)>=cfg.stability_threshold)
        out.append(HardenedFeature(target_k,f,mean,ci[0],ci[1],p,bh[f],bon[f],
                                    positive,pos_blocks,blocks,ok,len(gains)))
    out.sort(key=lambda x:(not x.accepted,-x.mean_gain))
    return (out[0] if out and out[0].accepted else None),out,raw
