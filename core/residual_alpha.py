"""V5.8.3 leakage-safe residual alpha / incremental signal discovery."""
from dataclasses import dataclass,asdict
from math import comb
import random
from .ensemble_weighting import combine_signal_probabilities
from .k_specific_policy import fit_k_policy
from .signal_metrics import multiclass_logloss
from .multiple_testing import benjamini_hochberg,bonferroni

@dataclass(frozen=True)
class ResidualConfig:
    residual_lambdas:tuple=(0.05,0.10,0.20,0.30,0.50)
    min_history:int=90
    validation_size:int=30
    min_gain:float=.005
    alpha:float=.05
    bh_alpha:float=.10
    bootstrap_rounds:int=800
    seed:int=20260926
    min_stability_gain:float=.0

@dataclass(frozen=True)
class ResidualPolicy:
    target_k:int
    signal:str
    residual_lambda:float
    mean_gain:float
    p_value:float
    bh_q_value:float
    bonferroni_p_value:float
    ci_low:float
    ci_high:float
    stable:bool
    accepted:bool
    n:int

def _blend(base,candidate,candidates,lam):
    out={c:(1-lam)*float(base.get(c,0.0))+lam*float(candidate.get(c,0.0)) for c in candidates}
    s=sum(max(0.0,v) for v in out.values())
    if s<=0:return {c:1.0/max(1,len(candidates)) for c in candidates}
    return {c:max(0.0,out[c])/s for c in candidates}

def _sign_pvalue(improvements):
    vals=[float(x) for x in improvements if x!=0]; n=len(vals); k=sum(x>0 for x in vals)
    if n==0:return 1.0
    return sum(comb(n,i)*.5**n for i in range(k,n+1))

def _candidate_stats(rows,signal,candidates,lam,rounds,seed):
    gains=[]
    for r in rows:
        cand=r["prob"].get(signal,{})
        if not cand:continue
        gains.append(multiclass_logloss(r["base_prob"],r["actual"])-multiclass_logloss(_blend(r["base_prob"],cand,candidates,lam),r["actual"]))
    if not gains:return {"n":0,"mean_gain":0.0,"p_value":1.0,"ci":{"low":0.0,"high":0.0,"mean":0.0},"improvements":[]}
    rng=random.Random(seed+len(gains)+int(lam*1000)); means=[]
    for _ in range(max(1,rounds)):
        sample=[gains[rng.randrange(len(gains))] for _ in gains]
        means.append(sum(sample)/len(sample))
    means.sort(); lo=means[int(.025*(len(means)-1))]; hi=means[int(.975*(len(means)-1))]
    return {"n":len(gains),"mean_gain":sum(gains)/len(gains),"p_value":_sign_pvalue(gains),
            "ci":{"low":lo,"high":hi,"mean":sum(means)/len(means)},"improvements":gains}

def discover(rows,signal_names,candidates,target_k,config=None):
    cfg=config or ResidualConfig(); rows=list(rows)
    if len(rows)<max(cfg.min_history,20): return None,[]
    v=min(cfg.validation_size,max(10,len(rows)//4)); fit_rows=rows[:-v]; eval_rows=rows[-v:]
    selected={}; family={}
    for name in signal_names:
        best_lam=cfg.residual_lambdas[0]; best_gain=float("-inf")
        for lam in cfg.residual_lambdas:
            st=_candidate_stats(fit_rows,name,candidates,lam,max(50,min(200,cfg.bootstrap_rounds)),cfg.seed)
            if st["mean_gain"]>best_gain: best_gain=st["mean_gain"]; best_lam=lam
        selected[name]=best_lam
        for lam in cfg.residual_lambdas:
            family[f"{name}|{lam:g}"]=_candidate_stats(eval_rows,name,candidates,lam,cfg.bootstrap_rounds,cfg.seed)
    bh=benjamini_hochberg({k:v["p_value"] for k,v in family.items()})
    bon=bonferroni({k:v["p_value"] for k,v in family.items()}); policies=[]
    for name in signal_names:
        lam=selected[name]; key=f"{name}|{lam:g}"; st=family[key]; gains=st["improvements"]; stable=False
        if len(gains)>=20:
            mid=len(gains)//2
            stable=sum(gains[:mid])/max(1,mid)>=cfg.min_stability_gain and sum(gains[mid:])/max(1,len(gains)-mid)>=cfg.min_stability_gain
        accepted=(st["mean_gain"]>=cfg.min_gain and st["ci"]["low"]>0 and bh[key]<=cfg.bh_alpha and bon[key]<=cfg.alpha and stable)
        policies.append(ResidualPolicy(target_k,name,lam,st["mean_gain"],st["p_value"],bh[key],bon[key],st["ci"]["low"],st["ci"]["high"],stable,accepted,st["n"]))
    policies.sort(key=lambda x:(not x.accepted,-x.mean_gain))
    return (policies[0] if policies and policies[0].accepted else None),policies

def apply_residual(base_prob,row,candidates,policy):
    if policy is None:return dict(base_prob)
    return _blend(base_prob,row.get("prob",{}).get(policy.signal,{}),candidates,policy.residual_lambda)

def build_point_in_time_rows(folds,signal_names,candidates,target_k,k_config=None,initial_train=60,final_holdout=60):
    n=len(folds); hs=max(0,n-final_holdout); rows=[]
    for j in range(max(initial_train,1),hs):
        hist=folds[:j]; policy=fit_k_policy(hist,signal_names,candidates,target_k,k_config)
        base=combine_signal_probabilities({s:folds[j]["prob"].get(s,{}) for s in signal_names},candidates,policy.weights)
        rows.append({"fold":j,"actual":folds[j]["actual"],"base_prob":base,"prob":folds[j]["prob"],"policy":asdict(policy)})
    return rows,hs

def evaluate_residual(folds,signal_names,candidates,target_k,k_config=None,config=None,initial_train=60,final_holdout=60):
    cfg=config or ResidualConfig(); rows,hs=build_point_in_time_rows(folds,signal_names,candidates,target_k,k_config,initial_train,final_holdout)
    chosen,discoveries=discover(rows,signal_names,candidates,target_k,cfg)
    outer=[(apply_residual(r["base_prob"],r,candidates,chosen),r["actual"]) for r in rows]; hold=[]
    if hs>=initial_train:
        fp=fit_k_policy(folds[:hs],signal_names,candidates,target_k,k_config)
        for j in range(hs,len(folds)):
            base=combine_signal_probabilities({s:folds[j]["prob"].get(s,{}) for s in signal_names},candidates,fp.weights)
            hold.append((apply_residual(base,{"prob":folds[j]["prob"]},candidates,chosen),folds[j]["actual"]))
    return {"target_k":target_k,"outer_rows":outer,"holdout_rows":hold,"discoveries":[asdict(x) for x in discoveries],
            "selected_residual":asdict(chosen) if chosen else None,"holdout_isolated":True,"n_discovery_rows":len(rows),"holdout_n":len(hold)}
