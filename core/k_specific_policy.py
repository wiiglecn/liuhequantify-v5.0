"""V5.8.2 true K-specific nested Set Prediction policy.

Each target K independently selects signal weights, horizon and structural
lambdas. Candidate complexity is accepted only when it improves the inner
validation tail over the K-specific marginal baseline.
"""
from dataclasses import dataclass,asdict
import math
from .ensemble_weighting import normalize_weights,combine_signal_probabilities
from .set_prediction import rank_set
from .pair_signal import transition_pair_scores
from .signal_metrics import hit_at_k,multiclass_logloss

@dataclass(frozen=True)
class KPolicyConfig:
    windows:tuple=(30,60,120,240)
    pair_lambdas:tuple=(0.0,0.02,0.05,0.10)
    diversity_lambdas:tuple=(0.0,0.02,0.05)
    min_history:int=60
    validation_size:int=30
    shrinkage:float=.35
    min_weight:float=.03
    temperature:float=.20
    complexity_penalty:float=.002
    min_gain:float=.005

@dataclass(frozen=True)
class KSpecificPolicy:
    target_k:int
    horizon:int
    weights:dict
    lambda_pair:float
    lambda_diversity:float
    score:float
    baseline_score:float
    gain:float
    structure_enabled:bool
    history_n:int

def _signal_score(rows,name,k,candidates):
    vals=[r for r in rows if r.get("actual") in candidates and name in r.get("prob",{})]
    if not vals:return 0.0
    hits=sum(hit_at_k(sorted(r["prob"][name],key=r["prob"][name].get,reverse=True),r["actual"],k) for r in vals)/len(vals)
    ll=sum(multiclass_logloss(r["prob"][name],r["actual"]) for r in vals)/len(vals)
    bonus=max(-5.0,min(2.0,-(ll-math.log(max(1,len(candidates))))))
    return hits+.02*bonus

def _weights_from_scores(scores,temp):
    if not scores:return {}
    m=max(scores.values()); e={n:math.exp((v-m)/max(1e-9,temp)) for n,v in scores.items()}
    return normalize_weights(e)

def _blend(local,global_w,shrinkage,min_weight):
    keys=set(local)|set(global_w)
    w={n:(1-shrinkage)*local.get(n,0.0)+shrinkage*global_w.get(n,0.0) for n in keys}
    return normalize_weights({n:max(min_weight,v) for n,v in w.items()})

def _fit_weights(train,names,k,candidates,cfg,regime=None):
    global_scores={n:_signal_score(train,n,k,candidates) for n in names}
    global_w=_weights_from_scores(global_scores,cfg.temperature)
    local=[r for r in train if regime is None or r.get("regime")==regime] or train
    local_scores={n:_signal_score(local,n,k,candidates) for n in names}
    local_w=_weights_from_scores(local_scores,cfg.temperature)
    return _blend(local_w,global_w,cfg.shrinkage,cfg.min_weight)

def _predict(row,names,candidates,weights,lp,ld,pair_scores):
    p=combine_signal_probabilities({n:row.get("prob",{}).get(n,{}) for n in names},candidates,weights)
    return rank_set({"ensemble":p},candidates, row["_target_k"],pair_scores,["ensemble"],lp,ld)

def fit_k_policy(history,signal_names,candidates,target_k,config=None,regime=None):
    cfg=config or KPolicyConfig()
    if len(history)<cfg.min_history: raise ValueError("insufficient history")
    names=list(signal_names); best=None
    for window in [w for w in cfg.windows if w<=len(history)] or [len(history)]:
        data=history[-window:]
        vn=min(cfg.validation_size,max(10,len(data)//5))
        if len(data)<=vn: continue
        train,valid=data[:-vn],data[-vn:]
        weights=_fit_weights(train,names,target_k,candidates,cfg,regime)
        # Baseline is the same K-specific weights with no structural terms.
        baseline_hits=0; model_hits={(lp,ld):0 for lp in cfg.pair_lambdas for ld in cfg.diversity_lambdas}
        for idx,row0 in enumerate(valid):
            row=dict(row0); row["_target_k"]=target_k
            prefix=train+valid[:idx]
            ps=transition_pair_scores([r.get("actual") for r in prefix],candidates)
            p=combine_signal_probabilities({n:row.get("prob",{}).get(n,{}) for n in names},candidates,weights)
            base=rank_set({"ensemble":p},candidates,target_k,{},["ensemble"],0,0)
            baseline_hits+=int(row["actual"] in base)
            for lp in cfg.pair_lambdas:
                for ld in cfg.diversity_lambdas:
                    pred=rank_set({"ensemble":p},candidates,target_k,ps,["ensemble"],lp,ld)
                    model_hits[(lp,ld)]+=int(row["actual"] in pred)
        n=len(valid); base_score=baseline_hits/n
        for (lp,ld),hits in model_hits.items():
            raw=hits/n
            complexity=(abs(lp)>0)+(abs(ld)>0)
            score=raw-cfg.complexity_penalty*complexity
            gain=raw-base_score
            enabled=(gain>=cfg.min_gain and (lp!=0 or ld!=0))
            if not enabled:
                lp2=ld2=0.0; raw2=base_score; score2=base_score
            else:
                lp2,ld2=lp,ld; raw2=raw; score2=score
            key=(score2,raw2,-window,-abs(lp2)-abs(ld2))
            candidate=(key,window,lp2,ld2,raw2,base_score,raw2-base_score,enabled)
            if best is None or key>best[0]: best=candidate
    if best is None: raise ValueError("no valid policy candidate")
    _,window,lp,ld,score,base,gain,enabled=best
    data=history[-window:]
    weights=_fit_weights(data,names,target_k,candidates,cfg,regime)
    return KSpecificPolicy(target_k,window,weights,lp,ld,float(score),float(base),float(gain),bool(enabled),len(history))

def predict_k_policy(fold,candidates,policy,signal_names,pair_scores=None):
    p=combine_signal_probabilities({n:fold.get("prob",{}).get(n,{}) for n in signal_names},candidates,policy.weights)
    return rank_set({"ensemble":p},candidates,policy.target_k,pair_scores or {},["ensemble"],policy.lambda_pair,policy.lambda_diversity)

def evaluate_k_policies(folds,candidates,signal_names,target_ks,config=None,initial_train=60,final_holdout=60):
    cfg=config or KPolicyConfig(); n=len(folds); hs=max(0,n-final_holdout)
    outer={k:[] for k in target_ks}; paths={k:[] for k in target_ks}
    for j in range(max(initial_train,1),hs):
        history=folds[:j]
        for k in target_ks:
            policy=fit_k_policy(history,signal_names,candidates,k,cfg)
            ps=transition_pair_scores([r.get("actual") for r in history],candidates)
            outer[k].append((predict_k_policy(folds[j],candidates,policy,signal_names,ps),folds[j]["actual"]))
            paths[k].append(asdict(policy)|{"fold":j})
    history=folds[:hs]; final={}; hold={k:[] for k in target_ks}
    for k in target_ks:
        policy=fit_k_policy(history,signal_names,candidates,k,cfg)
        ps=transition_pair_scores([r.get("actual") for r in history],candidates)
        final[k]=asdict(policy)
        for j in range(hs,n):
            hold[k].append((predict_k_policy(folds[j],candidates,policy,signal_names,ps),folds[j]["actual"]))
    return {"outer_by_k":outer,"holdout_by_k":hold,"policy_path":paths,"final_policy":final,"holdout_isolated":True}
