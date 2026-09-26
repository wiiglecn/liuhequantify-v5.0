"""V5.8 K-specific nested Set Prediction policy."""
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

@dataclass(frozen=True)
class KSpecificPolicy:
    target_k:int
    horizon:int
    weights:dict
    lambda_pair:float
    lambda_diversity:float
    score:float
    history_n:int

def _score_signals(rows,names,k,candidates):
    out={}
    vals=[r for r in rows if r.get("actual") in candidates]
    for n in names:
        if not vals: out[n]=0.0; continue
        hits=sum(hit_at_k(sorted(r["prob"].get(n,{}),key=r["prob"].get(n).get,reverse=True),r["actual"],k) for r in vals)/len(vals)
        ll=sum(multiclass_logloss(r["prob"].get(n,{}),r["actual"]) for r in vals)/len(vals)
        out[n]=hits+.02*max(-5.0,min(2.0,-(ll-math.log(max(1,len(candidates))))))
    return out

def _softmax(scores,temp):
    if not scores:return {}
    m=max(scores.values()); e={n:math.exp((v-m)/max(1e-9,temp)) for n,v in scores.items()}
    return normalize_weights(e)

def _blend(a,b,shrinkage,min_weight):
    keys=set(a)|set(b)
    w={n:(1-shrinkage)*a.get(n,0.0)+shrinkage*b.get(n,0.0) for n in keys}
    return normalize_weights({n:max(min_weight,v) for n,v in w.items()})

def fit_k_policy(history,signal_names,candidates,target_k,config=None):
    cfg=config or KPolicyConfig()
    if len(history)<cfg.min_history: raise ValueError("insufficient history")
    names=list(signal_names); best=None
    windows=[w for w in cfg.windows if w<=len(history)] or [len(history)]
    for window in windows:
        data=history[-window:]; vn=min(cfg.validation_size,max(10,len(data)//5))
        if len(data)<=vn: continue
        train,valid=data[:-vn],data[-vn:]
        global_w=_softmax(_score_signals(train,names,target_k,candidates),cfg.temperature)
        for lp in cfg.pair_lambdas:
            for ld in cfg.diversity_lambdas:
                weights=_blend(global_w,global_w,cfg.shrinkage,cfg.min_weight)
                score=0.0
                for idx,row in enumerate(valid):
                    probs={n:row["prob"].get(n,{}) for n in names}
                    prefix=train+valid[:idx]
                    ps=transition_pair_scores([r.get("actual") for r in prefix],candidates)
                    p=combine_signal_probabilities(probs,candidates,weights)
                    pred=rank_set({"ensemble":p},candidates,target_k,ps,["ensemble"],lp,ld)
                    score+=1.0 if row["actual"] in pred else 0.0
                score/=len(valid)
                key=(score,-window,-abs(lp),-abs(ld))
                if best is None or key>best[0]: best=(key,window,lp,ld)
    if best is None: raise ValueError("no valid policy candidate")
    _,window,lp,ld=best
    data=history[-window:]
    w=_softmax(_score_signals(data,names,target_k,candidates),cfg.temperature)
    w=_blend(w,w,cfg.shrinkage,cfg.min_weight)
    return KSpecificPolicy(target_k,window,w,lp,ld,float(best[0][0]),len(history))

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
        policy=fit_k_policy(history,signal_names,candidates,k,cfg); ps=transition_pair_scores([r.get("actual") for r in history],candidates)
        final[k]=asdict(policy)
        for j in range(hs,n): hold[k].append((predict_k_policy(folds[j],candidates,policy,signal_names,ps),folds[j]["actual"]))
    return {"outer_by_k":outer,"holdout_by_k":hold,"policy_path":paths,"final_policy":final,"holdout_isolated":True}
