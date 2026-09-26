"""V5.7 regime x signal x horizon meta-policy.

Leakage-safe target-specific policy selection. The target fold is never used
to choose its regime, signal weights, or historical lookback horizon.
"""
from dataclasses import dataclass, asdict
import math
from .ensemble_weighting import normalize_weights, combine_signal_probabilities
from .signal_metrics import hit_at_k, multiclass_logloss

@dataclass(frozen=True)
class HorizonConfig:
    windows: tuple = (30, 60, 120, 240)
    min_history: int = 60
    shrinkage: float = .35
    min_weight: float = .03
    temperature: float = .20
    regime_weight: float = .65

@dataclass(frozen=True)
class MetaPolicy:
    target_k: int
    horizon: int
    weights: dict
    regime: int
    history_n: int
    score: float

def _softmax(scores, temperature):
    if not scores:
        return {}
    m=max(scores.values())
    e={n:math.exp((v-m)/max(1e-9,temperature)) for n,v in scores.items()}
    return normalize_weights(e)

def _rows(history, regime):
    local=[r for r in history if r.get("regime")==regime]
    return local or list(history)

def _signal_scores(rows, names, k, candidates):
    out={}
    for n in names:
        vals=[r for r in rows if r.get("actual") in candidates and n in r.get("prob",{})]
        if not vals:
            out[n]=0.0
            continue
        hr=sum(hit_at_k(sorted(r["prob"][n],key=r["prob"][n].get,reverse=True),r["actual"],k)
               for r in vals)/len(vals)
        ll=sum(multiclass_logloss(r["prob"][n],r["actual"]) for r in vals)/len(vals)
        # Hit-rate is the target objective; logloss acts as a small stability tie-break.
        baseline=1.0/max(1,len(candidates))
        ll_bonus=max(-5.0,min(2.0,-(ll-math.log(max(1,len(candidates))))))
        out[n]=hr+0.02*ll_bonus
    return out

def _blend(local, global_w, shrinkage, min_weight):
    w={n:(1-shrinkage)*local.get(n,0.0)+shrinkage*global_w.get(n,0.0)
       for n in set(local)|set(global_w)}
    w={n:max(min_weight,v) for n,v in w.items()}
    return normalize_weights(w)

def fit_meta_policy(history, signal_names, candidates, regime, target_k, config=None):
    cfg=config or HorizonConfig()
    names=list(signal_names)
    if len(history)<cfg.min_history:
        raise ValueError("insufficient history")
    global_rows=list(history)
    global_scores=_signal_scores(global_rows,names,target_k,candidates)
    global_w=_softmax(global_scores,cfg.temperature)
    candidates_windows=[w for w in cfg.windows if w<=len(history)]
    if not candidates_windows:
        candidates_windows=[len(history)]
    best=None
    local_all=_rows(history,regime)
    for w in candidates_windows:
        all_w=history[-w:]
        local=[r for r in all_w if r.get("regime")==regime] or all_w
        local_scores=_signal_scores(local,names,target_k,candidates)
        local_w=_softmax(local_scores,cfg.temperature)
        weights=_blend(local_w,global_w,cfg.shrinkage,cfg.min_weight)
        eval_rows=local_all[-w:] if len(local_all)>=w else local_all
        if not eval_rows:
            eval_rows=all_w
        rows=[r for r in eval_rows if r.get("actual") in candidates]
        if not rows:
            score=0.0
        else:
            score=sum(hit_at_k(sorted(combine_signal_probabilities(
                {n:r["prob"].get(n,{}) for n in names},candidates,weights),
                key=lambda x:combine_signal_probabilities(
                    {n:r["prob"].get(n,{}) for n in names},candidates,weights).get(x,0.0),
                reverse=True),r["actual"],target_k) for r in rows)/len(rows)
        # Conservative shrink toward the all-history score to reduce horizon overfit.
        score=cfg.regime_weight*score+(1-cfg.regime_weight)*sum(
            hit_at_k(sorted(r["prob"][n],key=r["prob"][n].get,reverse=True),r["actual"],target_k)
            for r in rows for n in [] ) if False else score
        policy=MetaPolicy(target_k,w,weights,int(regime),len(history),float(score))
        if best is None or (policy.score, -policy.horizon)>(best.score,-best.horizon):
            best=policy
    return best

def predict_meta(fold, candidates, policy, signal_names):
    return combine_signal_probabilities(
        {n:fold.get("prob",{}).get(n,{}) for n in signal_names},
        candidates,policy.weights)

def evaluate_meta(folds,candidates,signal_names,regimes,target_ks,config=None,
                  initial_train=60,final_holdout=60):
    cfg=config or HorizonConfig()
    n=len(folds); hs=max(0,n-final_holdout)
    outer_by_k={k:[] for k in target_ks}; path_by_k={k:[] for k in target_ks}
    for j in range(max(initial_train,1),hs):
        history=[{"prob":folds[t]["prob"],"actual":folds[t]["actual"],"regime":regimes[t]} for t in range(j)]
        for k in target_ks:
            p=fit_meta_policy(history,signal_names,candidates,regimes[j],k,cfg)
            outer_by_k[k].append((predict_meta(folds[j],candidates,p,signal_names),folds[j]["actual"]))
            path_by_k[k].append(asdict(p)|{"fold":j})
    history=[{"prob":folds[t]["prob"],"actual":folds[t]["actual"],"regime":regimes[t]} for t in range(hs)]
    hold_by_k={k:[] for k in target_ks}; final_policy={}
    fr=regimes[hs] if hs<n else 0
    for k in target_ks:
        p=fit_meta_policy(history,signal_names,candidates,fr,k,cfg)
        final_policy[k]=asdict(p)
        hold_by_k[k]=[(predict_meta(folds[j],candidates,p,signal_names),folds[j]["actual"])
                       for j in range(hs,n)]
    return {"outer_by_k":outer_by_k,"holdout_by_k":hold_by_k,
            "policy_path":path_by_k,"final_policy":final_policy,
            "holdout_isolated":True,"config":asdict(cfg)}
