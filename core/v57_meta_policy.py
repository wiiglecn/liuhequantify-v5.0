"""V5.7 regime x signal x horizon meta-policy.

Leakage-safe target-specific policy selection. The target fold is never used
to choose its regime, signal weights, or historical lookback horizon.
"""
from dataclasses import dataclass, asdict
import math
from .ensemble_weighting import normalize_weights, combine_signal_probabilities
from .signal_metrics import hit_at_k, multiclass_logloss
from .regime_detection import RegimeDetector, RegimeConfig

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
    candidate_windows=[w for w in cfg.windows if w<=len(history)]
    if not candidate_windows:
        candidate_windows=[len(history)]
    best=None
    for w in candidate_windows:
        window=history[-w:]
        val_n=max(10,min(30,len(window)//5))
        if len(window)<=val_n:
            val_n=max(1,len(window)//3)
        train=window[:-val_n] if len(window)>val_n else window
        valid=window[-val_n:]
        global_scores_train=_signal_scores(train,names,target_k,candidates)
        global_w_train=_softmax(global_scores_train,cfg.temperature)
        local_train=[r for r in train if r.get("regime")==regime] or train
        local_scores=_signal_scores(local_train,names,target_k,candidates)
        local_w=_softmax(local_scores,cfg.temperature)
        weights=_blend(local_w,global_w_train,cfg.shrinkage,cfg.min_weight)
        rows=[r for r in valid if r.get("actual") in candidates]
        if rows:
            score=0.0
            for r in rows:
                p=combine_signal_probabilities(
                    {n:r["prob"].get(n,{}) for n in names},candidates,weights)
                score+=hit_at_k(sorted(p,key=p.get,reverse=True),r["actual"],target_k)
            score/=len(rows)
        else:
            score=0.0
        # Refit the selected horizon on all pre-target observations only.
        full_local=[r for r in window if r.get("regime")==regime] or window
        full_scores=_signal_scores(full_local,names,target_k,candidates)
        full_local_w=_softmax(full_scores,cfg.temperature)
        full_global_w=_softmax(_signal_scores(window,names,target_k,candidates),cfg.temperature)
        final_weights=_blend(full_local_w,full_global_w,cfg.shrinkage,cfg.min_weight)
        policy=MetaPolicy(target_k,w,final_weights,int(regime),len(history),float(score))
        if best is None or (policy.score,-policy.horizon)>(best.score,-best.horizon):
            best=policy
    return best

def predict_meta(fold, candidates, policy, signal_names):
    return combine_signal_probabilities(
        {n:fold.get("prob",{}).get(n,{}) for n in signal_names},
        candidates,policy.weights)

def _label_history_with_detector(folds, detector, signal_names, candidates):
    return [
        {"prob":f["prob"],"actual":f["actual"],
         "regime":detector.assign(f,signal_names,candidates)}
        for f in folds
    ]

def evaluate_meta(folds,candidates,signal_names,regimes=None,target_ks=(),
                  config=None,initial_train=60,final_holdout=60,
                  regime_config=None):
    cfg=config or HorizonConfig()
    rc=regime_config or RegimeConfig(min_history=initial_train)
    n=len(folds); hs=max(0,n-final_holdout)
    outer_by_k={k:[] for k in target_ks}; path_by_k={k:[] for k in target_ks}
    regime_path=[]
    for j in range(max(initial_train,1),hs):
        h=folds[max(0,j-rc.context_window*4):j] if rc.context_window else folds[:j]
        detector=RegimeDetector(rc).fit(h,signal_names,candidates)
        history=_label_history_with_detector(folds[:j],detector,signal_names,candidates)
        target_regime=detector.assign(folds[j],signal_names,candidates)
        regime_path.append({"fold":j,"regime":int(target_regime),
                            "history_regimes":[r["regime"] for r in history],
                            "detector":detector.to_dict()})
        for k in target_ks:
            p=fit_meta_policy(history,signal_names,candidates,target_regime,k,cfg)
            outer_by_k[k].append((predict_meta(folds[j],candidates,p,signal_names),folds[j]["actual"]))
            path_by_k[k].append(asdict(p)|{"fold":j})
    pre=folds[:hs]
    h=pre[max(0,len(pre)-rc.context_window*4):] if rc.context_window else pre
    detector=RegimeDetector(rc).fit(h,signal_names,candidates)
    history=_label_history_with_detector(pre,detector,signal_names,candidates)
    fr=detector.assign(folds[hs],signal_names,candidates) if hs<n else 0
    hold_by_k={k:[] for k in target_ks}; final_policy={}
    for k in target_ks:
        p=fit_meta_policy(history,signal_names,candidates,fr,k,cfg)
        final_policy[k]=asdict(p)
        hold_by_k[k]=[(predict_meta(folds[j],candidates,p,signal_names),folds[j]["actual"])
                       for j in range(hs,n)]
    return {"outer_by_k":outer_by_k,"holdout_by_k":hold_by_k,
            "policy_path":path_by_k,"final_policy":final_policy,
            "regime_path":regime_path,
            "final_detector":detector.to_dict(),
            "final_history_regimes":[r["regime"] for r in history],
            "final_holdout_regime":int(fr),
            "holdout_isolated":True,"config":asdict(cfg)}
