"""V5.6 leakage-safe regime-conditioned adaptive ensemble."""
from dataclasses import dataclass,asdict
import math
from .ensemble_weighting import normalize_weights,combine_signal_probabilities
from .signal_metrics import multiclass_logloss

@dataclass(frozen=True)
class AdaptiveConfig:
    decay:float=.97
    shrinkage:float=.35
    min_weight:float=.03
    temperature:float=1.0

def _weights(losses,temp):
    if not losses:return {}
    m=min(losses.values()); e={n:math.exp(-(v-m)/max(1e-9,temp)) for n,v in losses.items()}
    return normalize_weights(e)

def fit_adaptive_weights(history,signal_names,candidates,regime=None,config=None):
    cfg=config or AdaptiveConfig(); names=list(signal_names)
    local=[r for r in history if regime is None or r.get("regime")==regime] or list(history)
    def losses(rows):
        out={}
        for n in names:
            z=[multiclass_logloss([(r["prob"].get(n,{}),r["actual"])],candidates) for r in rows if r.get("actual") in candidates]
            out[n]=sum(z)/len(z) if z else 0.0
        return out
    lw=_weights(losses(local),cfg.temperature); gw=_weights(losses(history),cfg.temperature)
    w={n:(1-cfg.shrinkage)*lw.get(n,0)+cfg.shrinkage*gw.get(n,0) for n in names}
    w={n:max(cfg.min_weight,w.get(n,0)) for n in names}
    return normalize_weights(w)

def predict_adaptive(fold,candidates,weights,signal_names):
    return combine_signal_probabilities({n:fold.get("prob",{}).get(n,{}) for n in signal_names},candidates,weights)

def evaluate_adaptive(folds,candidates,signal_names,regimes,config=None,initial_train=60,final_holdout=60):
    cfg=config or AdaptiveConfig(); n=len(folds); hs=max(0,n-final_holdout); outer=[]; path=[]
    for j in range(max(initial_train,1),hs):
        hist=[{"prob":folds[t]["prob"],"actual":folds[t]["actual"],"regime":regimes[t]} for t in range(j)]
        w=fit_adaptive_weights(hist,signal_names,candidates,regimes[j],cfg)
        outer.append((predict_adaptive(folds[j],candidates,w,signal_names),folds[j]["actual"]))
        path.append({"fold":j,"regime":int(regimes[j]),"weights":w})
    hist=[{"prob":folds[t]["prob"],"actual":folds[t]["actual"],"regime":regimes[t]} for t in range(hs)]
    fr=regimes[hs] if hs<n else 0; fw=fit_adaptive_weights(hist,signal_names,candidates,fr,cfg)
    hold=[(predict_adaptive(folds[j],candidates,fw,signal_names),folds[j]["actual"]) for j in range(hs,n)]
    return {"outer_rows":outer,"holdout_rows":hold,"policy_path":path,"final_weights":fw,"holdout_isolated":True,"config":asdict(cfg)}
