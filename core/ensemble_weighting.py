"""V5.4 leakage-safe ensemble weight learning."""
from typing import Mapping,Sequence
from .evaluation_engine import normalize_probabilities
from .signal_metrics import multiclass_logloss

def normalize_weights(weights:Mapping[str,float])->dict:
    vals={k:max(0.0,float(v)) for k,v in weights.items()};s=sum(vals.values())
    if s<=1e-12:
        u=1.0/len(vals) if vals else 0.0;return {k:u for k in vals}
    return {k:v/s for k,v in vals.items()}

def combine_signal_probabilities(signal_predictions:Mapping[str,Mapping],candidates:Sequence,weights:Mapping[str,float])->dict:
    w=normalize_weights(weights);out={c:0.0 for c in candidates}
    for name,pred in signal_predictions.items():
        for c in candidates:out[c]+=w.get(name,0.0)*float(pred.get(c,0.0))
    return normalize_probabilities(out,candidates)

def fit_logloss_weights(rows:Sequence[tuple[Mapping[str,Mapping],object]],candidates:Sequence,
                        signal_names:Sequence[str],steps:int=120,lr:float=0.08)->dict:
    """Projected-gradient simplex optimizer with an analytic logloss gradient.
    This keeps V5.4.2 nested runs tractable on thousands of OOS folds.
    """
    names=list(signal_names); cands=list(candidates)
    if not names:return {}
    weights={n:1.0/len(names) for n in names}
    for _ in range(max(1,steps)):
        grad={n:0.0 for n in names}
        for signal_predictions,actual in rows:
            raw=combine_signal_probabilities(signal_predictions,cands,weights)
            pa=max(1e-12,float(raw.get(actual,0.0)))
            for n in names:
                grad[n] += -float(signal_predictions.get(n,{}).get(actual,0.0))/pa
        scale=1.0/max(1,len(rows))
        for n in names: weights[n]=max(0.0,weights[n]-lr*grad[n]*scale)
        weights=normalize_weights(weights)
    return weights

def ir_weights(ir_table:Mapping[str,Mapping[str,float]],signal_names:Sequence[str])->dict:
    return normalize_weights({n:max(0.0,float(ir_table.get(n,{}).get("ir",0.0))) for n in signal_names})
