"""V5.4 locked-policy research orchestration.

This adapter makes the intended boundary explicit:
inner rows -> fit calibration/weights -> frozen predictor -> outer rows.
It never consumes outer actuals while fitting the policy.
"""
from dataclasses import dataclass,asdict
from typing import Mapping,Sequence
from .calibration import fit_temperature,apply_temperature
from .ensemble_weighting import fit_logloss_weights,combine_signal_probabilities
from .signal_metrics import metric_summary

@dataclass(frozen=True)
class LockedPolicy:
    signal_names:tuple
    candidates:tuple
    weights:dict
    temperature:float

    def to_dict(self):
        return asdict(self)

def fit_locked_policy(inner_rows:Sequence[tuple[Mapping[str,Mapping],object]],
                      candidates:Sequence,signal_names:Sequence[str])->LockedPolicy:
    names=tuple(signal_names);cands=tuple(candidates)
    weights=fit_logloss_weights(inner_rows,cands,names)
    combined=[]
    for signal_predictions,actual in inner_rows:
        raw=combine_signal_probabilities(signal_predictions,cands,weights)
        combined.append((raw,actual))
    temperature=fit_temperature(combined,cands)
    return LockedPolicy(names,cands,weights,float(temperature))

def predict_locked(policy:LockedPolicy,signal_predictions:Mapping[str,Mapping])->dict:
    raw=combine_signal_probabilities(signal_predictions,policy.candidates,policy.weights)
    return apply_temperature(raw,policy.candidates,policy.temperature)

def evaluate_locked_policy(policy:LockedPolicy,
                           outer_rows:Sequence[tuple[Mapping[str,Mapping],object]])->dict:
    rows=[(predict_locked(policy,preds),actual) for preds,actual in outer_rows]
    result=metric_summary(rows)
    result["policy"]=policy.to_dict()
    result["outer_n"]=len(rows)
    return result
