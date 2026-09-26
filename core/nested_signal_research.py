"""V5.4.2 nested signal research engine.

Leakage boundary:
  historical OOS signal folds -> inner selection/correlation/weights/calibration
  -> frozen outer policy -> one outer observation.
Final holdout is evaluated with one policy fit only from pre-holdout OOS folds.
"""
from dataclasses import dataclass, asdict
from typing import Mapping, Sequence, Any
from .signal_metrics import metric_summary, binomial_two_sided_pvalue, bootstrap_metric_ci
from .signal_analysis import pearson, redundancy_groups
from .ensemble_weighting import fit_logloss_weights, combine_signal_probabilities
from .calibration import fit_temperature, apply_temperature

@dataclass(frozen=True)
class NestedResearchConfig:
    initial_train: int = 180
    final_holdout: int = 60
    inner_min_folds: int = 90
    inner_window: int = 180
    max_signals: int = 4
    redundancy_threshold: float = 0.85
    top_k: tuple = (1, 3, 6)

@dataclass(frozen=True)
class FoldPolicy:
    signals: tuple
    weights: dict
    temperature: float
    inner_metrics: dict
    redundancy: list

    def to_dict(self):
        return asdict(self)

def _actual_score_rows(folds, name):
    return [float(f["prob"].get(name, {}).get(f["actual"], 0.0)) for f in folds]

def _select_signals(inner, candidates, names, cfg):
    metrics = {}
    for name in names:
        rows=[(f["prob"].get(name, {}), f["actual"]) for f in inner]
        metrics[name]=metric_summary(rows, top_k=cfg.top_k)
    ordered=sorted(names, key=lambda n:(metrics[n]["logloss"], -metrics[n].get("hit_at_1",0.0), n))
    scalar={n:_actual_score_rows(inner,n) for n in names}
    corr={a:{} for a in names}
    for i,a in enumerate(names):
        for b in names[i:]:
            r=pearson(scalar[a],scalar[b])
            corr[a][b]=r
            corr.setdefault(b,{})[a]=r
    kept=[]
    for name in ordered:
        if len(kept)>=cfg.max_signals: break
        if all(abs(corr[name].get(k,0.0)) < cfg.redundancy_threshold for k in kept):
            kept.append(name)
    if not kept and ordered: kept=[ordered[0]]
    return kept, metrics, corr

def _rows(folds, names):
    return [({n:f["prob"].get(n,{}) for n in names}, f["actual"]) for f in folds]

def fit_policy(inner_folds, candidates, signal_names, cfg=None):
    cfg=cfg or NestedResearchConfig()
    if len(inner_folds)<cfg.inner_min_folds:
        raise ValueError(f"not enough inner OOS folds: {len(inner_folds)} < {cfg.inner_min_folds}")
    inner=list(inner_folds[-cfg.inner_window:]) if cfg.inner_window>0 else list(inner_folds)
    kept, metrics, corr=_select_signals(inner,candidates,signal_names,cfg)
    rows=_rows(inner,kept)
    weights=fit_logloss_weights(rows,candidates,kept)
    raw=[(combine_signal_probabilities(pred,candidates,weights),actual) for pred,actual in rows]
    temperature=fit_temperature(raw,candidates)
    return FoldPolicy(tuple(kept),weights,float(temperature),{k:metrics[k] for k in kept},
                      [[a,b,corr[a].get(b,0.0)] for i,a in enumerate(kept) for b in kept[i+1:]])

def predict_policy(policy, fold, candidates):
    pred={n:fold["prob"].get(n,{}) for n in policy.signals}
    raw=combine_signal_probabilities(pred,candidates,policy.weights)
    return apply_temperature(raw,candidates,policy.temperature)

def evaluate_nested_folds(folds, candidates, signal_names, cfg=None):
    cfg=cfg or NestedResearchConfig()
    candidates=list(candidates); folds=list(folds)
    if len(folds)<=cfg.final_holdout: raise ValueError("not enough folds for final holdout")
    holdout_start=len(folds)-cfg.final_holdout
    outer_start=max(cfg.initial_train, cfg.inner_min_folds)
    outer=[]
    policies=[]
    for j in range(outer_start, holdout_start):
        inner=folds[:j]
        policy=fit_policy(inner,candidates,signal_names,cfg)
        p=predict_policy(policy,folds[j],candidates)
        outer.append((p,folds[j]["actual"]))
        policies.append({"fold":j,**policy.to_dict()})
    metrics=metric_summary(outer,top_k=cfg.top_k)
    hits={k:int(round(metrics.get(f"hit_at_{k}",0.0)*len(outer))) for k in cfg.top_k}
    baselines={k:min(1.0,k/max(1,len(candidates))) for k in cfg.top_k}
    pvalues={k:binomial_two_sided_pvalue(hits[k],len(outer),baselines[k]) for k in cfg.top_k}
    final_inner=folds[:holdout_start]
    final_policy=fit_policy(final_inner,candidates,signal_names,cfg)
    holdout_rows=[]
    for f in folds[holdout_start:]:
        holdout_rows.append((predict_policy(final_policy,f,candidates),f["actual"]))
    holdout_metrics=metric_summary(holdout_rows,top_k=cfg.top_k)
    outer_ci={m:bootstrap_metric_ci(outer,m) for m in ("hit_at_1","hit_at_3","hit_at_6","logloss","brier","ece","information_gain")}
    holdout_ci={m:bootstrap_metric_ci(holdout_rows,m) for m in ("hit_at_1","hit_at_3","hit_at_6","logloss","brier","ece","information_gain")}
    return {
        "outer":metrics,
        "outer_bootstrap_ci":outer_ci,
        "outer_n":len(outer),
        "outer_p_value_vs_baseline":pvalues,
        "holdout_start":holdout_start,
        "final_holdout":holdout_metrics,
        "final_holdout_bootstrap_ci":holdout_ci,
        "final_policy":final_policy.to_dict(),
        "policy_path":policies,\n        "signal_outer_rows":signal_outer,\n        "signal_p_values":signal_p,
        "holdout_isolated":True,
        "selection_rule":"inner OOS logloss + redundancy filter; weights and temperature fit on inner OOS only",
    }
