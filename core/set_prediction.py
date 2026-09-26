"""V5.8 direct Set Prediction engine."""
import math
from .pair_signal import pair_score

def _feature(probabilities,candidate,names):
    return [float(probabilities.get(n,{}).get(candidate,0.0)) for n in names]

def set_objective(selected,candidate,probabilities,pair_scores,signal_names,
                  lambda_pair=0.0,lambda_diversity=0.0):
    p=sum(float(probabilities.get(n,{}).get(candidate,0.0)) for n in signal_names)/max(1,len(signal_names))
    pair_term=(lambda_pair*sum(pair_score(pair_scores,candidate,x) for x in selected)/len(selected)) if selected else 0.0
    diversity=0.0
    if selected and lambda_diversity:
        x=_feature(probabilities,candidate,signal_names)
        for y in selected:
            yy=_feature(probabilities,y,signal_names)
            diversity+=math.sqrt(sum((a-b)*(a-b) for a,b in zip(x,yy)))
        diversity=lambda_diversity*diversity/len(selected)
    return p+pair_term+diversity

def greedy_set(probabilities,candidates,k,pair_scores=None,signal_names=None,
               lambda_pair=0.0,lambda_diversity=0.0):
    cands=list(candidates); names=list(signal_names or probabilities.keys()); pair_scores=pair_scores or {}
    selected=[]; remaining=set(cands)
    while remaining and len(selected)<min(k,len(cands)):
        def key(x):
            score=set_objective(selected,x,probabilities,pair_scores,names,lambda_pair,lambda_diversity)
            return (score,-x) if isinstance(x,int) else (score,str(x))
        best=max(remaining,key=key); selected.append(best); remaining.remove(best)
    return selected

def rank_set(probabilities,candidates,k,pair_scores=None,signal_names=None,
             lambda_pair=0.0,lambda_diversity=0.0):
    return greedy_set(probabilities,candidates,k,pair_scores,signal_names,lambda_pair,lambda_diversity)
