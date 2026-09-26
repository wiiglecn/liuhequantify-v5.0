"""V5.8 point-in-time pair/transition structure signals."""
from collections import Counter
import math

def transition_pair_scores(actuals, candidates, context=60, smoothing=1.0):
    cands=list(candidates); cset=set(cands)
    seq=[x for x in actuals if x in cset]
    if context and len(seq)>context+1: seq=seq[-(context+1):]
    pair=Counter(); marg=Counter(); total=0
    for a,b in zip(seq,seq[1:]):
        marg[a]+=1; marg[b]+=1
        if a!=b: pair[tuple(sorted((a,b)))]+=1
        total+=1
    if total<=0 or len(cands)<2:
        return {(a,b):0.0 for i,a in enumerate(cands) for b in cands[i+1:]}
    n=len(cands); out={}
    denom=total+smoothing*n
    for i,a in enumerate(cands):
        for b in cands[i+1:]:
            co=pair.get(tuple(sorted((a,b))),0.0)+smoothing
            pa=(marg.get(a,0.0)+smoothing)/denom
            pb=(marg.get(b,0.0)+smoothing)/denom
            pp=co/max(1.0,total+smoothing*n*(n-1)/2)
            pmi=math.log(max(1e-12,pp)/max(1e-12,pa*pb))
            out[(a,b)]=max(-2.0,min(2.0,pmi))
    mean=sum(out.values())/len(out) if out else 0.0
    return {k:v-mean for k,v in out.items()}

def pair_score(pair_scores,a,b):
    if a==b:return 0.0
    return float(pair_scores.get(tuple(sorted((a,b))),0.0))

def build_pair_scores(history,candidates,actual_getter=lambda r:r):
    return transition_pair_scores([actual_getter(r) for r in history],candidates)
