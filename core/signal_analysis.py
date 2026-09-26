"""V5.4 signal redundancy and incremental-information diagnostics."""
from typing import Mapping,Sequence
from .signal_metrics import pearson

def signal_correlation(signal_rows:Mapping[str,Sequence[float]])->dict:
    names=list(signal_rows);out={a:{} for a in names}
    for i,a in enumerate(names):
        for b in names[i:]:
            r=pearson(signal_rows[a],signal_rows[b]);out[a][b]=r;out.setdefault(b,{})[a]=r
    return out

def redundancy_groups(signal_rows:Mapping[str,Sequence[float]],threshold:float=0.85)->list[list[str]]:
    corr=signal_correlation(signal_rows);names=list(signal_rows);groups=[];used=set()
    for a in names:
        if a in used:continue
        group=[a]
        for b in names:
            if b!=a and b not in used and abs(corr[a].get(b,0.0))>=threshold:group.append(b)
        if len(group)>1:used.update(group);groups.append(group)
    return groups

def incremental_information(base_scores:Sequence[float],candidate_scores:Sequence[float],actual:Sequence[float])->dict:
    n=min(len(base_scores),len(candidate_scores),len(actual))
    if n<3:return {"n":n,"base_corr":0.0,"candidate_corr":0.0,"residual_corr":0.0}
    x,z,y=list(map(float,base_scores[:n])),list(map(float,candidate_scores[:n])),list(map(float,actual[:n]))
    mx,mz=sum(x)/n,sum(z)/n;vx=sum((v-mx)**2 for v in x)
    if vx<=1e-12:residual=z
    else:
        beta=sum((x[i]-mx)*(z[i]-mz) for i in range(n))/vx;alpha=mz-beta*mx
        residual=[z[i]-(alpha+beta*x[i]) for i in range(n)]
    return {"n":n,"base_corr":pearson(x,y),"candidate_corr":pearson(z,y),"residual_corr":pearson(residual,y)}

def effective_signal_count(correlation:Mapping[str,Mapping[str,float]])->float:
    names=list(correlation)
    if not names:return 0.0
    vals=[abs(float(correlation[a].get(b,0.0))) for i,a in enumerate(names) for b in names[i+1:]]
    avg=sum(vals)/len(vals) if vals else 0.0
    return len(names)/max(1.0,1.0+(len(names)-1)*avg)
