"""V5.3 multiple-testing corrections."""
from math import comb
def bonferroni(p_values,alpha=0.05):
    m=len(p_values)
    return {k:min(1.0,float(v)*m) for k,v in p_values.items()}
def benjamini_hochberg(p_values):
    items=sorted(((k,float(v)) for k,v in p_values.items()),key=lambda x:x[1])
    m=len(items);out={};prev=1.0
    for rank,(k,p) in reversed(list(enumerate(items,1))):
        q=min(prev,p*m/rank);out[k]=q;prev=q
    return out
def binomial_tail(k,n,p):
    if n<=0:return 1.0
    return sum(comb(n,i)*(p**i)*((1-p)**(n-i)) for i in range(k,n+1))
