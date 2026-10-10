"""Similarity kernels: weighted cosine + spectral entropy similarity (Li et al. 2021)."""
import numpy as np
from numba import njit, prange

@njit(cache=True, fastmath=True)
def _clean(mz, it, floor, topk, power, ent_weight):
    n=len(mz)
    if n==0: return np.empty(0,np.float32), np.empty(0,np.float32)
    mx=0.0
    for i in range(n):
        if it[i]>mx: mx=it[i]
    if mx<=0: return np.empty(0,np.float32), np.empty(0,np.float32)
    thr=floor*mx; c=0
    for i in range(n):
        if it[i]>=thr: c+=1
    idx=np.empty(c,np.int64); j=0
    for i in range(n):
        if it[i]>=thr: idx[j]=i; j+=1
    if c>topk:
        v=np.empty(c,np.float32)
        for i in range(c): v[i]=it[idx[i]]
        o=np.argsort(v)[c-topk:]
        k2=np.empty(topk,np.int64)
        for i in range(topk): k2[i]=idx[o[i]]
        k2.sort(); idx=k2; c=topk
    om=np.empty(c,np.float32); oi=np.empty(c,np.float32)
    s=0.0
    for i in range(c):
        om[i]=mz[idx[i]]; v=it[idx[i]]**power; oi[i]=v; s+=v
    if s>0:
        for i in range(c): oi[i]/=s
    if ent_weight:
        S=0.0
        for i in range(c):
            if oi[i]>0: S-=oi[i]*np.log(oi[i])
        if S<3.0:
            w=0.25+0.25*S; s2=0.0
            for i in range(c): oi[i]=oi[i]**w; s2+=oi[i]
            if s2>0:
                for i in range(c): oi[i]/=s2
    return om, oi

@njit(cache=True, fastmath=True)
def entropy_sim(qmz,qp,cmz,cp,tol):
    i=0;j=0;n=len(qmz);m=len(cmz)
    SA=0.0
    for x in range(n):
        if qp[x]>0: SA-=qp[x]*np.log(qp[x])
    SB=0.0
    for x in range(m):
        if cp[x]>0: SB-=cp[x]*np.log(cp[x])
    SAB=0.0; tot=0.0
    buf=np.empty(n+m,np.float64); b=0
    while i<n and j<m:
        d=qmz[i]-cmz[j]
        if d<-tol: buf[b]=qp[i]; i+=1; b+=1
        elif d>tol: buf[b]=cp[j]; j+=1; b+=1
        else: buf[b]=qp[i]+cp[j]; i+=1; j+=1; b+=1
    while i<n: buf[b]=qp[i]; i+=1; b+=1
    while j<m: buf[b]=cp[j]; j+=1; b+=1
    for x in range(b): tot+=buf[x]
    if tot<=0: return 0.0
    for x in range(b):
        v=buf[x]/tot
        if v>0: SAB-=v*np.log(v)
    return 1.0-(2.0*SAB-SA-SB)/np.log(4.0)

@njit(cache=True, fastmath=True)
def cos_sim(qmz,qp,cmz,cp,tol):
    i=0;j=0;n=len(qmz);m=len(cmz); dot=0.0; na=0.0; nb=0.0
    for x in range(n): na+=qp[x]*qp[x]
    for x in range(m): nb+=cp[x]*cp[x]
    while i<n and j<m:
        d=qmz[i]-cmz[j]
        if d<-tol: i+=1
        elif d>tol: j+=1
        else: dot+=qp[i]*cp[j]; i+=1; j+=1
    if na<=0 or nb<=0: return 0.0
    return dot/np.sqrt(na*nb)

@njit(cache=True, fastmath=True, parallel=True)
def search(qmz,qp,cand,off,allmz,allin,tol,floor,topk,power,ent_weight,kind):
    out=np.zeros(len(cand),np.float32)
    for k in prange(len(cand)):
        c=cand[k]; a=off[c]; b=off[c+1]
        if b<=a: continue
        cm,cp=_clean(allmz[a:b],allin[a:b],floor,topk,power,ent_weight)
        if len(cm)==0: continue
        out[k]= entropy_sim(qmz,qp,cm,cp,tol) if kind==1 else cos_sim(qmz,qp,cm,cp,tol)
    return out

def prep(mz,it,floor=0.002,topk=256,power=0.5,ent_weight=False):
    return _clean(np.asarray(mz,np.float32),np.asarray(it,np.float32),floor,topk,power,ent_weight)

@njit(cache=True, fastmath=True)
def entropy_sim_shift(qmz,qp,cmz,cp,tol,shift):
    """Best of direct and mass-shifted entropy similarity."""
    a = entropy_sim(qmz,qp,cmz,cp,tol)
    if shift > -0.001 and shift < 0.001: return a
    sm = np.empty(len(cmz), np.float32)
    for i in range(len(cmz)): sm[i]=cmz[i]+shift
    b = entropy_sim(qmz,qp,sm,cp,tol)
    return a if a>b else b

@njit(cache=True, fastmath=True, parallel=True)
def search_shift(qmz,qp,cand,off,allmz,allin,tol,floor,topk,power,ent_weight,kind,shift):
    out=np.zeros(len(cand),np.float32)
    for k in prange(len(cand)):
        c=cand[k]; a=off[c]; b=off[c+1]
        if b<=a: continue
        cm,cp=_clean(allmz[a:b],allin[a:b],floor,topk,power,ent_weight)
        if len(cm)==0: continue
        out[k]=entropy_sim_shift(qmz,qp,cm,cp,tol,shift[k])
    return out
