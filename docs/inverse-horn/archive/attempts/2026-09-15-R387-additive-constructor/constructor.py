"""Spectra-only numerical additive Horn witness. No universal seed guarantee."""
import itertools
import numpy as np

TOL=1e-8

def check(a,b,c,q):
    if q is None: return float('inf')
    got=np.linalg.eigvalsh(np.diag(a)+(q*b)@q.T)[::-1]
    return float(np.max(np.abs(got-c)))

def orient(q):
    q=q.copy()
    if np.linalg.det(q)<0: q[:,0]*=-1
    return q

def elementary(a,b,c):
    n=len(a)
    if n==1: return np.eye(1)
    if n!=2: return None
    x=(a[0]-a[1])/2; y=(b[0]-b[1])/2; z=(c[0]-c[1])/2
    if x==0 or y==0: return np.eye(2)
    t=(z*z-x*x-y*y)/(2*x*y)
    if t < -1-1e-12 or t > 1+1e-12: return None
    theta=.5*np.arccos(np.clip(t,-1,1))
    u,v=np.cos(theta),np.sin(theta)
    return np.array([[u,-v],[v,u]])

def scaling(a,b,c,limit=6000,seed=387):
    n=len(a); scale=12*max(1.,np.max(np.abs(np.r_[a,b,c])))
    weights=[a/scale+1/3,b/scale+1/3,(-c)[::-1]/scale+1/3]
    rng=np.random.default_rng(seed)
    frames=[np.linalg.qr(rng.normal(size=(n,n)))[0] for _ in range(3)]
    last=float('inf')
    for step in range(limit+1):
        q=orient(frames[0].T@frames[1])
        last=check(a,b,c,q)
        if last<=TOL*.5:
            return q,{'method':'scaling','steps':step,'error':last}
        if step==limit: break
        total=sum((u*w)@u.T for u,w in zip(frames,weights))
        chol=np.linalg.cholesky(total)
        frames=[np.linalg.qr(np.linalg.solve(chol,u))[0] for u in frames]
    return None,{'method':'scaling','steps':limit,'error':last,'status':'UNKNOWN'}

def solve(a,b,c,blocks=True,limit=6000):
    a,b,c=(np.sort(np.asarray(x,dtype=float))[::-1] for x in (a,b,c))
    n=len(a)
    if abs(sum(a)+sum(b)-sum(c))>1e-10:
        return None,{'method':'trace','status':'UNKNOWN'}
    q=elementary(a,b,c)
    if q is not None and check(a,b,c,q)<=TOL*.5:
        return orient(q),{'method':'elementary','steps':0,'error':check(a,b,c,q)}
    candidates=0; child_steps=0
    if blocks and n==4:
        inds=set(range(n)); threshold=1e-12*max(1.,np.max(np.abs(np.r_[a,b,c])))
        for r in (1,2):
            for I,J,K in itertools.product(itertools.combinations(range(n),r),repeat=3):
                if abs(sum(a[list(I)])+sum(b[list(J)])-sum(c[list(K)]))>threshold: continue
                candidates+=1
                q=np.zeros((n,n)); good=True
                for rows,cols,target in ((I,J,K),(sorted(inds-set(I)),sorted(inds-set(J)),sorted(inds-set(K)))):
                    rows,cols,target=map(list,(rows,cols,target))
                    v,info=solve(a[rows],b[cols],c[target],blocks=False,limit=1000)
                    child_steps+=info.get('steps',0)
                    if v is None: good=False; break
                    q[np.ix_(rows,cols)]=v
                if good and check(a,b,c,q)<=TOL*.5:
                    return orient(q),{'method':'block','steps':child_steps,'candidates':candidates,'partition':[list(I),list(J),list(K)],'error':check(a,b,c,q)}
                if candidates>=8: break
            if candidates>=8: break
    q,info=scaling(a,b,c,limit=limit)
    info['block_candidates']=candidates; info['child_steps']=child_steps
    return q,info
