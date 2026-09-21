import json
from pathlib import Path
import numpy as np
p=Path(__file__).parent
rng=np.random.default_rng(387)
cases=[]; plants=[]
def add(name,a,b,q):
    a,b=(np.sort(np.asarray(x,dtype=float))[::-1] for x in (a,b))
    a-=a.mean(); b-=b.mean()
    c=np.linalg.eigvalsh(np.diag(a)+(q*b)@q.T)[::-1]
    cases.append(dict(name=name,alpha=a.tolist(),beta=b.tolist(),gamma=c.tolist()))
    plants.append(dict(name=name,Q=q.tolist()))
for i in range(4):
    a=rng.normal(size=4); b=rng.normal(size=4)
    q=np.linalg.qr(rng.normal(size=(4,4)))[0]
    add('generic'+str(i),a,b,q)
for i,(a,b) in enumerate([([2,2,-1,-3],[3,1,-2,-2]),([1,1,-1,-1],[2,2,-2,-2])]):
    add('repeated'+str(i),a,b,np.linalg.qr(rng.normal(size=(4,4)))[0])
def rotation(t): return np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
a=np.array([3.,1.,-1.,-3.]); b=np.array([4.,2.,-2.,-4.])
q=np.zeros((4,4)); q[:2,:2]=rotation(.41);q[2:,2:]=rotation(.73)
add('wall22',a,b,q)
q13=np.eye(4);q13[1:,1:]=np.linalg.qr(rng.normal(size=(3,3)))[0]
add('wall13',a,b,q13)
add('scalar',np.zeros(4),b,np.eye(4))
for t in (.1,.01,.001):
    m=np.eye(4);m[np.ix_([1,2],[1,2])]=rotation(t)
    add('nearwall'+str(t),a,b,m@q)
(p/'inputs.json').write_text(json.dumps(cases,indent=2)+'\n')
(p/'plants.json').write_text(json.dumps(plants,indent=2)+'\n')
print('Frozen',len(cases),'spectra-only cases; plants stored separately.')
