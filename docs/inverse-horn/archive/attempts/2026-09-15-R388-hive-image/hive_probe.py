import itertools
import numpy as np
from scipy.optimize import linprog
N=4
pts=[(p,q) for p in range(N+1) for q in range(N+1-p)]
def dist(a,b):
 p=a[0]-b[0];q=a[1]-b[1];return p*p+p*q+q*q
rh=[]
for y,z in itertools.combinations(pts,2):
 if dist(y,z)!=1:continue
 other=[x for x in pts if dist(x,y)==dist(x,z)==1]
 if len(other)==2:rh.append((y,z,*other))
inside=[(1,1),(1,2),(2,1)]
def problem(a,b,c):
 h={(0,0):0}
 for k in range(1,5):h[k,0]=sum(a[:k]);h[0,k]=sum(c[:k])
 for k in range(5):h[4-k,k]=sum(a)+sum(b[:k])
 mat=[];rhs=[]
 for y,z,x,w in rh:
  row=np.zeros(3);v=0
  for pt,sgn in [(y,1),(z,1),(x,-1),(w,-1)]:
   if pt in inside:row[inside.index(pt)]+=sgn
   else:v+=sgn*h[pt]
  mat.append(-row);rhs.append(v)
 return np.array(mat),np.array(rhs),h
if __name__=='__main__':
 a=[1,0,0,-1]
 for t,s in [(.5,.5),(1,.5),(1.5,.5),(1,1)]:
  m,r,h=problem(a,[t,s,-s,-t],a)
  lp=linprog([1,0,0],A_ub=m,b_ub=r,bounds=[(None,None)]*3,method='highs')
  print(t,s,lp.success,lp.x if lp.success else None)
 print('rhombi',len(rh))
