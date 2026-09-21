"""Exact replay of the independently supplied lift, not a surjectivity proof."""
import itertools,json
from pathlib import Path
import sympy as S
R=S.Rational; rt=S.sqrt(3)
A=S.diag(1,0,-1,0)
C=S.diag(S.Matrix([[R(3,4),rt/4],[rt/4,R(1,4)]]),S.diag(0,-1))
B=C-A; t=S.Symbol('t')
assert S.expand(A.charpoly(t).as_expr()-t**2*(t-1)*(t+1))==0
assert S.expand(C.charpoly(t).as_expr()-t**2*(t-1)*(t+1))==0
assert S.expand(B.charpoly(t).as_expr()-(t*t-1)*(t*t-R(1,4)))==0
L=S.diag((A[:2,:2]+C[:2,:2]+S.eye(2)/2)/2,S.zeros(2))
L21=S.diag(R(1,2),R(1,2),1,0)
L12=S.diag(R(1,2),R(1,2),0,1)
count=0
for m in (L,L-A,L-C,L21-B,L21+A,L12+B,L12+C):
 for r in range(1,5):
  for inds in itertools.combinations(range(4),r):
   value=S.simplify(m.extract(inds,inds).det())
   assert value.is_nonnegative,(m,inds,value)
   count+=1
assert S.trace(L)==R(3,2)
for m in (L21,L12):assert sorted(m.eigenvals().items(),key=lambda x:x[0])==[(0,1),(R(1,2),2),(1,1)]
pos=S.Matrix([R(1,2),rt/2,0,0]);neg=S.Matrix([-rt/2,R(1,2),0,0])
e3=S.eye(4)[:,2];e4=S.eye(4)[:,3]
for X,Y,u,v in ((A,C,neg,pos),(B,-A,pos,e3),(-B,-C,neg,e4)):
 assert S.simplify((u.T*u)[0])==S.simplify((v.T*v)[0])==1
 assert S.simplify((u.T*v)[0])==0
 assert S.simplify((u.T*X*u+v.T*Y*v)[0])==R(3,2)
a=[1,0,0,-1];b=[1,R(1,2),-R(1,2),-1]
h={(0,0):0}
for k in range(1,5): h[k,0]=sum(a[:k]);h[0,k]=sum(a[:k])
for k in range(5): h[4-k,k]=sum(b[:k])
h.update({(1,1):R(3,2),(1,2):R(3,2),(2,1):R(3,2)})
def d(a,b):
 p=a[0]-b[0];q=a[1]-b[1];return p*p+p*q+q*q
slacks=[]
for y,z in itertools.combinations(h,2):
 if d(y,z)!=1:continue
 ends=[x for x in h if d(x,y)==d(x,z)==1]
 if len(ends)!=2:continue
 slack=h[y]+h[z]-sum(h[x] for x in ends)
 assert slack>=0
 slacks.append(str(slack))
assert len(slacks)==18
out={'status':'EXACT_CANDIDATE_LIFT_VERIFIED','principal_minors_checked':count,'rhombi_checked':18,'rhombus_slacks':slacks,'all_three_internal_coordinates':'3/2','scope':'One exact hive preimage. No global surjectivity theorem or counterexample.'}
Path(__file__).with_name('certificate.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
