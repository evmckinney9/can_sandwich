"""Exact finite checks for the T4 counterexample; see proof for all witnesses."""
import itertools
import json
import sympy as s
x=s.Symbol('x')
A=s.ones(4)-s.eye(4)
C=s.diag(3,1,-1,-3)
B=C-A
p=x**4-16*x**2+8*x+16
assert A.trace()==B.trace()==C.trace()==0
assert s.expand(A.charpoly(x).as_expr()-(x-3)*(x+1)**3)==0
assert s.expand(B.charpoly(x).as_expr()-p)==0
assert B==B.T
values={str(z):int(p.subs(x,z)) for z in (4,2,0,-2)}
assert values=={'4':48,'2':-16,'0':16,'-2':-48}
assert s.gcd(p,s.diff(p,x))==1
assert s.Poly(p,x).count_roots(-s.oo,-2)==1
assert all(s.Poly(p,x).count_roots(lo,hi)==1 for lo,hi in ((-2,0),(0,2),(2,4)))
# A shifted rank-one tridiagonal matrix can only have support on a clique
# of the four-vertex path. Enumerate the supports, not numerical vectors.
supports=[]
for size in range(1,5):
    for support in itertools.combinations(range(4),size):
        if all(abs(i-j)<=1 for i,j in itertools.combinations(support,2)):
            supports.append(support)
assert len(supports)==7 and max(map(len,supports))==2
checks=0
for perm in itertools.permutations((3,1,-1,-3)):
    for support in supports:
        retained=[perm[i]+1 for i in range(4) if i not in support]
        assert len(retained)>=2
        assert all(p.subs(x,z)!=0 for z in retained)
        checks+=1
print(json.dumps({'result':'PASS','A':str(A),'B':str(B),'C':str(C),
                  'beta_polynomial':str(p),'excluded_eigenvalue_evaluations':values,
                  'permutation_support_checks':checks,
                  'scope':'Universal T4 counterexample, all 24 permutations'},indent=2))
