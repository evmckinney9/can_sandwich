"""Exact independent checks for the supplied T4 counterexample; no search."""
import itertools
import sympy as s

x = s.symbols('x')
J = s.ones(4)
A = J-s.eye(4)
C = s.diag(3,1,-1,-3)
B = C-A
p = x**4-16*x**2+8*x+16
assert A == A.T and B == B.T and C == C.T
assert A+B == C
assert A.trace() == B.trace() == C.trace() == 0
assert s.factor(A.charpoly(x).as_expr()) == (x-3)*(x+1)**3
assert B.charpoly(x).as_expr() == p
assert s.gcd(p, s.diff(p,x)) == 1
values = {z: p.subs(x,z) for z in (4,2,0,-2)}
assert values == {4:48,2:-16,0:16,-2:-48}
endpoints = [-5,-2,0,2,4]
signs = [s.sign(p.subs(x,z)) for z in endpoints]
assert signs == [1,-1,1,-1,1]
# Every feasible tridiagonal support is a clique in the path on four vertices.
allowed_supports = []
for size in range(5):
    for support in itertools.combinations(range(4), size):
        if all(abs(i-j)<=1 for i,j in itertools.combinations(support,2)):
            allowed_supports.append(support)
            assert len(support)<=2
            for perm in itertools.permutations([3,1,-1,-3]):
                retained = [perm[i]+1 for i in range(4) if i not in support]
                assert len(retained)>=2 and all(p.subs(x,z)!=0 for z in retained)
print('A =', A)
print('B =', B)
print('charpoly(A) =', s.factor(A.charpoly(x).as_expr()))
print('charpoly(B) =', p)
print('p(4), p(2), p(0), p(-2) =', values)
print('root bracket endpoints =', endpoints, 'signs =', signs)
print('allowed supports including empty =', allowed_supports)
print('all exact assertions passed; all 24 permutations covered')
