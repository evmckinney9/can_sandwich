"""Exact checks for the direct-sign Hermitian shortcut counterexample."""
import itertools
import json
import sympy as s

r = s.sqrt(2)
I = s.I
t = s.Symbol('t')
D = s.diag(3, 2, 1)
St = s.Matrix([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
T = D + D * St
assert St * St.T == s.eye(3) and St.det() == 1
assert T * T.T == s.diag(18, 8, 4)
assert T.det() == 24

# All signed Hermitian input traces are integers. The irrational
# coefficient of every signed output trace is one of -5,-1,1,5.
input_traces = {sum(e*x for e,x in zip(es, (3,2,1)))
                for es in itertools.product((-1,1), repeat=3)}
output_traces = [e1*3*r+e2*2*r+e3*2
                 for e1,e2,e3 in itertools.product((-1,1), repeat=3)]
checked = 0
for ia, ib, oc in itertools.product(input_traces, input_traces, output_traces):
    assert s.simplify(ia + ib - oc) != 0
    checked += 1
assert all(3*e1+2*e2 != 0 for e1,e2 in itertools.product((-1,1),repeat=2))

# Direct verification of the magic-basis lift, without relying on
# the asserted general exceptional-isomorphism correspondence.
X = s.Matrix([[0,1],[1,0]])
Y = s.Matrix([[0,-I],[I,0]])
Z = s.diag(1,-1)
kron = s.kronecker_product
E = s.Matrix([[1,I,0,0],[0,0,I,1],[0,0,I,-1],[1,-I,0,0]])/r
V = s.diag((1-I)/r, (1+I)/r)
Q = s.simplify(E.conjugate().T*kron(s.eye(2),V)*E)
HA = 3*kron(X,X)+2*kron(Y,Y)+kron(Z,Z)
HB = 3*kron(X,Y)-2*kron(Y,X)+kron(Z,Z)
A = s.simplify(E.conjugate().T*HA*E)
B = s.simplify(E.conjugate().T*HB*E)
assert E.conjugate().T*E == s.eye(4)
assert A == s.diag(2,0,4,-6)
assert B == B.T and all(x.is_real for x in B)
assert Q*Q.T == s.eye(4) and Q.det() == 1
assert s.simplify(Q*A*Q.T-B) == s.zeros(4)
gamma = (5*r-2, r+2, 2-r, -5*r-2)
assert s.expand((A+B).charpoly(t).as_expr()-s.prod(t-x for x in gamma)) == 0
assert all(s.simplify(gamma[i]-gamma[i+1]).is_positive for i in range(3))
assert s.discriminant((A+B).charpoly(t).as_expr(),t) != 0
print(json.dumps({"result":"PASS", "signed_trace_checks":checked,
                  "T":str(T), "A":str(A), "B":str(B), "Q":str(Q),
                  "gamma":[str(x) for x in gamma],
                  "scope":"Direct sign/permutation Hermitian conversion only"},indent=2))
