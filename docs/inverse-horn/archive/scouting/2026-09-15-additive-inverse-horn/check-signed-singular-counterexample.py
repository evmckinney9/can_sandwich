"""Exact independent check of the supplied restricted-reduction counterexample."""
import itertools
import sympy as s

I = s.I
X = s.Matrix([[0, 1], [1, 0]])
Y = s.Matrix([[0, -I], [I, 0]])
Z = s.diag(1, -1)
paulis = (X, Y, Z)
def H(T):
    return sum((T[i,j]*s.kronecker_product(paulis[i],paulis[j])
                for i in range(3) for j in range(3)), s.zeros(4))
Da = s.diag(3,2,1)
ST = s.Matrix([[0,1,0],[-1,0,0],[0,0,1]])
T = Da+Da*ST
assert ST*ST.T == s.eye(3) and ST.det() == 1
assert T*T.T == s.diag(18,8,4) and T.det() == 24
# Columns: i Psi+, Phi+, i Phi-, Psi-.
Q = s.Matrix([[0,1,I,0],[I,0,0,1],[I,0,0,-1],[0,1,-I,0]])/s.sqrt(2)
assert s.simplify(Q.H*Q) == s.eye(4)
alpha = s.diag(4,2,0,-6)
assert s.simplify(Q.H*H(Da)*Q) == alpha
U = (s.eye(2)-I*Z)/s.sqrt(2)
O = s.simplify(Q.H*s.kronecker_product(s.eye(2),U)*Q)
assert all(s.im(x) == 0 for x in O)
assert s.simplify(O.T*O) == s.eye(4) and s.simplify(O.det()) == 1
B = s.simplify(O*alpha*O.T)
M = s.simplify(Q.H*H(T)*Q)
assert M == alpha+B and M == M.T
gamma = [5*s.sqrt(2)-2, s.sqrt(2)+2, 2-s.sqrt(2), -5*s.sqrt(2)-2]
t = s.symbols('t')
assert s.expand(M.charpoly(t).as_expr()-s.prod(t-x for x in gamma)) == 0
input_traces = sorted({sum(e*a for e,a in zip(signs,[3,2,1]))
                       for signs in itertools.product([-1,1], repeat=3)})
radical_coefficients = sorted({3*p+2*q for p,q in itertools.product([-1,1],repeat=2)})
assert 0 not in radical_coefficients
print('T =', T)
print('O =', O)
print('B =', B)
print('M =', M)
print('charpoly M =', s.factor(M.charpoly(t).as_expr()))
print('input traces =', input_traces)
print('output sqrt(2) coefficients =', radical_coefficients)
print('all exact assertions passed')
