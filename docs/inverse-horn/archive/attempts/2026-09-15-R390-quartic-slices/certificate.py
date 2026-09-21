"""Symbolic identities supporting the proof; not a quantified solver."""
import json
import sympy as S
x,y,a,b,c,r,s,t,u=S.symbols('x y a b c r s t u', real=True)
f=x**4+a*x*x+b*x+c
assert S.expand(S.discriminant(S.diff(f,x),x)+16*(8*a**3+27*b*b))==0
assert S.expand(f.subs(x,r).subs(b,-4*r**3-2*a*r)-(c-3*r**4-a*r*r))==0
# All repeated critical points: a=-6r²,b=8r³. The bounds force c=-3r⁴.
assert S.expand(f.subs({a:-6*r*r,b:8*r**3,c:-3*r**4})-(x-r)**3*(x+3*r))==0
phi=y*(y*y-1)
assert phi.subs(y,2)==6 and phi.subs(y,3)==24
assert S.Matrix([[6,12],[24,72]]).det()==144
# Check all Hermite leading principal minors in the supplied formulation.
H=S.Matrix([[4,0,-2*a,-3*b],[0,-2*a,-3*b,2*a*a-4*c],
            [-2*a,-3*b,2*a*a-4*c,5*a*b],
            [-3*b,2*a*a-4*c,5*a*b,-2*a**3+6*a*c+3*b*b]])
assert H[:2,:2].det()==-8*a
assert S.expand(H[:3,:3].det()+4*(2*a**3-8*a*c+9*b*b))==0
assert S.expand(H.det()-S.discriminant(f,x))==0
print(json.dumps({'result':'PASS','scope':'Polynomial identities and anchor invertibility; see proof for convexity and compactness'},indent=2))
