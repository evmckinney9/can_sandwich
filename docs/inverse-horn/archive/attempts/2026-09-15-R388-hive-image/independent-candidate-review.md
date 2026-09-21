# Independent check: the proposed endpoint has a real preimage

2026-09-15. Reviewer context reused from the earlier bounded source and
numerical audits. The reviewer did not develop the proposed obstruction.
This check concerns only the frozen candidate with alpha=gamma=(1,0,0,-1),
beta=(1,1/2,-1/2,-1), and all three internal hive coordinates equal to 3/2.

## Verdict

**The candidate is attained exactly by a real symmetric triple.** It cannot
refute surjectivity of the compression-trace prescription. This does not
prove that prescription is onto every hive or that every forward output
is a hive. No numerical nonattainment inference is used.

## Explicit matrices

Set s=sqrt(3), and split R^4 into the first and last two coordinates:

```text
A = A+ ⊕ A−,       C = C+ ⊕ C−,
A+ = diag(1,0),    C+ = [[3/4,s/4],[s/4,1/4]],
A− = diag(-1,0),   C− = diag(0,-1),
B = C-A.
```

Both A+ and C+ are rank-one orthogonal projections. Consequently A and C
have spectrum (1,0,0,-1). Moreover

```text
B+ = C+−A+ = [[-1/4,s/4],[s/4,1/4]],  (B+)² = (1/4)I2,
B− = C−−A− = diag(1,-1).
```

Thus B has exactly the required spectrum (1,1/2,-1/2,-1). All matrices
are real symmetric, and C=A+B holds by definition.

## Compression-trace convention

For Hermitian X,Y define

```text
F(X,Y) = max { u*Xu + v*Yv : ||u||=||v||=1, u*v=0 }.
```

The source's internal coordinate is

```text
h(p,q) = max { Tr(AP)+Tr(CQ) :
               P,Q orthogonal projections, PQ=0, rank P=p, rank Q=q }.
```

Since Tr A=Tr C=0, taking complementary rank-one projections gives

```text
h(1,1) = F(A,C),
h(2,1) = F(B,−A),
h(1,2) = F(−B,−C).
```

For example, in h(2,1), write Q=vv* and the remaining one-dimensional
complement as ww*. Then P=I−vv*−ww*, and the objective becomes
v*(C−A)v−w*Aw. This also verifies the index and sign convention.

## h(1,1)=3/2

Define the common majorant

```text
L = ((A+ + C+ + (1/2)I2)/2) ⊕ 0_2.
```

It is positive semidefinite and has trace 3/2. On the first block,

```text
L−A = (B+ +(1/2)I2)/2,
L−C = (−B+ +(1/2)I2)/2,
```

which are positive semidefinite because B+ has eigenvalues ±1/2. On the
last block, L−A and L−C are also positive semidefinite. Hence for every
complex orthonormal u,v,

```text
u*Au+v*Cv <= u*Lu+v*Lv <= Tr L = 3/2.
```

For equality choose orthonormal u,v in the first block, with u an
eigenvector of A+−C+ of eigenvalue 1/2. Then

```text
u*A+u+v*C+v = Tr C+ + u*(A+−C+)u = 1+1/2.
```

These eigenvectors can be real. Thus this upper bound is attained.

## h(2,1)=3/2

Take

```text
L21 = (1/2)I2 ⊕ diag(1,0).
```

This dominates both B and −A. Its two largest eigenvalues sum to 3/2,
so for every complex orthonormal u,v,

```text
u*Bu+v*(−A)v <= u*L21u+v*L21v <= 3/2.
```

Equality is attained by taking u in the first block as a +1/2 eigenvector
of B+, and v=e3, where −A has eigenvalue 1. Therefore F(B,−A)=3/2.

## h(1,2)=3/2

Similarly,

```text
L12 = (1/2)I2 ⊕ diag(0,1)
```

dominates both −B and −C and has sum of its two largest eigenvalues 3/2.
Equality is attained by choosing u in the first block as a +1/2
eigenvector of −B+, and v=e4, where −C has eigenvalue 1. Thus
F(−B,−C)=3/2.

## Complete candidate coordinates and scope

The three computed entries are all interior entries at n=4. Boundary
entries follow from the listed spectra under the source convention:

```text
h(p,0), p=0..4: 0,1,1,1,0
h(0,q), q=0..4: 0,1,1,1,0
h(4−q,q), q=0..4: 0,1,3/2,1,0
h(1,1)=h(2,1)=h(1,2)=3/2.
```

Accordingly this is a preimage of the exact proposed point, not merely
another hive with the same boundary. All upper bounds apply over complex
vectors; the maximizing choices can be made real. The equality cases used
here include the repeated zero eigenspaces and do not assume genericity.

This is elementary exact block-matrix verification. No optimization,
numerical benchmark, corpus or source-code change was performed. The
compression-trace definition is the already reviewed Bercovici–Li (2024)
Introduction, equations (1)–(2); no additional source theorem is invoked.
An orthogonal independent certificate or author replay can check the
displayed projection and squared-matrix identities directly. No registry
entry was modified by this reviewer.
