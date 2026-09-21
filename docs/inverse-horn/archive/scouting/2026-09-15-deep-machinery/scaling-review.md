# Franks scaling: independent source/applicability review

2026-09-15. Bounded source audit; no new mathematical claim, computation,
production change, or claim promotion. Read PREFLIGHT, PROBLEM and
RESEARCH_CONTRACT and the registered R0241–R0243 statements. Author/evaluator
separation: this reviewer did not develop the proposed machinery route.

## Verdict

**A spectra-only approximate real additive constructor is already published.**
Calling its real preservation an unresolved Hermitian-to-real conversion would
understate the thesis. The constructive operation is alternating a common
Cholesky normalization with three separate orthogonalizations. It applies to
additive Horn-feasible spectra, including repeated spectra and feasible walls.
It does not by itself establish a deterministic, globally covering atomic
unitary-product constructor. This is an applicability distinction, not a
refutation of the route.

## Primary source statements checked

[Franks thesis, Chapter 4](https://sites.math.rutgers.edu/~wcf17/images/ThesisPhD_website.pdf):
Theorem 4.2 explicitly returns real symmetric matrices with prescribed positive
spectra and sum residual at most epsilon, almost surely. Algorithm 1 uses real
initial frames, common lower-triangular whitening, then separate upper-triangular
orthogonalization. Remark 4.3 shifts/rescales general additive data. Theorem 4.78
and Algorithm 5 give discrete random initialization and
O(epsilon^-2 m^2 log m) scaling steps under (4.18); Remark 4.77 only sketches
finite-precision analysis. Theorem 4.2 permits positive real, not just rational,
spectra. No simple-spectrum or strict Horn-interior premise appears.

[Franks 1801.01412v2](https://arxiv.org/html/1801.01412v2): Algorithm 1 and
Proposition 4 supply the normalized additive procedure, stated cost
O(b m^2 / epsilon^2), failure probability at most 1/3. Its input consists of
three nonincreasing rational lists in (1/4,1] totaling m; output has exact listed
spectra in the ideal arithmetic description and Frobenius sum error.
Proposition 72 supplies the operator-scaling reduction; Theorem 21 gives
polynomial finite-precision complexity in inverse tolerance, inverse smallest
marginals, and input bit size. Proposition 56 proves an iteration bound;
Remark 57 sketches rounding. These are approximation guarantees, not exact
finite matrix equality or logarithmic dependence on inverse tolerance.

## Rank-four operation to implement and independently check

The following is a transcription/adapter recipe, not a newly verified solver.
Its input is **additive** spectra h,k,l with trace(h)+trace(k)=trace(l), a
feasibility premise, and desired additive residual eta. It needs no input frame.

```text
b = 1 + max(0, -min(h), -min(k))
a = 1 + max(1, 2*b + max(l))
lambda1 = sort_desc((h+b)/a)
lambda2 = sort_desc((k+b)/a)
lambda3 = sort_desc((-l+a-2*b)/a)
tau = eta/a
choose nonsingular real U1,U2,U3 using the audited source initialization
repeat, subject to the selected source's proved iteration budget:
    Hi = Ui diag(lambdai) Ui^T
    L = chol(H1+H2+H3)                 # lower triangular
    Ui = solve(L,Ui), for each i
    Ui = orthogonal_Q_in_QR(Ui), for each i
    Hi = Ui diag(lambdai) Ui^T
    if certified_norm(H1+H2+H3-I4) < tau: return
return UNKNOWN if initialization, arithmetic, or budget fails

A = a*H1-b*I4
B = a*H2-b*I4
D = (a-2*b)*I4-a*H3
```

This normalization is positive and has total trace four on trace-compatible
data. The intended output has spectra h,k,l and residual
`||A+B-D|| <= eta`. Sorting the negated target list matters. The reconstruction
sign above follows by undoing the normalization; it must be independently
checked in an implementation. An acceptance test occurs only after a full
orthogonalization sweep. Do not combine the displayed sum-I4 normalization
with Algorithm 5's sum-I4/4 constants or budget without rescaling both tolerance
and marginals. The small fixed dimension makes each sweep elementary, but
does not remove tolerance or precision cost.

## Source transcription cautions

The following are literal source inconsistencies observed in the checked
versions, not grounds for rejecting their underlying theorems:

* Paper Algorithm 1's stopping expression omits the identity subtraction.
* Paper Algorithm 2 samples `[6*2^b]`; Proposition 56's proof uses
  `[6*2^(2b)]` with its polynomial degree estimate.
* Thesis Remark 4.3 prints the recovered third matrix with scalar `2b-a`;
  inversion of its displayed normalization needs `a-2b`.
* Thesis Algorithm 5 whitens toward `I/m` but its while-condition prints `I`.
  The theorem refers to “Algorithm 4.79”, while the displayed algorithm is 5
  and 4.79 labels a remark.

The theorem and a copied code listing must therefore be audited separately.
This review does not certify a specific finite-precision implementation or
provide an explicit constant hidden in the step bound.

## Can R0241–R0243 compose?

The registry accepts R0241 real SO(4) transport locally, and R0242 extends it
to supplied real symmetric trace-zero pairs with
`||A||op+||B||op < pi`, including repeated spectra. Original diagonal-factor
recovery is part of the accepted scope. Its flow maps an additive pair to a
symmetric unitary sandwich with eigenvalues `exp(i*spec(A+B))`. This is the
appropriate real target geometry; arbitrary unitary conjugation is insufficient.

R0243 accepts an endpoint packet only when its independent ball-arithmetic
checker proves a full-matrix bound. Its rational Cayley frames define exact
SO(4) matrices. The generator remains untrusted. The registry explicitly does
not accept universal finite-precision completion or certified trajectory error.

Consequently, for an input-only choice of phase lists h,k,l lying in this
domain and known additive-feasible, scaling can supply approximate additive
data for a real transport prototype. It does not furnish exact `spec(A+B)=l`
at a finite stopping time. The residual against D must be carried through to
the original target in the final certificate. R0243's existing check against
`exp(i(A+B))` alone does not perform that target comparison. This is a concrete
adapter/checker task; lack of that task is not evidence that composition fails.

## Exact first unproved input-only connection

For **the full atomic domain**, the first missing premise is a constructive,
covering choice, from C,G,T alone, of phase lifts (and any required branches or
pieces) whose exponentials represent the original factors and target, whose
trace-zero additive spectra are Horn-feasible, and whose factor norm sum is
strictly below pi. R0242 expressly does not claim all physical gate pairs can
be put there. Neither Franks source supplies this multiplicative-to-additive
input map or accounts for all quantum-degree Horn conditions.

For the **already selected strict-norm additive branch**, real preservation is
not the first gap. The first remaining implementation obligations are a total
input-only initialization/precision policy and certified propagation of the
nonzero additive residual to the actual target. A fixed random seed is not a
deterministic success proof. The source's bounded failure probability is not
the repository's finite deterministic witness guarantee. A finite exhaustive
initialization policy might be possible, but its coverage/cost have not been
checked here and are not asserted.

Useful next bounded operation: implement the displayed real sweeps on
additive-feasible spectral inputs, independently certify exact-frame or
interval-orthogonality representation plus additive residual, and only then
connect that certificate to R0243's original-factor endpoint check. Treat
global phase/QLR coverage as a separately named obligation. No newly proved
frontier reduction or global solver is claimed by this audit.

## Handoff

No CLAIMS change: source applicability review only. No tests or large
computations were run. Relevant primary theorem/algorithm sections were read;
the complete rounding proofs in cited follow-up work were not audited.
Proposed literature metadata are in `scaling-sources.toml` for the root agent's
single-writer registry update. The durable remaining obligations are stated
above; no route is marked refuted.
