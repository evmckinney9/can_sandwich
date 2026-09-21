# Change in research method

2026-09-15. Working method proposed in response to the user's criticism of
pedagogical research. This is not another mathematical claim ledger or an
additional approval protocol. The user's subsequent steering takes priority.

## The standard

A useful research contribution must leave the user with a more accurate,
operational understanding of the problem, whether the proposed method succeeds
or fails. Named theory, a runnable program, and a long list of caveats do not
by themselves meet that standard.

For each proposed step, explain these four things before promoting it:

1. What quantity or object do we already have?
2. What exact operation does the source or derivation permit?
3. What does its output guarantee, and why?
4. Which remaining assertion is our own and what evidence would decide it?

This is an explanatory discipline, not four more forms to fill out.

## Changes in behavior

- Start with the exact missing operation in the atomic problem. Choose papers
  to answer that operation, rather than choosing an operation to showcase a
  paper. A different formulation is useful only after its transfer is explicit.
- Explain one load-bearing connection with a small derivation or picture the
  reader can inspect. Avoid an inventory of named subjects and abbreviations.
- Distinguish source theorem, justified application, and proposed approximation
  in the prose where each is used. Do not rely on a caveat at the end to undo
  confidence created earlier.
- Before a numerical test, say what each possible result would mean. Include
  the possibility that the test cannot decide the mathematical question.
- Compare models and estimators before changing tolerances. For a discretized
  problem, explain why its accepted outputs approximate the original output.
- Use a small example only to isolate a mechanism. Do not present it as evidence
  of full rank-four coverage or quietly make it the next long special-case study.
- Report a failed inference at its exact scope. A failed implementation is not
  a failed theorem or a general route exclusion. Conversely, successful
  approximation is not an exact construction theorem.
- Keep the user involved at changes in the mathematical question. Offer an
  inspectable explanation and a meaningful choice, rather than another
  request to trust a promising direction.

## Demonstration: the R0386 boundary mistake

In the scalar radial case let h(r)>0 and u(r)=log h(r). Chern flatness is
Delta u=0. On an annulus epsilon<r<R its radial solutions are

    u(r)=a log r+b.

The connection is A=partial u=(a/2) dz/z. Counterclockwise parallel
transport is exp(-pi i a). Thus the desired phase exp(-2 pi i mu) fixes
**the logarithmic slope a=2mu**, not a single metric value.

If the two cutoff values are

    u(epsilon)=2mu log epsilon+c_inner,
    u(R)=2mu log R+c_outer,

subtracting them yields

    a=2mu+(c_outer-c_inner)/log(R/epsilon).

This gives an exactly solvable diagnostic. Zero PDE residual can coexist
with wrong holonomy. No matrix computation or convergence claim is needed.
It is the mechanism already identified in R0386's independent precheck,
now explained rather than buried among implementation caveats.

This does not prove the cause of the matrix pilot's error. Its interpolated
connection also fails loop closure, and combined grid/domain changes did not
isolate spatial error. The explanation only identifies the missing implication
that made the original numerical acceptance expectation unjustified.

## Matrix version: what would actually guarantee the local phase

With a diagonal real M and a single-valued holomorphic invertible G(z) on
a puncture neighborhood, set

    h(z)=G(z)^* |z|^(2M) G(z).

Direct differentiation gives

    A=h^-1 partial h
     =G^-1 M G dz/z + G^-1 dG.

This is the diagonal singular connection M dz/z in a holomorphic gauge.
Its peripheral holonomy is conjugate to exp(-2 pi i M). This is an
explanatory calculation in the source's local model, not a new global
construction or a theorem about all numerically imposed asymptotics.

A value of h on a cutoff curve does not supply this holomorphic factorization,
its normal derivative, or a quantitative error bound. Prescribing the full
normal form separately at each puncture still leaves global compatibility
and effective construction unresolved. Replacing metric Dirichlet values by
an asserted residue condition without solving that compatibility would repeat
the same mistake.

## What was learned and what was not

We can now explain precisely why the finite-domain test was not a test of the
published correspondence. We have not proved that parabolic balancing is
practical or impractical. R0386 remains a failed finite implementation.
The next research question must be selected on the strength of a justified
input-to-output connection; this lesson does not privilege the parabolic route.
