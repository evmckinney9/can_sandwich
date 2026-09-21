# Gate 0

Old obligation: spectra-only realization on the complete compact rank-four
domain, with termination, error control and practical cost.

Forward map proposed: spectral classes -> fixed real weighted flags and
truncated singular boundary metric -> Chern-flat numerical metric.
Inverse map proposed: its actual common-base holonomies -> real eigenframes
-> original-factor SO(4) sandwich. Coverage is not established by this map.

GO for this bounded prototype requires an actual checked frame, not just
flow descent, flatness, or unitary-looking eigenvalues. NO-GO means the
specified direct implementation fails its numerical gate; it does not
refute parabolic correspondence or every metric method.

FRONTIER_DELTA = NONE until a separately reviewed reduction is established.
User explicitly requests a prototype for go/no-go; one genuine input anchor
and a manufactured numerical control are bounded side evidence.

Complexity audit: six physical frame coordinates are replaced by sixteen
real metric components per interior mesh node. Algebraic elimination is
replaced by bounded Newton/Krylov iteration and ODE holonomy extraction.
Seed recognition, puncture limits, real structure, reconstruction errors,
boundary coverage and costs remain charged. No component is set to zero.
Any practical recommendation must account for the increased state size.
