//! Depth-two two-qubit realization in binary64 arithmetic.
//!
//! Given left, right, and target canonical gates, construct the local gate
//! between the first two that produces the target nonlocal action.
//!
//! Inputs are dimensionless monodromy triples. For `m = [m0, m1, m2]`, the
//! diagonal gate in the magic basis is
//! `D(m) = diag(exp(iπm1), exp(iπm0), exp(-iπ(m0+m1+m2)), exp(iπm2))`.
//! See the [researcher guide](https://github.com/evmckinney9/can_sandwich/blob/main/docs/researcher.md)
//! for the mathematical contract and how to compare a candidate algorithm.
#![allow(clippy::needless_range_loop, clippy::neg_cmp_op_on_partial_ord)]
use nalgebra::{Complex, Matrix4};
type C = Complex<f64>;
type ComplexMatrix = Matrix4<C>;
mod algebraic;
#[cfg(any(test, feature = "corpus"))]
pub mod corpus;
mod diagnostics;
mod numerical;
mod problem;
mod spectral;
#[cfg(not(feature = "diagnostics"))]
use algebraic::{Rung, Solution};
#[cfg(feature = "diagnostics")]
pub use algebraic::{Rung, Solution};
#[cfg(not(feature = "diagnostics"))]
use diagnostics::prof;
#[cfg(feature = "diagnostics")]
pub use diagnostics::prof;
use problem::Problem;
/// Largest root error, including the eigenbasis residual, of an accepted witness.
pub use spectral::SPECTRAL_TOLERANCE;
#[cfg(feature = "diagnostics")]
pub type Mat4 = ComplexMatrix;
#[cfg(feature = "diagnostics")]
pub use diagnostics::{branch_signature, certify_frame, solve_report};

/// The crate version, for bug reports.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Construct a real SO(4) matrix `O` for left gate `c`, right gate `g`, and target `t`.
///
/// The spectrum of `D(c)² O D(g)² Oᵀ` must match that of `s D(t)²` for one
/// global sign `s = ±1`, within numerical tolerance. The crate documentation
/// defines `D` and the input coordinates.
///
/// Returns `None` for nonfinite inputs or when the bounded search finds no
/// accepted witness. A decline does not establish that the input is infeasible.
pub fn solve(c: [f64; 3], g: [f64; 3], t: [f64; 3]) -> Option<Matrix4<f64>> {
    witness(c, g, t).map(|(_, solution)| solution.o)
}

/// Why [`solve_with_factors`] returned no result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeclineKind {
    /// The input is nonfinite, or the bounded search found no frame that
    /// passes spectral verification.
    NoWitness,
    /// A verified frame was found, but `L` and `R` failed their checks.
    Reconstruction,
}

/// A [`solve_with_factors`] decline with the inputs that produced it.
///
/// `Display` renders a bug report: the reason, the inputs, and the crate
/// version, followed by the issue URL. Debug formatting preserves the finite
/// input values so they can be copied into a reproducer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Decline {
    /// Why the solver declined.
    pub kind: DeclineKind,
    /// Left gate monodromy coordinates.
    pub c: [f64; 3],
    /// Right gate monodromy coordinates.
    pub g: [f64; 3],
    /// Target monodromy coordinates.
    pub t: [f64; 3],
}

impl std::fmt::Display for Decline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let reason = match self.kind {
            DeclineKind::NoWitness => "no witness",
            DeclineKind::Reconstruction => "endpoint reconstruction failed",
        };
        write!(
            f,
            "can_sandwich {VERSION} declined ({reason}) for c={:?}, g={:?}, t={:?}. \
             Please report this message at https://github.com/evmckinney9/can_sandwich/issues/1",
            self.c, self.g, self.t
        )
    }
}

impl std::error::Error for Decline {}

/// Return `(O, L, R, phase)` with all three matrices real SO(4), where
/// `D(c) O D(g) = exp(i phase) L D(t) R` in the magic basis.
///
/// Inputs follow the same convention as [`solve`], and `phase` is in radians.
/// The result is verified numerically, rather than certified in exact
/// arithmetic. A [`Decline`] does not establish that the input is infeasible.
#[allow(clippy::type_complexity)]
pub fn solve_with_factors(
    c: [f64; 3],
    g: [f64; 3],
    t: [f64; 3],
) -> Result<(Matrix4<f64>, Matrix4<f64>, Matrix4<f64>, f64), Decline> {
    let decline = |kind| Decline { kind, c, g, t };
    let (problem, solution) = witness(c, g, t).ok_or_else(|| decline(DeclineKind::NoWitness))?;
    solution
        .state
        .factors(&problem, t, solution.o)
        .ok_or_else(|| decline(DeclineKind::Reconstruction))
}

fn witness(c: [f64; 3], g: [f64; 3], t: [f64; 3]) -> Option<(Problem, Solution)> {
    if c.iter().chain(&g).chain(&t).any(|x| !x.is_finite()) {
        return None;
    }
    let tp = prof::start();
    let problem = Problem::new(c, g, t);
    prof::rec(prof::SEG_PREPARE, tp);
    // The algebraic dispatch settles almost every input: polish its frame and
    // return when the root error is already below the tail trigger. Every
    // other path is out of line.
    if let Some(solution) = algebraic::solve(&problem) {
        let primary = polish(&problem, c, g, t, solution);
        if primary.state.root_error <= DUAL_TRIGGER {
            return Some((problem, primary));
        }
        return tail(problem, c, g, t, Some(solution), Some(primary));
    }
    recover(problem, c, g, t)
}

/// Inputs without an algebraic frame: second-order vertex frames, the
/// numerical ladder, then the tail.
#[cold]
#[inline(never)]
fn recover(problem: Problem, c: [f64; 3], g: [f64; 3], t: [f64; 3]) -> Option<(Problem, Solution)> {
    let algebraic: Option<Solution> = None;
    // A second-order vertex frame that already meets the refinement target
    // needs neither the numerical ladder nor a polish.
    let vertex = std::cell::OnceCell::new();
    if algebraic.is_none()
        && let Some(v) = *vertex.get_or_init(|| vertex_quadratic(&problem))
        && v.state.root_error <= numerical::ROOT_TOLERANCE
    {
        return Some((problem, v));
    }
    let found = algebraic
        .or_else(|| {
            let o = numerical::solve(&problem, c, g, t)?;
            let state = spectral::verify(&problem, &o)?;
            Some(Solution {
                o,
                rung: Rung::Numerical,
                residual: state.error,
                state,
            })
        })
        // The second-order vertex frames, before a decline.
        .or_else(|| *vertex.get_or_init(|| vertex_quadratic(&problem)));
    let primary = found.map(|solution| polish(&problem, c, g, t, solution));
    if primary
        .as_ref()
        .is_some_and(|s| s.state.root_error <= DUAL_TRIGGER)
    {
        return primary.map(|solution| (problem, solution));
    }
    tail(problem, c, g, t, found, primary)
}

/// Witnesses above the tail trigger: sheet reflections, the stabilizer gauge
/// search, the dual chart and the stratum candidates.
#[cold]
#[inline(never)]
fn tail(
    problem: Problem,
    c: [f64; 3],
    g: [f64; 3],
    t: [f64; 3],
    found: Option<Solution>,
    primary: Option<Solution>,
) -> Option<(Problem, Solution)> {
    // Sheet reflections first: when one lands, the dual chart and the
    // stratum candidates are not needed. Otherwise the tail continues from
    // the unreflected primary.
    let sheet = reflected(&problem, c, g, t, primary);
    if sheet
        .as_ref()
        .is_some_and(|s| s.state.root_error <= DUAL_TRIGGER)
    {
        return sheet.map(|solution| (problem, solution));
    }
    let gauged = gauge_search(&problem, c, g, t, better(primary, sheet));
    if gauged
        .as_ref()
        .is_some_and(|s| s.state.root_error <= DUAL_TRIGGER)
    {
        return gauged.map(|solution| (problem, solution));
    }
    // Transpose duality: spec(D_c^2 O D_g^2 O^T) = spec(D_g^2 O^T D_c^2 O),
    // so O^T is a witness of the role-swapped problem (g, c, t) and back.
    // Refinement moves the frame by left rotations, i.e. on the c side only;
    // in the dual it moves the g side. Near a Horn wall or at a confluent
    // input the first witness can stall at a critical point of the left
    // action that the right action leaves, so the tail also polishes the
    // primary in the dual chart and tries the dual's own witness; the
    // smallest certified error wins.
    let dual = Problem::new(g, c, t);
    let mut best = primary;
    if let Some(p) = primary {
        best = better(
            best,
            transported(&problem, &dual, p)
                .map(|s| polish_with(&dual, g, c, t, s, false))
                .and_then(|s| transported(&dual, &problem, s)),
        );
    }
    // The dual's own algebraic witness: kept as is when it already beats
    // the candidates, polished only if the tail persists (or on a decline).
    let native = algebraic::solve(&dual);
    if let Some(native) = native {
        best = better(best, transported(&dual, &problem, native));
        if best
            .as_ref()
            .is_none_or(|b| b.state.root_error > DUAL_TRIGGER)
        {
            best = better(
                best,
                transported(&dual, &problem, polish_with(&dual, g, c, t, native, false)),
            );
        }
    }
    best = stratum_candidates(&problem, c, g, t, best, found, true);
    // The same stratum candidates in the dual chart, from the transported
    // witness and the dual's own construction (without the ladder, whose
    // decline budget the tail must not pay twice).
    if let Some(b) = best
        && b.state.root_error > DUAL_TRIGGER
    {
        let dual_best = transported(&problem, &dual, b);
        best = better(
            best,
            stratum_candidates(&dual, g, c, t, dual_best, native, false)
                .and_then(|s| transported(&dual, &problem, s)),
        );
    }
    best.map(|solution| (problem, solution))
}

/// Candidates from other strata for a near-miss witness: the dispatch with
/// its rung excluded, spectral retargets, plain iteration (from the witness
/// and from the unpolished construction), and optionally the ladder.
fn stratum_candidates(
    problem: &Problem,
    c: [f64; 3],
    g: [f64; 3],
    t: [f64; 3],
    mut best: Option<Solution>,
    found: Option<Solution>,
    ladder: bool,
) -> Option<Solution> {
    // A constructed frame at a target a hair off its stratum (a Horn wall
    // at distance ~1e-13) certifies at that distance, while the exact
    // witness lies in another stratum: rerun the dispatch without the rung
    // that produced the near-miss.
    if let Some(b) = best
        && b.state.root_error > DUAL_TRIGGER
        && !matches!(b.rung, Rung::Numerical)
    {
        best = better(
            best,
            algebraic::solve_excluding(problem, b.rung)
                .map(|s| polish_with(problem, c, g, t, s, false)),
        );
    }
    // Spectral retarget: the witness's sandwich S is a symmetric unitary,
    // so its real eigenframe carries it to diag(roots). Replacing those
    // roots by the exact target roots and peeling D_c off both sides gives
    // O diag(g) O^T up to the retarget, whose real Takagi factor with the
    // known spectrum diag(g) is the retargeted frame.
    if let Some(b) = best
        && b.state.root_error > DUAL_TRIGGER
    {
        // The plain refinement iteration from the witness, which follows the
        // spectral residual without the polish acceptance tests.
        // Both from the witness and from the unpolished construction.
        let iterated = [Some(b.o), found.map(|f| f.o)]
            .into_iter()
            .flatten()
            .filter_map(|o| problem.iterate(o, 0.0));
        // Horn-wall frames whose retained routed root misses its target
        // root by more than the refinement target but less than acceptance.
        let wall = problem.split(
            numerical::seed(c, g, t),
            1e-11,
            spectral::SPECTRAL_TOLERANCE,
            None,
        );
        for candidate in algebraic::retargets(problem, &b.o)
            .into_iter()
            .chain(iterated)
            .chain(wall)
        {
            if let Some(state) = spectral::verify(problem, &candidate) {
                let found = Solution {
                    o: candidate,
                    rung: b.rung,
                    residual: state.error,
                    state,
                };
                best = better(best, Some(polish_with(problem, c, g, t, found, false)));
            }
        }
    }
    // Still in the tail: the numerical ladder's own witness, polished.
    if ladder
        && let Some(b) = best
        && b.state.root_error > DUAL_TRIGGER
        && !matches!(b.rung, Rung::Numerical)
        && let Some(o) = numerical::solve(problem, c, g, t)
        && let Some(state) = spectral::verify(problem, &o)
    {
        let found = Solution {
            o,
            rung: Rung::Numerical,
            residual: state.error,
            state,
        };
        best = better(best, Some(polish_with(problem, c, g, t, found, false)));
    }
    best
}

/// Root error of the certified witness above which the transpose-dual
/// candidates are also tried. The root error tracks the graded spectral
/// error; the certified bound adds the eigenbasis coupling on top.
const DUAL_TRIGGER: f64 = 2e-14;

/// The certified candidate with the smaller root error (ties keep the first).
fn better(first: Option<Solution>, second: Option<Solution>) -> Option<Solution> {
    match (first, second) {
        (Some(a), Some(b)) => Some(if b.state.root_error < a.state.root_error {
            b
        } else {
            a
        }),
        (a, b) => a.or(b),
    }
}

/// The transposed witness, verified on the other problem of the dual pair.
fn transported(_from: &Problem, to: &Problem, solution: Solution) -> Option<Solution> {
    let o = solution.o.transpose();
    let state = spectral::verify(to, &o)?;
    Some(Solution {
        o,
        rung: solution.rung,
        residual: state.error,
        state,
    })
}

/// Post-solve refinement of a verified witness: Horn-wall split polish,
/// refinement, orthogonality repair, and root Gauss-Newton.
fn polish(
    problem: &Problem,
    c: [f64; 3],
    g: [f64; 3],
    t: [f64; 3],
    solution: Solution,
) -> Solution {
    polish_with(problem, c, g, t, solution, true)
}

/// `polish`, optionally without the Horn-wall split search. The split
/// search depends only on the problem, so a second polish on the same
/// problem would repeat it.
fn polish_with(
    problem: &Problem,
    c: [f64; 3],
    g: [f64; 3],
    t: [f64; 3],
    mut solution: Solution,
    split: bool,
) -> Solution {
    // Signed permutations and one-Givens frames are orthogonal to rounding
    // (their rows are unit (cos, sin) pairs), so drift repair cannot fire
    // on them until a polish replaces the frame.
    let mut constructed = matches!(solution.rung, Rung::Vertex | Rung::Edge);
    // Targets on a Horn wall, where a routed root equals a target root, can
    // stall above the refinement target in every free frame. Retaining that
    // root exactly while refining the complementary SO(3) block reaches it.
    // Routed roots within the multiplicity threshold (`1e-11`) qualify.
    if split && solution.state.error > numerical::ROOT_TOLERANCE {
        let current = problem.joint(&solution.o, solution.state, 0.0);
        if current.root_error > numerical::ROOT_TOLERANCE
            && let Some(o) = problem.split(
                numerical::seed(c, g, t),
                1e-11,
                numerical::ROOT_TOLERANCE,
                Some(2),
            )
            && let Some(state) = spectral::verify(problem, &o)
            && problem.joint(&o, state, 0.0).error
                < (current.root_error - 4.0 * current.off_diagonal_error).max(0.5 * current.error)
        {
            solution.o = o;
            solution.residual = state.error;
            solution.state = state;
            constructed = false;
        }
    }
    // Polish when the verified bound exceeds the refinement target. Measure
    // both frames with the joint eigenbasis and require improvement beyond
    // both of its residuals. A large residual can reflect orthogonality
    // drift, which refinement removes; then the error bound must halve.
    if solution.state.error > numerical::ROOT_TOLERANCE {
        let current = problem.joint(&solution.o, solution.state, 0.0);
        let lower_error = current.root_error - 4.0 * current.off_diagonal_error;
        if current.root_error > numerical::ROOT_TOLERANCE {
            let o = problem.refine(solution.o);
            if let Some(state) = spectral::verify(problem, &o)
                && problem.joint(&o, state, 0.0).error < lower_error.max(0.5 * current.error)
            {
                solution.o = o;
                solution.residual = state.error;
                solution.state = state;
                constructed = false;
            }
        }
    }
    // Restore orthogonality when its drift exceeds 1e-14. One polar
    // Newton step is exact to first order and moves the roots only by the
    // drift itself.
    let drift = if constructed {
        0.0
    } else {
        (solution.o.transpose() * solution.o - nalgebra::Matrix4::identity()).amax()
    };
    if drift > 1e-14 {
        let o = numerical::orthogonalize(numerical::orthogonalize(solution.o));
        if let Some(state) = spectral::verify(problem, &o)
            && state.root_error <= solution.state.root_error.max(drift)
        {
            solution.o = o;
            solution.residual = state.error;
            solution.state = state;
        }
    }
    // Frames still above 2e-14 after both polishes: Gauss–Newton on the roots.
    if solution.state.root_error > 1e-14 {
        let o = problem.newton_polish(solution.o);
        if let Some(state) = spectral::verify(problem, &o)
            && state.root_error < solution.state.root_error
        {
            solution.o = o;
            solution.residual = state.error;
            solution.state = state;
        }
    }
    solution
}

/// Near-double pairs of a diagonal phase list: distinct but closer than the
/// multiplicity threshold of the Horn-wall split (exactly equal pairs are
/// exact symmetries and reflect to the same product).
#[cold]
#[inline(never)]
fn near_pairs(d: &[C; 4]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for i in 0..4 {
        for j in i + 1..4 {
            let gap = (d[i] - d[j]).norm();
            if gap > 0.0 && gap < 1e-11 {
                out.push((i, j));
            }
        }
    }
    out
}

/// Quarter turns of `o` in the planes of near-double pairs: rows for the
/// left factor, columns for the right, and each left-right combination.
#[cold]
#[inline(never)]
fn reflections(problem: &Problem, o: &Matrix4<f64>) -> Vec<Matrix4<f64>> {
    let turn_rows = |mut m: Matrix4<f64>, (p, q): (usize, usize)| {
        for j in 0..4 {
            let (x, y) = (m[(p, j)], m[(q, j)]);
            m[(p, j)] = y;
            m[(q, j)] = -x;
        }
        m
    };
    let left = near_pairs(&problem.left);
    let right = near_pairs(&problem.right);
    let mut out = Vec::new();
    for &l in &left {
        out.push(turn_rows(*o, l));
    }
    for &r in &right {
        out.push(turn_rows(o.transpose(), r).transpose());
        for &l in &left {
            out.push(turn_rows(turn_rows(o.transpose(), r).transpose(), l));
        }
    }
    out
}

/// Sheet reflections of a tail witness (see `reflections`), polished; the
/// better of the input and the candidates.
#[cold]
#[inline(never)]
fn reflected(
    problem: &Problem,
    c: [f64; 3],
    g: [f64; 3],
    t: [f64; 3],
    mut best: Option<Solution>,
) -> Option<Solution> {
    // Sheet reflection. A near-double pair of D(c)^2 (rows) or D(g)^2
    // (columns) makes the quarter turn in its coordinate plane a symmetry
    // up to the pair gap. The turn exchanges the two sheets of the fold on
    // which the gap shifts the roots with opposite first-order signs, so a
    // witness stalled on the sheet whose shift points away from the target
    // has its solution next to the reflected frame. Gauss-Newton on the
    // roots screens each reflection; a screen that reaches the refinement
    // target ends the search, otherwise only the best screen is polished.
    let Some(b) = best else { return best };
    if b.state.root_error <= DUAL_TRIGGER {
        return best;
    }
    let mut screened: Option<Solution> = None;
    for o in reflections(problem, &b.o) {
        // The turn is a symmetry up to the pair gap, so a useful reflected
        // frame still verifies before any refinement.
        if spectral::verify(problem, &o).is_none() {
            continue;
        }
        let screen = [problem.newton_polish(o), o]
            .into_iter()
            .find_map(|o| spectral::verify(problem, &o).map(|state| (o, state)));
        if let Some((o, state)) = screen {
            let found = Solution {
                o,
                rung: b.rung,
                residual: state.error,
                state,
            };
            if state.root_error <= numerical::ROOT_TOLERANCE {
                return better(best, Some(polish_with(problem, c, g, t, found, false)));
            }
            screened = better(screened, Some(found));
        }
    }
    if let Some(s) = screened
        && s.state.root_error < b.state.root_error
    {
        best = better(best, Some(polish_with(problem, c, g, t, s, false)));
    }
    best
}

/// The best verified second-order vertex frame (see
/// `Problem::vertex_quadratic`), ahead of the numerical ladder.
fn vertex_quadratic(problem: &Problem) -> Option<Solution> {
    let mut best: Option<Solution> = None;
    for o in problem.vertex_quadratic() {
        if let Some(state) = spectral::verify(problem, &o) {
            best = better(
                best,
                Some(Solution {
                    o,
                    rung: Rung::Numerical,
                    residual: state.error,
                    state,
                }),
            );
        }
    }
    best
}

/// Root clusters closer than the multiplicity threshold (1e-11).
fn clusters(d: &[C; 4]) -> Vec<Vec<usize>> {
    let mut used = [false; 4];
    let mut out = Vec::new();
    for i in 0..4 {
        if used[i] {
            continue;
        }
        let mut group = vec![i];
        for j in i + 1..4 {
            if !used[j] && (d[i] - d[j]).norm() < 1e-11 {
                group.push(j);
                used[j] = true;
            }
        }
        used[i] = true;
        if group.len() > 1 {
            out.push(group);
        }
    }
    out
}

fn gauge_planes(groups: &[Vec<usize>]) -> Vec<(usize, usize)> {
    let mut planes = Vec::new();
    for group in groups {
        for a in 0..group.len() {
            for b in a + 1..group.len() {
                planes.push((group[a], group[b]));
            }
        }
    }
    planes
}

fn rotation(planes: &[(usize, usize)], p: &[f64]) -> Matrix4<f64> {
    let mut a = Matrix4::<f64>::zeros();
    for (&(i, j), &v) in planes.iter().zip(p) {
        a[(i, j)] = v;
        a[(j, i)] = -v;
    }
    let mut squarings = 0;
    while a.amax() > 0.25 {
        a /= 2.0;
        squarings += 1;
    }
    let mut out = Matrix4::<f64>::identity();
    let mut term = Matrix4::<f64>::identity();
    for n in 1..16 {
        term = term * a / n as f64;
        out += term;
    }
    for _ in 0..squarings {
        out = out * out;
    }
    out
}

/// Near-degenerate stabilizer gauge. Root clusters of D(c)^2 and D(g)^2
/// closer than 1e-11 make the product of their rotation groups a symmetry
/// up to the cluster gaps; along it the roots move only at the gap scale,
/// so a stalled witness is re-aligned by Gauss-Newton on the gauge angles
/// with the residual rescaled by 1e13, from a few deterministic starts.
#[cold]
#[inline(never)]
fn gauge_search(
    problem: &Problem,
    c: [f64; 3],
    g: [f64; 3],
    t: [f64; 3],
    best: Option<Solution>,
) -> Option<Solution> {
    let b = best?;
    if b.state.root_error <= DUAL_TRIGGER {
        return best;
    }
    let left = gauge_planes(&clusters(&problem.left));
    let right = gauge_planes(&clusters(&problem.right));
    let n = left.len() + right.len();
    if n == 0 {
        return best;
    }
    let frame =
        |p: &[f64]| rotation(&left, &p[..left.len()]) * b.o * rotation(&right, &p[left.len()..]);
    // Starts: the witness itself, then a quarter turn in each gauge plane
    // (the sheet reflections), then two pseudo-random gauges.
    let mut seed: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut starts: Vec<Vec<f64>> = vec![vec![0.0; n]];
    for q in 0..n {
        let mut p = vec![0.0; n];
        p[q] = std::f64::consts::FRAC_PI_2;
        starts.push(p);
    }
    for _ in 0..2 {
        starts.push(
            (0..n)
                .map(|_| {
                    seed = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 3.0
                })
                .collect(),
        );
    }
    let mut screened: Option<Solution> = None;
    for p0 in starts {
        // Along the gauge the sandwich is conjugated by the left rotation up to
        // the cluster gaps, so its eigenbasis is carried by that rotation: the
        // residual projects onto the carried basis instead of re-solving.
        let start = problem.state(&frame(&p0), 0.0);
        let carry = rotation(&left, &p0[..left.len()]).transpose() * start.eigenvectors;
        let residual = |p: &[f64]| -> [f64; 4] {
            let basis = rotation(&left, &p[..left.len()]) * carry;
            let roots = problem.roots_in_basis(&frame(p), &basis);
            std::array::from_fn(|k| (roots[k] * start.target[k].conj()).arg() * 1e13)
        };
        let mut p = p0;
        let mut r = residual(&p);
        let mut cost: f64 = r.iter().map(|x| x * x).sum();
        let mut damping = 1e-3;
        let mut jac = nalgebra::DMatrix::<f64>::zeros(4, n);
        let mut stale = true;
        for _ in 0..15 {
            // The Jacobian depends only on p: a rejected step keeps it.
            if stale {
                for q in 0..n {
                    let mut pq = p.clone();
                    pq[q] += 1e-3;
                    let rq = residual(&pq);
                    for k in 0..4 {
                        jac[(k, q)] = (rq[k] - r[k]) / 1e-3;
                    }
                }
                stale = false;
            }
            let rv = nalgebra::DVector::<f64>::from_row_slice(&r);
            let jt = jac.transpose();
            let mut normal = &jt * &jac;
            for q in 0..n {
                normal[(q, q)] *= 1.0 + damping;
                normal[(q, q)] += 1e-12;
            }
            let Some(step) = normal.lu().solve(&(-(&jt * rv))) else {
                break;
            };
            let trial: Vec<f64> = p.iter().zip(step.iter()).map(|(a, s)| a + s).collect();
            let rt = residual(&trial);
            let ct: f64 = rt.iter().map(|x| x * x).sum();
            if ct < cost {
                p = trial;
                r = rt;
                cost = ct;
                stale = true;
                damping = (damping * 0.3).max(1e-9);
                if cost < 1e-2 {
                    break;
                }
            } else {
                damping *= 10.0;
                if damping > 1e8 {
                    break;
                }
            }
        }
        let o = numerical::orthogonalize(frame(&p));
        if let Some(state) = spectral::verify(problem, &o) {
            let found = Solution {
                o,
                rung: b.rung,
                residual: state.error,
                state,
            };
            if state.root_error <= numerical::ROOT_TOLERANCE {
                return better(best, Some(polish_with(problem, c, g, t, found, false)));
            }
            screened = better(screened, Some(found));
        }
    }
    match screened {
        Some(s) if s.state.root_error < b.state.root_error => {
            better(best, Some(polish_with(problem, c, g, t, s, false)))
        }
        _ => best,
    }
}
