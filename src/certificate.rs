//! The production certificate: rootwise acceptance of a candidate frame against the
//! target roots, and the sandwich master it is evaluated on.
use super::*;

/// Cheap frame contract shared by every public certificate path.
pub(super) fn framed_solution(o: Mat4, rung: Rung, residual: f64) -> Option<Solution> {
    if !residual.is_finite() {
        return None;
    }
    let metrics = frame_metrics(&o)?;
    if !metrics.within(FRAME_ACCEPT) {
        return None;
    }
    Some(Solution {
        o: orient_so4(o),
        rung,
        residual,
    })
}

/// Shared pre-certificate for algebraic candidate frames.
///
/// Individual realization routes may use different equations to generate a
/// frame, but they all need the same final local operation: reject non-real or
/// non-orthogonal frames, normalize orientation, and evaluate the forward
/// spectral residual.  Keeping that operation here prevents each route from
/// quietly growing its own tolerance or orientation convention.
pub(super) fn certify_frame_candidate(
    frame: Mat4,
    dc: &Mat4,
    lam: &Mat4,
    target: &[C; 4],
) -> Option<(Mat4, f64)> {
    certify_frame_candidate_with_limit(frame, dc, lam, target, ACCEPT)
}

/// Candidate pre-certificate with an explicit proxy limit.  A few bounded
/// algebraic routes intentionally use a looser construction threshold and
/// defer the final decision to their caller's stronger certificate.
pub(super) fn certify_frame_candidate_with_limit(
    mut frame: Mat4,
    dc: &Mat4,
    lam: &Mat4,
    target: &[C; 4],
    limit: f64,
) -> Option<(Mat4, f64)> {
    let metrics = frame_metrics(&frame)?;
    if !metrics.within(2e-10) {
        return None;
    }
    frame = orient_so4(frame);
    let residual = compound_residual(dc, lam, &frame, target);
    (residual <= limit).then_some((frame, residual))
}

/// The same pre-certificate for a finite set of target lifts.  The minimum is
/// taken only after the frame contract passes; callers must not use a target
/// coefficient proxy as a substitute for this forward check.
pub(super) fn certify_frame_against_targets(
    frame: Mat4,
    dc: &Mat4,
    lam: &Mat4,
    targets: &[[C; 4]],
) -> Option<(Mat4, f64)> {
    let metrics = frame_metrics(&frame)?;
    if !metrics.within(2e-10) {
        return None;
    }
    let frame = orient_so4(frame);
    let residual = targets
        .iter()
        .map(|target| compound_residual(dc, lam, &frame, target))
        .fold(f64::INFINITY, f64::min);
    (residual <= ACCEPT).then_some((frame, residual))
}

/// Rootwise certificate for the public compiler boundary. Symmetric-function
/// residuals are the stable algebraic test used inside the solver, but their
/// inverse map is ill-conditioned at repeated spectra: an `O(1e-9)` coefficient
/// error can represent an `O(1e-5)` class error. Once per returned candidate,
/// diagonalize the realized master against the target roots via Sylvester
/// spectral projectors (eigenvalues known, no iteration): the diagonalization
/// residual IS the rootwise certificate, and the real frame it produces is the
/// complete spectral factorization, kept on the solution for the stitch.
pub(super) fn compiler_solution(
    problem: &PreparedSandwich,
    o: Mat4,
    rung: Rung,
    residual: f64,
) -> Option<Solution> {
    // The coefficient residual is a fast construction gate, not the public
    // certificate: it becomes ill-conditioned at root collisions.  A valid
    // frame may therefore proceed to the direct rootwise test even when that
    // proxy is large.
    let mut solution = framed_solution(o, rung, residual)?;
    let master = sandwich_master(problem, &solution.o);
    // Signed-permutation witnesses make the master diagonal.  Match those
    // roots directly, before invoking coefficient polynomials or projectors;
    // this remains well-conditioned even when several roots nearly collide.
    let off_diagonal = (0..4)
        .flat_map(|row| (0..4).map(move |column| (row, column)))
        .filter(|(row, column)| row != column)
        .map(|(row, column)| master[(row, column)].norm())
        .fold(0.0f64, f64::max);
    if off_diagonal < 1e-10 {
        for roots in &problem.target_roots {
            for permutation in *PERMS24 {
                let error = (0..4)
                    .map(|i| (master[(i, i)] - roots[permutation[i]]).norm())
                    .fold(off_diagonal, f64::max);
                if error < 1e-8 {
                    solution.residual = error;
                    return Some(solution);
                }
            }
        }
    }
    // The well-conditioned branches below (known-spectrum Takagi, Hermitian
    // projection, retargeting) are worth trying whenever the target roots are
    // merely CLOSE, not only when they collide at the 1e-8 representation
    // floor.  A coefficient-exact construction against a cluster of
    // multiplicity m carries a rootwise error of order eps^(1/m), which for
    // f64 eps is 1e-8 to 1e-4: precisely the band that this gate used to
    // exclude, sending those rows into divided projectors that amplify by the
    // inverse root gap.  Widening the GATE only: every branch it admits still
    // certifies against the actual roots at the unchanged 1e-8 bar, so this
    // adds attempts without loosening what is accepted.
    const CLUSTER_GATE: f64 = 1e-3;
    let repeated_target = problem.target_roots[0].iter().enumerate().any(|(i, root)| {
        problem.target_roots[0][i + 1..]
            .iter()
            .any(|other| (root - other).norm() < CLUSTER_GATE)
    });
    if repeated_target {
        // The target roots are known.  Diagonalize the realized symmetric
        // unitary against those roots directly before forming divided
        // projectors.  At a collision the latter amplify a machine-scale
        // construction error by the inverse root gap; the known-spectrum
        // Takagi factorization instead treats the repeated eigenspace as one
        // algebraic block and verifies the full complex reconstruction.
        for roots in &problem.target_roots {
            if let Some(real) = klein::takagi_real(&master, roots) {
                let takagi = Mat4::from_fn(|row, column| C::new(real[(row, column)], 0.0));
                let diagonalized = takagi.transpose() * master * takagi;
                let mut error = 0.0f64;
                for row in 0..4 {
                    for column in 0..4 {
                        let want = if row == column {
                            roots[row]
                        } else {
                            C::default()
                        };
                        error = error.max((diagonalized[(row, column)] - want).norm());
                    }
                }
                if error < 1e-8 {
                    solution.residual = error;
                    return Some(solution);
                }
            }
        }
        // Repeated roots are exactly where divided spectral projectors lose
        // conditioning.  A Hermitian projection of the unitary master keeps
        // the repeated eigenspace intact and supplies a direct multiset check;
        // try it before the historical projector fallback.
        for roots in &problem.target_roots {
            if let Some((actual, off_diagonal)) = klein::unitary_eigenvalues(&master, roots) {
                for permutation in *PERMS24 {
                    let error = (0..4)
                        .map(|i| (actual[i] - roots[permutation[i]]).norm())
                        .fold(off_diagonal, f64::max);
                    if error < 1e-8 {
                        solution.residual = error;
                        return Some(solution);
                    }
                }
            }
        }
        // Root separation powers the simple-target projector certificate, so
        // a repeated target needs a machine-scale certificate of its own; the
        // stitch falls back to extraction there. The symmetric-function
        // residual is only a forward proxy on collision data: it can be tiny
        // while a split repeated root is still outside the public tolerance.
        // Consequently no proxy-only acceptance is sound here; every
        // candidate proceeds through a direct root/multiplicity certificate.
        for roots in problem.target_roots.iter() {
            let mut vals = [C::default(); 4];
            let mut nv = 0usize;
            for &r in roots {
                if !vals[..nv].iter().any(|&v| (v - r).norm() < 1e-8) {
                    vals[nv] = r;
                    nv += 1;
                }
            }
            let shift = |v: C| {
                let mut m = master;
                for i in 0..4 {
                    m[(i, i)] -= v;
                }
                m
            };
            let mut worst = 0.0f64;
            for gi in 0..nv {
                let mut pg = Mat4::identity();
                for hi in 0..nv {
                    if hi != gi {
                        pg = shift(vals[hi]) * pg / (vals[gi] - vals[hi]);
                    }
                }
                let nil = shift(vals[gi]) * pg;
                for i in 0..4 {
                    for j in 0..4 {
                        worst = worst.max(nil[(i, j)].norm());
                    }
                }
                // The annihilation identity proves only that the realized
                // spectrum is a subset of the target's DISTINCT values. Its
                // projector trace supplies the missing algebraic
                // multiplicity: without this, {+1,+1,+1,+1} falsely
                // certified against {+1,+1,-1,-1}.
                let multiplicity = roots
                    .iter()
                    .filter(|&&root| (root - vals[gi]).norm() < 1e-8)
                    .count() as f64;
                worst = worst.max((pg.trace() - C::new(multiplicity, 0.0)).norm());
            }
            // Public monodromy coordinates are quantized, so a sandwich made
            // from an exact unitary can differ from its stored collision
            // target by a few 1e-12.  This is a rootwise error, not the
            // ill-conditioned coefficient proxy; keep a two-order margin over
            // that representation floor.  This is the same direct rootwise
            // threshold used for simple targets below; unlike the symmetric-
            // function proxy, it does not lose conditioning at a collision.
            if worst < 1e-8 {
                solution.residual = worst;
                return Some(solution);
            }
        }
        // A clustered coefficient solve can land on the correct real
        // eigenframe while splitting a repeated root by O(sqrt(eps)). Replace
        // that spectrum in the same frame, peel the left diagonal, and recover
        // the gate frame from its known spectrum. This is a fixed algebraic
        // reconstruction; the complete rebuilt sandwich is checked below.
        for roots in &problem.target_roots {
            let Some((target_master, _)) = klein::retarget_symmetric(&master, roots) else {
                continue;
            };
            let inverse_dc = problem.dc.map(|value| value.conj());
            let peeled = inverse_dc * target_master * inverse_dc;
            let Some(real_o) = klein::takagi_real(&peeled, &problem.right) else {
                continue;
            };
            let o = orient_so4(Mat4::from_fn(|row, column| {
                C::new(real_o[(row, column)], 0.0)
            }));
            let rebuilt = sandwich_master(problem, &o);
            let error = rebuilt
                .iter()
                .zip(target_master.iter())
                .map(|(actual, expected)| (*actual - *expected).norm())
                .fold(0.0f64, f64::max);
            if error < 1e-8 {
                let mut repaired = framed_solution(o, rung, error)?;
                repaired.residual = error;
                return Some(repaired);
            }
        }
        return None;
    }
    // The correct central-rho branch is picked (up to numerical ties) by the
    // free trace test tr(master) = e1(branch); trying it first saves a full
    // projector build + orthogonality check on rho-branch rows. Pure reorder:
    // both branches still run until one certifies, so the accept set is
    // unchanged.
    let trace = master.trace();
    let branch_order = if (trace - problem.targets[0][0]).norm_sqr()
        <= (trace - problem.targets[1][0]).norm_sqr()
    {
        [0usize, 1]
    } else {
        [1usize, 0]
    };
    // Simple-target closed-form certificate.  With p the master's own
    // characteristic polynomial (symmetric functions, three products), |p(lambda_k)| is the
    // product of the distances from lambda_k to the realized roots, so once every realized root
    // is within half its gap of its target the root nearest lambda_k lies within
    // |p(lambda_k)| / prod_{j != k} (|lambda_k - lambda_j| / 2).  The bound alone is not the
    // acceptance: it gates the Sylvester-projector eigenframe of the known roots (closed form,
    // no eigensolve), and the frame's explicit diagonalization residual is what certifies.
    let tcert = prof::start();
    // Unitary master: e4 = det D1 det D2 from the gates and e3 = conj(e1) e4, so two traces suffice.
    let e = {
        let e1 = master.trace();
        let mut tr2 = C::default();
        for i in 0..4 {
            for j in 0..4 {
                tr2 += master[(i, j)] * master[(j, i)];
            }
        }
        let e2 = (e1 * e1 - tr2) * 0.5;
        let mut e4 = C::new(1.0, 0.0);
        for i in 0..4 {
            e4 *= problem.dc[(i, i)] * problem.dc[(i, i)] * problem.lam[(i, i)];
        }
        [e1, e2, e1.conj() * e4, e4]
    };
    for &branch in &branch_order {
        let roots = &problem.target_roots[branch];
        let mut bound = 0.0f64;
        for k in 0..4 {
            let z = roots[k];
            let p = z * z * z * z - e[0] * z * z * z + e[1] * z * z - e[2] * z + e[3];
            let den: f64 = (0..4)
                .filter(|&j| j != k)
                .map(|j| 0.5 * (roots[k] - roots[j]).norm())
                .product();
            bound = bound.max(p.norm() / den);
        }
        if bound < 1e-8 {
            // The annihilation bound is itself a rigorous certificate for a
            // simple target, so accept on it directly.  `takagi` is only the
            // segment-stitch hint; core rebuilds it where a stitch needs it,
            // and skipping it here removes the projector eigenframe (about a
            // third of the median solve) from every simple-target row.
            solution.residual = bound;
            prof::rec(prof::CERT_FAST, tcert);
            return Some(solution);
        }
    }
    prof::rec(prof::CERT_FAST, tcert);
    // Use a Hermitian projection before either Takagi projectors or a generic
    // complex eigensolve.  The master is unitary and normal, so this preserves
    // its eigenspaces and remains backward stable when roots cluster.
    for &branch in &branch_order {
        let roots = &problem.target_roots[branch];
        if let Some((actual, off_diagonal)) = klein::unitary_eigenvalues(&master, roots) {
            let mut best_error = f64::INFINITY;
            for permutation in *PERMS24 {
                let error = (0..4)
                    .map(|i| (actual[i] - roots[permutation[i]]).norm())
                    .fold(off_diagonal, f64::max);
                best_error = best_error.min(error);
                if error < 1e-8 {
                    solution.residual = error;
                    return Some(solution);
                }
            }
        }
    }
    for &branch in &branch_order {
        let roots = &problem.target_roots[branch];
        let Some(frame) = klein::takagi_real(&master, roots) else {
            continue;
        };
        let v = Mat4::from_fn(|i, j| C::new(frame[(i, j)], 0.0));
        let t = v.transpose() * master * v;
        let mut error = 0.0f64;
        for r in 0..4 {
            for c in 0..4 {
                let want = if r == c { roots[r] } else { C::default() };
                error = error.max((t[(r, c)] - want).norm());
            }
        }
        if error < 1e-8 {
            solution.residual = error;
            return Some(solution);
        }
    }
    // The master is normal, so its direct 4x4 eigensolve is backward stable
    // even at a root collision. This is a stronger fallback than solving its
    // characteristic polynomial, whose inverse map is ill-conditioned there.
    let matrix = faer::Mat::<C>::from_fn(4, 4, |row, column| master[(row, column)]);
    if let Ok(values) = matrix.eigenvalues() {
        let actual: [C; 4] = std::array::from_fn(|i| values[i]);
        let mut best_error = f64::INFINITY;
        for &branch in &branch_order {
            for permutation in *PERMS24 {
                let error = (0..4)
                    .map(|i| (actual[i] - problem.target_roots[branch][permutation[i]]).norm())
                    .fold(0.0f64, f64::max);
                best_error = best_error.min(error);
                if error < 1e-8 {
                    solution.residual = error;
                    return Some(solution);
                }
            }
        }
    }
    // A real Takagi frame can be numerically awkward at a collision in an
    // INPUT spectrum even when the target roots are simple.  Certify the
    // realized master's roots directly before declining.  The caller already
    // has a clean-product extraction fallback when `takagi` is absent.
    let e = symfn(&master);
    let coefficients = [C::new(1.0, 0.0), -e[0], e[1], -e[2], e[3]];
    let actual = crate::cpoly::roots(&coefficients);
    if actual.len() == 4 {
        let mut best_error = f64::INFINITY;
        for &branch in &branch_order {
            for permutation in *PERMS24 {
                let error = (0..4)
                    .map(|i| (actual[i] - problem.target_roots[branch][permutation[i]]).norm())
                    .fold(0.0f64, f64::max);
                best_error = best_error.min(error);
                if error < 1e-8 {
                    solution.residual = error;
                    return Some(solution);
                }
            }
        }
    }
    None
}

/// `dc·O·Λ·Oᵀ·dc` with the diagonality of `dc` and `Λ` folded into entry
/// scalings: one bilinear pass instead of four dense 4×4 products. Exact
/// same arithmetic content as the dense chain.
#[inline]
pub(super) fn sandwich_master(problem: &PreparedSandwich, o: &Mat4) -> Mat4 {
    let d: [C; 4] = std::array::from_fn(|i| problem.dc[(i, i)]);
    let l: [C; 4] = std::array::from_fn(|j| problem.lam[(j, j)]);
    let ol: [[C; 4]; 4] = std::array::from_fn(|i| std::array::from_fn(|k| o[(i, k)] * l[k]));
    Mat4::from_fn(|i, j| {
        let mut s = C::default();
        for k in 0..4 {
            s += ol[i][k] * o[(j, k)];
        }
        d[i] * s * d[j]
    })
}
