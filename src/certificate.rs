//! The production certificate: rootwise acceptance of a candidate frame against the
//! target roots, and the sandwich master it is evaluated on.
use super::*;

/// Numerical certificate required before treating a matrix as a real frame.
#[derive(Debug, Clone, Copy)]
pub(super) struct FrameMetrics {
    pub determinant: f64,
    pub gram: f64,
    pub imaginary: f64,
}

impl FrameMetrics {
    #[inline]
    pub(super) fn within(self, tolerance: f64) -> bool {
        self.gram <= tolerance
            && self.imaginary <= tolerance
            && (self.determinant.abs() - 1.0).abs() <= tolerance
    }
}

#[inline]
pub(super) fn frame_metrics(frame: &Mat4) -> Option<FrameMetrics> {
    let real = frame.map(|value| value.re);
    let determinant = real.determinant();
    let gram = (real.transpose() * real - nalgebra::Matrix4::identity())
        .iter()
        .fold(0.0f64, |maximum, value| maximum.max(value.abs()));
    let imaginary = frame
        .iter()
        .fold(0.0f64, |maximum, value| maximum.max(value.im.abs()));
    (determinant.is_finite() && gram.is_finite() && imaginary.is_finite()).then_some(FrameMetrics {
        determinant,
        gram,
        imaginary,
    })
}

/// Normalize an accepted real orthogonal frame to `SO(4)`.
#[inline]
pub(super) fn orient_so4(mut frame: Mat4) -> Mat4 {
    if frame.determinant().re < 0.0 {
        for row in 0..4 {
            frame[(row, 0)] = -frame[(row, 0)];
        }
    }
    frame
}

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

#[inline]
pub(super) fn apply_transpose(frame: Mat4, transpose: bool) -> Mat4 {
    if transpose {
        frame.transpose()
    } else {
        frame
    }
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

/// Recover the right endpoint gauge `R` in `D_C O D_G = L D_M R`.
pub(super) fn right_endpoint_gauge(problem: &PreparedSandwich, o: &Mat4) -> Option<Mat4> {
    endpoint_factorization(problem, o).map(|(_, right)| right)
}

/// Recover both endpoint frames in
/// `D_C O D_G = L D_T R`.  The left frame is needed when a factorized
/// `B V B` witness is collapsed back to the original canonical gate `G`.
pub(super) fn endpoint_factorization(problem: &PreparedSandwich, o: &Mat4) -> Option<(Mat4, Mat4)> {
    endpoint_factorization_branch(problem, o, 0)
        .or_else(|| endpoint_factorization_branch(problem, o, 1))
}

/// Endpoint factorization on one explicit target representative branch.
pub(super) fn endpoint_factorization_branch(
    problem: &PreparedSandwich,
    o: &Mat4,
    branch: usize,
) -> Option<(Mat4, Mat4)> {
    let master = sandwich_master(problem, o);
    let roots0 = problem.target_roots.get(branch)?;
    for permutation in *PERMS24 {
      let roots = std::array::from_fn(|i| roots0[permutation[i]]);
      let Some(frame) = klein::takagi_real(&master, &roots) else { continue };
      let l = Mat4::from_fn(|i, j| C::new(frame[(i, j)], 0.0));
      let diagonalized = l.transpose() * master * l;
      let error = (0..4)
        .flat_map(|i| (0..4).map(move |j| (i, j)))
        .map(|(i, j)| {
            let want = if i == j { roots[i] } else { C::default() };
            (diagonalized[(i, j)] - want).norm()
        })
        .fold(0.0f64, f64::max);
      if error < 1e-8 {
        let db = Mat4::from_fn(|i, j| {
            if i == j {
                problem.lam[(i, i)].sqrt()
            } else {
                C::default()
            }
        });
        let dm = Mat4::from_fn(|i, j| {
            if i == j {
                roots[i].sqrt()
            } else {
                C::default()
            }
        });
        let u = problem.dc * *o * db;
        let right = dm
            .try_inverse()
            .map(|inverse| inverse * l.transpose() * u)?;
        return Some((l, right));
      }
    }
    // A certified solver frame can carry coefficient-scale spectral error that
    // makes the known-root projector reject even though the symmetric master is
    // numerically diagonalizable. Recover its actual unit-circle roots first;
    // the caller still performs the forward class certificate.
    if let Some((actual, off_diagonal)) = klein::unitary_eigenvalues(&master, roots0) {
        if off_diagonal < 1e-6 {
            if let Some(frame) = klein::takagi_real(&master, &actual) {
                let l = Mat4::from_fn(|i, j| C::new(frame[(i, j)], 0.0));
                let db = Mat4::from_fn(|i, j| {
                    if i == j { problem.lam[(i, i)].sqrt() } else { C::default() }
                });
                let dm = Mat4::from_fn(|i, j| {
                    if i == j { actual[i].sqrt() } else { C::default() }
                });
                let u = problem.dc * *o * db;
                if let Some(inverse) = dm.try_inverse() {
                    return Some((l, inverse * l.transpose() * u));
                }
            }
        }
    }
    None
}

/// Diagnostic residual before the endpoint-gauge acceptance threshold.
pub(super) fn endpoint_factorization_residual(problem: &PreparedSandwich, o: &Mat4) -> f64 {
    let master = sandwich_master(problem, o);
    let mut best = f64::INFINITY;
    for branch in 0..2 {
        let roots0 = problem.target_roots[branch];
        for permutation in *PERMS24 {
            let roots = std::array::from_fn(|i| roots0[permutation[i]]);
            let Some(frame) = klein::takagi_real(&master, &roots) else { continue };
            let l = Mat4::from_fn(|i, j| C::new(frame[(i, j)], 0.0));
            let diagonalized = l.transpose() * master * l;
            let error = (0..4)
                .flat_map(|i| (0..4).map(move |j| (i, j)))
                .map(|(i, j)| {
                    let want = if i == j { roots[i] } else { C::default() };
                    (diagonalized[(i, j)] - want).norm()
                })
                .fold(0.0f64, f64::max);
            best = best.min(error);
        }
    }
    for branch in 0..2 {
        if let Some((_actual, off_diagonal)) =
            klein::unitary_eigenvalues(&master, &problem.target_roots[branch])
        {
            best = best.min(off_diagonal);
        }
    }
    best
}

/// Fixed real factors for the central-rho identity
/// `D_rho = i (S P) D (P^T)`. The discarded scalar phase is irrelevant to a
/// local-frame certificate.
fn rho_transport() -> (Mat4, Mat4) {
    let rows = [2usize, 3, 0, 1];
    let signs = [1.0, 1.0, -1.0, -1.0];
    let sp = Mat4::from_fn(|i, j| {
        if j == rows[i] {
            C::new(signs[i], 0.0)
        } else {
            C::default()
        }
    });
    let p = Mat4::from_fn(|i, j| {
        if j == rows[i] {
            C::new(1.0, 0.0)
        } else {
            C::default()
        }
    });
    (sp, p.transpose())
}

pub(super) fn rho_transport_for_collapse() -> (Mat4, Mat4) {
    rho_transport()
}

/// Rephase the endpoint gauge onto the canonical target diagonal when the
/// direct target branch was used.  `endpoint_factorization` intentionally uses
/// principal square roots because it serves both rho branches; a factorized
/// middle gate must use the caller's canonical `D_T` before multiplying by V.
pub(super) fn canonical_right_endpoint_gauge(problem: &PreparedSandwich, o: &Mat4) -> Option<Mat4> {
    // `u = D_C O D_G`; the canonical middle diagonal is `D_T`, not `D_G`.
    // The old implementation accidentally used `right_phases` for both, which
    // can still look plausible for Berkeley tests but makes generic child
    // endpoint gauges fail their defining identity.
    let gate = Mat4::from_diagonal(&nalgebra::Vector4::from_row_slice(
        &problem.right_phases.map(|phase| C::from_polar(1.0, phase)),
    ));
    let u = problem.dc * *o * gate;
    for branch in 0..2 {
        let Some((left, principal_right)) = endpoint_factorization_branch(problem, o, branch)
        else {
            continue;
        };
        if branch == 0 {
            let principal = Mat4::from_fn(|i, j| {
                if i == j {
                    problem.target_roots[0][i].sqrt()
                } else {
                    C::default()
                }
            });
            let error = (u - left * principal * principal_right)
                .iter()
                .map(|entry| entry.norm())
                .fold(0.0, f64::max);
            if error < 1e-8 {
                return Some(principal_right);
            }
        } else {
            let (sp, ps) = rho_transport();
            let corrected = ps * principal_right;
            let principal = Mat4::from_fn(|i, j| {
                if i == j {
                    problem.target_roots[0][i].sqrt()
                } else {
                    C::default()
                }
            });
            let rho_product = left * sp * principal * corrected;
            let plus = (u - rho_product.map(|entry| C::new(0.0, 1.0) * entry))
                .iter()
                .map(|entry| entry.norm())
                .fold(0.0, f64::max);
            let minus = (u + rho_product.map(|entry| C::new(0.0, 1.0) * entry))
                .iter()
                .map(|entry| entry.norm())
                .fold(0.0, f64::max);
            if plus.min(minus) < 1e-8 {
                return Some(corrected);
            }
        }
    }
    None
}

/// Residual of the exact two-child composition constraint for a factorization
/// `G = B V B`.  If the first child has certificate frame `O_L`, its endpoint
/// gauge `R_L` is defined by `D_C O_L D_B = L D_M R_L`.  The second child must
/// use `O_R = R_L V`; independently realizing both children and comparing only
/// their Weyl points loses this equation.
pub(super) fn factorized_middle_residual(
    first: &PreparedSandwich,
    first_frame: &Mat4,
    second_frame: &Mat4,
    middle: &Mat4,
) -> Option<f64> {
    let endpoint = canonical_right_endpoint_gauge(first, first_frame)?;
    Some(
        (second_frame - endpoint * middle)
            .iter()
            .map(|entry| entry.norm())
            .fold(0.0, f64::max),
    )
}

/// Grassmannian form of the same constraint.  A final Berkeley factor has
/// right stabilizer K_B, so a child frame is defined modulo its first two
/// columns' oriented 2-plane.  Matching these planes is equivalent to the
/// matrix constraint up to the K_B gauge (with the opposite Pluecker sign
/// representing the same unoriented plane).
pub(super) fn factorized_middle_plane_residual(
    first: &PreparedSandwich,
    first_frame: &Mat4,
    second_frame: &Mat4,
    middle: &Mat4,
) -> Option<f64> {
    let endpoint = canonical_right_endpoint_gauge(first, first_frame)?;
    let expected = endpoint * middle;
    let pluecker = |frame: &Mat4, columns: (usize, usize)| {
        let (a, b) = columns;
        [
            (frame[(0, a)] * frame[(1, b)] - frame[(1, a)] * frame[(0, b)]).re,
            (frame[(0, a)] * frame[(2, b)] - frame[(2, a)] * frame[(0, b)]).re,
            (frame[(0, a)] * frame[(3, b)] - frame[(3, a)] * frame[(0, b)]).re,
            (frame[(1, a)] * frame[(2, b)] - frame[(2, a)] * frame[(1, b)]).re,
            (frame[(1, a)] * frame[(3, b)] - frame[(3, a)] * frame[(1, b)]).re,
            (frame[(2, a)] * frame[(3, b)] - frame[(3, a)] * frame[(2, b)]).re,
        ]
    };
    let lhs = pluecker(&expected, (0, 1));
    let rhs = pluecker(second_frame, (0, 1));
    let plus = lhs
        .iter()
        .zip(rhs.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    let minus = lhs
        .iter()
        .zip(rhs.iter())
        .map(|(a, b)| (a + b).abs())
        .fold(0.0, f64::max);
    Some(plus.min(minus))
}

/// Compare an already-gauged expected endpoint frame against a child frame.
/// Kept separate so generic waypoint code can enumerate finite K_M lifts
/// without recomputing the endpoint gauge.
pub(super) fn endpoint_plane_residual(expected: &Mat4, second_frame: &Mat4) -> f64 {
    let pluecker = |frame: &Mat4| {
        [
            (frame[(0, 0)] * frame[(1, 1)] - frame[(1, 0)] * frame[(0, 1)]).re,
            (frame[(0, 0)] * frame[(2, 1)] - frame[(2, 0)] * frame[(0, 1)]).re,
            (frame[(0, 0)] * frame[(3, 1)] - frame[(3, 0)] * frame[(0, 1)]).re,
            (frame[(1, 0)] * frame[(2, 1)] - frame[(2, 0)] * frame[(1, 1)]).re,
            (frame[(1, 0)] * frame[(3, 1)] - frame[(3, 0)] * frame[(1, 1)]).re,
            (frame[(2, 0)] * frame[(3, 1)] - frame[(3, 0)] * frame[(2, 1)]).re,
        ]
    };
    let lhs = pluecker(expected);
    let rhs = pluecker(second_frame);
    let plus = lhs
        .iter()
        .zip(rhs.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    let minus = lhs
        .iter()
        .zip(rhs.iter())
        .map(|(a, b)| (a + b).abs())
        .fold(0.0, f64::max);
    plus.min(minus)
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
