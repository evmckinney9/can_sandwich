//! The production certificate: rootwise acceptance of a candidate frame against the
//! target roots, and the sandwich master it is evaluated on.
use super::{ACCEPT, C, FRAME_ACCEPT, Mat4, Problem, Rung, Solution, compound_residual, klein};

#[cfg(feature = "diagnostics")]
use super::PERMS24;

/// Numerical certificate required before treating a matrix as a real frame.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FrameMetrics {
    pub determinant: f64,
    pub gram: f64,
    pub imaginary: f64,
}

impl FrameMetrics {
    #[inline]
    pub(crate) fn within(self, tolerance: f64) -> bool {
        self.gram <= tolerance
            && self.imaginary <= tolerance
            && (self.determinant.abs() - 1.0).abs() <= tolerance
    }
}

#[inline]
pub(crate) fn frame_metrics(frame: &Mat4) -> Option<FrameMetrics> {
    if frame.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
        return None;
    }
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
pub(crate) fn orient_so4(mut frame: Mat4) -> Mat4 {
    if frame.determinant().re < 0.0 {
        for row in 0..4 {
            frame[(row, 0)] = -frame[(row, 0)];
        }
    }
    frame
}

#[inline]
pub(crate) fn apply_transpose(frame: Mat4, transpose: bool) -> Mat4 {
    if transpose { frame.transpose() } else { frame }
}

/// Shared pre-certificate for algebraic candidate frames.
///
/// Individual realization routes may use different equations to generate a
/// frame, but they all need the same final local operation: reject non-real or
/// non-orthogonal frames, normalize orientation, and evaluate the forward
/// spectral residual.  Keeping that operation here prevents each route from
/// quietly growing its own tolerance or orientation convention.
pub(crate) fn certify_frame_candidate(
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
pub(crate) fn certify_frame_candidate_with_limit(
    mut frame: Mat4,
    dc: &Mat4,
    lam: &Mat4,
    target: &[C; 4],
    limit: f64,
) -> Option<(Mat4, f64)> {
    let metrics = frame_metrics(&frame)?;
    if !metrics.within(FRAME_ACCEPT) {
        return None;
    }
    frame = orient_so4(frame);
    let residual = compound_residual(dc, lam, &frame, target);
    (residual <= limit).then_some((frame, residual))
}

/// The same pre-certificate for a finite set of target lifts.  The minimum is
/// taken only after the frame contract passes; callers must not use a target
/// coefficient proxy as a substitute for this forward check.
pub(crate) fn certify_frame_against_targets(
    frame: Mat4,
    dc: &Mat4,
    lam: &Mat4,
    targets: &[[C; 4]],
) -> Option<(Mat4, f64)> {
    let metrics = frame_metrics(&frame)?;
    if !metrics.within(FRAME_ACCEPT) {
        return None;
    }
    let frame = orient_so4(frame);
    let residual = targets
        .iter()
        .map(|target| compound_residual(dc, lam, &frame, target))
        .fold(f64::INFINITY, f64::min);
    (residual <= ACCEPT).then_some((frame, residual))
}

/// Verify the real frame once and retain its spectral decomposition.
pub(crate) fn compiler_solution(
    problem: &Problem,
    o: Mat4,
    rung: Rung,
    residual: f64,
) -> Option<Solution> {
    if !residual.is_finite() || !frame_metrics(&o)?.within(FRAME_ACCEPT) {
        return None;
    }
    let o = orient_so4(o);
    let certify = |o: Mat4| {
        let real = o.map(|z| z.re);
        let state = crate::spectral::verify(problem, &real)?;
        Some(Solution {
            o: real.map(|x| C::new(x, 0.0)),
            rung,
            residual: state.error,
            verified: Some(state),
        })
    };
    if let Some(verified) = certify(o) {
        return Some(verified);
    }
    // Near repeated target roots, coefficient-based construction can split a
    // cluster. Retarget in the same eigenbasis, then reconstruct the gate frame.
    let roots = problem.target_roots[0];
    if !(0..4).any(|i| (i + 1..4).any(|j| (roots[i] - roots[j]).norm() < 1e-3)) {
        return None;
    }
    let master = sandwich_master(problem, &o);
    for roots in &problem.target_roots {
        let Some((target_master, _)) = klein::retarget_symmetric(&master, roots) else {
            continue;
        };
        let inverse_dc = problem.dc.map(|z| z.conj());
        let peeled = inverse_dc * target_master * inverse_dc;
        let Some(real) = klein::takagi_real(&peeled, &problem.right) else {
            continue;
        };
        let o = orient_so4(real.map(|x| C::new(x, 0.0)));
        if let Some(verified) = certify(o) {
            return Some(verified);
        }
    }
    None
}

/// Recover both endpoint frames in
/// `D_C O D_G = L D_T R`.  The left frame is needed when a factorized
/// `B V B` witness is collapsed back to the original canonical gate `G`.
#[cfg(feature = "diagnostics")]
pub(crate) fn endpoint_factorization(problem: &Problem, o: &Mat4) -> Option<(Mat4, Mat4)> {
    endpoint_factorization_branch(problem, o, 0)
        .or_else(|| endpoint_factorization_branch(problem, o, 1))
}

/// Endpoint factorization on one explicit target representative branch.
#[cfg(feature = "diagnostics")]
pub(crate) fn endpoint_factorization_branch(
    problem: &Problem,
    o: &Mat4,
    branch: usize,
) -> Option<(Mat4, Mat4)> {
    let master = sandwich_master(problem, o);
    let roots0 = problem.target_roots.get(branch)?;
    for permutation in *PERMS24 {
        let roots = std::array::from_fn(|i| roots0[permutation[i]]);
        let Some(frame) = klein::takagi_real(&master, &roots) else {
            continue;
        };
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
    if let Some((actual, off_diagonal)) = klein::unitary_eigenvalues(&master, roots0)
        && off_diagonal < 1e-6
        && let Some(frame) = klein::takagi_real(&master, &actual)
    {
        let l = Mat4::from_fn(|i, j| C::new(frame[(i, j)], 0.0));
        let db = Mat4::from_fn(|i, j| {
            if i == j {
                problem.lam[(i, i)].sqrt()
            } else {
                C::default()
            }
        });
        let dm = Mat4::from_fn(|i, j| {
            if i == j {
                actual[i].sqrt()
            } else {
                C::default()
            }
        });
        let u = problem.dc * *o * db;
        if let Some(inverse) = dm.try_inverse() {
            return Some((l, inverse * l.transpose() * u));
        }
    }
    None
}

/// Diagnostic residual before the endpoint-gauge acceptance threshold.
#[cfg(feature = "diagnostics")]
pub(crate) fn endpoint_factorization_residual(problem: &Problem, o: &Mat4) -> f64 {
    let master = sandwich_master(problem, o);
    let mut best = f64::INFINITY;
    for branch in 0..2 {
        let roots0 = problem.target_roots[branch];
        for permutation in *PERMS24 {
            let roots = std::array::from_fn(|i| roots0[permutation[i]]);
            let Some(frame) = klein::takagi_real(&master, &roots) else {
                continue;
            };
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
#[cfg(feature = "diagnostics")]
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

#[cfg(feature = "diagnostics")]
pub(crate) fn rho_transport_for_collapse() -> (Mat4, Mat4) {
    rho_transport()
}

/// Rephase the endpoint gauge onto the canonical target diagonal when the
/// direct target branch was used.  `endpoint_factorization` intentionally uses
/// principal square roots because it serves both rho branches; a factorized
/// middle gate must use the caller's canonical `D_T` before multiplying by V.
#[cfg(feature = "diagnostics")]
pub(crate) fn canonical_right_endpoint_gauge(problem: &Problem, o: &Mat4) -> Option<Mat4> {
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

/// Compare an already-gauged expected endpoint frame against a child frame.
/// Kept separate so generic waypoint code can enumerate finite K_M lifts
/// without recomputing the endpoint gauge.
#[cfg(feature = "diagnostics")]
pub(crate) fn endpoint_plane_residual(expected: &Mat4, second_frame: &Mat4) -> f64 {
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
pub(crate) fn sandwich_master(problem: &Problem, o: &Mat4) -> Mat4 {
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
