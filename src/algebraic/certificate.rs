//! The production certificate: rootwise acceptance of a candidate frame against the
//! target roots, and the sandwich master it is evaluated on.
use super::{ACCEPT, C, FRAME_ACCEPT, Mat4, Problem, Rung, Solution, compound_residual, klein};

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
            o: real,
            rung,
            residual: state.error,
            state,
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
