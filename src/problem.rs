//! Canonical input and output contracts for one depth-two sandwich.
//!
//! Every realization formula must consume [`PreparedSandwich`].  Every fast
//! residual that assumes a real orthogonal frame must first use
//! [`frame_metrics`].  Keeping these contracts separate from the chart atlas
//! prevents individual rungs from inventing their own Weyl lift, target
//! branch, or notion of an acceptable frame.

use super::{eigphases, esym4, rho_weyl, weyl_from_monodromy, Mat4, C};

/// Exact spectral partition used by the dispatch spine.
///
/// This is deliberately stricter than conditioning clusters inside confluent
/// formulas: a merely close pair is generic algebra and must not be collapsed
/// into a repeated eigenvalue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SpectrumKind {
    Distinct,
    Pair211,
    Pair22,
    Triple31,
    Scalar4,
}

/// Independent distance-to-stratum label. This does not alter the historical
/// dispatch partition; it prevents near multiplicities from being advertised
/// as exact closed-form inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum SpectrumProximity {
    Exact,
    Near,
    Distinct,
}

impl SpectrumKind {
    #[inline]
    pub(super) fn is_repeated(self) -> bool {
        self != Self::Distinct
    }

    #[inline]
    pub(super) fn is_rank_two(self) -> bool {
        matches!(self, Self::Pair211 | Self::Pair22)
    }
}

pub(super) fn spectrum_kind(s: &[C; 4]) -> SpectrumKind {
    let mut used = [false; 4];
    let mut counts = [0usize; 4];
    let mut groups = 0usize;
    for i in 0..4 {
        if used[i] {
            continue;
        }
        let mut count = 0usize;
        for j in i..4 {
            if !used[j] && (s[j] - s[i]).norm_sqr() <= f64::EPSILON {
                used[j] = true;
                count += 1;
            }
        }
        counts[groups] = count;
        groups += 1;
    }
    counts[..groups].sort_unstable_by(|a, b| b.cmp(a));
    match &counts[..groups] {
        [4] => SpectrumKind::Scalar4,
        [3, 1] => SpectrumKind::Triple31,
        [2, 2] => SpectrumKind::Pair22,
        [2, 1, 1] => SpectrumKind::Pair211,
        _ => SpectrumKind::Distinct,
    }
}

pub(super) fn spectrum_proximity(s: &[C; 4]) -> SpectrumProximity {
    let mut minimum = f64::INFINITY;
    for i in 0..4 {
        for j in (i + 1)..4 {
            minimum = minimum.min((s[i] - s[j]).norm());
        }
    }
    if minimum <= 64.0 * f64::EPSILON {
        SpectrumProximity::Exact
    } else if minimum <= 1.0e-7 {
        SpectrumProximity::Near
    } else {
        SpectrumProximity::Distinct
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct StratumSignature {
    pub(super) c: SpectrumKind,
    pub(super) g: SpectrumKind,
    pub(super) target: [SpectrumKind; 2],
    pub(super) c_proximity: SpectrumProximity,
    pub(super) g_proximity: SpectrumProximity,
    pub(super) target_proximity: [SpectrumProximity; 2],
    /// A repeated value in the routed product table `c_i g_j`, even when both
    /// input spectra themselves are simple.  This is a separate confluence
    /// mechanism from input/target spectral multiplicity.
    pub(super) routed_collision: bool,
}

impl StratumSignature {
    pub(super) fn new(c: &[C; 4], g: &[C; 4], target: &[[C; 4]; 2]) -> Self {
        Self {
            c: spectrum_kind(c),
            g: spectrum_kind(g),
            target: std::array::from_fn(|branch| spectrum_kind(&target[branch])),
            c_proximity: spectrum_proximity(c),
            g_proximity: spectrum_proximity(g),
            target_proximity: std::array::from_fn(|branch| spectrum_proximity(&target[branch])),
            routed_collision: routed_product_collision(c, g),
        }
    }

    #[inline]
    pub(super) fn has_confluence(self) -> bool {
        self.c.is_repeated()
            || self.g.is_repeated()
            || self.target.iter().any(|kind| kind.is_repeated())
    }

}

/// Detect exact routed-product collisions without collapsing near collisions
/// into a repeated spectrum.  Such collisions are the natural next branch for
/// simple/simple inputs: `c_i g_j = c_k g_l` can lower the rank of the
/// characteristic map even though `c` and `g` are individually distinct.
fn routed_product_collision(c: &[C; 4], g: &[C; 4]) -> bool {
    let products: [C; 16] = std::array::from_fn(|n| c[n / 4] * g[n % 4]);
    for i in 0..products.len() {
        for j in (i + 1)..products.len() {
            if (products[i] - products[j]).norm_sqr() <= 16.0 * f64::EPSILON {
                return true;
            }
        }
    }
    false
}

/// Canonical spectral form of one depth-two sandwich.
///
/// The Weyl fold, central `rho` lift, unit-circle roots, characteristic
/// coefficients, and diagonal master-object factors are constructed exactly
/// once here.
#[derive(Clone, Copy)]
pub(super) struct PreparedSandwich {
    pub(super) left_phases: [f64; 4],
    pub(super) right_phases: [f64; 4],
    pub(super) left: [C; 4],
    pub(super) right: [C; 4],
    pub(super) target_roots: [[C; 4]; 2],
    pub(super) targets: [[C; 4]; 2],
    pub(super) routed: [[C; 4]; 4],
    pub(super) dc: Mat4,
    pub(super) lam: Mat4,
    pub(super) strata: StratumSignature,
}

impl PreparedSandwich {
    pub(super) fn new(c: [f64; 3], g: [f64; 3], t: [f64; 3]) -> Self {
        let left_phases = eigphases(weyl_from_monodromy(c));
        let right_phases = eigphases(weyl_from_monodromy(g));
        let target_weyl = weyl_from_monodromy(t);
        let target_phases = [eigphases(target_weyl), eigphases(rho_weyl(target_weyl))];

        let left = left_phases.map(|phase| C::from_polar(1.0, 2.0 * phase));
        let right = right_phases.map(|phase| C::from_polar(1.0, 2.0 * phase));
        let target0 = target_phases[0].map(|phase| C::from_polar(1.0, 2.0 * phase));
        // The central rho reflection is exact on roots: negate and swap the
        // two pairs. Its odd elementary coefficients change sign.
        let target_roots = [
            target0,
            [-target0[2], -target0[3], -target0[0], -target0[1]],
        ];
        let target0_coefficients = esym4(target0);
        let targets = [
            target0_coefficients,
            [
                -target0_coefficients[0],
                target0_coefficients[1],
                -target0_coefficients[2],
                target0_coefficients[3],
            ],
        ];
        let routed = std::array::from_fn(|i| std::array::from_fn(|j| left[i] * right[j]));
        let dc = Mat4::from_diagonal(&nalgebra::Vector4::from_row_slice(
            &left_phases.map(|phase| C::from_polar(1.0, phase)),
        ));
        let lam = Mat4::from_diagonal(&nalgebra::Vector4::from_row_slice(&right));
        let strata = StratumSignature::new(&left, &right, &target_roots);

        Self {
            left_phases,
            right_phases,
            left,
            right,
            target_roots,
            targets,
            routed,
            dc,
            lam,
            strata,
        }
    }
}

/// Numerical certificate required before treating a matrix as a real frame.
///
/// `compound_residual` intentionally reads only real entries and is sound only
/// after this certificate bounds the discarded imaginary part and the complete
/// Gram defect.
#[derive(Debug, Clone, Copy)]
pub(super) struct FrameMetrics {
    pub determinant: f64,
    pub gram: f64,
    pub imaginary: f64,
}

impl FrameMetrics {
    /// Whether the candidate is a real orthogonal frame to the requested
    /// absolute tolerance.  The determinant magnitude is part of the same
    /// contract; callers may repair its sign only after this test passes.
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
///
/// Right-multiplying by a diagonal sign leaves `O Lambda O^T` unchanged because
/// `Lambda` is diagonal, so this repairs orientation without changing the
/// spectral certificate.
#[inline]
pub(super) fn orient_so4(mut frame: Mat4) -> Mat4 {
    if frame.determinant().re < 0.0 {
        for row in 0..4 {
            frame[(row, 0)] = -frame[(row, 0)];
        }
    }
    frame
}

#[cfg(test)]
mod tests {
    use super::{routed_product_collision, spectrum_proximity, C, SpectrumProximity};

    #[test]
    fn detects_collision_with_simple_inputs() {
        let c = [
            C::new(1.0, 0.0),
            C::new(0.0, 1.0),
            C::new(-1.0, 0.0),
            C::new(0.0, -1.0),
        ];
        let g = c;
        assert!(routed_product_collision(&c, &g));
    }

    #[test]
    fn does_not_collapse_near_collision() {
        let c = [
            C::new(1.0, 0.0),
            C::new(0.0, 1.0),
            C::new(-1.0, 0.0),
            C::new(0.0, -1.0),
        ];
        let g = [
            C::from_polar(1.0, 0.13),
            C::from_polar(1.0, 1.71),
            C::from_polar(1.0, 3.29),
            C::from_polar(1.0, 4.91),
        ];
        assert!(!routed_product_collision(&c, &g));
    }

    #[test]
    fn separates_exact_near_and_distinct_spectra() {
        let exact = [C::new(1.0, 0.0); 4];
        assert_eq!(spectrum_proximity(&exact), SpectrumProximity::Exact);
        let near = [
            C::from_polar(1.0, 0.0),
            C::from_polar(1.0, 1.0e-8),
            C::from_polar(1.0, 2.0),
            C::from_polar(1.0, 4.0),
        ];
        assert_eq!(spectrum_proximity(&near), SpectrumProximity::Near);
        let distinct = [
            C::from_polar(1.0, 0.0),
            C::from_polar(1.0, 0.2),
            C::from_polar(1.0, 2.0),
            C::from_polar(1.0, 4.0),
        ];
        assert_eq!(spectrum_proximity(&distinct), SpectrumProximity::Distinct);
    }
}
