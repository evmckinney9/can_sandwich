//! Complex univariate polynomial helpers for the identity-construction rungs.
//! Convention: coefficients HIGHEST-DEGREE FIRST (matches np.poly / the
//! prototypes s86-s96). Small fixed degrees (<= 16); no allocation cleverness.
use nalgebra::Complex;

pub type C = Complex<f64>;

/// All complex roots via the companion matrix (faer eigenvalues).
/// Leading near-zero coefficients are trimmed relative to the max magnitude,
/// and the polynomial is normalized (scale-invariant) for conditioning: the
/// edge polynomials carry factors like P²w_d²zB² that otherwise degrade the
/// companion eigensolve (cost: a missed production row, 104160).
pub fn roots(co: &[C]) -> Vec<C> {
    let maxc = co.iter().map(|c| c.norm()).fold(0.0f64, f64::max);
    if maxc == 0.0 {
        return vec![];
    }
    let normed: Vec<C> = co.iter().map(|&c| c / maxc).collect();
    let co = &normed[..];
    let mut start = 0usize;
    while start < co.len() && co[start].norm() < 1e-13 {
        start += 1;
    }
    let co = &co[start..];
    let n = co.len().saturating_sub(1);
    if n == 0 {
        return vec![];
    }
    let lead = co[0];
    let comp = faer::Mat::<C>::from_fn(n, n, |i, j| {
        if j == n - 1 {
            -co[n - i] / lead
        } else if i == j + 1 {
            C::new(1.0, 0.0)
        } else {
            C::default()
        }
    });
    match comp.eigenvalues() {
        Ok(ev) => ev,
        Err(_) => vec![],
    }
}
