//! ARCHIVED PROTOTYPE: not compiled into the production solver.
//!
//! Mumford-form chart-free generic-interior rung (additive prototype).
//!
//! Implements the three-level secular-weight chain (s234 / ledger blocks
//! "THE SECULAR-WEIGHT FORM" and "THE EXPLICIT WITNESS-COORDINATE CHAIN").
//! Not wired into any production tier; call `mumford_recover` directly.
//! Reference: .claude/scripts/research/prototypes/extremal_section/s234_interior_piece_prototype.py
//!
//! Zero-strata rows (~63%+ of production corpus) are NOT this piece's job:
//! the generic formulas degenerate there BY DESIGN (mu hits delta = forced-pin
//! regime). Route by decline (None return), not by prediction.

use crate::can_sandwich::{compound_residual, esym4, Mat4, Rung, Solution, ACCEPT, C};

const TOL: f64 = 1e-7;
const REALITY: f64 = 1e-6;

fn is_finite_c(x: C) -> bool {
    x.re.is_finite() && x.im.is_finite()
}

/// Bilinear dot product: sum x_k * y_k (no conjugation).
fn bdot(x: &[C; 4], y: &[C; 4]) -> C {
    x[0] * y[0] + x[1] * y[1] + x[2] * y[2] + x[3] * y[3]
}

/// Product over all roots.
fn chi(z: C, roots: &[C; 4]) -> C {
    roots.iter().fold(C::new(1.0, 0.0), |acc, &r| acc * (z - r))
}

/// Derivative-product excluding the root nearest z.
fn chip(z: C, roots: &[C; 4]) -> C {
    roots
        .iter()
        .filter(|&&r| (z - r).norm() > 1e-11)
        .fold(C::new(1.0, 0.0), |acc, &r| acc * (z - r))
}

/// All 8 sign-triples (±1, ±1, ±1) for the three free sign positions.
fn sign_triples() -> [[f64; 3]; 8] {
    let mut out = [[1.0f64; 3]; 8];
    for i in 0..8 {
        for b in 0..3 {
            if (i >> b) & 1 == 1 {
                out[i][b] = -1.0;
            }
        }
    }
    out
}

fn rdot(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

/// Find the 4th real column orthogonal to v, u, s via modified Gram-Schmidt.
/// Picks the standard-basis vector with largest residual after projection.
fn fourth_column(v: &[f64; 4], u: &[f64; 4], s: &[f64; 4]) -> Option<[f64; 4]> {
    let norm4 = |x: &[f64; 4]| rdot(x, x).sqrt();
    let proj_sub = |x: &mut [f64; 4], q: &[f64; 4]| {
        let d = rdot(x, q);
        for k in 0..4 {
            x[k] -= d * q[k];
        }
    };

    let mut q0 = *v;
    let n0 = norm4(&q0);
    if n0 < 1e-10 {
        return None;
    }
    for k in 0..4 {
        q0[k] /= n0;
    }

    let mut q1 = *u;
    proj_sub(&mut q1, &q0);
    let n1 = norm4(&q1);
    let q1 = if n1 > 1e-10 {
        let mut q = q1;
        for k in 0..4 {
            q[k] /= n1;
        }
        q
    } else {
        [0.0; 4]
    };

    let mut q2 = *s;
    proj_sub(&mut q2, &q0);
    proj_sub(&mut q2, &q1);
    let n2 = norm4(&q2);
    let q2 = if n2 > 1e-10 {
        let mut q = q2;
        for k in 0..4 {
            q[k] /= n2;
        }
        q
    } else {
        [0.0; 4]
    };

    let mut best = [0.0f64; 4];
    let mut best_n = 0.0;
    for e in 0..4 {
        let mut x = [0.0f64; 4];
        x[e] = 1.0;
        proj_sub(&mut x, &q0);
        proj_sub(&mut x, &q1);
        proj_sub(&mut x, &q2);
        let n = norm4(&x);
        if n > best_n {
            best_n = n;
            best = x;
        }
    }
    if best_n < 1e-10 {
        return None;
    }
    for k in 0..4 {
        best[k] /= best_n;
    }
    Some(best)
}

/// Chart-free generic-interior rung via three-level secular-weight chain.
///
/// `gate[k] = exp(2i·ep[k])`, `a[k] = exp(i·eb[k])`, `w` = target eigenvalues,
/// `mu`/`nu` = level-1/level-2 witness spectra. Returns `None` on zero-strata rows
/// (Cauchy denominators blow up by design; ~63%+ of production corpus).
pub fn mumford_recover(
    gate: &[C; 4],
    a: &[C; 4],
    w: &[C; 4],
    mu: &[C; 4],
    nu: &[C; 4],
) -> Option<Solution> {
    let c0 = gate[3];
    let rho = [gate[0] - gate[3], gate[1] - gate[3], gate[2] - gate[3]];
    let d2: [C; 4] = std::array::from_fn(|k| a[k] * a[k]);
    let delta: [C; 4] = std::array::from_fn(|k| c0 * d2[k]);
    let dc = Mat4::from_diagonal(&nalgebra::Vector4::from_row_slice(a));
    let lam = Mat4::from_diagonal(&nalgebra::Vector4::from_row_slice(gate));
    let target = esym4(*w);

    // Level-1 secular weights: P[k] = -chi_mu(delta[k]) / (rho[0] * chip_delta(delta[k])).
    let p: [C; 4] = std::array::from_fn(|k| {
        let denom = rho[0] * chip(delta[k], &delta);
        if denom.norm() < 1e-20 {
            C::new(f64::NAN, 0.0)
        } else {
            -chi(delta[k], mu) / denom
        }
    });
    // v2[k] = P[k] / d2[k] must be real (reality gate on v).
    let v2: [C; 4] = std::array::from_fn(|k| p[k] / d2[k]);
    if v2.iter().any(|x| !is_finite_c(*x) || x.im.abs() > REALITY) {
        return None;
    }
    let vabs: [f64; 4] = std::array::from_fn(|k| v2[k].re.abs().sqrt());

    // Level-2 secular weights: R[m] = -chi_nu(mu[m]) / (rho[1] * chip_mu(mu[m])).
    let r_sec: [C; 4] = std::array::from_fn(|m| {
        let denom = rho[1] * chip(mu[m], mu);
        if denom.norm() < 1e-20 {
            C::new(f64::NAN, 0.0)
        } else {
            -chi(mu[m], nu) / denom
        }
    });
    if r_sec.iter().any(|x| !is_finite_c(*x)) {
        return None;
    }

    // Level-3 secular weights: r[m] = -chi_w(nu[m]) / (rho[2] * chip_nu(nu[m])).
    let r_lev3: [C; 4] = std::array::from_fn(|m| {
        let denom = rho[2] * chip(nu[m], nu);
        if denom.norm() < 1e-20 {
            C::new(f64::NAN, 0.0)
        } else {
            -chi(nu[m], w) / denom
        }
    });
    if r_lev3.iter().any(|x| !is_finite_c(*x)) {
        return None;
    }

    let sq_r2: [C; 4] = std::array::from_fn(|m| r_sec[m].sqrt());
    let sq_r3: [C; 4] = std::array::from_fn(|m| r_lev3[m].sqrt());

    let mut best: Option<(Mat4, f64)> = None;

    // Outer loop: 8 sign-triples for v (first component fixed +1, 3 free).
    for sv in sign_triples() {
        let v: [f64; 4] = [vabs[0], sv[0] * vabs[1], sv[1] * vabs[2], sv[2] * vabs[3]];
        let dv: [C; 4] = std::array::from_fn(|k| a[k] * v[k]);

        // Cauchy frame Phi: column m = Dv/(delta - mu[m]), bilinear-normalized.
        // phi[m][k] = row k of column m.
        let mut phi = [[C::new(0.0, 0.0); 4]; 4];
        let mut phi_ok = true;
        for m in 0..4 {
            let col: [C; 4] = std::array::from_fn(|k| dv[k] / (delta[k] - mu[m]));
            if col.iter().any(|x| !is_finite_c(*x)) {
                phi_ok = false;
                break;
            }
            let n2 = bdot(&col, &col);
            if n2.norm() < 1e-30 {
                phi_ok = false;
                break;
            }
            let n = n2.sqrt();
            for k in 0..4 {
                phi[m][k] = col[k] / n;
            }
        }
        if !phi_ok {
            continue;
        }

        // V1[m] = Phi^T @ (Dc^{-1} v) -- bilinear (Phi.T, not Phi^H).
        let dc_inv_v: [C; 4] = std::array::from_fn(|k| C::new(v[k], 0.0) / a[k]);
        let v1: [C; 4] = std::array::from_fn(|m| (0..4).map(|k| phi[m][k] * dc_inv_v[k]).sum());

        // Middle loop: 8 sign-triples for t2 (weights on mu eigenvectors).
        for su in sign_triples() {
            let t2: [C; 4] = [
                sq_r2[0],
                C::new(su[0], 0.0) * sq_r2[1],
                C::new(su[1], 0.0) * sq_r2[2],
                C::new(su[2], 0.0) * sq_r2[3],
            ];
            // Closure C2: sum_m t2[m] * V1[m] = 0.
            let c2: C = (0..4).map(|m| t2[m] * v1[m]).sum();
            if c2.norm() > TOL {
                continue;
            }

            // Du = Phi @ t2; then u = Dc^{-1} Du must be real.
            let du: [C; 4] = std::array::from_fn(|k| (0..4).map(|m| phi[m][k] * t2[m]).sum());
            let u_c: [C; 4] = std::array::from_fn(|k| du[k] / a[k]);
            if u_c.iter().any(|x| !is_finite_c(*x) || x.im.abs() > REALITY) {
                continue;
            }
            let u: [f64; 4] = std::array::from_fn(|k| u_c[k].re);

            // Woodbury two-step Sherman-Morrison for Psi columns.
            // M2 = diag(delta) + rho[0]*Dv*Dv^T + rho[1]*Du*Du^T.
            // Psi[:,m] = (M2 - nu[m]*I)^{-1} Du, bilinear-normalized.
            let mut psi = [[C::new(0.0, 0.0); 4]; 4];
            let mut psi_ok = true;
            for m in 0..4 {
                let l: [C; 4] = std::array::from_fn(|k| delta[k] - nu[m]);
                if l.iter().any(|x| x.norm() < 1e-15) {
                    psi_ok = false;
                    break;
                }
                // Step 1: invert (diag(L) + rho[0]*Dv*Dv^T) applied to Du.
                let sigma: C = (0..4).map(|k| dv[k] * dv[k] / l[k]).sum();
                let tau: C = (0..4).map(|k| dv[k] * du[k] / l[k]).sum();
                let da = C::new(1.0, 0.0) + rho[0] * sigma;
                if da.norm() < 1e-20 {
                    psi_ok = false;
                    break;
                }
                let a_inv_du: [C; 4] =
                    std::array::from_fn(|k| du[k] / l[k] - rho[0] * tau / da * (dv[k] / l[k]));
                // Step 2: add rho[1]*Du*Du^T rank-1 correction.
                let xi: C = (0..4).map(|k| du[k] * du[k] / l[k]).sum();
                let beta = xi - rho[0] * tau * tau / da;
                let db = C::new(1.0, 0.0) + rho[1] * beta;
                if db.norm() < 1e-20 {
                    psi_ok = false;
                    break;
                }
                let x: [C; 4] = std::array::from_fn(|k| a_inv_du[k] / db);
                if x.iter().any(|v| !is_finite_c(*v)) {
                    psi_ok = false;
                    break;
                }
                let n2 = bdot(&x, &x);
                if n2.norm() < 1e-30 {
                    psi_ok = false;
                    break;
                }
                let n = n2.sqrt();
                for k in 0..4 {
                    psi[m][k] = x[k] / n;
                }
            }
            if !psi_ok {
                continue;
            }

            // Vt[m] = Psi^T @ (Dc^{-1} v); Ut[m] = Psi^T @ (Dc^{-1} u).
            let vt: [C; 4] = std::array::from_fn(|m| (0..4).map(|k| psi[m][k] * dc_inv_v[k]).sum());
            let dc_inv_u: [C; 4] = std::array::from_fn(|k| C::new(u[k], 0.0) / a[k]);
            let ut: [C; 4] = std::array::from_fn(|m| (0..4).map(|k| psi[m][k] * dc_inv_u[k]).sum());

            // Inner loop: 8 sign-triples for t3.
            for ss in sign_triples() {
                let t3: [C; 4] = [
                    sq_r3[0],
                    C::new(ss[0], 0.0) * sq_r3[1],
                    C::new(ss[1], 0.0) * sq_r3[2],
                    C::new(ss[2], 0.0) * sq_r3[3],
                ];
                // Closure C3: sum_m t3[m]*Vt[m] = 0 AND sum_m t3[m]*Ut[m] = 0.
                let c3v: C = (0..4).map(|m| t3[m] * vt[m]).sum();
                if c3v.norm() > TOL {
                    continue;
                }
                let c3u: C = (0..4).map(|m| t3[m] * ut[m]).sum();
                if c3u.norm() > TOL {
                    continue;
                }

                // s = Dc^{-1} (Psi @ t3) must be real.
                let ds: [C; 4] = std::array::from_fn(|k| (0..4).map(|m| psi[m][k] * t3[m]).sum());
                let s_c: [C; 4] = std::array::from_fn(|k| ds[k] / a[k]);
                if s_c.iter().any(|x| !is_finite_c(*x) || x.im.abs() > REALITY) {
                    continue;
                }
                let s: [f64; 4] = std::array::from_fn(|k| s_c[k].re);

                // Frame completion: 4th column orthogonal to v, u, s.
                let Some(x4) = fourth_column(&v, &u, &s) else {
                    continue;
                };

                // Assemble O = [v | u | s | x4] and fix det sign.
                let mut o = Mat4::zeros();
                for i in 0..4 {
                    o[(i, 0)] = C::new(v[i], 0.0);
                    o[(i, 1)] = C::new(u[i], 0.0);
                    o[(i, 2)] = C::new(s[i], 0.0);
                    o[(i, 3)] = C::new(x4[i], 0.0);
                }
                // orient_so4 is private; inline: negate col 3 if det < 0.
                if o.map(|z| z.re).determinant() < 0.0 {
                    for i in 0..4 {
                        o[(i, 3)] = -o[(i, 3)];
                    }
                }

                let r = compound_residual(&dc, &lam, &o, &target);
                if r < ACCEPT {
                    match &best {
                        None => best = Some((o, r)),
                        Some((_, br)) if r < *br => best = Some((o, r)),
                        _ => {}
                    }
                }
            }
        }
    }

    best.map(|(o, residual)| Solution {
        o,
        rung: Rung::Interior,
        residual,
    })
}

/// Circular interlacing check: do the 8 angles (4 nu + 4 w) alternate type
/// on the circle? Normalises to [0, 2*pi) and checks the sorted sequence.
fn interlaces_circular(nu_angles: &[f64; 4], w_angles: &[f64; 4]) -> bool {
    let tau = 2.0 * std::f64::consts::PI;
    let norm = |x: f64| x.rem_euclid(tau);
    // 0 = nu, 1 = w.
    let mut pts: Vec<(f64, u8)> = nu_angles
        .iter()
        .map(|&a| (norm(a), 0u8))
        .chain(w_angles.iter().map(|&a| (norm(a), 1u8)))
        .collect();
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    (0..8).all(|i| pts[i].1 != pts[(i + 1) % 8].1)
}

/// Deterministic nu-picker for the Mumford-form interior rung.
///
/// Proportional placement: nu[m] = exp(i*(phi[idx] + s*arc[idx])) where
/// idx = (m+offset)%4, arc[m] is the m-th circular arc between consecutive
/// w-eigenphases, and s ∈ (0,1) is the unique fractional position satisfying
/// the det-pin constraint `prod(nu) = prod(delta)*gate[0]*gate[1]/gate[3]^2`.
///
/// Why this is always solvable: sum(phi + s*arc) = sum(phi) + s*2*pi spans a
/// full period as s ∈ (0,1), so there is exactly one s that matches
/// arg(P_target) mod 2*pi. Circular interlacing (nu strictly between
/// consecutive w-phases) holds for all s ∈ (0,1) by construction.
///
/// `offset` in {0,1}: cyclic rotation of the arc assignment. Same nu SET and
/// same product (det-pin preserved), different nu VECTOR -- gives the companion
/// solve a genuinely different candidate for the retry.
pub fn nu_pick(gate: &[C; 4], a: &[C; 4], w: &[C; 4], offset: usize) -> [C; 4] {
    let tau = 2.0 * std::f64::consts::PI;

    // Sort w eigenphases into [0, 2*pi).
    let mut phi: [f64; 4] = std::array::from_fn(|k| w[k].arg().rem_euclid(tau));
    phi.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Arc lengths; last arc wraps [phi[3], phi[0]+2*pi].
    let arc: [f64; 4] = std::array::from_fn(|m| {
        let phi_next = if m < 3 { phi[m + 1] } else { phi[0] + tau };
        phi_next - phi[m]
    });

    // Det-pin target: prod(nu) = prod(delta)*gate[0]*gate[1]/gate[3]^2.
    // prod(delta) = prod(gate[3]*a[k]^2) = gate[3]^4 * prod(a[k])^2.
    let p_delta: C = a
        .iter()
        .fold(C::new(1.0, 0.0), |acc, &ak| acc * gate[3] * ak * ak);
    let p_target: C = p_delta * gate[0] * gate[1] / (gate[3] * gate[3]);

    // Unique s in [0,1) s.t. sum(phi) + s*tau = arg(P_target) + 2*pi*k.
    // s = (arg(P_target) - sum(phi)) rem_euclid tau / tau.
    let s_min: f64 = phi.iter().sum();
    let s = (p_target.arg() - s_min).rem_euclid(tau) / tau;

    // nu[m] = exp(i*(phi[idx] + s*arc[idx])) with idx = (m+offset)%4.
    std::array::from_fn(|m| {
        let idx = (m + offset) % 4;
        let angle = phi[idx] + s * arc[idx];
        C::new(angle.cos(), angle.sin())
    })
}

/// Tier scaffold for the Mumford-form interior rung.
///
/// Calls `nu_pick` with offset 0 then 1 (the two sanctioned nu placements
/// per the bounded-enumeration rule). For each nu, calls `solve_mu` to obtain
/// candidate mu vectors, feeds each to `mumford_recover`, and returns the first
/// accept. Declines (None) only when all candidates at both offsets are
/// exhausted.
///
/// `solve_mu(gate, a, w, nu)` is the placeholder for the float deg-16
/// companion eigensolve arriving from the companion agent. Keeping it as a
/// closure parameter means the wiring is a single call-site change.
///
/// Not wired into any existing production tier chain.
pub fn tier_mumford(
    gate: &[C; 4],
    a: &[C; 4],
    w: &[C; 4],
    solve_mu: impl Fn(&[C; 4], &[C; 4], &[C; 4], &[C; 4]) -> Vec<[C; 4]>,
) -> Option<Solution> {
    for offset in 0..2usize {
        let nu = nu_pick(gate, a, w, offset);
        for mu in solve_mu(gate, a, w, &nu) {
            if let Some(sol) = mumford_recover(gate, a, w, &mu, &nu) {
                return Some(sol);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::can_sandwich::ACCEPT;

    fn c(re: f64, im: f64) -> C {
        C::new(re, im)
    }

    // Seed-11 synthetic row: random SO(4) frame + random (eb, ep).
    // Python s234 prototype: forward spec error 8.545e-14, msg "ok".
    #[test]
    fn test_seed11_accept() {
        let gate = [
            c(9.71374502484236735e-01, -2.37553311750229312e-01),
            c(-8.41008912738173464e-01, 5.41021264549699010e-01),
            c(7.80584673150464425e-01, 6.25050052429869551e-01),
            c(9.78902286952058276e-01, -2.04328932361597077e-01),
        ];
        let a = [
            c(4.79767064296249823e-01, 8.77395899247630506e-01),
            c(9.96975077459966275e-01, 7.77219076174428258e-02),
            c(-4.07137078064051194e-01, 9.13367067320399184e-01),
            c(9.52835921008089337e-01, 3.03485926587487853e-01),
        ];
        let mu = [
            c(-3.56205604450301305e-01, 9.34407602365366574e-01),
            c(-8.16355040189586600e-01, -5.77550385989878512e-01),
            c(9.22295250078664641e-01, 3.86486056258610822e-01),
            c(9.98597070558034261e-01, -5.29517768626527330e-02),
        ];
        let nu = [
            c(-9.93536633236181577e-01, 1.13511930728499577e-01),
            c(9.96114809187814698e-02, -9.95026408126520767e-01),
            c(7.13997174784282373e-01, 7.00148580231412576e-01),
            c(9.80800298164841711e-01, 1.95014807437173365e-01),
        ];
        let w = [
            c(-9.38951747022464644e-01, -3.44048858105150845e-01),
            c(1.01284953147205273e-01, -9.94857456254898853e-01),
            c(3.75875786628658337e-01, 9.26670056183044544e-01),
            c(9.79920781325626100e-01, 1.99387217058102140e-01),
        ];
        let result = mumford_recover(&gate, &a, &w, &mu, &nu);
        assert!(result.is_some(), "seed-11 must return Some");
        let sol = result.unwrap();
        assert!(
            sol.residual < ACCEPT,
            "seed-11 residual {:.3e} >= ACCEPT {:.3e}",
            sol.residual,
            ACCEPT
        );
    }

    // Haar12345 production (eb,ep) with a fresh random SO(4) frame (seed 99).
    // Zero-stratum workaround: real production O_true gives mu=delta exactly;
    // seed-99 frame gives min mu-delta gap 7.325e-04 (genuinely generic).
    // Python s234 prototype: forward spec error 4.302e-15, msg "ok".
    #[test]
    fn test_haar_generic_seed99_accept() {
        let gate = [
            c(7.81790744554717421e-01, 6.23540882162974652e-01),
            c(-7.58334432345166842e-01, 6.51865698376386082e-01),
            c(-9.86059793182763600e-01, -1.66391358755090923e-01),
            c(9.91511191681765047e-01, -1.30021370435041222e-01),
        ];
        let a = [
            c(7.43378301087965432e-01, 6.68871214413933490e-01),
            c(3.75408523435267594e-01, 9.26859450257778938e-01),
            c(6.63020307098132478e-02, -9.97799599480654775e-01),
            c(9.15437572551085821e-01, -4.02459998958623721e-01),
        ];
        let mu = [
            c(2.32916643473974339e-01, 9.72496702921309963e-01),
            c(6.83001284510025397e-01, -7.30417172140452209e-01),
            c(-9.59790701734385632e-01, -2.80716598839854725e-01),
            c(-8.68198280932854671e-01, 4.96217437203928635e-01),
        ];
        let nu = [
            c(7.74268334597621788e-01, 6.32857445274543906e-01),
            c(-3.33550378390960411e-01, 9.42732276457769025e-01),
            c(-8.56582255870265707e-01, -5.16010502730522069e-01),
            c(-9.57018387600664933e-01, 2.90027250089062549e-01),
        ];
        let w = [
            c(-9.20807771895155169e-01, -3.90016726843196160e-01),
            c(-3.46919965976816125e-01, 9.37894736741093249e-01),
            c(6.65231664081360474e-01, 7.46637015626430589e-01),
            c(9.99631981877393772e-01, -2.71274917357757807e-02),
        ];
        let result = mumford_recover(&gate, &a, &w, &mu, &nu);
        assert!(result.is_some(), "haar-generic-seed99 must return Some");
        let sol = result.unwrap();
        assert!(
            sol.residual < ACCEPT,
            "haar-generic-seed99 residual {:.3e} >= ACCEPT {:.3e}",
            sol.residual,
            ACCEPT
        );
    }

    // Helper: product of unit-circle complex numbers.
    fn cprod(xs: &[C; 4]) -> C {
        xs.iter().fold(C::new(1.0, 0.0), |acc, &x| acc * x)
    }

    // det-pin target: prod(delta)*gate[0]*gate[1]/gate[3]^2.
    fn det_pin_target(gate: &[C; 4], a: &[C; 4]) -> C {
        let p_delta = a
            .iter()
            .fold(C::new(1.0, 0.0), |acc, &ak| acc * gate[3] * ak * ak);
        p_delta * gate[0] * gate[1] / (gate[3] * gate[3])
    }

    #[test]
    fn test_nu_pick_seed11() {
        let gate = [
            c(9.71374502484236735e-01, -2.37553311750229312e-01),
            c(-8.41008912738173464e-01, 5.41021264549699010e-01),
            c(7.80584673150464425e-01, 6.25050052429869551e-01),
            c(9.78902286952058276e-01, -2.04328932361597077e-01),
        ];
        let a = [
            c(4.79767064296249823e-01, 8.77395899247630506e-01),
            c(9.96975077459966275e-01, 7.77219076174428258e-02),
            c(-4.07137078064051194e-01, 9.13367067320399184e-01),
            c(9.52835921008089337e-01, 3.03485926587487853e-01),
        ];
        let w = [
            c(-9.38951747022464644e-01, -3.44048858105150845e-01),
            c(1.01284953147205273e-01, -9.94857456254898853e-01),
            c(3.75875786628658337e-01, 9.26670056183044544e-01),
            c(9.79920781325626100e-01, 1.99387217058102140e-01),
        ];
        let tau = 2.0 * std::f64::consts::PI;
        let w_phi: [f64; 4] = std::array::from_fn(|k| w[k].arg().rem_euclid(tau));
        let p_target = det_pin_target(&gate, &a);

        for offset in 0..2usize {
            let nu = nu_pick(&gate, &a, &w, offset);
            // Interlacing.
            let nu_phi: [f64; 4] = std::array::from_fn(|k| nu[k].arg().rem_euclid(tau));
            assert!(
                interlaces_circular(&nu_phi, &w_phi),
                "seed-11 offset={offset}: nu must circularly interlace with w"
            );
            // Det-pin.
            let residual = (cprod(&nu) - p_target).norm();
            assert!(
                residual < 1e-12,
                "seed-11 offset={offset}: det-pin residual {residual:.3e} >= 1e-12"
            );
        }
    }

    #[test]
    fn test_nu_pick_haar_seed99() {
        let gate = [
            c(7.81790744554717421e-01, 6.23540882162974652e-01),
            c(-7.58334432345166842e-01, 6.51865698376386082e-01),
            c(-9.86059793182763600e-01, -1.66391358755090923e-01),
            c(9.91511191681765047e-01, -1.30021370435041222e-01),
        ];
        let a = [
            c(7.43378301087965432e-01, 6.68871214413933490e-01),
            c(3.75408523435267594e-01, 9.26859450257778938e-01),
            c(6.63020307098132478e-02, -9.97799599480654775e-01),
            c(9.15437572551085821e-01, -4.02459998958623721e-01),
        ];
        let w = [
            c(-9.20807771895155169e-01, -3.90016726843196160e-01),
            c(-3.46919965976816125e-01, 9.37894736741093249e-01),
            c(6.65231664081360474e-01, 7.46637015626430589e-01),
            c(9.99631981877393772e-01, -2.71274917357757807e-02),
        ];
        let tau = 2.0 * std::f64::consts::PI;
        let w_phi: [f64; 4] = std::array::from_fn(|k| w[k].arg().rem_euclid(tau));
        let p_target = det_pin_target(&gate, &a);

        for offset in 0..2usize {
            let nu = nu_pick(&gate, &a, &w, offset);
            // Interlacing.
            let nu_phi: [f64; 4] = std::array::from_fn(|k| nu[k].arg().rem_euclid(tau));
            assert!(
                interlaces_circular(&nu_phi, &w_phi),
                "haar-seed99 offset={offset}: nu must circularly interlace with w"
            );
            // Det-pin.
            let residual = (cprod(&nu) - p_target).norm();
            assert!(
                residual < 1e-12,
                "haar-seed99 offset={offset}: det-pin residual {residual:.3e} >= 1e-12"
            );
        }
    }
}
