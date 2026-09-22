//! Two-word accumulation of the existing chart coefficient formulas.
use super::C;
use std::ops::{Add, Div, Mul, Sub};

#[derive(Clone, Copy, Default)]
pub(super) struct D(f64, f64);
impl D {
    fn new(a: f64, b: f64) -> Self {
        let s = a + b;
        Self(s, b - (s - a))
    }
    pub(super) fn value(self) -> f64 {
        self.0 + self.1
    }
    fn sqrt(self) -> Self {
        let first = D::from(self.value().sqrt());
        first + (self - first * first) / (D::from(2.0) * first)
    }
}

impl From<f64> for D {
    fn from(x: f64) -> Self {
        Self(x, 0.0)
    }
}
impl Add for D {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        let s = self.0 + other.0;
        let v = s - self.0;
        let e = (self.0 - (s - v)) + (other.0 - v);
        Self::new(s, e + self.1 + other.1)
    }
}
impl Sub for D {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        self + Self(-other.0, -other.1)
    }
}
impl Mul for D {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        let p = self.0 * other.0;
        let e = self.0.mul_add(other.0, -p);
        Self::new(
            p,
            e + self.0 * other.1 + self.1 * other.0 + self.1 * other.1,
        )
    }
}
impl Div for D {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        let first = self.0 / other.0;
        let residual = self - other * D::from(first);
        D::from(first) + D::from(residual.value() / other.0)
    }
}

#[derive(Clone, Copy, Default)]
struct Z {
    re: D,
    im: D,
}
impl From<C> for Z {
    fn from(z: C) -> Self {
        Self {
            re: z.re.into(),
            im: z.im.into(),
        }
    }
}
impl Add for Z {
    type Output = Self;
    fn add(self, z: Self) -> Self {
        Self {
            re: self.re + z.re,
            im: self.im + z.im,
        }
    }
}
impl Sub for Z {
    type Output = Self;
    fn sub(self, z: Self) -> Self {
        Self {
            re: self.re - z.re,
            im: self.im - z.im,
        }
    }
}
impl Mul for Z {
    type Output = Self;
    fn mul(self, z: Self) -> Self {
        Self {
            re: self.re * z.re - self.im * z.im,
            im: self.re * z.im + self.im * z.re,
        }
    }
}
impl Mul<f64> for Z {
    type Output = Self;
    fn mul(self, x: f64) -> Self {
        Self {
            re: self.re * D::from(x),
            im: self.im * D::from(x),
        }
    }
}

fn normalized(roots: &[C; 4]) -> [Z; 4] {
    let unit = roots.map(|z| {
        let z = Z::from(z);
        let inv = D::from(1.0) / (z.re * z.re + z.im * z.im).sqrt();
        Z {
            re: z.re * inv,
            im: z.im * inv,
        }
    });
    let product = unit[0] * unit[1] * unit[2] * unit[3];
    // A first-order fourth-root correction reduces determinant drift from
    // floating conversion and the existing snapped-spectrum candidate pass.
    // This changes only candidate-generation coefficients, not the original
    // PreparedSandwich spectra used by both public rootwise certificates.
    let one = Z::from(C::new(1.0, 0.0));
    let factor = one - (product - one) * 0.25;
    unit.map(|z| z * factor)
}

pub(super) fn rows_at(
    lam: &[C; 4],
    mu: &[C; 4],
    i: usize,
    k: usize,
    tp: f64,
    target: &[C; 4],
) -> ([[D; 4]; 3], [D; 3]) {
    let lam = normalized(lam);
    let mu = normalized(mu);
    let target = normalized(target);
    let others: Vec<usize> = (0..4).filter(|&j| j != i).collect();
    let ki = others.iter().position(|&j| j == k).unwrap();
    let a = mu[i] * (1.0 - tp) + mu[k] * tp;
    let cc = mu[i] * tp + mu[k] * (1.0 - tp);
    let b2 = (mu[i] - mu[k]) * (mu[i] - mu[k]) * (tp * (1.0 - tp));
    let mut mup: [Z; 3] = std::array::from_fn(|q| mu[others[q]]);
    mup[ki] = cc;
    let lo: [Z; 3] = std::array::from_fn(|q| lam[others[q]]);
    let mut c1 = [[Z::default(); 3]; 3];
    let mut c2 = c1;
    for j in 0..3 {
        for l in 0..3 {
            c1[j][l] = lo[j] * mup[l];
            c2[j][l] = lam[i] * lo[j] * a * mup[l];
            if l == ki {
                c2[j][l] = c2[j][l] - lam[i] * lo[j] * b2;
            }
        }
    }
    for j in 0..3 {
        for q in j + 1..3 {
            for l in 0..3 {
                for p in l + 1..3 {
                    c2[3 - j - q][3 - l - p] =
                        c2[3 - j - q][3 - l - p] + lo[j] * lo[q] * mup[l] * mup[p];
                }
            }
        }
    }
    let restrict = |cm: [[Z; 3]; 3]| -> ([Z; 4], Z) {
        let row = [
            cm[0][0] - cm[0][2] - cm[2][0] + cm[2][2],
            cm[0][1] - cm[0][2] - cm[2][1] + cm[2][2],
            cm[1][0] - cm[1][2] - cm[2][0] + cm[2][2],
            cm[1][1] - cm[1][2] - cm[2][1] + cm[2][2],
        ];
        (row, cm[0][2] + cm[1][2] + cm[2][0] + cm[2][1] - cm[2][2])
    };
    let (r1, b1) = restrict(c1);
    let (r2, b2) = restrict(c2);
    let k1 = lam[i] * a;
    let e1 = target[0] + target[1] + target[2] + target[3];
    let mut e2 = Z::default();
    for a in 0..4 {
        for b in a + 1..4 {
            e2 = e2 + target[a] * target[b];
        }
    }
    let rhs1 = e1 - k1 - b1;
    let rhs2 = e2 - b2;
    (
        [r1.map(|z| z.re), r1.map(|z| z.im), r2.map(|z| z.re)],
        [rhs1.re, rhs1.im, rhs2.re],
    )
}
