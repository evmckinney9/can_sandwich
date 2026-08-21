//! Fixed-action projective Spin selector.
//!
//! The bounded dense-action prefix is a production fallback constructor. Its
//! exhaustion is not an infeasibility certificate; every returned frame is
//! independently certified by the caller.
//! Fixing one rational right Spin quaternion leaves two projective quadrics
//! and one quartic in the left quaternion.  Over one retained coordinate the
//! quadrics form a four-dimensional quotient algebra.  The norm of the
//! quartic in that algebra is the degree-16 selector.

use nalgebra::Matrix4;

use super::{poly_roots, Mat4, PreparedSandwich, Rung, Solution, ACCEPT, C};

const N: usize = 17;
const AFFINE_CHARTS: [[usize; 4]; 4] = [[0, 1, 2, 3], [1, 0, 2, 3], [2, 0, 1, 3], [3, 0, 1, 2]];

#[derive(Clone, Copy, Debug)]
struct Poly {
    coefficients: [f64; N],
    len: usize,
}

impl Poly {
    const ZERO: Self = Self {
        coefficients: [0.0; N],
        len: 1,
    };

    fn constant(value: f64) -> Self {
        let mut result = Self::ZERO;
        result.coefficients[0] = value;
        result
    }

    fn monomial(degree: usize, value: f64) -> Self {
        let mut result = Self::ZERO;
        if degree < N && value != 0.0 {
            result.coefficients[degree] = value;
            result.len = degree + 1;
        }
        result
    }

    fn add(self, right: Self) -> Self {
        Self {
            coefficients: std::array::from_fn(|index| {
                self.coefficients[index] + right.coefficients[index]
            }),
            len: self.len.max(right.len),
        }
    }

    fn sub(self, right: Self) -> Self {
        Self {
            coefficients: std::array::from_fn(|index| {
                self.coefficients[index] - right.coefficients[index]
            }),
            len: self.len.max(right.len),
        }
    }

    fn scale(self, value: f64) -> Self {
        if value == 0.0 {
            return Self::ZERO;
        }
        Self {
            coefficients: std::array::from_fn(|index| self.coefficients[index] * value),
            len: self.len,
        }
    }

    fn mul(self, right: Self) -> Self {
        let mut result = [0.0; N];
        let len = (self.len + right.len - 1).min(N);
        for left_degree in 0..self.len.min(N) {
            for right_degree in 0..right.len.min(N - left_degree) {
                result[left_degree + right_degree] +=
                    self.coefficients[left_degree] * right.coefficients[right_degree];
            }
        }
        Self {
            coefficients: result,
            len,
        }
    }

    fn eval(self, value: f64) -> f64 {
        self.coefficients[..self.len]
            .iter()
            .rev()
            .fold(0.0, |sum, coefficient| sum.mul_add(value, *coefficient))
    }
}

type PMat = [[Poly; 4]; 4];
type Quadric = [[[f64; 3]; 3]; 3];
type Quartic = [[[f64; 5]; 5]; 5];
type Biquadric = [[Poly; 3]; 3];
type Biquartic = [[Poly; 5]; 5];

fn eval_biquadric(value: &Biquadric, t: f64, y: f64, z: f64) -> f64 {
    let yp = [1.0, y, y * y];
    let zp = [1.0, z, z * z];
    (0..3)
        .flat_map(|i| (0..3).map(move |j| (i, j)))
        .map(|(i, j)| value[i][j].eval(t) * yp[i] * zp[j])
        .sum()
}

fn eval_biquartic(value: &Biquartic, t: f64, y: f64, z: f64) -> f64 {
    let yp = [1.0, y, y * y, y.powi(3), y.powi(4)];
    let zp = [1.0, z, z * z, z.powi(3), z.powi(4)];
    (0..5)
        .flat_map(|i| (0..5).map(move |j| (i, j)))
        .map(|(i, j)| value[i][j].eval(t) * yp[i] * zp[j])
        .sum()
}

#[cfg(test)]
fn pmat_zero() -> PMat {
    [[Poly::ZERO; 4]; 4]
}

#[cfg(test)]
fn pmat_identity() -> PMat {
    std::array::from_fn(|row| {
        std::array::from_fn(|column| Poly::constant((row == column) as u8 as f64))
    })
}

#[cfg(test)]
fn pmat_add(left: &PMat, right: &PMat) -> PMat {
    std::array::from_fn(|row| {
        std::array::from_fn(|column| left[row][column].add(right[row][column]))
    })
}

#[cfg(test)]
fn pmat_scale_poly(value: &PMat, scale: Poly) -> PMat {
    std::array::from_fn(|row| std::array::from_fn(|column| value[row][column].mul(scale)))
}

fn pmat_mul(left: &PMat, right: &PMat) -> PMat {
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            (0..4).fold(Poly::ZERO, |sum, inner| {
                sum.add(left[row][inner].mul(right[inner][column]))
            })
        })
    })
}

fn pmat_eval(value: &PMat, parameter: f64) -> Matrix4<f64> {
    Matrix4::from_fn(|row, column| value[row][column].eval(parameter))
}

fn determinant3(value: [[f64; 3]; 3]) -> f64 {
    value[0][0] * (value[1][1] * value[2][2] - value[1][2] * value[2][1])
        - value[0][1] * (value[1][0] * value[2][2] - value[1][2] * value[2][0])
        + value[0][2] * (value[1][0] * value[2][1] - value[1][1] * value[2][0])
}

fn adjugate(value: &Matrix4<f64>) -> Matrix4<f64> {
    Matrix4::from_fn(|row, column| {
        let mut minor = [[0.0; 3]; 3];
        let mut minor_row = 0;
        for source_row in 0..4 {
            if source_row == column {
                continue;
            }
            let mut minor_column = 0;
            for source_column in 0..4 {
                if source_column == row {
                    continue;
                }
                minor[minor_row][minor_column] = value[(source_row, source_column)];
                minor_column += 1;
            }
            minor_row += 1;
        }
        let sign = if (row + column) % 2 == 0 { 1.0 } else { -1.0 };
        sign * determinant3(minor)
    })
}

fn pmat_determinant(value: &PMat) -> Poly {
    let mut result = Poly::ZERO;
    for permutation in super::PERMS24.iter() {
        let inversions = (0..4)
            .flat_map(|left| (left + 1..4).map(move |right| (left, right)))
            .filter(|&(left, right)| permutation[left] > permutation[right])
            .count();
        let term = value[0][permutation[0]]
            .mul(value[1][permutation[1]])
            .mul(value[2][permutation[2]])
            .mul(value[3][permutation[3]]);
        result = if inversions % 2 == 0 {
            result.add(term)
        } else {
            result.sub(term)
        };
    }
    result
}

#[derive(Clone, Copy)]
struct Double {
    hi: f64,
    lo: f64,
}

impl Double {
    const ZERO: Self = Self { hi: 0.0, lo: 0.0 };
    const ONE: Self = Self { hi: 1.0, lo: 0.0 };

    fn add(self, right: Self) -> Self {
        let sum = self.hi + right.hi;
        let virtual_right = sum - self.hi;
        let error = (self.hi - (sum - virtual_right)) + (right.hi - virtual_right);
        let tail = error + self.lo + right.lo;
        let hi = sum + tail;
        Self {
            hi,
            lo: tail - (hi - sum),
        }
    }

    fn scale(self, sign: f64) -> Self {
        Self {
            hi: sign * self.hi,
            lo: sign * self.lo,
        }
    }

    fn mul_f64(self, right: f64) -> Self {
        let product = self.hi * right;
        let error = self.hi.mul_add(right, -product) + self.lo * right;
        let hi = product + error;
        Self {
            hi,
            lo: error - (hi - product),
        }
    }

    fn to_f64(self) -> f64 {
        self.hi + self.lo
    }
}

/// Determinant of the four-dimensional quotient multiplication matrix with
/// compensated coefficient arithmetic. The quotient entries are still the
/// same `f64` equations; this only prevents the 24 signed determinant terms
/// from losing their low bits before the degree-sixteen root solve.
fn pmat_determinant_compensated(value: &PMat) -> Poly {
    let mut result = [Double::ZERO; N];
    for permutation in super::PERMS24.iter() {
        let inversions = (0..4)
            .flat_map(|left| (left + 1..4).map(move |right| (left, right)))
            .filter(|&(left, right)| permutation[left] > permutation[right])
            .count();
        let sign = if inversions % 2 == 0 { 1.0 } else { -1.0 };
        let mut term = [Double::ZERO; N];
        term[0] = Double::ONE;
        let mut term_len = 1;
        for row in 0..4 {
            let factor = value[row][permutation[row]];
            let mut product = [Double::ZERO; N];
            for left_degree in 0..term_len {
                for right_degree in 0..factor.len.min(N - left_degree) {
                    product[left_degree + right_degree] = product[left_degree + right_degree]
                        .add(term[left_degree].mul_f64(factor.coefficients[right_degree]));
                }
            }
            term = product;
            term_len = (term_len + factor.len - 1).min(N);
        }
        for degree in 0..term_len {
            result[degree] = result[degree].add(term[degree].scale(sign));
        }
    }
    Poly {
        coefficients: result.map(Double::to_f64),
        len: N,
    }
}

fn left_quaternion(value: [f64; 4]) -> Matrix4<f64> {
    let [w, x, y, z] = value;
    Matrix4::from_row_slice(&[w, -x, -y, -z, x, w, -z, y, y, z, w, -x, z, -y, x, w])
}

fn right_conjugate(value: [f64; 4]) -> Matrix4<f64> {
    let [w, x, y, z] = value;
    Matrix4::from_row_slice(&[w, x, y, z, -x, w, -z, y, -y, z, w, -x, -z, -y, x, w])
}

fn exponent_of_coordinate(coordinate: usize) -> [usize; 3] {
    let mut exponent = [0; 3];
    if coordinate > 0 {
        exponent[coordinate - 1] = 1;
    }
    exponent
}

fn add_quadratic_monomial(polynomial: &mut Quadric, left: usize, right: usize, coefficient: f64) {
    let mut exponent = exponent_of_coordinate(left);
    let right_exponent = exponent_of_coordinate(right);
    for index in 0..3 {
        exponent[index] += right_exponent[index];
    }
    polynomial[exponent[0]][exponent[1]][exponent[2]] += coefficient;
}

fn square_quadric(value: &Quadric) -> Quartic {
    multiply_quadrics(value, value)
}

fn multiply_quadrics(left: &Quadric, right: &Quadric) -> Quartic {
    let mut result = [[[0.0; 5]; 5]; 5];
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                let left_value = left[a][b][c];
                if left_value == 0.0 {
                    continue;
                }
                for d in 0..3 {
                    for e in 0..3 {
                        for f in 0..3 {
                            result[a + d][b + e][c + f] += left_value * right[d][e][f];
                        }
                    }
                }
            }
        }
    }
    result
}

fn add_scaled_quadric(target: &mut Quadric, source: &Quadric, scale: f64) {
    for t in 0..3 {
        for y in 0..3 {
            for z in 0..3 {
                target[t][y][z] += scale * source[t][y][z];
            }
        }
    }
}

fn add_scaled_quartic(target: &mut Quartic, source: &Quartic, scale: f64) {
    for t in 0..5 {
        for y in 0..5 {
            for z in 0..5 {
                target[t][y][z] += scale * source[t][y][z];
            }
        }
    }
}

fn projective_adjoint(order: [usize; 4]) -> [[Quadric; 3]; 3] {
    let mut result = [[[[[0.0; 3]; 3]; 3]; 3]; 3];
    let terms: [[&[(usize, usize, f64)]; 3]; 3] = [
        [
            &[(0, 0, 1.0), (1, 1, 1.0), (2, 2, -1.0), (3, 3, -1.0)],
            &[(1, 2, 2.0), (0, 3, -2.0)],
            &[(1, 3, 2.0), (0, 2, 2.0)],
        ],
        [
            &[(1, 2, 2.0), (0, 3, 2.0)],
            &[(0, 0, 1.0), (1, 1, -1.0), (2, 2, 1.0), (3, 3, -1.0)],
            &[(2, 3, 2.0), (0, 1, -2.0)],
        ],
        [
            &[(1, 3, 2.0), (0, 2, -2.0)],
            &[(2, 3, 2.0), (0, 1, 2.0)],
            &[(0, 0, 1.0), (1, 1, -1.0), (2, 2, -1.0), (3, 3, 1.0)],
        ],
    ];
    let mut inverse = [0usize; 4];
    for (local, original) in order.into_iter().enumerate() {
        inverse[original] = local;
    }
    for row in 0..3 {
        for column in 0..3 {
            for &(left, right, coefficient) in terms[row][column] {
                add_quadratic_monomial(
                    &mut result[row][column],
                    inverse[left],
                    inverse[right],
                    coefficient,
                );
            }
        }
    }
    result
}

fn adjoint_quaternion(value: [f64; 4]) -> [[f64; 3]; 3] {
    let [w, x, y, z] = value;
    [
        [
            w * w + x * x - y * y - z * z,
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            w * w - x * x + y * y - z * z,
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            w * w - x * x - y * y + z * z,
        ],
    ]
}

fn walsh_spectrum(value: &[C; 4]) -> [C; 4] {
    [
        value[0] + value[1] + value[2] + value[3],
        value[0] + value[1] - value[2] - value[3],
        value[0] - value[1] + value[2] - value[3],
        value[0] - value[1] - value[2] + value[3],
    ]
}

fn complementary_pair_spectrum(value: &[C; 4]) -> ([C; 3], [C; 3]) {
    let pairs = [
        value[0] * value[1],
        value[0] * value[2],
        value[0] * value[3],
        value[2] * value[3],
        value[3] * value[1],
        value[1] * value[2],
    ];
    (
        std::array::from_fn(|index| pairs[index] + pairs[index + 3]),
        std::array::from_fn(|index| pairs[index] - pairs[index + 3]),
    )
}

struct ProjectiveChart {
    p_rotation: [[Quadric; 3]; 3],
    p_rotation_squared: [[Quartic; 3]; 3],
    norm_rotation: [[Quartic; 3]; 3],
    norm_squared: Quartic,
    norm: Quadric,
}

static PROJECTIVE_CHARTS: std::sync::LazyLock<[ProjectiveChart; 4]> =
    std::sync::LazyLock::new(|| {
        std::array::from_fn(|index| {
            let p_rotation = projective_adjoint(AFFINE_CHARTS[index]);
            let mut norm = [[[0.0; 3]; 3]; 3];
            for coordinate in 0..4 {
                add_quadratic_monomial(&mut norm, coordinate, coordinate, 1.0);
            }
            ProjectiveChart {
                p_rotation_squared: std::array::from_fn(|row| {
                    std::array::from_fn(|column| square_quadric(&p_rotation[row][column]))
                }),
                norm_rotation: std::array::from_fn(|row| {
                    std::array::from_fn(|column| multiply_quadrics(&norm, &p_rotation[row][column]))
                }),
                norm_squared: square_quadric(&norm),
                p_rotation,
                norm,
            }
        })
    });

struct SpinBasis {
    chart: usize,
    q_rotation: [[f64; 3]; 3],
}

#[derive(Clone)]
struct FixedActionProblem {
    alpha: [C; 4],
    gate: [C; 4],
    target: [C; 4],
    first_constant: C,
    first_weights: [[C; 3]; 3],
    second_plus_weights: [[C; 3]; 3],
    second_mixed_weights: [[C; 3]; 3],
}

impl FixedActionProblem {
    fn new(alpha: &[C; 4], gate: &[C; 4], target: &[C; 4]) -> Self {
        let alpha_walsh = walsh_spectrum(alpha);
        let gate_walsh = walsh_spectrum(gate);
        let (alpha_plus, alpha_minus) = complementary_pair_spectrum(alpha);
        let (gate_plus, gate_minus) = complementary_pair_spectrum(gate);
        Self {
            alpha: *alpha,
            gate: *gate,
            target: *target,
            first_constant: 0.25 * alpha_walsh[0] * gate_walsh[0] - target[0],
            first_weights: std::array::from_fn(|row| {
                std::array::from_fn(|column| 0.25 * alpha_walsh[row + 1] * gate_walsh[column + 1])
            }),
            second_plus_weights: std::array::from_fn(|row| {
                std::array::from_fn(|column| 0.25 * alpha_plus[row] * gate_plus[column])
            }),
            second_mixed_weights: std::array::from_fn(|row| {
                std::array::from_fn(|column| 0.5 * alpha_minus[row] * gate_minus[column])
            }),
        }
    }
}

fn spin_basis(q: [f64; 4], order: [usize; 4]) -> SpinBasis {
    SpinBasis {
        chart: AFFINE_CHARTS
            .iter()
            .position(|candidate| *candidate == order)
            .expect("fixed projective chart"),
        q_rotation: adjoint_quaternion(q),
    }
}

fn first_character_equations(
    problem: &FixedActionProblem,
    basis: &SpinBasis,
) -> (Quadric, Quadric) {
    let chart = &PROJECTIVE_CHARTS[basis.chart];
    // H (O o O) H^T = 4 diag(1, Ad(p) o Ad(q)).
    let mut real = [[[0.0; 3]; 3]; 3];
    let mut imaginary = [[[0.0; 3]; 3]; 3];
    add_scaled_quadric(&mut real, &chart.norm, problem.first_constant.re);
    add_scaled_quadric(&mut imaginary, &chart.norm, problem.first_constant.im);
    for row in 0..3 {
        for column in 0..3 {
            let weight = problem.first_weights[row][column] * basis.q_rotation[row][column];
            add_scaled_quadric(&mut real, &chart.p_rotation[row][column], weight.re);
            add_scaled_quadric(&mut imaginary, &chart.p_rotation[row][column], weight.im);
        }
    }
    (real, imaginary)
}

fn second_character_equation(problem: &FixedActionProblem, basis: &SpinBasis) -> Quartic {
    let chart = &PROJECTIVE_CHARTS[basis.chart];
    // Lambda^2(O)=Ad(p) direct-sum Ad(q) in the Hodge basis.  Complementary
    // pair sums contract P o P + Q o Q; differences contract P o Q.
    let mut q4 = [[[0.0; 5]; 5]; 5];
    for row in 0..3 {
        for column in 0..3 {
            let plus_weight = problem.second_plus_weights[row][column];
            add_scaled_quartic(
                &mut q4,
                &chart.p_rotation_squared[row][column],
                plus_weight.re,
            );
            add_scaled_quartic(
                &mut q4,
                &chart.norm_squared,
                plus_weight.re * basis.q_rotation[row][column].powi(2),
            );
            let mixed_weight =
                problem.second_mixed_weights[row][column] * basis.q_rotation[row][column];
            add_scaled_quartic(&mut q4, &chart.norm_rotation[row][column], mixed_weight.re);
        }
    }
    add_scaled_quartic(&mut q4, &chart.norm_squared, -problem.target[1].re);
    q4
}

fn spectral_equations(
    problem: &FixedActionProblem,
    q: [f64; 4],
    order: [usize; 4],
) -> (Quadric, Quadric, Quartic) {
    let basis = spin_basis(q, order);
    let (q1, q2) = first_character_equations(problem, &basis);
    let q4 = second_character_equation(problem, &basis);
    (q1, q2, q4)
}

fn quadric_matrix(value: &Quadric) -> Matrix4<f64> {
    let mut matrix = Matrix4::zeros();
    matrix[(0, 0)] = value[0][0][0];
    for coordinate in 1..4 {
        let mut exponent = [0usize; 3];
        exponent[coordinate - 1] = 1;
        let coefficient = value[exponent[0]][exponent[1]][exponent[2]];
        matrix[(0, coordinate)] = 0.5 * coefficient;
        matrix[(coordinate, 0)] = 0.5 * coefficient;
    }
    for left in 1..4 {
        for right in left..4 {
            let mut exponent = [0usize; 3];
            exponent[left - 1] += 1;
            exponent[right - 1] += 1;
            let coefficient = value[exponent[0]][exponent[1]][exponent[2]];
            if left == right {
                matrix[(left, right)] = coefficient;
            } else {
                matrix[(left, right)] = 0.5 * coefficient;
                matrix[(right, left)] = 0.5 * coefficient;
            }
        }
    }
    matrix
}

/// Outward-only definiteness certificate for a symmetric matrix.
///
/// The eigensolver supplies an orthogonal trial basis.  Gershgorin in the
/// explicitly recomputed transformed matrix then supplies the certificate;
/// the wide margin makes every ill-conditioned case decline to "unknown".
fn certified_definite(value: &Matrix4<f64>) -> bool {
    let scale = value
        .iter()
        .fold(0.0_f64, |maximum, entry| maximum.max(entry.abs()));
    if !scale.is_finite() || scale == 0.0 {
        return false;
    }
    let eigen = nalgebra::linalg::SymmetricEigen::new(value.clone_owned());
    let transformed = eigen.eigenvectors.transpose() * value * eigen.eigenvectors;
    let margin = 1e-10 * scale;
    let positive = (0..4).all(|row| {
        transformed[(row, row)]
            - (0..4)
                .filter(|&column| column != row)
                .map(|column| transformed[(row, column)].abs())
                .sum::<f64>()
            > margin
    });
    let negative = (0..4).all(|row| {
        -transformed[(row, row)]
            - (0..4)
                .filter(|&column| column != row)
                .map(|column| transformed[(row, column)].abs())
                .sum::<f64>()
            > margin
    });
    positive || negative
}

fn definite_pencil_member(left: &Matrix4<f64>, right: &Matrix4<f64>, t: f64) -> bool {
    if !t.is_finite() {
        return false;
    }
    let denominator = 1.0_f64.max(t.abs());
    certified_definite(&(left.scale(denominator.recip()) + right.scale(t / denominator)))
}

/// Necessary and sufficient exact criterion, implemented as an outward-only
/// floating certificate, for the two first-character quadrics to have a
/// common real projective point.
///
/// For four real homogeneous coordinates, the joint numerical range of two
/// quadratic forms is convex (Dines--Brickman).  Hence the forms have no
/// common unit zero iff their real pencil contains a definite matrix.  Pencil
/// inertia changes only at a root of `det(A+tB)`, so one sample per real-root
/// interval is complete in exact arithmetic.  Numerically, an uncertain root
/// or definiteness test only preserves the action; it can never manufacture a
/// realization or an infeasibility conclusion.
fn first_character_supported(left: &Quadric, right: &Quadric) -> bool {
    let left = quadric_matrix(left);
    let right = quadric_matrix(right);
    if certified_definite(&left) || certified_definite(&right) {
        return false;
    }

    let pencil: PMat = std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            Poly::constant(left[(row, column)]).add(Poly::monomial(1, right[(row, column)]))
        })
    });
    let determinant = pmat_determinant(&pencil);
    let scale = determinant.coefficients[..5]
        .iter()
        .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
    let quartic_coefficients: [f64; 5] = determinant.coefficients[..5]
        .try_into()
        .expect("quartic pencil determinant");
    let mut quartic = [C::new(0.0, 0.0); 4];
    let algebraic = (determinant.coefficients[4].abs() >= 1e-13 * scale)
        .then(|| super::klein::quartic_roots(&quartic_coefficients, &mut quartic))
        .flatten();
    let breakpoints: Vec<C> = match algebraic {
        Some(count) => quartic[..count].to_vec(),
        None => poly_roots(&determinant.coefficients[..5]),
    };
    let mut roots: Vec<f64> = breakpoints
        .into_iter()
        .filter(|root| root.re.is_finite() && root.im.abs() <= 1e-7 * (1.0 + root.re.abs()))
        .map(|root| root.re)
        .collect();
    roots.sort_by(f64::total_cmp);
    roots.dedup_by(|left, right| (*left - *right).abs() <= 1e-8 * (1.0 + left.abs()));

    let mut samples = Vec::with_capacity(roots.len() + 3);
    samples.extend([0.0, 1.0, -1.0]);
    if let Some(&first) = roots.first() {
        samples.push(first - 1.0 - first.abs());
    }
    for pair in roots.windows(2) {
        samples.push(0.5 * pair[0] + 0.5 * pair[1]);
    }
    if let Some(&last) = roots.last() {
        samples.push(last + 1.0 + last.abs());
    }
    !samples
        .into_iter()
        .any(|t| definite_pencil_member(&left, &right, t))
}

/// Roots of the projective selector through the shared fallible companion
/// backend.  A near-defective degree-sixteen companion can make nalgebra's
/// fixed-size QR iteration run for an effectively unbounded time.  The faer
/// backend used by `poly_roots` reports non-convergence, allowing this action
/// to decline and the exact action scheduler to continue.
fn selector_roots(selector: &Poly) -> Vec<C> {
    poly_roots(&selector.coefficients[..selector.len])
}

fn convert_quadric(value: &Quadric) -> Biquadric {
    std::array::from_fn(|y| {
        std::array::from_fn(|z| {
            (0..3).fold(Poly::ZERO, |sum, t| {
                sum.add(Poly::monomial(t, value[t][y][z]))
            })
        })
    })
}

fn convert_quartic(value: &Quartic) -> Biquartic {
    std::array::from_fn(|y| {
        std::array::from_fn(|z| {
            (0..5).fold(Poly::ZERO, |sum, t| {
                sum.add(Poly::monomial(t, value[t][y][z]))
            })
        })
    })
}

fn vec_add(left: &[Poly; 4], right: &[Poly; 4]) -> [Poly; 4] {
    std::array::from_fn(|index| left[index].add(right[index]))
}

fn vec_scale(value: &[Poly; 4], scale: Poly) -> [Poly; 4] {
    std::array::from_fn(|index| value[index].mul(scale))
}

fn quotient_matrices(left: &Biquadric, right: &Biquadric) -> Option<(PMat, PMat)> {
    let leading = [
        [left[2][0].coefficients[0], left[0][2].coefficients[0]],
        [right[2][0].coefficients[0], right[0][2].coefficients[0]],
    ];
    let det = leading[0][0] * leading[1][1] - leading[0][1] * leading[1][0];
    let scale = leading
        .iter()
        .flatten()
        .fold(0.0_f64, |m, value| m.max(value.abs()));
    if std::env::var_os("SPIN_DEBUG").is_some() {
        eprintln!(
            "spin quotient leading={leading:?} det={det:.3e} normalized_det={:.3e}",
            det / (scale * scale).max(1e-300)
        );
    }
    if !det.is_finite() || det.abs() <= 1e-12 * scale * scale {
        return None;
    }
    let inverse = [
        [leading[1][1] / det, -leading[0][1] / det],
        [-leading[1][0] / det, leading[0][0] / det],
    ];
    let basis = [(0, 0), (1, 0), (0, 1), (1, 1)];
    let y2 = std::array::from_fn(|index| {
        let (y, z) = basis[index];
        left[y][z]
            .scale(-inverse[0][0])
            .add(right[y][z].scale(-inverse[0][1]))
    });
    let z2 = std::array::from_fn(|index| {
        let (y, z) = basis[index];
        left[y][z]
            .scale(-inverse[1][0])
            .add(right[y][z].scale(-inverse[1][1]))
    });
    let yv = [Poly::ZERO, Poly::constant(1.0), Poly::ZERO, Poly::ZERO];
    let zv = [Poly::ZERO, Poly::ZERO, Poly::constant(1.0), Poly::ZERO];
    let yz = [Poly::ZERO, Poly::ZERO, Poly::ZERO, Poly::constant(1.0)];
    let u_base = vec_add(
        &vec_add(&vec_scale(&zv, y2[0]), &vec_scale(&yz, y2[1])),
        &vec_scale(&z2, y2[2]),
    );
    let v_base = vec_add(
        &vec_add(&vec_scale(&yv, z2[0]), &vec_scale(&y2, z2[1])),
        &vec_scale(&yz, z2[2]),
    );
    let uv_det = 1.0 - y2[3].coefficients[0] * z2[3].coefficients[0];
    if std::env::var_os("SPIN_DEBUG").is_some() {
        eprintln!(
            "spin quotient yz=({:.3e},{:.3e}) uv_det={uv_det:.3e}",
            y2[3].coefficients[0], z2[3].coefficients[0]
        );
    }
    if !uv_det.is_finite() || uv_det.abs() <= 1e-12 {
        return None;
    }
    let u = vec_add(&u_base, &vec_scale(&v_base, y2[3])).map(|p| p.scale(1.0 / uv_det));
    let v = vec_add(&vec_scale(&u_base, z2[3]), &v_base).map(|p| p.scale(1.0 / uv_det));
    let my_columns = [yv, y2, yz, u];
    let mz_columns = [zv, yz, z2, v];
    let my = std::array::from_fn(|row| std::array::from_fn(|column| my_columns[column][row]));
    let mz = std::array::from_fn(|row| std::array::from_fn(|column| mz_columns[column][row]));
    Some((my, mz))
}

fn pmat_vec_mul(matrix: &PMat, vector: &[Poly; 4]) -> [Poly; 4] {
    std::array::from_fn(|row| {
        (0..4).fold(Poly::ZERO, |sum, column| {
            sum.add(matrix[row][column].mul(vector[column]))
        })
    })
}

/// Reduce the quartic to one element of the four-dimensional quotient algebra,
/// then form its regular representation.
///
/// Evaluating the quartic directly in the regular representation spends 14
/// polynomial 4-by-4 matrix products.  Reduction needs the same Horner shape
/// but only matrix-vector products.  Linearity of the regular representation
/// then gives
///
/// `M_h = h_0 I + h_1 M_y + h_2 M_z + h_3 M_y M_z`,
///
/// requiring one matrix product. This is an algebraic factorization, not a
/// change in the selector.
fn evaluate_quartic(value: &Biquartic, my: &PMat, mz: &PMat) -> PMat {
    let scalar_add = |mut vector: [Poly; 4], scalar: Poly| {
        vector[0] = vector[0].add(scalar);
        vector
    };

    let row = |y_degree: usize| {
        let mut result = [Poly::ZERO; 4];
        for z_degree in (0..=4 - y_degree).rev() {
            if z_degree != 4 - y_degree {
                result = pmat_vec_mul(mz, &result);
            }
            result = scalar_add(result, value[y_degree][z_degree]);
        }
        result
    };

    let mut result = row(4);
    for y_degree in (0..4).rev() {
        result = vec_add(&pmat_vec_mul(my, &result), &row(y_degree));
    }
    let my_mz = pmat_mul(my, mz);
    std::array::from_fn(|matrix_row| {
        std::array::from_fn(|column| {
            let identity = if matrix_row == column {
                result[0]
            } else {
                Poly::ZERO
            };
            identity
                .add(my[matrix_row][column].mul(result[1]))
                .add(mz[matrix_row][column].mul(result[2]))
                .add(my_mz[matrix_row][column].mul(result[3]))
        })
    })
}

#[cfg(test)]
fn evaluate_quartic_reference(value: &Biquartic, my: &PMat, mz: &PMat) -> PMat {
    let mut my_powers = [pmat_zero(); 5];
    let mut mz_powers = [pmat_zero(); 5];
    my_powers[0] = pmat_identity();
    mz_powers[0] = pmat_identity();
    for degree in 1..5 {
        my_powers[degree] = pmat_mul(&my_powers[degree - 1], my);
        mz_powers[degree] = pmat_mul(&mz_powers[degree - 1], mz);
    }
    let mut result = pmat_zero();
    for y in 0..5 {
        for z in 0..5 - y {
            let monomial = pmat_mul(&my_powers[y], &mz_powers[z]);
            result = pmat_add(&result, &pmat_scale_poly(&monomial, value[y][z]));
        }
    }
    result
}

fn real_frame(p: [f64; 4], q: [f64; 4]) -> Mat4 {
    let value = left_quaternion(p) * right_conjugate(q);
    Mat4::from_fn(|row, column| C::new(value[(row, column)], 0.0))
}

/// Recover the two unit Spin quaternions from a certified real frame.
///
/// The sixteen matrices `L(e_a) R(conj(e_b))` are Frobenius-orthogonal, so
/// their coefficients form the rank-one matrix `p q^T`.  This is an exact
/// inverse of `real_frame` up to the central simultaneous sign.
#[cfg(any(test, feature = "research-spin"))]
fn spin_factors(frame: &Mat4) -> Option<([f64; 4], [f64; 4])> {
    let mut coefficients = [[0.0; 4]; 4];
    for a in 0..4 {
        let mut left_basis = [0.0; 4];
        left_basis[a] = 1.0;
        for b in 0..4 {
            let mut right_basis = [0.0; 4];
            right_basis[b] = 1.0;
            let basis = left_quaternion(left_basis) * right_conjugate(right_basis);
            let mut inner = 0.0;
            for row in 0..4 {
                for column in 0..4 {
                    inner += basis[(row, column)] * frame[(row, column)].re;
                }
            }
            coefficients[a][b] = inner / 4.0;
        }
    }

    let pivot = (0..4).max_by(|&left, &right| {
        let left_norm = (0..4)
            .map(|row| coefficients[row][left].powi(2))
            .sum::<f64>();
        let right_norm = (0..4)
            .map(|row| coefficients[row][right].powi(2))
            .sum::<f64>();
        left_norm.total_cmp(&right_norm)
    })?;
    let pivot_norm = (0..4)
        .map(|row| coefficients[row][pivot].powi(2))
        .sum::<f64>()
        .sqrt();
    if !pivot_norm.is_finite() || pivot_norm <= 1e-14 {
        return None;
    }
    let p = std::array::from_fn(|row| coefficients[row][pivot] / pivot_norm);
    let mut q = std::array::from_fn(|column| {
        (0..4)
            .map(|row| p[row] * coefficients[row][column])
            .sum::<f64>()
    });
    let q_norm = q.iter().map(|value| value * value).sum::<f64>().sqrt();
    if !q_norm.is_finite() || q_norm <= 1e-14 {
        return None;
    }
    for value in &mut q {
        *value /= q_norm;
    }

    let reconstructed = real_frame(p, q);
    let error = (0..4)
        .flat_map(|row| (0..4).map(move |column| (row, column)))
        .map(|(row, column)| (reconstructed[(row, column)] - frame[(row, column)]).norm())
        .fold(0.0_f64, f64::max);
    (error < 1e-8).then_some((p, q))
}

fn spectral_data(c: [f64; 3], g: [f64; 3], t: [f64; 3]) -> ([C; 4], [C; 4], [[C; 4]; 2]) {
    let problem = PreparedSandwich::new(c, g, t);
    (problem.left, problem.right, problem.targets)
}

fn spin_characters(alpha: &[C; 4], gate: &[C; 4], o: &Mat4) -> (C, C) {
    let mut first = C::new(0.0, 0.0);
    for row in 0..4 {
        for column in 0..4 {
            first += alpha[row] * gate[column] * o[(row, column)].re.powi(2);
        }
    }
    let mut second = C::new(0.0, 0.0);
    for row0 in 0..4 {
        for row1 in row0 + 1..4 {
            for column0 in 0..4 {
                for column1 in column0 + 1..4 {
                    let minor = o[(row0, column0)].re * o[(row1, column1)].re
                        - o[(row0, column1)].re * o[(row1, column0)].re;
                    second +=
                        alpha[row0] * alpha[row1] * gate[column0] * gate[column1] * minor.powi(2);
                }
            }
        }
    }
    (first, second)
}

fn spin_residual(alpha: &[C; 4], gate: &[C; 4], target: &[C; 4], o: &Mat4) -> f64 {
    let (first, second) = spin_characters(alpha, gate, o);
    (first - target[0]).norm().max((second - target[1]).norm())
}

fn solve_fixed_action_equations(
    problem: &FixedActionProblem,
    q: [f64; 4],
    expected_p: Option<[f64; 4]>,
    order: [usize; 4],
    q1_raw: &Quadric,
    q2_raw: &Quadric,
    q4_raw: &Quartic,
) -> Option<(Mat4, f64)> {
    let debug = std::env::var_os("SPIN_DEBUG").is_some();
    if debug {
        eprintln!("spin alpha={:?}", problem.alpha);
        eprintln!("spin gate={:?}", problem.gate);
        eprintln!("spin target={:?}", problem.target);
    }
    let q1 = convert_quadric(q1_raw);
    let q2 = convert_quadric(q2_raw);
    let q4 = convert_quartic(q4_raw);
    if debug {
        if let Some(p) = expected_p.filter(|p| p[order[0]].abs() > 1e-14) {
            let (t, y, z) = (
                p[order[1]] / p[order[0]],
                p[order[2]] / p[order[0]],
                p[order[3]] / p[order[0]],
            );
            eprintln!(
                "spin expected raw equations=({:.3e},{:.3e},{:.3e})",
                eval_biquadric(&q1, t, y, z),
                eval_biquadric(&q2, t, y, z),
                eval_biquartic(&q4, t, y, z),
            );
        }
    }
    let (my, mz) = quotient_matrices(&q1, &q2)?;
    let multiplication_q4 = evaluate_quartic(&q4, &my, &mz);
    let selector = pmat_determinant(&multiplication_q4);
    let roots = selector_roots(&selector);
    if debug {
        let scale = selector
            .coefficients
            .iter()
            .fold(0.0_f64, |m, value| m.max(value.abs()));
        eprintln!(
            "spin selector scale={scale:.3e} coefficients={:?}",
            selector.coefficients
        );
        eprintln!("spin roots={roots:?}");
        if let Some(p) = expected_p.filter(|p| p[order[0]].abs() > 1e-14) {
            let expected_t = p[order[1]] / p[order[0]];
            let expected_y = p[order[2]] / p[order[0]];
            let expected_z = p[order[3]] / p[order[0]];
            let evaluation = [1.0, expected_y, expected_z, expected_y * expected_z];
            let expected_my = pmat_eval(&my, expected_t);
            let expected_mz = pmat_eval(&mz, expected_t);
            let expected_matrix = pmat_eval(&multiplication_q4, expected_t);
            let left_defect = |matrix: &Matrix4<f64>, eigenvalue: f64| {
                (0..4)
                    .map(|column| {
                        ((0..4)
                            .map(|row| evaluation[row] * matrix[(row, column)])
                            .sum::<f64>()
                            - eigenvalue * evaluation[column])
                            .abs()
                    })
                    .fold(0.0_f64, f64::max)
            };
            let annihilation = (0..4)
                .map(|column| {
                    (0..4)
                        .map(|row| evaluation[row] * expected_matrix[(row, column)])
                        .sum::<f64>()
                        .abs()
                })
                .fold(0.0_f64, f64::max);
            eprintln!(
                "spin expected t={expected_t:.16e} selector={:.3e} normalized={:.3e} direct_det={:.3e} quotient_defect=({:.3e},{:.3e}) annihilation={annihilation:.3e}",
                selector.eval(expected_t),
                selector.eval(expected_t) / scale.max(1e-300),
                expected_matrix.determinant(),
                left_defect(&expected_my, expected_y),
                left_defect(&expected_mz, expected_z),
            );
        }
    }
    let recover = |roots: Vec<C>| -> Option<(Mat4, f64)> {
        let mut best = None;
        for root in roots {
            if root.im.abs() > 1e-7 * (1.0 + root.re.abs()) || !root.re.is_finite() {
                continue;
            }
            let t = root.re;
            let my_value = pmat_eval(&my, t);
            let mz_value = pmat_eval(&mz, t);
            let q4_value = pmat_eval(&multiplication_q4, t);
            let adj = adjugate(&q4_value);
            let column = (0..4).max_by(|&left, &right| {
                adj.column(left)
                    .norm_squared()
                    .total_cmp(&adj.column(right).norm_squared())
            })?;
            let packet = adj.column(column).into_owned();
            let pivot = (0..4)
                .max_by(|&left, &right| packet[left].abs().total_cmp(&packet[right].abs()))?;
            if packet[pivot].abs() <= 1e-14 * packet.norm() {
                continue;
            }
            let y = (my_value * packet)[pivot] / packet[pivot];
            let z = (mz_value * packet)[pivot] / packet[pivot];
            if !y.is_finite() || !z.is_finite() {
                continue;
            }
            let mut local_p = [1.0, t, y, z];
            let norm = local_p
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt();
            if !norm.is_finite() || norm <= 0.0 {
                continue;
            }
            for value in &mut local_p {
                *value /= norm;
            }
            let mut p = [0.0; 4];
            for local in 0..4 {
                p[order[local]] = local_p[local];
            }
            let o = real_frame(p, q);
            let residual = spin_residual(&problem.alpha, &problem.gate, &problem.target, &o);
            if debug {
                eprintln!(
                    "spin candidate t={t:.16e} y={y:.16e} z={z:.16e} equations=({:.3e},{:.3e},{:.3e}) residual={residual:.3e}",
                    eval_biquadric(&q1, t, y, z),
                    eval_biquadric(&q2, t, y, z),
                    eval_biquartic(&q4, t, y, z),
                );
            }
            if residual < ACCEPT && best.as_ref().is_none_or(|(_, old)| residual < *old) {
                best = Some((o, residual));
            }
        }
        best
    };
    if let Some(solution) = recover(roots) {
        return Some(solution);
    }
    if debug {
        eprintln!("spin retrying identical selector with compensated determinant coefficients");
    }
    let compensated = pmat_determinant_compensated(&multiplication_q4);
    recover(selector_roots(&compensated))
}

fn solve_fixed_action_expected(
    problem: &FixedActionProblem,
    q: [f64; 4],
    expected_p: Option<[f64; 4]>,
    order: [usize; 4],
) -> Option<(Mat4, f64)> {
    let (q1, q2, q4) = spectral_equations(problem, q, order);
    solve_fixed_action_equations(problem, q, expected_p, order, &q1, &q2, &q4)
}

fn solve_prepared_action(problem: &FixedActionProblem, q: [f64; 4]) -> Option<(Mat4, f64)> {
    let first_order = AFFINE_CHARTS[0];
    let gate_basis = spin_basis(q, first_order);
    let (first_real, first_imaginary) = first_character_equations(problem, &gate_basis);
    if !first_character_supported(&first_real, &first_imaginary) {
        return None;
    }
    let second = second_character_equation(problem, &gate_basis);
    // Algebraically, one generic chart plus three coordinate walls covers
    // projective p-space. Numerically the overlapping full charts are still
    // load-bearing: the degree-16 companion can lose a generic real root in a
    // badly scaled affine coordinate and recover it after repivoting.
    solve_fixed_action_equations(
        problem,
        q,
        None,
        first_order,
        &first_real,
        &first_imaginary,
        &second,
    )
    .or_else(|| {
        AFFINE_CHARTS[1..]
            .iter()
            .copied()
            .find_map(|order| solve_fixed_action_expected(problem, q, None, order))
    })
}

#[cfg(test)]
fn solve_fixed_action(
    alpha: &[C; 4],
    gate: &[C; 4],
    target: &[C; 4],
    q: [f64; 4],
) -> Option<(Mat4, f64)> {
    solve_prepared_action(&FixedActionProblem::new(alpha, gate, target), q)
}

#[cfg(any(test, feature = "research-spin"))]
fn stereographic_quaternion(value: [f64; 3]) -> [f64; 4] {
    let norm_squared = value.iter().map(|entry| entry * entry).sum::<f64>();
    let denominator = 1.0 + norm_squared;
    [
        (1.0 - norm_squared) / denominator,
        2.0 * value[0] / denominator,
        2.0 * value[1] / denominator,
        2.0 * value[2] / denominator,
    ]
}

/// Deterministic dense action sequence on the unit three-sphere.
///
/// The orbit `k*(sqrt(2),sqrt(3),sqrt(5)) mod 1` is dense in the three-torus
/// by Kronecker's theorem. The Hopf-coordinate map below is continuous and
/// surjective onto `S^3`, so these actions meet every open regular action
/// image. Index zero keeps the identity action as the natural first probe.
fn spread_quaternion(index: usize) -> [f64; 4] {
    if index == 0 {
        return [1.0, 0.0, 0.0, 0.0];
    }
    let k = index as f64;
    let fraction = |value: f64| value - value.floor();
    let radius_squared = fraction(0.5 + k * std::f64::consts::SQRT_2);
    let left_angle = std::f64::consts::TAU * fraction(k * 3.0_f64.sqrt());
    let right_angle = std::f64::consts::TAU * fraction(k * 5.0_f64.sqrt());
    let (left_sine, left_cosine) = left_angle.sin_cos();
    let (right_sine, right_cosine) = right_angle.sin_cos();
    let left_radius = (1.0 - radius_squared).sqrt();
    let right_radius = radius_squared.sqrt();
    [
        left_radius * left_cosine,
        left_radius * left_sine,
        right_radius * right_cosine,
        right_radius * right_sine,
    ]
}

/// Research route for the theorem-backed fair Spin scheduler. A bounded
/// denominator is only an acceleration prefix: exhaustion returns `None`,
/// never an infeasibility decision. Both full-half projections and both
/// central target lifts are tried.
#[cfg(any(test, feature = "research-spin"))]
fn try_action(
    problems: &[FixedActionProblem; 4],
    action: [f64; 4],
    attempts: &mut usize,
) -> Option<Solution> {
    for mode in 0..4 {
        *attempts += 1;
        if let Some(solution) = try_action_mode(problems, action, mode) {
            return Some(solution);
        }
    }
    None
}

fn try_action_mode(
    problems: &[FixedActionProblem; 4],
    action: [f64; 4],
    mode: usize,
) -> Option<Solution> {
    let swapped = mode % 2 == 1;
    solve_prepared_action(&problems[mode], action).map(|(o, residual)| Solution {
        takagi: None,
        rho_branch: false,
        orbit_rep: 0,
        o: if swapped { o.transpose() } else { o },
        rung: Rung::Spin,
        residual,
    })
}

fn action_problems(
    alpha: &[C; 4],
    gate: &[C; 4],
    targets: &[[C; 4]; 2],
) -> [FixedActionProblem; 4] {
    std::array::from_fn(|mode| {
        if mode % 2 == 0 {
            FixedActionProblem::new(alpha, gate, &targets[mode / 2])
        } else {
            FixedActionProblem::new(gate, alpha, &targets[mode / 2])
        }
    })
}

#[cfg(feature = "research-spin")]
pub fn solve_dyadic(
    c: [f64; 3],
    g: [f64; 3],
    t: [f64; 3],
    maximum_denominator: i32,
) -> (Option<Solution>, usize) {
    let (alpha, gate, targets) = spectral_data(c, g, t);
    let problems = action_problems(&alpha, &gate, &targets);

    let mut attempts = 0usize;
    let mut denominator = 1_i32;
    while denominator <= maximum_denominator {
        for x in -denominator..=denominator {
            for y in -denominator..=denominator {
                for z in -denominator..=denominator {
                    if denominator > 1 && x % 2 == 0 && y % 2 == 0 && z % 2 == 0 {
                        continue;
                    }
                    let action = stereographic_quaternion([
                        x as f64 / denominator as f64,
                        y as f64 / denominator as f64,
                        z as f64 / denominator as f64,
                    ]);
                    if let Some(solution) = try_action(&problems, action, &mut attempts) {
                        return (Some(solution), attempts);
                    }
                }
            }
        }
        denominator *= 2;
    }
    (None, attempts)
}

/// Dense Kronecker--Hopf action order for the projective Spin fallback.
/// Every quaternion is tried in both target lifts and factor orientations. A
/// bounded prefix is only an acceleration experiment; exhaustion never
/// certifies infeasibility.
#[cfg(any(test, feature = "research-spin"))]
pub fn solve_spread(
    c: [f64; 3],
    g: [f64; 3],
    t: [f64; 3],
    maximum_actions: usize,
) -> (Option<Solution>, usize, usize) {
    let (alpha, gate, targets) = spectral_data(c, g, t);
    let problems = action_problems(&alpha, &gate, &targets);
    let mut attempts = 0usize;
    for index in 0..maximum_actions {
        if let Some(solution) = try_action(&problems, spread_quaternion(index), &mut attempts) {
            return (Some(solution), attempts, index + 1);
        }
    }
    (None, attempts, maximum_actions)
}

/// Enumerate every algebraic candidate in a bounded dense-action prefix.
/// Near a root collision the coefficient residual can admit a candidate that
/// the rootwise compiler rejects, so production must not let that first proxy
/// hit hide later actions or target/orientation modes.
#[cfg(feature = "research-spin")]
pub fn spread_candidates(
    c: [f64; 3],
    g: [f64; 3],
    t: [f64; 3],
    maximum_actions: usize,
) -> Vec<Solution> {
    let (alpha, gate, targets) = spectral_data(c, g, t);
    let problems = action_problems(&alpha, &gate, &targets);
    let mut candidates = Vec::new();
    for index in 0..maximum_actions {
        let action = spread_quaternion(index);
        for mode in 0..4 {
            if let Some(solution) = try_action_mode(&problems, action, mode) {
                candidates.push(solution);
            }
        }
    }
    candidates
}

/// Diagnostic inverse: recover the true right Spin action from an existing
/// certified witness and replay only that fixed-action algebraic kernel.
/// This distinguishes action-selection failure from kernel incompleteness.
#[cfg(feature = "research-spin")]
pub fn replay_frame_action(
    c: [f64; 3],
    g: [f64; 3],
    t: [f64; 3],
    frame: &Mat4,
) -> Option<Solution> {
    let (p, q) = spin_factors(frame)?;
    let (alpha, gate, targets) = spectral_data(c, g, t);
    if std::env::var_os("SPIN_DEBUG").is_some() {
        eprintln!("spin recovered p={p:?} q={q:?}");
        for (index, target) in targets.iter().enumerate() {
            eprintln!(
                "spin supplied-frame target[{index}] residual={:.3e}",
                spin_residual(&alpha, &gate, target, frame)
            );
        }
    }
    for target in &targets {
        let problem = FixedActionProblem::new(&alpha, &gate, target);
        for order in AFFINE_CHARTS {
            if let Some((o, residual)) = solve_fixed_action_expected(&problem, q, Some(p), order) {
                return Some(Solution {
                    takagi: None,
                    rho_branch: false,
                    orbit_rep: 0,
                    o,
                    rung: Rung::Spin,
                    residual,
                });
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::can_sandwich::{eigphases, esym4, weyl_from_monodromy};

    fn dense_fixture() -> ([C; 4], [C; 4], [C; 4]) {
        let alpha = [
            C::new(-21.0 / 221.0, 220.0 / 221.0),
            C::new(-3741.0 / 8741.0, 7900.0 / 8741.0),
            C::new(-82512.0 / 83137.0, 10175.0 / 83137.0),
            C::new(
                146985891768.0 / 160600814257.0,
                -64712975215.0 / 160600814257.0,
            ),
        ];
        let gate = [
            C::new(48.0 / 577.0, 575.0 / 577.0),
            C::new(-232.0 / 857.0, 825.0 / 857.0),
            C::new(-21.0 / 29.0, -20.0 / 29.0),
            C::new(8319731.0 / 14340181.0, -11680020.0 / 14340181.0),
        ];
        let first = C::new(
            319113923785107667807701200.0 / 121844609829979135404338495189.0,
            -2037564135460516246602443896320.0 / 2071358367109645301873754418213.0,
        );
        let second = C::new(
            -319936062719280224561019133032160578328078.0
                / 143306126644568955208912313309549136531289.0,
            0.0,
        );
        (alpha, gate, [first, second, first.conj(), C::new(1.0, 0.0)])
    }

    fn planted_case(index: usize) -> (FixedActionProblem, [f64; 4], [f64; 4]) {
        let k = index as f64 + 1.0;
        let spectrum = |offset: f64| {
            let first = 0.83 * (0.37 * k + offset).sin();
            let second = 0.71 * (0.53 * k + 1.7 * offset).cos();
            let third = 0.59 * (0.79 * k - 0.4 * offset).sin();
            [first, second, third, -first - second - third].map(|angle| C::from_polar(1.0, angle))
        };
        let unit_quaternion = |offset: f64| {
            let mut value = [
                (0.31 * k + offset).sin(),
                (0.47 * k - 0.7 * offset).cos(),
                (0.67 * k + 1.3 * offset).sin(),
                (0.89 * k - 1.1 * offset).cos(),
            ];
            let norm = value.iter().map(|entry| entry * entry).sum::<f64>().sqrt();
            for entry in &mut value {
                *entry /= norm;
            }
            value
        };
        let alpha = spectrum(0.19);
        let gate = spectrum(0.61);
        let p = unit_quaternion(0.23);
        let q = unit_quaternion(0.77);
        let frame = real_frame(p, q);
        let (first, second) = spin_characters(&alpha, &gate, &frame);
        let target = [first, second, first.conj(), C::new(1.0, 0.0)];
        (FixedActionProblem::new(&alpha, &gate, &target), p, q)
    }

    #[test]
    fn degree_sixteen_kernel_solves_dense_exact_fixture() {
        let (alpha, gate, target) = dense_fixture();
        let q = [3.0 / 5.0, -8.0 / 15.0, -8.0 / 15.0, -4.0 / 15.0];
        let (_, residual) = solve_fixed_action(&alpha, &gate, &target, q)
            .expect("the exact rational Spin action must realize the dense fixture");
        assert!(residual < ACCEPT, "residual={residual:.3e}");
    }

    #[test]
    fn planted_actions_satisfy_compiled_equations() {
        fn homogeneous_quadric(value: &Quadric, coordinates: [f64; 4]) -> f64 {
            let [x, t, y, z] = coordinates;
            (0..3)
                .flat_map(|td| (0..3).flat_map(move |yd| (0..3).map(move |zd| (td, yd, zd))))
                .filter(|&(td, yd, zd)| td + yd + zd <= 2)
                .map(|(td, yd, zd)| {
                    value[td][yd][zd]
                        * x.powi((2 - td - yd - zd) as i32)
                        * t.powi(td as i32)
                        * y.powi(yd as i32)
                        * z.powi(zd as i32)
                })
                .sum()
        }

        fn homogeneous_quartic(value: &Quartic, coordinates: [f64; 4]) -> f64 {
            let [x, t, y, z] = coordinates;
            (0..5)
                .flat_map(|td| (0..5).flat_map(move |yd| (0..5).map(move |zd| (td, yd, zd))))
                .filter(|&(td, yd, zd)| td + yd + zd <= 4)
                .map(|(td, yd, zd)| {
                    value[td][yd][zd]
                        * x.powi((4 - td - yd - zd) as i32)
                        * t.powi(td as i32)
                        * y.powi(yd as i32)
                        * z.powi(zd as i32)
                })
                .sum()
        }

        for index in 0..64 {
            let (problem, p, q) = planted_case(index);
            for order in AFFINE_CHARTS {
                let local = [p[order[0]], p[order[1]], p[order[2]], p[order[3]]];
                let (first_real, first_imaginary, second) = spectral_equations(&problem, q, order);
                let residual = homogeneous_quadric(&first_real, local)
                    .abs()
                    .max(homogeneous_quadric(&first_imaginary, local).abs())
                    .max(homogeneous_quartic(&second, local).abs());
                assert!(
                    residual < 1e-12,
                    "index={index} order={order:?} equation residual={residual:.3e}"
                );
            }
        }
    }

    #[test]
    fn compensated_selector_recovers_cancellation_fixtures() {
        for index in [0, 14, 113, 155, 179, 207, 254, 314, 328, 427, 469, 487] {
            let (problem, _, q) = planted_case(index);
            let (_, residual) = solve_prepared_action(&problem, q)
                .unwrap_or_else(|| panic!("compensated selector lost planted action {index}"));
            assert!(residual < ACCEPT, "index={index} residual={residual:.3e}");
        }
    }

    #[test]
    fn spectral_data_passes_characteristic_coefficients_not_roots() {
        let t = [0.31, 0.17, 0.06];
        let (_, _, targets) = spectral_data([0.27, 0.13, 0.05], [0.23, 0.11, 0.03], t);
        let roots = eigphases(weyl_from_monodromy(t)).map(|phase| C::from_polar(1.0, 2.0 * phase));
        let expected = esym4(roots);
        let error = (0..4)
            .map(|index| (targets[0][index] - expected[index]).norm())
            .fold(0.0_f64, f64::max);
        assert!(error < 1e-14, "target coefficient error={error:.3e}");
        assert!((targets[0][3] - C::new(1.0, 0.0)).norm() < 1e-14);
    }

    #[test]
    fn quotient_element_reduction_matches_the_monomial_oracle() {
        let (alpha, gate, target) = dense_fixture();
        let problem = FixedActionProblem::new(&alpha, &gate, &target);
        let q = [3.0 / 5.0, -8.0 / 15.0, -8.0 / 15.0, -4.0 / 15.0];
        for order in AFFINE_CHARTS {
            let (q1, q2, q4) = spectral_equations(&problem, q, order);
            let (my, mz) = quotient_matrices(&convert_quadric(&q1), &convert_quadric(&q2))
                .expect("dense fixture quotient chart");
            let q4 = convert_quartic(&q4);
            let reduced = evaluate_quartic(&q4, &my, &mz);
            let oracle = evaluate_quartic_reference(&q4, &my, &mz);
            let scale = oracle
                .iter()
                .flatten()
                .flat_map(|entry| entry.coefficients)
                .fold(1.0_f64, |maximum, value| maximum.max(value.abs()));
            let error = reduced
                .iter()
                .flatten()
                .zip(oracle.iter().flatten())
                .flat_map(|(left, right)| {
                    left.coefficients
                        .into_iter()
                        .zip(right.coefficients)
                        .map(|(a, b)| (a - b).abs())
                })
                .fold(0.0_f64, f64::max);
            assert!(error <= 2e-10 * scale, "order={order:?} error={error:.3e}");
        }
    }

    #[test]
    fn first_character_pencil_separates_only_empty_real_intersections() {
        let mut positive = [[[0.0; 3]; 3]; 3];
        for coordinate in 0..4 {
            add_quadratic_monomial(&mut positive, coordinate, coordinate, 1.0);
        }
        assert!(!first_character_supported(&positive, &[[[0.0; 3]; 3]; 3]));

        let mut left = [[[0.0; 3]; 3]; 3];
        let mut right = [[[0.0; 3]; 3]; 3];
        for (coordinate, coefficient) in [1.0, -1.0, 1.0, -1.0].into_iter().enumerate() {
            add_quadratic_monomial(&mut left, coordinate, coordinate, coefficient);
        }
        for (coordinate, coefficient) in [1.0, 1.0, -1.0, -1.0].into_iter().enumerate() {
            add_quadratic_monomial(&mut right, coordinate, coordinate, coefficient);
        }
        // (1,1,1,1) is a common isotropic vector.
        assert!(first_character_supported(&left, &right));

        let (alpha, gate, target) = dense_fixture();
        let problem = FixedActionProblem::new(&alpha, &gate, &target);
        let q = [3.0 / 5.0, -8.0 / 15.0, -8.0 / 15.0, -4.0 / 15.0];
        let (left, right, _) = spectral_equations(&problem, q, AFFINE_CHARTS[0]);
        assert!(first_character_supported(&left, &right));
    }

    #[test]
    fn frobenius_spin_factorization_roundtrips_a_dense_frame() {
        let mut p = [0.37, -0.41, 0.71, 0.43];
        let mut q = [-0.29, 0.61, -0.17, 0.71];
        for value in [&mut p, &mut q] {
            let norm = value.iter().map(|entry| entry * entry).sum::<f64>().sqrt();
            for entry in value.iter_mut() {
                *entry /= norm;
            }
        }
        let frame = real_frame(p, q);
        let (recovered_p, recovered_q) = spin_factors(&frame).expect("Spin factorization");
        let recovered = real_frame(recovered_p, recovered_q);
        let error = (0..4)
            .flat_map(|row| (0..4).map(move |column| (row, column)))
            .map(|(row, column)| (frame[(row, column)] - recovered[(row, column)]).norm())
            .fold(0.0_f64, f64::max);
        assert!(error < 1e-14, "Spin roundtrip error={error:.3e}");
    }

    #[test]
    #[ignore = "microbenchmark for the retained-action algebraic kernel"]
    fn benchmark_degree_sixteen_kernel() {
        let (alpha, gate, target) = dense_fixture();
        let q = [3.0 / 5.0, -8.0 / 15.0, -8.0 / 15.0, -4.0 / 15.0];
        let repeats = 1001;
        let mut samples = Vec::with_capacity(repeats);
        for _ in 0..repeats {
            let started = std::time::Instant::now();
            let result = solve_fixed_action(&alpha, &gate, &target, q);
            std::hint::black_box(result);
            samples.push(started.elapsed().as_nanos() as f64 / 1000.0);
        }
        samples.sort_by(f64::total_cmp);
        let average = samples.iter().sum::<f64>() / samples.len() as f64;
        eprintln!(
            "fixed Spin action: average={average:.3} us slowest={:.3} us median={:.3} us",
            samples[samples.len() - 1],
            samples[samples.len() / 2],
        );
    }

    #[test]
    #[ignore = "stage census for the retained-action algebraic kernel"]
    fn benchmark_degree_sixteen_kernel_stages() {
        let (alpha, gate, target) = dense_fixture();
        let problem = FixedActionProblem::new(&alpha, &gate, &target);
        let q = [3.0 / 5.0, -8.0 / 15.0, -8.0 / 15.0, -4.0 / 15.0];
        let order = AFFINE_CHARTS[0];
        let repeats = 1001;
        let mut totals = [0u128; 6];
        let mut slowest = [0u128; 6];
        let record = |index: usize,
                      started: std::time::Instant,
                      totals: &mut [u128; 6],
                      slowest: &mut [u128; 6]| {
            let elapsed = started.elapsed().as_nanos();
            totals[index] += elapsed;
            slowest[index] = slowest[index].max(elapsed);
        };

        // Exclude one-time lazy chart construction from the steady-state
        // algebra census. The ordinary benchmark above intentionally retains
        // the cold-start sample in its reported slowest time.
        std::hint::black_box(&*PROJECTIVE_CHARTS);
        for _ in 0..repeats {
            let started = std::time::Instant::now();
            let (q1_raw, q2_raw, q4_raw) = spectral_equations(&problem, q, order);
            record(0, started, &mut totals, &mut slowest);

            let started = std::time::Instant::now();
            assert!(first_character_supported(&q1_raw, &q2_raw));
            record(1, started, &mut totals, &mut slowest);

            let started = std::time::Instant::now();
            let q1 = convert_quadric(&q1_raw);
            let q2 = convert_quadric(&q2_raw);
            let q4 = convert_quartic(&q4_raw);
            let (my, mz) = quotient_matrices(&q1, &q2).expect("dense quotient chart");
            record(2, started, &mut totals, &mut slowest);

            let started = std::time::Instant::now();
            let multiplication_q4 = evaluate_quartic(&q4, &my, &mz);
            record(3, started, &mut totals, &mut slowest);

            let started = std::time::Instant::now();
            let selector = pmat_determinant(&multiplication_q4);
            record(4, started, &mut totals, &mut slowest);

            let started = std::time::Instant::now();
            let roots = selector_roots(&selector);
            std::hint::black_box(roots);
            record(5, started, &mut totals, &mut slowest);
        }
        for (name, index) in [
            ("equations", 0),
            ("support", 1),
            ("quotient", 2),
            ("norm matrix", 3),
            ("selector determinant", 4),
            ("degree-16 roots", 5),
        ] {
            eprintln!(
                "fixed Spin stage {name}: average={:.3} us slowest={:.3} us",
                totals[index] as f64 / repeats as f64 / 1000.0,
                slowest[index] as f64 / 1000.0,
            );
        }
    }

    #[test]
    #[ignore = "planted latency census for the compiled fixed-action kernel"]
    fn benchmark_planted_fixed_action_census() {
        let mut samples = Vec::with_capacity(512);
        let mut failures = Vec::new();
        for index in 0..512 {
            let (problem, _, q) = planted_case(index);
            let started = std::time::Instant::now();
            let result = solve_prepared_action(&problem, q);
            samples.push(started.elapsed().as_nanos() as f64 / 1000.0);
            if result.is_none() {
                failures.push(index);
            }
        }
        samples.sort_by(f64::total_cmp);
        let average = samples.iter().sum::<f64>() / samples.len() as f64;
        eprintln!(
            "planted fixed actions: solved={}/{} average={average:.3} us slowest={:.3} us median={:.3} us failures={failures:?}",
            samples.len() - failures.len(),
            samples.len(),
            samples[samples.len() - 1],
            samples[samples.len() / 2],
        );
    }

    #[test]
    #[ignore = "research benchmark for theorem-backed dyadic action enumeration"]
    fn benchmark_dyadic_action_search_on_dense_fixture() {
        let (alpha, gate, target) = dense_fixture();
        let started = std::time::Instant::now();
        let mut attempts = 0usize;
        let mut slowest = 0.0_f64;
        let mut found = None;
        'denominators: for denominator in [1_i32, 2, 4, 8] {
            for x in -denominator..=denominator {
                for y in -denominator..=denominator {
                    for z in -denominator..=denominator {
                        if denominator > 1 && x % 2 == 0 && y % 2 == 0 && z % 2 == 0 {
                            continue;
                        }
                        let action = [
                            x as f64 / denominator as f64,
                            y as f64 / denominator as f64,
                            z as f64 / denominator as f64,
                        ];
                        let call_started = std::time::Instant::now();
                        let result = solve_fixed_action(
                            &alpha,
                            &gate,
                            &target,
                            stereographic_quaternion(action),
                        );
                        let elapsed = call_started.elapsed().as_nanos() as f64 / 1000.0;
                        slowest = slowest.max(elapsed);
                        attempts += 1;
                        if let Some((_, residual)) = result {
                            found = Some((denominator, action, residual));
                            break 'denominators;
                        }
                    }
                }
            }
        }
        let total = started.elapsed().as_nanos() as f64 / 1000.0;
        let hit = found.expect("the dyadic action search should meet the open action image");
        eprintln!(
            "dyadic Spin search: attempts={attempts} hit={hit:?} total={total:.3} us average={:.3} us slowest={slowest:.3} us",
            total / attempts as f64,
        );
    }

    #[test]
    #[ignore = "research benchmark for the dense Kronecker--Hopf action order"]
    fn benchmark_spread_action_search_on_dense_fixture() {
        let (alpha, gate, target) = dense_fixture();
        let started = std::time::Instant::now();
        let mut samples = Vec::new();
        let mut found = None;
        for index in 0..1000 {
            let call_started = std::time::Instant::now();
            let result = solve_fixed_action(&alpha, &gate, &target, spread_quaternion(index));
            samples.push(call_started.elapsed().as_nanos() as f64 / 1000.0);
            if let Some((_, residual)) = result {
                found = Some((index, residual));
                break;
            }
        }
        let hit = found.expect("the spread action order should meet the open action image");
        let total = started.elapsed().as_nanos() as f64 / 1000.0;
        let average = samples.iter().sum::<f64>() / samples.len() as f64;
        let slowest = samples.iter().copied().fold(0.0_f64, f64::max);
        eprintln!(
            "spread Spin search: attempts={} hit={hit:?} total={total:.3} us average={average:.3} us slowest={slowest:.3} us",
            samples.len(),
        );
    }

    #[test]
    #[ignore = "research benchmark for the folded production counterexample"]
    fn benchmark_spread_on_folded_counterexample() {
        let input = (
            [
                0.320_387_857_027_5,
                0.265_146_172_188_37,
                -0.066_006_359_790_49,
            ],
            [
                0.293_629_518_344_32,
                0.236_744_755_386_73,
                -0.151_493_215_321_9,
            ],
            [
                0.455_019_768_623_71,
                0.049_468_366_219_53,
                0.029_722_997_536_99,
            ],
        );
        let started = std::time::Instant::now();
        let (solution, attempts, actions) = solve_spread(input.0, input.1, input.2, 1024);
        let elapsed = started.elapsed().as_nanos() as f64 / 1000.0;
        let solution = solution.expect("the dense action order must realize the folded fixture");
        eprintln!(
            "folded spread counterexample: actions={actions} kernel_calls={attempts} total={elapsed:.3} us residual={:.3e}",
            solution.residual,
        );
    }

    #[test]
    fn spread_selector_handles_near_defective_degree_sixteen_companion() {
        // Haar row 20946 made nalgebra's fixed-size companion QR iteration
        // run for tens of seconds on the second spread action.  The shared
        // fallible faer backend must either recover or decline that action;
        // here it recovers a certified witness within the first two actions.
        let input = (
            [
                0.381_610_277_759_55,
                0.095_645_415_199_63,
                0.033_404_301_767_87,
            ],
            [
                0.385_899_736_354_26,
                0.214_714_996_535_49,
                -0.187_951_922_599_11,
            ],
            [
                0.497_155_126_507_68,
                0.230_588_165_760_59,
                -0.269_908_123_688_72,
            ],
        );
        let (solution, _, actions) = solve_spread(input.0, input.1, input.2, 2);
        let solution = solution.expect("two-action spread prefix must recover the regression row");
        assert_eq!(actions, 2);
        assert!(
            solution.residual < ACCEPT,
            "residual={:.3e}",
            solution.residual
        );
    }
}
