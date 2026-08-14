//! Small, ownership-safe wrapper around FLINT/Arb's certified polynomial
//! root isolator.  This module intentionally exposes one operation only: all
//! distinct real roots of a real power polynomial.  Input `f64` coefficients
//! are exact dyadic rationals, so FLINT first computes `f/gcd(f,f')` over
//! `QQ`; Arb then isolates the square-free roots at 256 bits.  Multiplicity is
//! irrelevant to witness selection and is no longer a decline stratum.

use std::{ffi::c_int, ptr};

type Slong = i64;

#[repr(C)]
#[derive(Clone, Copy)]
struct MantissaPtr {
    alloc: Slong,
    limbs: *mut u64,
}

#[repr(C)]
union Mantissa {
    inline: [u64; 2],
    pointer: MantissaPtr,
}

#[repr(C)]
struct Arf {
    exponent: Slong,
    size: Slong,
    mantissa: Mantissa,
}

#[repr(C)]
struct Mag {
    exponent: Slong,
    mantissa: u64,
}

#[repr(C)]
struct Arb {
    midpoint: Arf,
    radius: Mag,
}

#[repr(C)]
struct Acb {
    real: Arb,
    imag: Arb,
}

#[repr(C)]
struct AcbPoly {
    coefficients: *mut Acb,
    allocation: Slong,
    length: Slong,
}

#[repr(C)]
struct Fmpq {
    numerator: Slong,
    denominator: Slong,
}

#[repr(C)]
struct FmpqPoly {
    coefficients: *mut Slong,
    allocation: Slong,
    length: Slong,
    denominator: Slong,
}

#[link(name = "flint")]
unsafe extern "C" {
    fn acb_set_d_d(value: *mut Acb, real: f64, imaginary: f64);

    fn acb_poly_init(poly: *mut AcbPoly);
    fn acb_poly_clear(poly: *mut AcbPoly);
    fn acb_poly_set_fmpq_poly(poly: *mut AcbPoly, source: *const FmpqPoly, precision: Slong);
    fn acb_poly_find_roots(
        roots: *mut Acb,
        poly: *const AcbPoly,
        initial: *const Acb,
        max_iterations: Slong,
        prec: Slong,
    ) -> Slong;
    fn acb_poly_validate_real_roots(roots: *const Acb, poly: *const AcbPoly, prec: Slong) -> c_int;

    fn _acb_vec_init(length: Slong) -> *mut Acb;
    fn _acb_vec_clear(values: *mut Acb, length: Slong);
    fn arb_contains_zero(value: *const Arb) -> c_int;
    fn arf_get_d(value: *const Arf, rounding: c_int) -> f64;

    fn fmpq_init(value: *mut Fmpq);
    fn fmpq_clear(value: *mut Fmpq);
    fn fmpq_set_si(value: *mut Fmpq, numerator: Slong, denominator: u64);
    fn fmpq_mul_2exp(result: *mut Fmpq, value: *const Fmpq, exponent: u64);
    fn fmpq_div_2exp(result: *mut Fmpq, value: *const Fmpq, exponent: u64);

    fn fmpq_poly_init(poly: *mut FmpqPoly);
    fn fmpq_poly_clear(poly: *mut FmpqPoly);
    fn fmpq_poly_set_coeff_fmpq(poly: *mut FmpqPoly, degree: Slong, value: *const Fmpq);
    fn fmpq_poly_derivative(result: *mut FmpqPoly, source: *const FmpqPoly);
    fn fmpq_poly_gcd(result: *mut FmpqPoly, left: *const FmpqPoly, right: *const FmpqPoly);
    fn fmpq_poly_div(
        result: *mut FmpqPoly,
        numerator: *const FmpqPoly,
        denominator: *const FmpqPoly,
    );
}

struct Polynomial(AcbPoly);

impl Polynomial {
    fn new() -> Self {
        let mut value = std::mem::MaybeUninit::<AcbPoly>::uninit();
        unsafe {
            acb_poly_init(value.as_mut_ptr());
            Self(value.assume_init())
        }
    }

    fn set_rational_polynomial(&mut self, source: &RationalPolynomial, precision: Slong) {
        unsafe { acb_poly_set_fmpq_poly(&mut self.0, &source.0, precision) }
    }
}

struct Rational(Fmpq);

impl Rational {
    fn from_f64(value: f64) -> Option<Self> {
        if !value.is_finite() {
            return None;
        }
        let mut rational = std::mem::MaybeUninit::<Fmpq>::uninit();
        unsafe { fmpq_init(rational.as_mut_ptr()) };
        let mut rational = Self(unsafe { rational.assume_init() });
        if value == 0.0 {
            unsafe { fmpq_set_si(&mut rational.0, 0, 1) };
            return Some(rational);
        }

        let bits = value.to_bits();
        let negative = bits >> 63 != 0;
        let exponent_bits = ((bits >> 52) & 0x7ff) as i32;
        let fraction = bits & ((1u64 << 52) - 1);
        let (mantissa, exponent) = if exponent_bits == 0 {
            (fraction, 1 - 1023 - 52)
        } else {
            (fraction | (1u64 << 52), exponent_bits - 1023 - 52)
        };
        let signed = if negative {
            -(mantissa as i64)
        } else {
            mantissa as i64
        };
        unsafe { fmpq_set_si(&mut rational.0, signed, 1) };
        let source = std::ptr::addr_of!(rational.0);
        let destination = std::ptr::addr_of_mut!(rational.0);
        if exponent >= 0 {
            unsafe { fmpq_mul_2exp(destination, source, exponent as u64) };
        } else {
            unsafe { fmpq_div_2exp(destination, source, (-exponent) as u64) };
        }
        Some(rational)
    }
}

impl Drop for Rational {
    fn drop(&mut self) {
        unsafe { fmpq_clear(&mut self.0) }
    }
}

struct RationalPolynomial(FmpqPoly);

impl RationalPolynomial {
    fn new() -> Self {
        let mut value = std::mem::MaybeUninit::<FmpqPoly>::uninit();
        unsafe {
            fmpq_poly_init(value.as_mut_ptr());
            Self(value.assume_init())
        }
    }

    fn from_f64(coefficients: &[f64]) -> Option<Self> {
        let mut polynomial = Self::new();
        for (degree, &coefficient) in coefficients.iter().enumerate() {
            let rational = Rational::from_f64(coefficient)?;
            unsafe { fmpq_poly_set_coeff_fmpq(&mut polynomial.0, degree as Slong, &rational.0) };
        }
        Some(polynomial)
    }

    fn degree(&self) -> usize {
        self.0.length.saturating_sub(1) as usize
    }

    fn square_free_part(&self) -> Self {
        let mut derivative = Self::new();
        let mut divisor = Self::new();
        let mut result = Self::new();
        unsafe {
            fmpq_poly_derivative(&mut derivative.0, &self.0);
            fmpq_poly_gcd(&mut divisor.0, &self.0, &derivative.0);
            fmpq_poly_div(&mut result.0, &self.0, &divisor.0);
        }
        result
    }

    fn gcd(&self, other: &Self) -> Self {
        let mut result = Self::new();
        unsafe { fmpq_poly_gcd(&mut result.0, &self.0, &other.0) };
        result
    }
}

impl Drop for RationalPolynomial {
    fn drop(&mut self) {
        unsafe { fmpq_poly_clear(&mut self.0) }
    }
}

impl Drop for Polynomial {
    fn drop(&mut self) {
        unsafe { acb_poly_clear(&mut self.0) }
    }
}

fn isolate_rational_roots(rational: &RationalPolynomial, seeds: &[(f64, f64)]) -> Option<Vec<f64>> {
    const PRECISION: Slong = 256;
    const ARF_ROUND_NEAREST: c_int = 4;

    let square_free = rational.square_free_part();
    let degree = square_free.degree();
    if degree == 0 {
        return Some(Vec::new());
    }
    let mut polynomial = Polynomial::new();
    polynomial.set_rational_polynomial(&square_free, PRECISION);

    let roots = unsafe { _acb_vec_init(degree as Slong) };
    if roots.is_null() {
        return None;
    }
    let initial = if seeds.len() == degree
        && seeds
            .iter()
            .all(|(re, im)| re.is_finite() && im.is_finite())
    {
        let values = unsafe { _acb_vec_init(degree as Slong) };
        if values.is_null() {
            unsafe { _acb_vec_clear(roots, degree as Slong) };
            return None;
        }
        for (index, &(real, imaginary)) in seeds.iter().enumerate() {
            unsafe { acb_set_d_d(values.add(index), real, imaginary) };
        }
        values
    } else {
        ptr::null_mut()
    };
    let isolated = unsafe { acb_poly_find_roots(roots, &polynomial.0, initial, 0, PRECISION) };
    if !initial.is_null() {
        unsafe { _acb_vec_clear(initial, degree as Slong) };
    }
    let real_valid = isolated == degree as Slong
        && unsafe { acb_poly_validate_real_roots(roots, &polynomial.0, PRECISION) != 0 };
    if !real_valid {
        unsafe { _acb_vec_clear(roots, degree as Slong) };
        return None;
    }

    let mut real = Vec::new();
    for index in 0..degree {
        let root = unsafe { &*roots.add(index) };
        if unsafe { arb_contains_zero(&root.imag) != 0 } {
            let midpoint = unsafe { arf_get_d(&root.real.midpoint, ARF_ROUND_NEAREST) };
            if midpoint.is_finite() {
                real.push(midpoint);
            }
        }
    }
    unsafe { _acb_vec_clear(roots, degree as Slong) };
    real.sort_by(f64::total_cmp);
    Some(real)
}

/// Isolate every real root of `sum_k coefficients[k] x^k`.
///
/// This avoids a basis conversion when an eliminant has already been derived
/// coefficientwise in the power basis.  Repeated factors are removed exactly
/// over `QQ`; unresolved square-free input declines instead of returning a
/// partial root set. Seeds only accelerate `acb_poly_find_roots`; validation
/// is identical to the unseeded path and a partial isolation still declines.
pub(super) fn real_roots_power_seeded(
    coefficients: &[f64],
    seeds: &[(f64, f64)],
) -> Option<Vec<f64>> {
    let mut length = coefficients.len();
    while length > 1 && coefficients[length - 1] == 0.0 {
        length -= 1;
    }
    if length < 2 || coefficients[..length].iter().any(|x| !x.is_finite()) {
        return Some(Vec::new());
    }
    let rational = RationalPolynomial::from_f64(&coefficients[..length])?;
    isolate_rational_roots(&rational, seeds)
}

/// Isolate the common real roots of several dyadic power polynomials.
///
/// FLINT computes their gcd exactly over `QQ`; Arb then isolates every
/// distinct real root of that gcd.  This is the native operation required by
/// the singular quartic subresultant selector: a specialization is critical
/// exactly when every coefficient of the first nonzero subresultant vanishes.
pub(super) fn common_real_roots_power(polynomials: &[&[f64]]) -> Option<Vec<f64>> {
    let mut common: Option<RationalPolynomial> = None;
    for coefficients in polynomials {
        let mut length = coefficients.len();
        while length > 1 && coefficients[length - 1] == 0.0 {
            length -= 1;
        }
        if coefficients[..length].iter().any(|x| !x.is_finite()) {
            return None;
        }
        let polynomial = RationalPolynomial::from_f64(&coefficients[..length])?;
        common = Some(match common {
            Some(value) => value.gcd(&polynomial),
            None => polynomial,
        });
    }
    match common {
        Some(polynomial) => isolate_rational_roots(&polynomial, &[]),
        None => Some(Vec::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::{common_real_roots_power, real_roots_power_seeded};

    #[test]
    fn isolates_power_cubic() {
        // (x+1/2)(x-1/4)(x-3/4)
        let roots = real_roots_power_seeded(&[3.0 / 32.0, -5.0 / 16.0, -0.5, 1.0], &[]).unwrap();
        let expected = [-0.5, 0.25, 0.75];
        assert_eq!(roots.len(), expected.len());
        for (actual, expected) in roots.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-14);
        }

        let seeded = real_roots_power_seeded(
            &[3.0 / 32.0, -5.0 / 16.0, -0.5, 1.0],
            &[(-0.49, 0.0), (0.24, 0.0), (0.76, 0.0)],
        )
        .unwrap();
        assert_eq!(seeded.len(), expected.len());
        for (actual, expected) in seeded.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-14);
        }
    }

    #[test]
    fn exact_square_free_reduction_keeps_repeated_real_roots() {
        // (x-1/4)^2 (x+1/2) = x^3 - 3/16 x + 1/32.
        // Every coefficient is dyadic, so the FLINT gcd is exact.
        let roots = real_roots_power_seeded(&[1.0 / 32.0, -3.0 / 16.0, 0.0, 1.0], &[])
            .expect("the square-free part must isolate");
        assert_eq!(roots.len(), 2);
        assert!((roots[0] + 0.5).abs() < 1e-14);
        assert!((roots[1] - 0.25).abs() < 1e-14);
    }

    #[test]
    fn isolates_exact_common_roots() {
        // gcd((x-1/4)(x+1/2), (x-1/4)(x-3/4)) = x-1/4.
        let left = [-1.0 / 8.0, 1.0 / 4.0, 1.0];
        let right = [3.0 / 16.0, -1.0, 1.0];
        let roots = common_real_roots_power(&[&left, &right]).expect("common root isolation");
        assert_eq!(roots.len(), 1);
        assert!((roots[0] - 0.25).abs() < 1e-14);
    }
}
