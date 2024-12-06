//! Straightforward implementation of a polynomial type with basic arithmetic operations.

use std::ops::{Add, Mul, Neg, Sub};

use num::Zero;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Polynomial<T> {
    pub coeffs: Vec<T>,
}

impl<T> Polynomial<T> {
    /// Creates a new polynomial with the given coefficients.
    /// p(x) = coeffs\[0\] + coeffs\[1\] * x + coeffs\[2\] * x^2 + ...
    pub fn new(coeffs: Vec<T>) -> Self {
        Polynomial { coeffs }
    }

    /// Returns the degree of the polynomial.
    ///
    /// ## Panics
    /// Panics if the input is an empty vector.
    pub fn deg(&self) -> usize {
        self.coeffs.len() - 1
    }

    /// Computes the pseudo remainder `r` of the polynomial `p` by another polynomial `d`.
    /// i.e. `r = p - q * d` where `q` is the quotient of the division.
    pub fn pseudo_remainder(&self, other: &Self) -> Self
    where
        T: Zero + Clone + Copy + Sub<Output = T> + Mul<Output = T>,
    {
        if other.is_zero() {
            panic!("Division by zero");
        }

        if self.is_zero() {
            return Polynomial::zero();
        }

        if self.deg() < other.deg() {
            return self.clone();
        }

        // Polynomial Pseudo-Division (Wu's method, https://en.wikipedia.org/wiki/Wu%27s_method_of_characteristic_set)
        // Variable names are using the notations from the reference: https://aszanto.math.ncsu.edu/MA722/ln-02.pdf
        let mut r = self.clone(); // r = f
        let s = other.deg();
        let b_s = *other.coeffs.last().unwrap();
        while !r.is_zero() && r.deg() >= s {
            // deg_y(r) - s
            let pow_y = r.deg() - s;
            let lc_r = *r.coeffs.last().unwrap();
            // r' = b_s r - lc_r * g * x^pow_y
            for i in 0..r.deg() {
                let term = if i < pow_y {
                    T::zero()
                } else {
                    lc_r * other.coeffs[i - pow_y]
                };
                r.coeffs[i] = b_s * r.coeffs[i] - term;
            }
            r.coeffs.pop();
            trim_zeros(&mut r.coeffs);
        }

        r
    }
}

fn trim_zeros<T: Zero>(v: &mut Vec<T>) {
    while let Some(&t) = v.last().as_ref() {
        if t.is_zero() {
            v.pop();
        } else {
            break;
        }
    }
}

impl<T> Zero for Polynomial<T>
where
    T: Zero + Clone + Copy,
{
    fn zero() -> Self {
        Polynomial { coeffs: vec![] }
    }

    fn is_zero(&self) -> bool {
        self.coeffs.is_empty() || self.coeffs.iter().all(|c| c.is_zero())
    }
}

// ... impl ops::* over T for Polynomial<T> ...

impl<T> Add<T> for Polynomial<T>
where
    T: Zero + Clone + Copy,
{
    type Output = Self;

    fn add(self, other: T) -> Self {
        if self.is_zero() {
            return Polynomial::new(vec![other]);
        }

        let mut result = self;
        result.coeffs[0] = result.coeffs[0] + other;
        result
    }
}

impl<T> Sub<T> for Polynomial<T>
where
    T: Zero + Clone + Copy + Neg<Output = T>,
{
    type Output = Self;

    fn sub(self, other: T) -> Self {
        self + (-other)
    }
}

impl<T> Mul<T> for Polynomial<T>
where
    T: Zero + Clone + Copy + Mul<Output = T>,
{
    type Output = Self;

    fn mul(self, other: T) -> Self {
        if other.is_zero() {
            return Polynomial::zero();
        }

        Polynomial {
            coeffs: self.coeffs.iter().map(|&c| c * other).collect(),
        }
    }
}

// ... impl ops::* for Polynomial<T> ...

impl<T> Add for Polynomial<T>
where
    T: Zero + Clone + Copy,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let mut result = vec![T::zero(); std::cmp::max(self.coeffs.len(), other.coeffs.len())];
        for (i, r_i) in result.iter_mut().enumerate() {
            if i < self.coeffs.len() {
                *r_i = *r_i + self.coeffs[i];
            }
            if i < other.coeffs.len() {
                *r_i = *r_i + other.coeffs[i];
            }
        }
        trim_zeros(&mut result);
        Polynomial { coeffs: result }
    }
}

impl<T> Neg for Polynomial<T>
where
    T: Zero + Clone + Copy + Neg<Output = T>,
{
    type Output = Self;

    fn neg(self) -> Self {
        Polynomial {
            coeffs: self.coeffs.iter().map(|&c| -c).collect(),
        }
    }
}

impl<T> Sub for Polynomial<T>
where
    T: Zero + Clone + Copy + Neg<Output = T>,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self + (-other)
    }
}

impl<T> Mul for Polynomial<T>
where
    T: Zero + Clone + Copy + Mul<Output = T>,
{
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let mut result = vec![T::zero(); self.coeffs.len() + other.coeffs.len() - 1];
        for i in 0..self.coeffs.len() {
            for j in 0..other.coeffs.len() {
                result[i + j] = result[i + j] + self.coeffs[i] * other.coeffs[j];
            }
        }
        trim_zeros(&mut result);
        Polynomial { coeffs: result }
    }
}

#[cfg(test)]
mod test {

    use std::vec;

    use super::*;

    #[test]
    fn test_zero() {
        let p = Polynomial::<i32>::zero();
        assert!(p.is_zero());
    }

    #[test]
    fn test_deg() {
        let p = Polynomial::new(vec![1, 2, 3]);
        assert_eq!(p.deg(), 2);
    }

    #[test]
    fn test_add() {
        let p1 = Polynomial::new(vec![1, 2, 3]);
        let p2 = Polynomial::new(vec![4, 5]);
        let r = Polynomial::new(vec![5, 7, 3]);
        assert_eq!(p1 + p2, r);

        let p1 = Polynomial::new(vec![2, 1]);
        let p2 = Polynomial::new(vec![5, 4, 3]);
        let r = Polynomial::new(vec![7, 5, 3]);
        assert_eq!(p1 + p2, r);

        let p1 = Polynomial::new(vec![1, 2, 3]);
        let p2 = Polynomial::zero();
        let r = Polynomial::new(vec![1, 2, 3]);
        assert_eq!(p1 + p2, r);

        let p1 = Polynomial::zero();
        let p2 = Polynomial::new(vec![1, 2, 3]);
        let r = Polynomial::new(vec![1, 2, 3]);
        assert_eq!(p1 + p2, r);
    }

    #[test]
    fn test_add_t() {
        let p = Polynomial::new(vec![1, 2, 3]);
        let r = Polynomial::new(vec![2, 2, 3]);
        assert_eq!(p + 1, r);

        let p = Polynomial::zero();
        let r = Polynomial::new(vec![1]);
        assert_eq!(p + 1, r);
    }

    #[test]
    fn test_sub() {
        let p1 = Polynomial::new(vec![1, 2, 3]);
        let p2 = Polynomial::new(vec![4, 5]);
        let r = Polynomial::new(vec![-3, -3, 3]);
        assert_eq!(p1 - p2, r);

        let p1 = Polynomial::new(vec![2, 1]);
        let p2 = Polynomial::new(vec![5, 4, 3]);
        let r = Polynomial::new(vec![-3, -3, -3]);
        assert_eq!(p1 - p2, r);

        let p1 = Polynomial::new(vec![1, 2, 3]);
        let p2 = Polynomial::zero();
        let r = Polynomial::new(vec![1, 2, 3]);
        assert_eq!(p1 - p2, r);

        let p1 = Polynomial::zero();
        let p2 = Polynomial::new(vec![1, 2, 3]);
        let r = Polynomial::new(vec![-1, -2, -3]);
        assert_eq!(p1 - p2, r);
    }

    #[test]
    fn test_sub_t() {
        let p = Polynomial::new(vec![1, 2, 3]);
        let r = Polynomial::new(vec![0, 2, 3]);
        assert_eq!(p - 1, r);

        let p = Polynomial::zero();
        let r = Polynomial::new(vec![-1]);
        assert_eq!(p - 1, r);
    }

    #[test]
    fn test_mul() {
        let p1 = Polynomial::new(vec![1, 2, 3]);
        let p2 = Polynomial::new(vec![4, 5]);
        let r = Polynomial::new(vec![4, 13, 22, 15]);
        assert_eq!(p1 * p2, r);

        let p1 = Polynomial::new(vec![2, 1]);
        let p2 = Polynomial::new(vec![5, 4, 3]);
        let r = Polynomial::new(vec![10, 13, 10, 3]);
        assert_eq!(p1 * p2, r);

        let p1 = Polynomial::new(vec![1, 2, 3]);
        let p2 = Polynomial::zero();
        let r = Polynomial::zero();
        assert_eq!(p1 * p2, r);
    }

    #[test]
    fn test_mul_t() {
        let p = Polynomial::new(vec![1, 2, 3]);
        let r = Polynomial::new(vec![2, 4, 6]);
        assert_eq!(p * 2, r);

        let p = Polynomial::zero();
        let r = Polynomial::zero();
        assert_eq!(p * 2, r);
    }

    #[test]
    fn test_divide() {
        // examples from https://en.wikipedia.org/wiki/Polynomial_long_division
        let p1 = Polynomial::new(vec![-4, 0, -2, 1]);
        let p2 = Polynomial::new(vec![-3, 1]);
        let r = p1.pseudo_remainder(&p2);
        assert_eq!(r, Polynomial::new(vec![5]));

        let p1 = Polynomial::new(vec![-42, 0, -12, 1]);
        let p2 = Polynomial::new(vec![1, -2, 1]);
        let r = p1.pseudo_remainder(&p2);
        assert_eq!(r, Polynomial::new(vec![-32, -21]));
    }
}
