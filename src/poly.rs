//! Straightforward implementation of a polynomial type with basic arithmetic operations.

use std::ops::{Add, Mul, Neg, Sub};

use num::Zero;

pub trait PolynomialDivider<T>: Zero {
    fn deg(&self) -> usize;
    fn leading_coefficient(&self) -> T;
    fn coefficient(&self, idx: usize) -> T;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Polynomial<T> {
    pub(crate) coeffs: Vec<T>,
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
    pub fn pseudo_remainder<D>(&self, other: &D) -> Self
    where
        D: PolynomialDivider<T> + Zero,
        T: Zero + Clone,
        for<'a> &'a T: Mul<Output = T> + Sub<Output = T> + Add<Output = T>,
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
        let b_s = other.leading_coefficient();
        while !r.is_zero() && r.deg() >= s {
            // deg_y(r) - s
            let pow_y = r.deg() - s;
            let lc_r = r.coeffs.last().cloned().unwrap();
            // r' = b_s r - lc_r * g * x^pow_y
            for i in 0..r.deg() {
                let term: T = if i < pow_y {
                    T::zero()
                } else {
                    &lc_r * &other.coefficient(i - pow_y)
                };
                r.coeffs[i] = &(&b_s * &r.coeffs[i]) - &term;
            }
            r.coeffs.pop();
            trim_zeros(&mut r.coeffs);
        }

        r
    }

    pub fn mapv<U, F>(&self, mut f: F) -> Polynomial<U>
    where
        F: FnMut(&T) -> U,
    {
        Polynomial {
            coeffs: self.coeffs.iter().map(|c| f(c)).collect(),
        }
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
    T: Zero + Clone,
    for<'a> &'a T: Add<Output = T> + Mul<Output = T>,
{
    fn zero() -> Self {
        Polynomial { coeffs: vec![] }
    }

    fn is_zero(&self) -> bool {
        self.coeffs.is_empty() || self.coeffs.iter().all(|c| c.is_zero())
    }
}

impl<T> PolynomialDivider<T> for Polynomial<T>
where
    T: Zero + Clone,
    for<'a> &'a T: Add<Output = T> + Mul<Output = T>,
{
    fn deg(&self) -> usize {
        self.deg()
    }

    fn leading_coefficient(&self) -> T {
        self.coeffs.last().cloned().unwrap_or(T::zero())
    }

    fn coefficient(&self, idx: usize) -> T {
        self.coeffs.get(idx).cloned().unwrap_or(T::zero())
    }
}

// ... impl ops::* over T for Polynomial<T> ...

impl<T> Add<T> for Polynomial<T>
where
    T: Zero + Clone,
    for<'a> &'a T: Add<Output = T> + Mul<Output = T>,
{
    type Output = Self;

    fn add(self, other: T) -> Self {
        if self.is_zero() {
            return Polynomial::new(vec![other]);
        }

        let mut result = self;
        result.coeffs[0] = &result.coeffs[0] + &other;
        result
    }
}

impl<T> Sub<T> for Polynomial<T>
where
    T: Zero + Clone + Neg<Output = T>,
    for<'a> &'a T: Add<Output = T> + Mul<Output = T>,
{
    type Output = Self;

    fn sub(self, other: T) -> Self {
        self + (-other)
    }
}

impl<T> Mul<T> for Polynomial<T>
where
    T: Zero + Clone,
    for<'a> &'a T: Add<Output = T> + Mul<Output = T>,
{
    type Output = Self;

    fn mul(self, other: T) -> Self {
        if other.is_zero() {
            return Polynomial::zero();
        }

        Polynomial {
            coeffs: self.coeffs.iter().map(|c| c * &other).collect(),
        }
    }
}

// ... impl ops::* for Polynomial<T> ...

impl<T> Add for Polynomial<T>
where
    T: Zero + Clone,
    for<'a> &'a T: Add<Output = T>,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let mut result = vec![T::zero(); std::cmp::max(self.coeffs.len(), other.coeffs.len())];
        for (i, r_i) in result.iter_mut().enumerate() {
            if i < self.coeffs.len() {
                *r_i = &*r_i + &self.coeffs[i];
            }
            if i < other.coeffs.len() {
                *r_i = &*r_i + &other.coeffs[i];
            }
        }
        trim_zeros(&mut result);
        Polynomial { coeffs: result }
    }
}

impl<T> Neg for Polynomial<T>
where
    T: Zero + Clone,
    for<'a> &'a T: Neg<Output = T>,
{
    type Output = Self;

    fn neg(self) -> Self {
        Polynomial {
            coeffs: self.coeffs.iter().map(|c| -c).collect(),
        }
    }
}

impl<T> Sub for Polynomial<T>
where
    T: Zero + Clone,
    for<'a> &'a T: Neg<Output = T> + Add<Output = T>,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self + (-other)
    }
}

impl<T> Mul for Polynomial<T>
where
    T: Zero + Clone,
    for<'a> &'a T: Mul<Output = T> + Add<Output = T>,
{
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let mut result = vec![T::zero(); self.coeffs.len() + other.coeffs.len() - 1];
        for i in 0..self.coeffs.len() {
            for j in 0..other.coeffs.len() {
                result[i + j] = &result[i + j] + &(&self.coeffs[i] * &other.coeffs[j]);
            }
        }
        trim_zeros(&mut result);
        Polynomial { coeffs: result }
    }
}

// ... SparsePolynomial<T> ...

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct SparsePolynomial<T> {
    pub terms: Vec<(usize, T)>,
}

impl<T> SparsePolynomial<T> {
    pub fn deg(&self) -> usize {
        self.terms.iter().map(|(d, _)| *d).max().unwrap_or(0)
    }

    /// Sets the coefficient of the term with the given degree.
    pub fn set(&mut self, coeff: T, degree: usize)
    where
        T: Zero,
    {
        if coeff.is_zero() {
            self.terms.retain(|(d, _)| *d != degree);
        } else {
            let idx = self.terms.iter().position(|(d, _)| *d == degree);
            match idx {
                Some(i) => self.terms[i] = (degree, coeff),
                None => self.terms.push((degree, coeff)),
            }
        }
    }

    fn trim(&mut self)
    where
        T: Zero,
    {
        self.terms.retain(|(_, c)| !c.is_zero());
    }
}

impl<T> PolynomialDivider<T> for SparsePolynomial<T>
where
    T: Zero + Clone,
    for<'a> &'a T: Add<Output = T> + Mul<Output = T>,
{
    fn deg(&self) -> usize {
        self.deg()
    }

    fn leading_coefficient(&self) -> T {
        self.coefficient(self.deg())
    }

    fn coefficient(&self, idx: usize) -> T {
        self.terms
            .iter()
            .find(|(d, _)| *d == idx)
            .map(|(_, c)| c.clone())
            .unwrap_or(T::zero())
    }
}

impl<T> Zero for SparsePolynomial<T>
where
    T: Zero + Clone,
    for<'a> &'a T: Add<Output = T> + Mul<Output = T>,
{
    fn zero() -> Self {
        SparsePolynomial { terms: vec![] }
    }

    fn is_zero(&self) -> bool {
        self.terms.is_empty() || self.terms.iter().all(|(_, c)| c.is_zero())
    }
}

impl<T> Add for SparsePolynomial<T>
where
    T: Zero + Clone,
    for<'a> &'a T: Add<Output = T> + Mul<Output = T>,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let mut result = self;

        for (i, coeff) in other.terms {
            let sum = &result.coefficient(i) + &coeff;
            result.set(sum, i);
        }
        result.trim();
        result
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
    fn test_add_sparse() {
        let mut p1 = SparsePolynomial::zero();
        p1.set(3, 2);
        p1.set(2, 1);
        p1.set(1, 0);

        let mut p2 = SparsePolynomial::zero();
        p2.set(5, 1);
        p2.set(4, 0);

        let mut r = SparsePolynomial::zero();
        r.set(3, 2);
        r.set(7, 1);
        r.set(5, 0);

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

        // Polynomial 1
        let p1 = Polynomial::new(vec![-4, 0, -2, 1]);
        let p2 = Polynomial::new(vec![-3, 1]);
        let r = p1.pseudo_remainder(&p2);
        assert_eq!(r, Polynomial::new(vec![5]));

        // SparsePolynomial 1
        let mut sp2 = SparsePolynomial::zero();
        sp2.set(-3, 0);
        sp2.set(1, 1);
        let r = p1.pseudo_remainder(&sp2);
        assert_eq!(r, Polynomial::new(vec![5]));

        // Polynomial 2
        let p1 = Polynomial::new(vec![-42, 0, -12, 1]);
        let p2 = Polynomial::new(vec![1, -2, 1]);
        let r = p1.pseudo_remainder(&p2);
        assert_eq!(r, Polynomial::new(vec![-32, -21]));

        // SparsePolynomial 2
        let mut sp2 = SparsePolynomial::zero();
        sp2.set(1, 0);
        sp2.set(-2, 1);
        sp2.set(1, 2);
        let r = p1.pseudo_remainder(&sp2);
        assert_eq!(r, Polynomial::new(vec![-32, -21]));
    }
}
