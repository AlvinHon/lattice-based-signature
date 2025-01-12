//! Defines a ring structure for quotient ring Zp[x]/(x^n + 1) and its
//! polynomial operations.

use num::{BigInt, Zero};
use poly_ring::Polynomial;
use rand::Rng;

use crate::field::Elem;

/// A ring structure for quotient ring Zp[x]/(x^n + 1) where p is a prime number,
/// and n is a power of 2 s.t. p = 1 mod 2n.
#[derive(Clone, Debug)]
pub struct Ring<const P: u32, const N: usize> {
    phantom: std::marker::PhantomData<()>, // to make the struct non-generic
}

impl<const P: u32, const N: usize> Ring<P, N> {
    /// Generate a random polynomial with degree = n - 1, and coefficients are in field Zp.
    pub fn rand_polynomial<R: Rng>(&self, rng: &mut R) -> Polynomial<Elem<P>, N> {
        let bound = Self::bound();
        self.rand_polynomial_within(rng, &bound)
    }

    /// Generate a random polynomial with degree = n - 1, and coefficients are within [-bound, bound].
    pub fn rand_polynomial_within<R: Rng>(
        &self,
        rng: &mut R,
        bound: &BigInt,
    ) -> Polynomial<Elem<P>, N> {
        let lower = -bound;
        let upper = bound + 1; // +1 to include the upper bound

        let mut coeffs = (0..N)
            .map(|_| Elem::gen_range(rng, &lower, &upper))
            .collect::<Vec<_>>();
        // if leading coefficient is 0, then generate a new polynomial
        while coeffs[N - 1] == Elem::zero() {
            coeffs[N - 1] = Elem::gen_range(rng, &lower, &upper);
        }

        Polynomial::new(coeffs)
    }

    #[inline]
    pub fn convert_polynomial_to_bytes(&self, p: Polynomial<Elem<P>, N>) -> Vec<u8> {
        p.iter().flat_map(|x| x.to_bytes_le()).collect()
    }

    #[inline]
    pub fn mul(
        &self,
        p1: &Polynomial<Elem<P>, N>,
        p2: &Polynomial<Elem<P>, N>,
    ) -> Polynomial<Elem<P>, N> {
        p1.clone() * p2.clone()
    }

    #[inline]
    pub fn add(
        &self,
        p1: &Polynomial<Elem<P>, N>,
        p2: &Polynomial<Elem<P>, N>,
    ) -> Polynomial<Elem<P>, N> {
        p1.clone() + p2.clone()
    }

    #[inline]
    pub fn sub(
        &self,
        p1: Polynomial<Elem<P>, N>,
        p2: Polynomial<Elem<P>, N>,
    ) -> Polynomial<Elem<P>, N> {
        p1 - p2
    }

    /// Check if the polynomial has coefficients within [-(p-1)/2, (p-1)/2].
    pub fn is_valid(&self, p: &Polynomial<Elem<P>, N>) -> bool {
        let bound = Elem::<P>::from(Self::bound());
        self.is_valid_within(p, &bound.to_signed())
    }

    /// Check if the polynomial has coefficients within [-bound, bound].
    pub fn is_valid_within(&self, p: &Polynomial<Elem<P>, N>, bound: &BigInt) -> bool {
        p.iter()
            .map(|c| c.to_signed())
            .all(|c| c >= -bound && &c <= bound)
    }

    /// Coefficients are within [-(p-1)/2, (p-1)/2]
    #[inline]
    fn bound() -> BigInt {
        BigInt::from((P - 1) / 2)
    }
}

impl<const P: u32, const N: usize> Default for Ring<P, N> {
    /// Create a new ring structure for Zp[x]/(x^n + 1).
    /// Panics if p % (n * 2) != 1
    fn default() -> Self {
        assert!(P % (N * 2) as u32 == 1u32);
        Self {
            phantom: std::marker::PhantomData,
        }
    }
}

#[cfg(test)]
mod test {
    use crate::params::set_1;

    #[test]
    fn test_arithmetics() {
        let rng = &mut rand::thread_rng();
        let params = set_1();
        let r = params.r;

        let a = r.rand_polynomial(rng);
        let y1 = r.rand_polynomial(rng);
        let y2 = r.rand_polynomial(rng);

        // y1 + y2
        let y1_plus_y2 = r.add(&y1, &y2);
        // y1 - y2
        let y1_minus_y2 = r.sub(y1.clone(), y2.clone());
        // a * y1
        let ay1 = r.mul(&a, &y1);
        // a * y2
        let ay2 = r.mul(&a, &y2);

        // a * (y1 + y2) == a * y1 + a * y2
        let lhs = r.mul(&a, &y1_plus_y2);
        let rhs = r.add(&ay1, &ay2);
        assert!(lhs == rhs);

        // a * (y1 - y2) == a * y1 - a * y2
        let lhs = r.mul(&a, &y1_minus_y2);
        let rhs = r.sub(ay1.clone(), ay2.clone());
        assert!(lhs == rhs);
    }
}
