use num::{bigint::RandBigInt, BigInt, BigUint, One, Zero};
use rand::Rng;

use crate::poly::{negacyclic_convolution, Polynomial, SparsePolynomial};

#[derive(Clone, Debug)]
pub struct Ring {
    // prime number congruent to 1 mod 2n
    pub(crate) p: u32,
    // an integer that is a power of 2
    pub(crate) n: u32,
}

impl Ring {
    pub fn new(p: u32, n: u32) -> Self {
        assert!(p % (n * 2u32) == 1u32);
        Ring { p, n }
    }

    /// Generate a random polynomial with degree = n - 1, and coefficients are in field Zp.
    pub fn rand_polynomial<R: Rng>(&self, rng: &mut R) -> Polynomial<BigInt> {
        let bound = self.bound();
        self.rand_polynomial_within(rng, &bound)
    }

    /// Generate a random polynomial with degree = n - 1, and coefficients are within [-bound, bound].
    pub fn rand_polynomial_within<R: Rng>(
        &self,
        rng: &mut R,
        bound: &BigInt,
    ) -> Polynomial<BigInt> {
        let one = BigInt::one();
        let mut coeffs = (0..self.n)
            // +1 to include the upper bound
            .map(|_| rng.gen_bigint_range(&(-bound), &(bound + &one)))
            .collect::<Vec<_>>();

        // if leading coefficient is 0, then generate a new polynomial
        while coeffs[self.n as usize - 1] == BigInt::zero() {
            coeffs[self.n as usize - 1] = rng.gen_bigint_range(&(-bound), &(bound + &one));
        }

        Polynomial::new(coeffs)
    }

    #[inline]
    pub fn convert_polynomial_to_bytes(&self, p: Polynomial<BigInt>) -> Vec<u8> {
        p.coeffs
            .iter()
            .flat_map(|x| x.to_signed_bytes_le())
            .collect()
    }

    pub fn mul(&self, p1: &Polynomial<BigInt>, p2: &Polynomial<BigInt>) -> Polynomial<BigInt> {
        negacyclic_convolution(self.n, p1, p2)
    }

    pub fn add(&self, p1: &Polynomial<BigInt>, p2: &Polynomial<BigInt>) -> Polynomial<BigInt> {
        self.reduce_modulo(p1.add(p2))
    }

    pub fn sub(&self, p1: Polynomial<BigInt>, p2: Polynomial<BigInt>) -> Polynomial<BigInt> {
        self.reduce_modulo(p1 - p2)
    }

    fn reduce_modulo(&self, p: Polynomial<BigInt>) -> Polynomial<BigInt> {
        let divider = SparsePolynomial {
            terms: vec![(self.n as usize, BigInt::one()), (0, BigInt::one())],
        };
        // Zp[x] / (x^n + 1)
        p.pseudo_remainder(&divider)
        // TODO check if need coefficients reduced modulo p
    }

    pub fn is_valid(&self, p: &Polynomial<BigInt>) -> bool {
        let bound = self.bound();
        self.is_valid_within(p, &bound)
    }

    pub fn is_valid_within(&self, p: &Polynomial<BigInt>, bound: &BigInt) -> bool {
        p.coeffs.iter().all(|c| c >= &(-bound) && c <= bound)
    }

    /// Coefficients are within [-(p-1)/2, (p-1)/2]
    fn bound(&self) -> BigInt {
        BigInt::from((self.p - &BigUint::one()) / BigUint::from(2u64))
    }
}

#[cfg(test)]
mod test {
    use crate::{poly::Polynomial, ring::Ring};
    use num::{BigInt, One};

    #[test]
    fn test_zp() {
        let rng = &mut rand::thread_rng();
        let p = 8383489u32;
        let n = 512u32;

        let r = Ring::new(p, n);

        let a = r.rand_polynomial(rng);
        let p = Polynomial::new(vec![BigInt::from(0), BigInt::one()]);
        let rhs = r.mul(&a, &p);

        // (x^n + 1)-cyclic lattice is an ideal in Z[x]/(x^n + 1)
        // (v_{0} + ... v_{n-2} x^{n-2} + v_{n-1} x^{n-1}) x = -v_{n-1} + v_{0} x + ... + v_{n-2} x^{n-1}
        assert!(a.coeffs[n as usize - 1] == -&rhs.coeffs[0]);
        for i in 0..n as usize - 1 {
            assert!(a.coeffs[i] == rhs.coeffs[i + 1]);
        }
    }

    #[test]
    fn test_arithmetics() {
        let rng = &mut rand::thread_rng();
        let p = 8383489u32;
        let n = 512u32;

        let r = Ring::new(p, n);

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
