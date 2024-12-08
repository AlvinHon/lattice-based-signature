use num::{bigint::RandBigInt, BigInt, BigUint, One};
use rand::Rng;

use crate::poly::{Polynomial, SparsePolynomial};

pub struct Ring {
    // prime number congruent to 1 mod 2n
    pub p: BigUint,
    // an integer that is a power of 2
    pub n: u32,
}

impl Ring {
    pub fn new<B: Into<BigUint>>(p: B, n: u32) -> Self {
        let p = p.into();
        assert!(&p % &(n * BigUint::from(2u64)) == 1u64.into());
        Ring { p, n }
    }

    /// Generate a random polynomial with degree = n - 1, and coefficients are in field Zp.
    pub fn rand_polynomial<R: Rng>(&self, rng: &mut R) -> Polynomial<BigInt> {
        let mut coeffs = Vec::new();
        let bound = self.bound();
        let one = BigInt::one();
        for _ in 0..self.n {
            // +1 to include the upper bound
            let c = rng.gen_bigint_range(&(-&bound), &(&bound + &one));
            coeffs.push(c);
        }

        Polynomial::new(coeffs)
    }

    pub fn mul(&self, p1: Polynomial<BigInt>, p2: Polynomial<BigInt>) -> Polynomial<BigInt> {
        let p = p1 * p2;
        self.apply_ring_ops(p)
    }

    pub fn add(&self, p1: Polynomial<BigInt>, p2: Polynomial<BigInt>) -> Polynomial<BigInt> {
        let p = p1 + p2;
        self.apply_ring_ops(p)
    }

    pub fn sub(&self, p1: Polynomial<BigInt>, p2: Polynomial<BigInt>) -> Polynomial<BigInt> {
        let p = p1 - p2;
        self.apply_ring_ops(p)
    }

    fn apply_ring_ops(&self, p: Polynomial<BigInt>) -> Polynomial<BigInt> {
        let divider = SparsePolynomial {
            terms: vec![(self.n as usize, BigInt::one()), (0, BigInt::one())],
        };
        // Zp[x] / (x^n + 1)
        let p = p.pseudo_remainder(&divider);
        // field operations
        let bound = self.bound();
        let coeffs = p.coeffs.into_iter().map(|c| c % &bound).collect();

        Polynomial::new(coeffs)
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
        BigInt::from((&self.p - &BigUint::one()) / BigUint::from(2u64))
    }
}
