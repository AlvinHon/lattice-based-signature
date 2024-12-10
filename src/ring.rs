use num::{bigint::RandBigInt, BigInt, BigUint, One};
use rand::Rng;

use crate::poly::{Polynomial, SparsePolynomial};

#[derive(Clone, Debug)]
pub struct Ring {
    // prime number congruent to 1 mod 2n
    pub(crate) p: BigUint,
    // an integer that is a power of 2
    pub(crate) n: u32,
}

impl Ring {
    pub fn new<B: Into<BigUint>>(p: B, n: u32) -> Self {
        let p = p.into();
        assert!(&p % &(n * BigUint::from(2u64)) == 1u64.into());
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
        let mut coeffs = Vec::new();
        let one = BigInt::one();
        for _ in 0..self.n {
            // +1 to include the upper bound
            let c = rng.gen_bigint_range(&(-bound), &(bound + &one));
            coeffs.push(c);
        }

        // TODO if leading coefficient is 0, then generate a new polynomial

        Polynomial::new(coeffs)
    }

    pub fn mul(&self, p1: Polynomial<BigInt>, p2: Polynomial<BigInt>) -> Polynomial<BigInt> {
        let p = p1 * p2;
        self.reduce_modulo(p)
    }

    pub fn add(&self, p1: Polynomial<BigInt>, p2: Polynomial<BigInt>) -> Polynomial<BigInt> {
        let p = p1 + p2;
        self.reduce_modulo(p)
    }

    pub fn sub(&self, p1: Polynomial<BigInt>, p2: Polynomial<BigInt>) -> Polynomial<BigInt> {
        let p = p1 - p2;
        self.reduce_modulo(p)
    }

    fn reduce_modulo(&self, p: Polynomial<BigInt>) -> Polynomial<BigInt> {
        let divider = SparsePolynomial {
            terms: vec![(self.n as usize, BigInt::one()), (0, BigInt::one())],
        };
        // Zp[x] / (x^n + 1)
        p.pseudo_remainder(&divider)
        // coefficients reduced modulo p
        // let bound = self.bound();
        // let coeffs = p
        //     .coeffs
        //     .into_iter()
        //     .map(|c| {
        //         let sign = c.sign();
        //         c % &bound
        //     })
        //     .collect();

        // Polynomial::new(coeffs)
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

#[cfg(test)]
mod test {
    use crate::{poly::Polynomial, ring::Ring};
    use num::{BigInt, BigUint, One};
    use num_prime::{nt_funcs::is_prime, Primality, PrimalityTestConfig};

    #[test]
    fn test_zp() {
        let rng = &mut rand::thread_rng();
        let p = 8383489u64;
        let n = 512u32;

        let r = Ring::new(p, n);

        let a = r.rand_polynomial(rng);
        let p = Polynomial::new(vec![BigInt::from(0), BigInt::one()]);
        let rhs = r.mul(a.clone(), p);

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
        let p = 8383489u64;
        let n = 512u32;
        assert!(
            is_prime(&BigUint::from(p), Some(PrimalityTestConfig::default())) == Primality::Yes
        );
        assert!(p % (2 * n as u64) == 1);

        let r = Ring::new(p, n);

        let a = r.rand_polynomial(rng);
        let y1 = r.rand_polynomial(rng);
        let y2 = r.rand_polynomial(rng);

        // y1 + y2
        let y1_plus_y2 = r.add(y1.clone(), y2.clone());
        // y1 - y2
        let y1_minus_y2 = r.sub(y1.clone(), y2.clone());
        // a * y1
        let ay1 = r.mul(a.clone(), y1.clone());
        // a * y2
        let ay2 = r.mul(a.clone(), y2.clone());

        // a * (y1 + y2) == a * y1 + a * y2
        let lhs = r.mul(a.clone(), y1_plus_y2.clone());
        let rhs = r.add(ay1.clone(), ay2.clone());
        assert!(lhs == rhs);

        // a * (y1 - y2) == a * y1 - a * y2
        let lhs = r.mul(a.clone(), y1_minus_y2.clone());
        let rhs = r.sub(ay1.clone(), ay2.clone());
        assert!(lhs == rhs);
    }
}
