//! Define a finite field and its operations.

use std::ops::{Add, Mul, Neg, Sub};

use num::{bigint::RandBigInt, BigInt, BigUint, One, Zero};
use num_bigint::ToBigInt;
use rand::Rng;

/// A finite field element. It encapsulates a positive integer modulo a prime number P.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Elem<const P: u32> {
    pub(crate) value: BigUint,
}

impl<const P: u32> From<u32> for Elem<P> {
    fn from(value: u32) -> Self {
        Elem {
            value: BigUint::from(value),
        }
    }
}

impl<const P: u32> Elem<P> {
    pub fn to_signed(&self) -> BigInt {
        let mid = BigInt::from((P - 1) / 2);
        let v = self.value.to_bigint().unwrap();
        if v > mid {
            v - BigInt::from(P)
        } else {
            v
        }
    }

    pub fn from_signed(value: BigInt) -> Self {
        let v = if value < BigInt::zero() {
            value + BigInt::from(P)
        } else {
            value
        };
        Elem {
            value: v.to_biguint().unwrap(),
        }
    }

    #[inline]
    pub fn to_bytes_le(&self) -> Vec<u8> {
        self.value.to_bytes_le()
    }

    #[inline]
    pub fn gen_range<R: Rng>(rng: &mut R, lower: &BigInt, upper: &BigInt) -> Self {
        rng.gen_bigint_range(lower, upper).into()
    }
}

impl<const P: u32> From<BigInt> for Elem<P> {
    fn from(value: BigInt) -> Self {
        Elem::from_signed(value)
    }
}

impl<const P: u32> From<Elem<P>> for BigInt {
    fn from(value: Elem<P>) -> Self {
        value.to_signed()
    }
}

impl<const P: u32> Zero for Elem<P> {
    fn zero() -> Self {
        Elem {
            value: BigUint::from(0u64),
        }
    }

    fn is_zero(&self) -> bool {
        self.value.is_zero()
    }
}

impl<const P: u32> One for Elem<P> {
    fn one() -> Self {
        Elem {
            value: BigUint::from(1u64),
        }
    }
}

impl<const P: u32> Add for Elem<P> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Elem {
            value: (self.value + other.value) % P,
        }
    }
}

impl<const P: u32> Add for &Elem<P> {
    type Output = Elem<P>;

    fn add(self, other: Self) -> Elem<P> {
        Elem {
            value: (&self.value + &other.value) % P,
        }
    }
}

impl<const P: u32> Neg for Elem<P> {
    type Output = Self;

    fn neg(self) -> Self {
        Elem {
            value: P - self.value,
        }
    }
}

impl<const P: u32> Neg for &Elem<P> {
    type Output = Elem<P>;

    fn neg(self) -> Elem<P> {
        Elem {
            value: P - &self.value,
        }
    }
}

impl<const P: u32> Sub for Elem<P> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Elem {
            value: (self.value + P - other.value) % P,
        }
    }
}

impl<const P: u32> Sub for &Elem<P> {
    type Output = Elem<P>;

    fn sub(self, other: Self) -> Elem<P> {
        Elem {
            value: (&self.value + P - &other.value) % P,
        }
    }
}

impl<const P: u32> Mul for Elem<P> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Elem {
            value: (self.value * other.value) % P,
        }
    }
}

impl<const P: u32> Mul for &Elem<P> {
    type Output = Elem<P>;

    fn mul(self, other: Self) -> Elem<P> {
        Elem {
            value: (&self.value * &other.value) % P,
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{
        field::Elem,
        params::set_1,
        poly::{negacyclic_convolution, Polynomial, SparsePolynomial},
    };
    use num::One;
    use num_bigint::Sign;

    use super::*;

    #[test]
    fn test_zp() {
        let rng = &mut rand::thread_rng();
        let params = set_1();
        let n = params.n() as u32;

        let a = params.r.rand_polynomial(rng);
        let p = Polynomial::new(vec![Elem::zero(), Elem::one()]);
        let divider = SparsePolynomial {
            terms: vec![(n as usize, Elem::one()), (0, Elem::one())],
        };
        let start_time = std::time::Instant::now();
        let rhs = (a.clone() * p.clone()).pseudo_remainder(&divider);
        println!("Time elapsed: {:?}", start_time.elapsed());

        // (x^n + 1)-cyclic lattice is an ideal in Z[x]/(x^n + 1)
        // (v_{0} + ... v_{n-2} x^{n-2} + v_{n-1} x^{n-1}) x = -v_{n-1} + v_{0} x + ... + v_{n-2} x^{n-1}")
        assert!(a.coeffs[n as usize - 1] == -&rhs.coeffs[0]);
        for i in 0..n as usize - 1 {
            assert!(a.coeffs[i] == rhs.coeffs[i + 1]);
        }

        let start_time = std::time::Instant::now();
        let lhs = negacyclic_convolution(n, &a, &p);
        println!("Time elapsed: {:?}", start_time.elapsed());
        assert!(lhs == rhs);

        let start_time = std::time::Instant::now();
        let lhs = negacyclic_convolution(n, &p, &a);
        println!("Time elapsed: {:?}", start_time.elapsed());
        assert!(lhs == rhs);
    }

    #[test]
    fn test_signed() {
        const P1: u32 = 8383489;

        // bound
        let bound = Elem::<P1> {
            value: BigUint::from(P1 - 1),
        };
        assert!(bound.to_signed() == BigInt::from(-1));
        assert!(Elem::from_signed(bound.to_signed()) == bound);

        // zero
        let zero = Elem::<P1>::zero();
        assert!(zero.to_signed() == BigInt::zero());
        assert!(Elem::from_signed(zero.to_signed()) == zero);

        // (P1 - 1) / 2
        let bound = Elem::<P1> {
            value: BigUint::from((P1 - 1) / 2),
        };
        assert!(bound.to_signed() == BigInt::from((P1 - 1) / 2));
        assert!(Elem::from_signed(bound.to_signed()) == bound);

        // -(P1 - 1) / 2
        let bound = Elem::<P1> {
            value: BigUint::from((P1 - 1) / 2 + 1),
        };
        assert!(bound.to_signed() == -BigInt::from((P1 - 1) / 2));
        assert!(Elem::from_signed(bound.to_signed()) == bound);

        // 1 and -1
        let one = Elem::<P1> {
            value: BigUint::one(),
        };
        let neg_one = -&one;

        assert!(one.to_signed().sign() == Sign::Plus);
        assert!(neg_one.to_signed().sign() == Sign::Minus);
        assert!(&one + &neg_one == Elem::zero());
        assert!(one.to_signed() + neg_one.to_signed() == BigInt::zero());
    }
}
