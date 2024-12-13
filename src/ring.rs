use num::{BigInt, One, Zero};
use rand::Rng;

use crate::{
    field::Elem,
    poly::{negacyclic_convolution, Polynomial, SparsePolynomial},
};

#[derive(Clone, Debug)]
pub struct Ring<const P: u32> {
    // an integer that is a power of 2
    pub(crate) n: u32,
}

impl<const P: u32> Ring<P> {
    pub fn new(n: u32) -> Self {
        assert!(P % (n * 2u32) == 1u32);
        Ring { n }
    }

    /// Generate a random polynomial with degree = n - 1, and coefficients are in field Zp.
    pub fn rand_polynomial<R: Rng>(&self, rng: &mut R) -> Polynomial<Elem<P>> {
        let bound = Self::bound();
        self.rand_polynomial_within(rng, &bound)
    }

    /// Generate a random polynomial with degree = n - 1, and coefficients are within [-bound, bound].
    pub fn rand_polynomial_within<R: Rng>(
        &self,
        rng: &mut R,
        bound: &BigInt,
    ) -> Polynomial<Elem<P>> {
        let lower = -bound;
        let upper = bound + 1; // +1 to include the upper bound

        self.rand_polynomial_with(rng, |rng| Elem::gen_range(rng, &lower, &upper))
    }

    fn rand_polynomial_with<R: Rng, F>(&self, rng: &mut R, f: F) -> Polynomial<Elem<P>>
    where
        F: Fn(&mut R) -> Elem<P>,
    {
        let mut coeffs = (0..self.n).map(|_| f(rng)).collect::<Vec<_>>();
        // if leading coefficient is 0, then generate a new polynomial
        while coeffs[self.n as usize - 1] == Elem::zero() {
            coeffs[self.n as usize - 1] = f(rng);
        }

        Polynomial::new(coeffs)
    }

    #[inline]
    pub fn convert_polynomial_to_bytes(&self, p: Polynomial<Elem<P>>) -> Vec<u8> {
        p.coeffs.iter().flat_map(|x| x.to_bytes_le()).collect()
    }

    pub fn mul(&self, p1: &Polynomial<Elem<P>>, p2: &Polynomial<Elem<P>>) -> Polynomial<Elem<P>> {
        negacyclic_convolution(self.n, p1, p2)
    }

    pub fn add(&self, p1: &Polynomial<Elem<P>>, p2: &Polynomial<Elem<P>>) -> Polynomial<Elem<P>> {
        self.reduce_modulo(p1.add(p2))
    }

    pub fn sub(&self, p1: Polynomial<Elem<P>>, p2: Polynomial<Elem<P>>) -> Polynomial<Elem<P>> {
        self.reduce_modulo(p1 - p2)
    }

    fn reduce_modulo(&self, p: Polynomial<Elem<P>>) -> Polynomial<Elem<P>> {
        let divider = SparsePolynomial {
            terms: vec![(self.n as usize, Elem::<P>::one()), (0, Elem::<P>::one())],
        };
        // Zp[x] / (x^n + 1)
        p.pseudo_remainder(&divider)
        // TODO check if need coefficients reduced modulo p
    }

    pub fn is_valid(&self, p: &Polynomial<Elem<P>>) -> bool {
        let bound = Elem::<P>::from(Self::bound());
        self.is_valid_within(p, &bound.to_signed())
    }

    pub fn is_valid_within(&self, p: &Polynomial<Elem<P>>, bound: &BigInt) -> bool {
        p.coeffs
            .iter()
            .map(|c| c.to_signed())
            .all(|c| c >= -bound && &c <= bound)
    }

    /// Coefficients are within [-(p-1)/2, (p-1)/2]
    #[inline]
    fn bound() -> BigInt {
        BigInt::from((P - 1) / 2)
    }
}

#[cfg(test)]
mod test {
    use crate::{field::Elem, params::set_1, poly::Polynomial};
    use num::{One, Zero};

    #[test]
    fn test_zp() {
        let rng = &mut rand::thread_rng();

        let params = set_1();
        let r = params.r;
        let n = r.n;

        let a = r.rand_polynomial(rng);
        let p = Polynomial::new(vec![Elem::zero(), Elem::one()]);
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
