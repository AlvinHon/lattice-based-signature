use digest::Digest;
use num::BigInt;

use crate::{field::Elem, hash::hash, params::Params, poly::Polynomial};

use super::signature::Signature;

#[derive(Clone, Debug)]
pub struct VerificationKey<const P: u32> {
    pub(crate) a: Polynomial<Elem<P>>,
    pub(crate) t: Polynomial<Elem<P>>,
}

impl<const P: u32> VerificationKey<P> {
    pub fn verify<H: Digest>(
        &self,
        params: &Params<P>,
        message: &[u8],
        signature: &Signature<P>,
    ) -> bool {
        let Params { r, k } = params;
        let Signature { z1, z2, c } = signature;
        let bound = BigInt::from(k - 32);

        if !r.is_valid_within(z1, &bound) || !r.is_valid_within(z2, &bound) {
            return false;
        }

        let c_prime = {
            let tc = r.mul(&self.t, c);
            let az1 = r.mul(&self.a, z1);
            let az1_z2 = r.add(&az1, z2);
            let az1_z2_tc = r.sub(az1_z2, tc);
            let az1_z2_tc_bytes = params.r.convert_polynomial_to_bytes(az1_z2_tc);
            hash::<_, H>(params.r.n as usize, &[&az1_z2_tc_bytes, message].concat())
        };

        c == &c_prime
    }
}
