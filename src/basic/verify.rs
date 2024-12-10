use digest::Digest;
use num::BigInt;

use crate::{hash::hash, params::Params, poly::Polynomial};

use super::signature::Signature;

#[derive(Clone, Debug)]
pub struct VerificationKey {
    pub(crate) a: Polynomial<BigInt>,
    pub(crate) t: Polynomial<BigInt>,
}

impl VerificationKey {
    pub fn verify<H: Digest>(
        &self,
        params: &Params,
        message: &[u8],
        signature: &Signature,
    ) -> bool {
        let Params { r, k } = params;
        let Signature { z1, z2, c } = signature;
        let bound = BigInt::from(k - 32);

        if !r.is_valid_within(z1, &bound) || !r.is_valid_within(z2, &bound) {
            return false;
        }

        let c_prime = {
            let tc = r.mul(self.t.clone(), c.clone());
            let az1_z2 = r.add(self.a.clone() * z1.clone(), z2.clone());
            let az1_z2_tc = r.sub(az1_z2, tc);
            let az1_z2_tc_bytes = params.r.convert_polynomial_to_bytes(az1_z2_tc);
            hash::<_, H>(params.r.n as usize, &[&az1_z2_tc_bytes, message].concat())
        };

        c == &c_prime
    }
}
