use crate::{hash::hash, params::Params, poly::Polynomial};
use digest::Digest;
use num::BigInt;
use rand::Rng;

use super::{signature::Signature, verify::VerificationKey};

#[derive(Clone, Debug)]
pub struct SigningKey {
    pub(crate) s1: Polynomial<BigInt>,
    pub(crate) s2: Polynomial<BigInt>,
}

impl SigningKey {
    pub fn sign<R: Rng, H: Digest>(
        &self,
        rng: &mut R,
        params: &Params,
        vk: &VerificationKey,
        message: &[u8],
    ) -> Signature {
        let bound_k = BigInt::from(params.k);
        let bound_k_32 = BigInt::from(params.k - 32);
        loop {
            let y1 = params.r.rand_polynomial_within(rng, &bound_k);
            let y2 = params.r.rand_polynomial_within(rng, &bound_k);
            let c = {
                let ay1 = params.r.mul(&vk.a, &y1);
                let ay1_y2 = params.r.add(&ay1, &y2);
                let ay1_y2_bytes = params.r.convert_polynomial_to_bytes(ay1_y2);
                hash::<_, H>(params.r.n as usize, &[&ay1_y2_bytes, message].concat())
            };
            let s1c = params.r.mul(&self.s1, &c);
            let s2c = params.r.mul(&self.s2, &c);
            let z1 = params.r.add(&s1c, &y1);
            let z2 = params.r.add(&s2c, &y2);
            if params.r.is_valid_within(&z1, &bound_k_32)
                && params.r.is_valid_within(&z2, &bound_k_32)
            {
                return Signature { z1, z2, c };
            }
        }
    }
}
