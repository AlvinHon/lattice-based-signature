//! Defines the `SigningKey` struct and its methods.

use crate::{field::Elem, hash::hash, params::Params, poly::Polynomial};
use digest::Digest;
use rand::Rng;

use super::{signature::Signature, verify::VerificationKey};

#[derive(Clone, Debug)]
pub struct SigningKey<const P: u32> {
    pub(crate) s1: Polynomial<Elem<P>>,
    pub(crate) s2: Polynomial<Elem<P>>,
}

impl<const P: u32> SigningKey<P> {
    /// Sign a message using the key pair.
    ///
    /// This method requires digest `H` from [Digest] because signature scheme
    /// does not specify how the message is hashed into some bytes. Hence, it
    /// is up to the caller to provide a hash algorithm.
    pub fn sign<R: Rng, H: Digest>(
        &self,
        rng: &mut R,
        params: &Params<P>,
        vk: &VerificationKey<P>,
        message: &[u8],
    ) -> Signature<P> {
        let bound_k = Elem::<P>::from(params.k).to_signed();
        let bound_k_32 = Elem::<P>::from(params.k - 32).to_signed();
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
