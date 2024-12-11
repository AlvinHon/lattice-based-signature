pub mod sign;
pub mod signature;
pub mod verify;

use crate::params::Params;
use num::{BigInt, One};
use rand::Rng;
use sign::SigningKey;
use verify::VerificationKey;

pub trait RandKeyGen {
    fn gen_signing_key(&mut self, params: &Params) -> SigningKey;
    fn gen_verifying_key(&mut self, params: &Params, sk: &SigningKey) -> VerificationKey;
}

impl<R: Rng> RandKeyGen for R {
    fn gen_signing_key(&mut self, params: &Params) -> SigningKey {
        let one = BigInt::one();
        let s1 = params.r.rand_polynomial_within(self, &one);
        let s2 = params.r.rand_polynomial_within(self, &one);

        SigningKey { s1, s2 }
    }

    fn gen_verifying_key(&mut self, params: &Params, sk: &SigningKey) -> VerificationKey {
        let a = params.r.rand_polynomial(self);
        let t = {
            let t_as1 = params.r.mul(&a, &sk.s1);
            params.r.add(t_as1, sk.s2.clone())
        };

        VerificationKey { a, t }
    }
}

#[cfg(test)]
mod test {
    use ripemd::Ripemd160;

    use crate::params::Params;

    use super::*;

    #[test]
    fn test_basic_signature_scheme_set_1() {
        let rng = &mut rand::thread_rng();
        let params = Params::set_1();
        let sk = rng.gen_signing_key(&params);
        let vk = rng.gen_verifying_key(&params, &sk);

        // random 20-byte message
        let message = (0..20).map(|_| rng.gen()).collect::<Vec<u8>>();

        let sig = sk.sign::<_, Ripemd160>(rng, &params, &vk, &message);
        assert!(vk.verify::<Ripemd160>(&params, &message, &sig));
    }

    #[test]
    fn test_basic_signature_scheme_set_2() {
        let rng = &mut rand::thread_rng();
        let params = Params::set_2();
        let sk = rng.gen_signing_key(&params);
        let vk = rng.gen_verifying_key(&params, &sk);

        // random 20-byte message
        let message = (0..20).map(|_| rng.gen()).collect::<Vec<u8>>();

        let sig = sk.sign::<_, Ripemd160>(rng, &params, &vk, &message);
        assert!(vk.verify::<Ripemd160>(&params, &message, &sig));
    }
}
