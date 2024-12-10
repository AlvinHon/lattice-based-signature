use num::BigUint;

use crate::ring::Ring;

#[derive(Clone, Debug)]
pub struct Params {
    pub(crate) r: Ring,
    pub(crate) k: u32,
}

impl Params {
    pub fn new<B: Into<BigUint>>(p: B, n: u32, k: u32) -> Self {
        assert!(k > 32);
        let r = Ring::new(p, n);
        Params { r, k }
    }

    pub fn set_1() -> Self {
        let p = 8383489u64;
        let n = 512u32;
        let k = 16384u32; // 2^14
        Params::new(p, n, k)
    }

    pub fn set_2() -> Self {
        let p = 16760833u64;
        let n = 1024u32;
        let k = 32768u32; // 2^15
        Params::new(p, n, k)
    }
}
