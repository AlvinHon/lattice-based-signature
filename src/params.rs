use crate::ring::Ring;

#[derive(Clone, Debug)]
pub struct Params<const P: u32> {
    pub(crate) r: Ring<P>,
    pub(crate) k: u32,
}

impl<const P: u32> Params<P> {
    pub fn new(n: u32, k: u32) -> Self {
        assert!(k > 32);
        let r = Ring::new(n);
        Params { r, k }
    }
}

pub fn set_1() -> Params<8383489> {
    const P: u32 = 8383489u32;
    let n = 512u32;
    let k = 16384u32; // 2^14
    Params::<P>::new(n, k)
}

pub fn set_2() -> Params<16760833> {
    const P: u32 = 16760833u32;
    let n = 1024u32;
    let k = 32768u32; // 2^15
    Params::<P>::new(n, k)
}
