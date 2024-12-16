use crate::ring::Ring;

/// Parameters for the signature scheme.
#[derive(Clone, Debug)]
pub struct Params<const P: u32> {
    pub(crate) r: Ring<P>,
    pub(crate) k: u32,
}

impl<const P: u32> Params<P> {
    /// Create a new parameter set.
    /// Panics if k <= 32 and if p % (n * 2) != 1
    pub fn new(n: u32, k: u32) -> Self {
        assert!(k > 32);
        let r = Ring::new(n);
        Params { r, k }
    }
}

/// Set 1 parameters. As defined in the paper, it gives approximately:
/// - 8950 bits of signature
/// - 1620 bits of secret key
/// - 11800 bits of public key
pub fn set_1() -> Params<8383489> {
    const P: u32 = 8383489u32;
    let n = 512u32;
    let k = 16384u32; // 2^14
    Params::<P>::new(n, k)
}

/// Set 1 parameters. As defined in the paper, it gives approximately:
/// - 18800 bits of signature
/// - 3250 bits of secret key
/// - 25000 bits of public key
pub fn set_2() -> Params<16760833> {
    const P: u32 = 16760833u32;
    let n = 1024u32;
    let k = 32768u32; // 2^15
    Params::<P>::new(n, k)
}
