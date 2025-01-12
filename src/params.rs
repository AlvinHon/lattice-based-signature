use crate::ring::Ring;

/// Parameters for the signature scheme.
#[derive(Clone, Debug)]
pub struct Params<const P: u32, const N: usize> {
    pub(crate) r: Ring<P, N>,
    pub(crate) k: u32,
}

impl<const P: u32, const N: usize> Params<P, N> {
    /// Create a new parameter set.
    /// Panics if k <= 32 and if p % (n * 2) != 1
    pub fn new(k: u32) -> Self {
        assert!(k > 32);
        Params {
            r: Ring::default(),
            k,
        }
    }

    /// Return the prime modulus P in the ring Zp[x]/(x^n + 1).
    #[inline]
    pub fn p(&self) -> u32 {
        P
    }

    /// Return the value of N in the ring Zp[x]/(x^n + 1).
    #[inline]
    pub fn n(&self) -> usize {
        N
    }
}

/// Set 1 parameters. As defined in the paper, it gives approximately:
/// - 8950 bits of signature
/// - 1620 bits of secret key
/// - 11800 bits of public key
pub fn set_1() -> Params<8383489, 512> {
    const P: u32 = 8383489u32;
    const N: usize = 512;
    let k = 16384u32; // 2^14
    Params::<P, N>::new(k)
}

/// Set 1 parameters. As defined in the paper, it gives approximately:
/// - 18800 bits of signature
/// - 3250 bits of secret key
/// - 25000 bits of public key
pub fn set_2() -> Params<16760833, 1024> {
    const P: u32 = 16760833u32;
    const N: usize = 1024;
    let k = 32768u32; // 2^15
    Params::<P, N>::new(k)
}
