use crate::poly::Polynomial;
use num::BigInt;

#[derive(Clone, Debug)]
pub struct Signature {
    pub z1: Polynomial<BigInt>,
    pub z2: Polynomial<BigInt>,
    pub c: Polynomial<BigInt>,
}
