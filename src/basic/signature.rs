use crate::{field::Elem, poly::Polynomial};

#[derive(Clone, Debug)]
pub struct Signature<const P: u32> {
    pub z1: Polynomial<Elem<P>>,
    pub z2: Polynomial<Elem<P>>,
    pub c: Polynomial<Elem<P>>,
}
