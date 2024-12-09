use num::{One, Zero};
use std::ops::Neg;

use crate::poly::Polynomial;

/// Convert a 160-bit string to a polynomial of degree n - 1 for n >= 512
/// that have all zero coefficients except for at most 32 coefficients that are +-1.
pub fn hash<T>(mut p: Polynomial<T>, n: usize, r: &[u8]) -> Polynomial<T>
where
    T: Zero + One + Neg<Output = T>,
{
    assert_eq!(r.len(), 20);
    assert!(n >= 512);
    assert!(p.coeffs.len() >= n);

    for i in (0..160).step_by(5) {
        let r0 = bit(r, i);
        let pos_num = byte_at_bit(r, i + 1);

        // divide coefficients into 16-bit strings, and assign
        // a 1 or -1 according to the position number.
        let coeff_i = (i / 5) * 16 + pos_num as usize;
        if r0 == 0 {
            // put a -1 in position number of the 16-bit string
            p.coeffs[coeff_i] = -T::one();
        } else {
            // put a 1 in position number of the 16-bit string
            p.coeffs[coeff_i] = T::one();
        }
    }

    // if the polynomial is of degree greater than 512, then
    // all of its higher-order terms will be 0.
    if p.coeffs.len() > n {
        p.truncate(n);
    }

    p
}

/// Return the i-th bit of a byte string.
#[inline]
fn bit(bs: &[u8], i: usize) -> u8 {
    (bs[i / 8] >> (i % 8)) & 1
}

/// Return a value (0-15) with starting bit at i-th position.
#[inline]
fn byte_at_bit(bs: &[u8], i: usize) -> u8 {
    let r1 = bit(bs, i);
    let r2 = bit(bs, i + 1);
    let r3 = bit(bs, i + 2);
    let r4 = bit(bs, i + 3);

    // r1 r2 r3 r4 (where we read the 4-digit string as a number between 0 and 15)
    (r4 | r3 << 1 | r2 << 2 | r1 << 3) % 16
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_bit() {
        let bs = [0b1010_1010, 0b0101_0101];
        assert_eq!(bit(&bs, 0), 0);
        assert_eq!(bit(&bs, 1), 1);
        assert_eq!(bit(&bs, 7), 1);

        assert_eq!(bit(&bs, 8), 1);
        assert_eq!(bit(&bs, 9), 0);
        assert_eq!(bit(&bs, 15), 0);

        assert_eq!(byte_at_bit(&bs, 4), 5); // r1 = 0, r2 = 1, r3 = 0, r4 = 1 => 0b0101 = 5
        assert_eq!(byte_at_bit(&bs, 8), 10); // r1 = 1, r2 = 0, r3 = 1, r4 = 0 => 0b1010 = 10
    }
}
