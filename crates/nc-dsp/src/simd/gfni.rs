//! Affine matrices for logical and arithmetic byte shifts, shared by CDEF
//! and loop filters. GFNI stores the row producing bit i in byte 7-i.

/// Copy input bit i+shift, extending with zero or the sign bit. A shift
/// of eight gives zero (logical) or repeats the sign in every output bit.
pub(crate) const fn shift_matrix(shift: i32, signed: bool) -> i64 {
    assert!(shift >= 0 && shift <= 8);
    let mut matrix = 0u64;
    let mut bit = 0;
    while bit < 8 {
        let source = bit + shift;
        let row = if source < 8 {
            1u8 << source
        } else if signed {
            128
        } else {
            0
        };
        matrix |= (row as u64) << ((7 - bit) * 8);
        bit += 1;
    }
    matrix as i64
}
