//! Tests of [`super`]: fixed vectors from dav2d's C (commit 237229e, 8 bpc)
//! and the kernel selection of `recon_tmpl.c`. The full differential test
//! against dav2d (every set, type, stride, transpose and end of block) was
//! run outside the repository when porting.

use super::*;
use crate::av2_dec::itx::tests::pattern;
use nc_common::BitDepth;
use nc_common::fnv1a;

const BUFFER: usize = 32 * 32;
const EIGHT: BitDepth = BitDepth::EIGHT;

fn hash(cf: &[i16]) -> u64 {
    fnv1a(cf.iter().flat_map(|c| c.to_le_bytes()))
}

#[test]
fn matches_dav2d_vectors() {
    let mut cf: Vec<i16> = (0..BUFFER).map(pattern).collect();
    secondary4(&mut cf, 16, true, kernel4(3, 1), 7, EIGHT);
    assert_eq!(hash(&cf), 0x1c4c_cc1c_f6b9_5d30);

    let mut cf: Vec<i16> = (0..BUFFER).map(pattern).collect();
    let (kernel, map_idx) = kernel8(10, 2);
    secondary8(&mut cf, 32, false, map_idx, kernel, 31, EIGHT);
    assert_eq!(hash(&cf), 0x4d4d_fdd4_83bc_6163);
}

#[test]
fn apply_secondary_selects_dav2d_kernels() {
    // stx = type 2 (0-based 1) of set 3.
    let stx = (3 << STX_SET_SHIFT) | 2;
    let mut expected: Vec<i16> = (0..BUFFER).map(pattern).collect();
    let mut cf = expected.clone();
    // 4x16: a 4-wide block takes the 4x4 kernel at column stride 16.
    secondary4(&mut expected, 16, true, kernel4(3, 1), 5, EIGHT);
    assert_eq!(
        apply_secondary(&mut cf, RectTxSize::Tx4x16, stx, true, 5, EIGHT),
        EOB_AFTER_STX4
    );
    assert_eq!(cf, expected);

    let mut expected: Vec<i16> = (0..BUFFER).map(pattern).collect();
    let mut cf = expected.clone();
    // 64x8: the 8x8 kernel at column stride 8.
    let (kernel, map_idx) = kernel8(3, 1);
    secondary8(&mut expected, 8, false, map_idx, kernel, 20, EIGHT);
    assert_eq!(
        apply_secondary(&mut cf, RectTxSize::Tx64x8, stx, false, 20, EIGHT),
        EOB_AFTER_STX8
    );
    assert_eq!(cf, expected);
}

#[test]
fn only_horizontal_intra_modes_keep_the_orientation() {
    assert!(transposed(None));
    assert!(transposed(Some(0)));
    for mode in [HOR_PRED, HOR_DOWN_PRED, VERT_LEFT_PRED, SMOOTH_H_PRED] {
        assert!(!transposed(Some(mode)));
    }
}

/// The `i32` coefficients at 8 bits give the `i16` results, and at 10 bits
/// the outputs clamp to `±2^17` rather than the 8-bit `±2^15`.
#[test]
fn wide_coefficients_clamp_to_the_depth() {
    let narrow: Vec<i16> = (0..BUFFER).map(pattern).collect();
    let mut wide: Vec<i32> = narrow.iter().map(|&c| c.into()).collect();
    let mut narrow = narrow;
    let stx = (5 << STX_SET_SHIFT) | 1;
    let narrow_eob = apply_secondary(&mut narrow, RectTxSize::Tx16x16, stx, true, 31, EIGHT);
    let wide_eob = apply_secondary(&mut wide, RectTxSize::Tx16x16, stx, true, 31, EIGHT);
    assert_eq!(narrow_eob, wide_eob);
    assert_eq!(
        wide,
        narrow.iter().map(|&c| i32::from(c)).collect::<Vec<_>>()
    );

    let ten = BitDepth::TEN;
    let mut big = vec![0; BUFFER];
    big[..STX8_INPUTS].fill(1_i32 << 17);
    apply_secondary(&mut big, RectTxSize::Tx8x8, stx, false, 31, ten);
    let bound = 1 << 17;
    assert!(big[..64].iter().all(|&c| (-bound..bound).contains(&c)));
    assert!(big[..64].iter().any(|&c| c.abs() > i32::from(i16::MAX)));
}
