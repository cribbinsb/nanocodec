//! Tests of [`super`]: fixed vectors from dav2d's C (commit 237229e, 8 bpc)
//! and the kernel selection of `recon_tmpl.c`. The full differential test
//! against dav2d (every set, type, stride, transpose and end of block) was
//! run outside the repository when porting.

use super::*;
use crate::av2_dec::itx::tests::pattern;
use nc_common::fnv1a;

const BUFFER: usize = 32 * 32;

fn hash(cf: &[i16]) -> u64 {
    fnv1a(cf.iter().flat_map(|c| c.to_le_bytes()))
}

#[test]
fn matches_dav2d_vectors() {
    let mut cf: Vec<i16> = (0..BUFFER).map(pattern).collect();
    secondary4(&mut cf, 16, true, kernel4(3, 1), 7);
    assert_eq!(hash(&cf), 0x1c4c_cc1c_f6b9_5d30);

    let mut cf: Vec<i16> = (0..BUFFER).map(pattern).collect();
    let (kernel, map_idx) = kernel8(10, 2);
    secondary8(&mut cf, 32, false, map_idx, kernel, 31);
    assert_eq!(hash(&cf), 0x4d4d_fdd4_83bc_6163);
}

#[test]
fn apply_secondary_selects_dav2d_kernels() {
    // stx = type 2 (0-based 1) of set 3.
    let stx = (3 << STX_SET_SHIFT) | 2;
    let mut expected: Vec<i16> = (0..BUFFER).map(pattern).collect();
    let mut cf = expected.clone();
    // 4x16: a 4-wide block takes the 4x4 kernel at column stride 16.
    secondary4(&mut expected, 16, true, kernel4(3, 1), 5);
    assert_eq!(
        apply_secondary(&mut cf, RectTxSize::Tx4x16, stx, true, 5),
        EOB_AFTER_STX4
    );
    assert_eq!(cf, expected);

    let mut expected: Vec<i16> = (0..BUFFER).map(pattern).collect();
    let mut cf = expected.clone();
    // 64x8: the 8x8 kernel at column stride 8.
    let (kernel, map_idx) = kernel8(3, 1);
    secondary8(&mut expected, 8, false, map_idx, kernel, 20);
    assert_eq!(
        apply_secondary(&mut cf, RectTxSize::Tx64x8, stx, false, 20),
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
