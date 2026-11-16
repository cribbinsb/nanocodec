//! Geometric mask fingerprints independently prepared from `VVdeC`'s full masks.
use super::*;
use nc_common::{FNV1A_OFFSET, fnv1a_step};
#[test]
fn compact_line_matches_all_reference_masks() {
    let mut hash = FNV1A_OFFSET;
    for width in [8, 16, 32, 64] {
        for height in [8, 16, 32, 64] {
            for split in 0..64 {
                let mask = VvcGeometric::new((width, height), split);
                for y in 0..height {
                    for x in 0..width {
                        hash = fnv1a_step(hash, u64::from(mask.weight(x, y)));
                    }
                }
            }
        }
    }
    // VVdeC builds six 112x112 masks, then applies table offsets/mirroring.
    assert_eq!(hash, 0x297e_69e7_485b_2021);
}
#[test]
fn chroma_blend_samples_the_luma_mask_before_rounding() {
    let dsp = Dsp::new();
    let mask = VvcGeometric::new((16, 16), 7);
    let first = [64 * 31 + 17; 16];
    let second = [64 * 183 + 45; 16];
    let mut result = [0; 16];
    dsp.vvc_geometric_blend(mask, (4, 4), (4, 4), true, (&first, &second), &mut result);
    for (i, &pixel) in result.iter().enumerate() {
        let weight = i64::from(mask.weight(4 + 2 * (i % 4), 4 + 2 * (i / 4)));
        let expected =
            (weight * i64::from(first[i]) + (8 - weight) * i64::from(second[i]) + 256) / 512;
        assert_eq!(i64::from(pixel), expected);
    }
}
