//! Palette prediction: each sample is the palette colour its index names.
//!
//! Ported from dav2d `src/ipred_tmpl.c` (`pal_pred_c`; commit 237229e,
//! BSD-2-Clause, see `docs/licenses/dav2d`).

use nc_common::Sample;

/// Bits of one index in the packed index array.
const INDEX_BITS: u32 = 4;
/// Mask of one index (palettes have at most eight colours).
const INDEX_MASK: u8 = 7;

/// Writes the `w` x `h` block at `offset` in `dst` (rows `stride` apart)
/// from palette `pal` and `idx`, which packs two indices per byte (the left
/// sample in the low nibble), `w / 2` bytes per row (`pal_pred`).
pub fn pal_pred<T: Sample>(
    dst: &mut [T],
    offset: usize,
    stride: usize,
    pal: &[T; 8],
    idx: &[u8],
    w: usize,
    h: usize,
) {
    let packed_row = w / 2;
    for (y, indices) in idx.chunks_exact(packed_row).take(h).enumerate() {
        let row = &mut dst[offset + y * stride..][..w];
        for (pair, &i) in row.as_chunks_mut::<2>().0.iter_mut().zip(indices) {
            pair[0] = pal[usize::from(i & INDEX_MASK)];
            pair[1] = pal[usize::from(i >> INDEX_BITS)];
        }
    }
}
