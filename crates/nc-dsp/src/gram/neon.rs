//! NEON Gram rows: eight values per vector, widening multiply-accumulates
//! into four 32-bit lanes per pair, reduced in 64 bits once per pair.

use super::GRAM_ROW_ALIGN;
use core::arch::aarch64::{
    int16x8_t, vaddlvq_s32, vdupq_n_s32, vget_low_s16, vld1q_s16, vmlal_high_s16, vmlal_s16,
};

/// Values per vector.
const LANES: usize = 8;

/// Eight values of a row.
#[target_feature(enable = "neon")]
fn load(values: &[i16; LANES]) -> int16x8_t {
    // SAFETY: NEON is enabled; the load reads exactly the 16 bytes of `values`.
    unsafe { vld1q_s16(values.as_ptr()) }
}

/// NEON [`super::GramRowsFn`]: each 32-bit lane takes two products of
/// every eighth column, so it sums at most 2^10 products of at most 2^20.
#[target_feature(enable = "neon")]
pub(super) unsafe fn gram_rows(rows: &[&[i16]], out: &mut [i64]) {
    debug_assert!(GRAM_ROW_ALIGN.is_multiple_of(LANES));
    let mut entry = 0;
    for (i, first) in rows.iter().enumerate() {
        let (a, _) = first.as_chunks::<LANES>();
        for second in &rows[i..] {
            let (b, _) = second.as_chunks::<LANES>();
            let mut sum = vdupq_n_s32(0);
            for (a, b) in a.iter().zip(b) {
                let (a, b) = (load(a), load(b));
                sum = vmlal_high_s16(vmlal_s16(sum, vget_low_s16(a), vget_low_s16(b)), a, b);
            }
            out[entry] += vaddlvq_s32(sum);
            entry += 1;
        }
    }
}
