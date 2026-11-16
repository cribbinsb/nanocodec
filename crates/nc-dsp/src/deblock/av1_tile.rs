//! Shared grouped eight-line tile IO for AV1 and VP9 filtering.
#![allow(clippy::inline_always)] // Tile IO must retain its caller's ISA features.
use super::av1_packed::Tile;
use crate::lpf_bytes::{GroupedBytes, LANES};

pub(crate) struct GroupedTile<B>(core::marker::PhantomData<B>);
impl<B: GroupedBytes> Tile<B> for GroupedTile<B> {
    #[inline(always)]
    unsafe fn load<const LINES: usize>(
        data: *const u8,
        edge: usize,
        across: usize,
        along: usize,
        before: usize,
    ) -> [B; LANES] {
        debug_assert_eq!(LINES, LANES * B::GROUPS);
        let origin = edge - before * across;
        // SAFETY: caller validates all complete eight-line tiles.
        unsafe {
            // Keep grouped loads in the ISA caller: Array::from_fn outlined on
            // x86 and called a vector-construction helper for every row.
            let mut rows = [B::splat(0); LANES];
            for (i, row) in rows.iter_mut().enumerate() {
                let at = if along == 1 {
                    origin + i * across
                } else {
                    origin + i * along
                };
                *row = B::load_groups(data.add(at), LANES * along);
            }
            if along == 1 { rows } else { B::transpose(rows) }
        }
    }
    #[inline(always)]
    unsafe fn store<const LINES: usize>(
        data: *mut u8,
        edge: usize,
        across: usize,
        along: usize,
        before: usize,
        values: [B; LANES],
        changed: core::ops::Range<usize>,
    ) {
        debug_assert_eq!(LINES, LANES * B::GROUPS);
        let origin = edge - before * across;
        // SAFETY: caller validates all complete eight-line tiles. Vertical
        // stores preserve outer taps by transposing the complete tile back.
        unsafe {
            if along == 1 {
                for i in changed {
                    let at = origin + i * across;
                    values[i].store_groups(data.add(at), LANES);
                }
            } else {
                for (i, row) in B::transpose(values).into_iter().enumerate() {
                    let at = origin + i * along;
                    row.store_groups(data.add(at), LANES * along);
                }
            }
        }
    }
}
