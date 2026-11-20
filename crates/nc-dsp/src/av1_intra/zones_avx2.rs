//! AVX2 whole-block AV1 directional prediction of blocks up to 16x16
//! ([`ZoneBlock`], AV1 §7.11.2.4), after dav1d 1.5.4
//! `src/x86/ipred_avx2.asm` (`ipred_z1_8bpc`, `ipred_z2_8bpc`,
//! `ipred_z3_8bpc`, BSD-2-Clause, see `THIRD_PARTY.md`): a block is
//! predicted 16 samples at a time (one row of 16, two of 8 or four of 4),
//! each sample's edge index and weight computed in its 16-bit lane, the two
//! samples it blends gathered by `pshufb` from the first 32 samples of the
//! line held in a register, and blended by one `pmaddubsw` against the
//! `(32 - w, w)` pairs with the `pmulhrsw` rounding shift.
//!
//! The same gather serves every zone and resolution (an upsampled line's
//! pairs are just indices two apart): zone 1 clamps its index at
//! `maxBaseX`, whose sample the line repeats once past it, so no sample is
//! selected separately; zone 3 is zone 1 with the roles of row and column
//! swapped in the lanes, without dav1d's transpose; zone 2 gathers from
//! both lines and picks per lane (`base_x < minBaseX`), as dav1d's blend.
//! The per-row runs of [`crate::intra_runs`] serve larger and
//! full-resolution square blocks; a short rectangular row cost more in run
//! calls than its interpolation.
//!
//! Codec reason for an AV1-only kernel (the run family's other members,
//! HEVC, VVC and AV2 angular prediction, read one reference line per block
//! at full resolution): AV1's edge upsampling and zone 2's per-sample
//! choice between the two lines.

#![allow(clippy::inline_always)]

use core::arch::x86_64::{
    __m256i, _mm_cvtsi32_si128, _mm_cvtsi128_si32, _mm_extract_epi32, _mm_storel_epi64,
    _mm_storeu_si128, _mm_unpackhi_epi64, _mm256_add_epi16, _mm256_and_si256, _mm256_blendv_epi8,
    _mm256_castsi256_si128, _mm256_cmpgt_epi8, _mm256_cmpgt_epi16, _mm256_loadu_si256,
    _mm256_maddubs_epi16, _mm256_max_epi16, _mm256_min_epi16, _mm256_mulhrs_epi16,
    _mm256_mullo_epi16, _mm256_or_si256, _mm256_packus_epi16, _mm256_permute2x128_si256,
    _mm256_permute4x64_epi64, _mm256_set1_epi8, _mm256_set1_epi16, _mm256_setr_epi16,
    _mm256_sll_epi16, _mm256_slli_epi16, _mm256_sra_epi16, _mm256_srli_epi16, _mm256_sub_epi16,
};

use nc_common::PlaneViewMut;

use crate::intra_prediction::av1_block::{ZONE_TABLE, ZONE2_MARGIN, Zone, ZoneBlock};

/// Samples (16-bit lanes) per vector.
const LANES: usize = 16;
/// Fractional bits of an edge position.
const POSITION_BITS: i32 = 6;
/// Mask of a position's fraction.
const POSITION_MASK: i16 = (1 << POSITION_BITS) - 1;
/// Interpolation weights sum to this.
const WEIGHT_SUM: i16 = 32;
/// `pmulhrsw` by this is the `(sum + 16) >> 5` rounding of the blend.
const ROUND_SHIFT_5: i16 = 1 << 10;
/// Table indices of a `pshufb` half.
const HALF_TABLE: i8 = 16;

/// The column and the row (within one vector) of each lane for a block
/// `width` wide.
///
/// # Safety
///
/// AVX2 must be enabled in the caller.
#[inline(always)]
unsafe fn lane_grid(width: usize) -> (__m256i, __m256i) {
    // SAFETY: AVX2 is enabled (caller contract).
    unsafe {
        match width {
            4 => (
                _mm256_setr_epi16(0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3),
                _mm256_setr_epi16(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3),
            ),
            8 => (
                _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7),
                _mm256_setr_epi16(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1),
            ),
            _ => (
                _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
                _mm256_set1_epi16(0),
            ),
        }
    }
}

/// A line's first [`ZONE_TABLE`] samples as two `pshufb` tables: samples
/// 0–15 and 16–31, each in both 128-bit halves.
#[derive(Clone, Copy)]
struct Table {
    low: __m256i,
    high: __m256i,
}

impl Table {
    /// # Safety
    ///
    /// AVX2 must be enabled in the caller.
    #[inline(always)]
    unsafe fn load(line: &[u8]) -> Self {
        let line = &line[..ZONE_TABLE];
        // SAFETY: AVX2 is enabled (caller contract); `line` holds the 32
        // bytes read.
        unsafe {
            let all = _mm256_loadu_si256(line.as_ptr().cast());
            Self {
                low: _mm256_permute2x128_si256::<0x00>(all, all),
                high: _mm256_permute2x128_si256::<0x11>(all, all),
            }
        }
    }

    /// The blend of samples `index` and `index + 1` at `weight` (0..=31)
    /// in each lane, rounded: `(s0 (32 - w) + s1 w + 16) >> 5`. Indices
    /// must lie in `0..ZONE_TABLE - 1`.
    ///
    /// # Safety
    ///
    /// AVX2 must be enabled in the caller.
    #[inline(always)]
    unsafe fn interpolate(self, index: __m256i, weight: __m256i) -> __m256i {
        // SAFETY: AVX2 is enabled (caller contract).
        unsafe {
            // Byte pair (index, index + 1) per lane, then the samples there.
            let pairs = _mm256_add_epi16(
                _mm256_mullo_epi16(index, _mm256_set1_epi16(0x0101)),
                _mm256_set1_epi16(0x0100),
            );
            let in_high = _mm256_cmpgt_epi8(pairs, _mm256_set1_epi8(HALF_TABLE - 1));
            let samples = _mm256_blendv_epi8(
                core::arch::x86_64::_mm256_shuffle_epi8(self.low, pairs),
                core::arch::x86_64::_mm256_shuffle_epi8(self.high, pairs),
                in_high,
            );
            let weights = _mm256_or_si256(
                _mm256_sub_epi16(_mm256_set1_epi16(WEIGHT_SUM), weight),
                _mm256_slli_epi16::<8>(weight),
            );
            _mm256_mulhrs_epi16(
                _mm256_maddubs_epi16(samples, weights),
                _mm256_set1_epi16(ROUND_SHIFT_5),
            )
        }
    }
}

/// The weight of each lane's 1/64 `position` on a line of resolution
/// `1 << upsample`: `((position << upsample) & 63) >> 1`.
///
/// # Safety
///
/// AVX2 must be enabled in the caller.
#[inline(always)]
unsafe fn weight(position: __m256i, upsample: u32) -> __m256i {
    // SAFETY: AVX2 is enabled (caller contract).
    unsafe {
        let scaled = _mm256_sll_epi16(position, _mm_cvtsi32_si128(upsample as i32));
        _mm256_srli_epi16::<1>(_mm256_and_si256(scaled, _mm256_set1_epi16(POSITION_MASK)))
    }
}

/// Each lane's whole-sample index of `position` on a line of resolution
/// `1 << upsample` (an arithmetic shift: zone 2 positions are negative).
///
/// # Safety
///
/// AVX2 must be enabled in the caller.
#[inline(always)]
unsafe fn base(position: __m256i, upsample: u32) -> __m256i {
    // SAFETY: AVX2 is enabled (caller contract).
    unsafe {
        let shift = POSITION_BITS - upsample as i32;
        _mm256_sra_epi16(position, _mm_cvtsi32_si128(shift))
    }
}

/// The `W` samples of picture row `row`.
#[inline(always)]
fn row_of<const W: usize>(data: &mut [u8], stride: usize, row: usize) -> &mut [u8] {
    &mut data[row * stride..][..W]
}

/// Write the 16 results (16-bit lanes, 0..=255) to rows `row..` of the
/// picture `data` (rows `stride` apart), `W` samples per row.
///
/// # Safety
///
/// AVX2 must be enabled in the caller.
#[inline(always)]
unsafe fn store<const W: usize>(values: __m256i, data: &mut [u8], stride: usize, row: usize) {
    // SAFETY: AVX2 is enabled (caller contract); each store covers the
    // checked subslice it writes.
    unsafe {
        let wide = _mm256_permute4x64_epi64::<0b1000>(_mm256_packus_epi16(values, values));
        let packed = _mm256_castsi256_si128(wide);
        match W {
            16 => _mm_storeu_si128(row_of::<W>(data, stride, row).as_mut_ptr().cast(), packed),
            8 => {
                _mm_storel_epi64(row_of::<W>(data, stride, row).as_mut_ptr().cast(), packed);
                _mm_storel_epi64(
                    row_of::<W>(data, stride, row + 1).as_mut_ptr().cast(),
                    _mm_unpackhi_epi64(packed, packed),
                );
            }
            _ => {
                let quads = [
                    _mm_cvtsi128_si32(packed),
                    _mm_extract_epi32::<1>(packed),
                    _mm_extract_epi32::<2>(packed),
                    _mm_extract_epi32::<3>(packed),
                ];
                for (k, quad) in quads.into_iter().enumerate() {
                    row_of::<W>(data, stride, row + k).copy_from_slice(&quad.to_le_bytes());
                }
            }
        }
    }
}

/// [`ZoneBlock::predict`] for a block that [fits](ZoneBlock::fits).
///
/// Every index stays inside the 32-sample tables: zones 1 and 3 clamp at
/// `max_base` (`fits`: `max_base + 1 < 32`); zone 2 clamps both indices
/// below at `-2` (lanes taking the other line) and reads at most `2w - 1`
/// above and `2h - 1` left from index `-2`, at most 23 with `w, h <= 12`
/// upsampled and 17 with `w, h <= 16`. Every intermediate fits 16 bits:
/// positions are at most `16 · 1023 + 15 · 64` in magnitude.
///
/// # Panics
///
/// If the block does not fit or a line is shorter than [`ZONE_TABLE`].
///
/// # Safety
///
/// The CPU must support AVX2.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn zones(block: &ZoneBlock<'_, u8>, dst: &mut PlaneViewMut<'_>) {
    let (w, h) = (dst.width(), dst.height());
    assert!(block.fits((w, h)) && h * w >= LANES && h.is_multiple_of(LANES / w.max(1)));
    // SAFETY: AVX2 is enabled (this function's target feature).
    unsafe {
        match w {
            4 => zones_w::<4>(block, dst),
            8 => zones_w::<8>(block, dst),
            16 => zones_w::<16>(block, dst),
            _ => unreachable!("AV1 transform widths are powers of two from 4"),
        }
    }
}

/// [`zones`] for blocks `W` wide: `LANES / W` rows per vector.
///
/// # Safety
///
/// AVX2 must be enabled in the caller.
#[inline(always)]
unsafe fn zones_w<const W: usize>(block: &ZoneBlock<'_, u8>, dst: &mut PlaneViewMut<'_>) {
    let rows = LANES / W;
    let h = dst.height();
    let stride = dst.stride();
    let data = dst.data_mut();
    // SAFETY: AVX2 is enabled (caller contract); every memory access is a
    // checked slice.
    unsafe {
        let (column, row) = lane_grid(W);
        let one = _mm256_set1_epi16(1);
        match block.zone {
            Zone::Above {
                dx,
                upsample,
                max_base,
            } => {
                let line = Table::load(block.above);
                // Lane position (row + 1) dx; index (position >> fb) + (column << up).
                let mut position = _mm256_mullo_epi16(_mm256_add_epi16(row, one), splat(dx));
                let step = splat(dx * rows as i32);
                let across = _mm256_sll_epi16(column, _mm_cvtsi32_si128(upsample as i32));
                let last = splat(max_base);
                for r in (0..h).step_by(rows) {
                    let index =
                        _mm256_min_epi16(_mm256_add_epi16(base(position, upsample), across), last);
                    let values = line.interpolate(index, weight(position, upsample));
                    store::<W>(values, data, stride, r);
                    position = _mm256_add_epi16(position, step);
                }
            }
            Zone::Left {
                dy,
                upsample,
                max_base,
            } => {
                let line = Table::load(block.left);
                // Lane position (column + 1) dy; index (position >> fb) + (row << up).
                let position = _mm256_mullo_epi16(_mm256_add_epi16(column, one), splat(dy));
                let weights = weight(position, upsample);
                let shift = _mm_cvtsi32_si128(upsample as i32);
                let mut index =
                    _mm256_add_epi16(base(position, upsample), _mm256_sll_epi16(row, shift));
                let step = splat((rows as i32) << upsample);
                let last = splat(max_base);
                for r in (0..h).step_by(rows) {
                    let values = line.interpolate(_mm256_min_epi16(index, last), weights);
                    store::<W>(values, data, stride, r);
                    index = _mm256_add_epi16(index, step);
                }
            }
            Zone::Both {
                dx,
                dy,
                upsample_above,
                upsample_left,
            } => {
                let (above, left) = (Table::load(block.above), Table::load(block.left));
                let margin = splat(ZONE2_MARGIN as i32);
                let lowest = splat(-(ZONE2_MARGIN as i32));
                let min_base_above = splat(-(1 << upsample_above));
                // Lane x = (column << 6) - (row + 1) dx and y = (row << 6) - (column + 1) dy.
                let mut x = _mm256_sub_epi16(
                    _mm256_slli_epi16::<POSITION_BITS>(column),
                    _mm256_mullo_epi16(_mm256_add_epi16(row, one), splat(dx)),
                );
                let mut y = _mm256_sub_epi16(
                    _mm256_slli_epi16::<POSITION_BITS>(row),
                    _mm256_mullo_epi16(_mm256_add_epi16(column, one), splat(dy)),
                );
                // y moves by whole samples from row to row: one weight per lane.
                let left_weights = weight(y, upsample_left);
                let (x_step, y_step) = (
                    splat(dx * rows as i32),
                    splat((rows as i32) << POSITION_BITS),
                );
                for r in (0..h).step_by(rows) {
                    let base_x = base(x, upsample_above);
                    let from_left = _mm256_cmpgt_epi16(min_base_above, base_x);
                    let above_index = _mm256_add_epi16(_mm256_max_epi16(base_x, lowest), margin);
                    let left_index =
                        _mm256_add_epi16(_mm256_max_epi16(base(y, upsample_left), lowest), margin);
                    let values = _mm256_blendv_epi8(
                        above.interpolate(above_index, weight(x, upsample_above)),
                        left.interpolate(left_index, left_weights),
                        from_left,
                    );
                    store::<W>(values, data, stride, r);
                    x = _mm256_sub_epi16(x, x_step);
                    y = _mm256_add_epi16(y, y_step);
                }
            }
        }
    }
}

/// `value` in every lane (block geometry: at most `16 · 1023` in magnitude).
///
/// # Safety
///
/// AVX2 must be enabled in the caller.
#[inline(always)]
unsafe fn splat(value: i32) -> __m256i {
    // SAFETY: AVX2 is enabled (caller contract).
    unsafe { _mm256_set1_epi16(value as i16) }
}
