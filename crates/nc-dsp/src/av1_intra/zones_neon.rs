//! NEON whole-block AV1 directional prediction of blocks up to 16x16
//! ([`ZoneBlock`], AV1 §7.11.2.4), after dav1d 1.5.4
//! `src/arm/64/ipred.S` (`ipred_z1_fill1`, `ipred_z2_fill1`,
//! `ipred_z3_fill1`, BSD-2-Clause, see `THIRD_PARTY.md`): the same layout
//! and technique as the AVX2 kernel ([`super::zones_avx2`]), a block
//! predicted 16 samples at a time (one row of 16, two of 8 or four of 4),
//! each sample's edge index and weight computed in a 16-bit lane (two
//! 8-lane registers), the two samples it blends gathered by `tbl` from the
//! first 32 samples of the line held in a register pair, and blended by
//! `umull` / `umlal` against `(32 - w, w)` with a rounding narrow by 5.
//!
//! As on AVX2, one gather serves every zone and resolution: zone 1 clamps
//! its index at `maxBaseX` (the line repeats that sample once past it),
//! zone 3 swaps the roles of row and column in the lanes, zone 2 gathers
//! from both lines and picks per lane (`base_x < minBaseX`).
//!
//! Codec reason for an AV1-only kernel: as [`super::zones_avx2`].

#![allow(clippy::inline_always)]

use core::arch::aarch64::{
    int16x8_t, uint8x16_t, uint8x16x2_t, vaddq_s16, vaddq_u8, vandq_s16, vbslq_u8, vcgtq_s16,
    vcombine_u8, vdupq_n_s16, vdupq_n_u8, vget_high_u8, vget_low_u8, vgetq_lane_u32, vld1q_s16,
    vld1q_u8_x2, vmaxq_s16, vminq_s16, vmlal_high_u8, vmlal_u8, vmovn_u16, vmull_high_u8, vmull_u8,
    vmulq_s16, vqtbl2q_u8, vreinterpretq_s16_u16, vreinterpretq_u16_s16, vreinterpretq_u32_u8,
    vrshrn_high_n_u16, vrshrn_n_u16, vshlq_s16, vshrq_n_s16, vst1_u8, vst1q_u8, vsubq_s16,
    vsubq_u8,
};

use nc_common::PlaneViewMut;

use crate::intra_prediction::av1_block::{ZONE_TABLE, ZONE2_MARGIN, Zone, ZoneBlock};

/// Samples (16-bit lanes, over two registers) per step.
const LANES: usize = 16;
/// 16-bit lanes per register.
const HALF: usize = 8;
/// Fractional bits of an edge position.
const POSITION_BITS: i16 = 6;
/// Mask of a position's fraction.
const POSITION_MASK: i16 = (1 << POSITION_BITS) - 1;
/// Interpolation weights sum to this.
const WEIGHT_SUM: u8 = 32;
/// The blend's rounding shift: `(sum + 16) >> 5`.
const ROUND_BITS: i32 = 5;

/// Sixteen 16-bit lanes: samples 0–7 and 8–15 of a step.
#[derive(Clone, Copy)]
struct Lanes {
    low: int16x8_t,
    high: int16x8_t,
}

impl Lanes {
    /// # Safety
    ///
    /// NEON must be enabled in the caller.
    #[inline(always)]
    unsafe fn load(values: &[i16; LANES]) -> Self {
        // SAFETY: NEON is enabled (caller contract); each load reads 8 of
        // the array's 16 values.
        unsafe {
            Self {
                low: vld1q_s16(values.as_ptr()),
                high: vld1q_s16(values[HALF..].as_ptr()),
            }
        }
    }

    /// `value` in every lane (block geometry: at most `16 · 1023` in
    /// magnitude).
    ///
    /// # Safety
    ///
    /// NEON must be enabled in the caller.
    #[inline(always)]
    unsafe fn splat(value: i32) -> Self {
        // SAFETY: NEON is enabled (caller contract).
        let v = unsafe { vdupq_n_s16(value as i16) };
        Self { low: v, high: v }
    }

    /// Applies a two-operand lane operation to both halves.
    #[inline(always)]
    fn zip(self, other: Self, op: impl Fn(int16x8_t, int16x8_t) -> int16x8_t) -> Self {
        Self {
            low: op(self.low, other.low),
            high: op(self.high, other.high),
        }
    }

    /// # Safety
    ///
    /// NEON must be enabled in the caller.
    #[inline(always)]
    unsafe fn add(self, other: Self) -> Self {
        // SAFETY: NEON is enabled (caller contract).
        self.zip(other, |a, b| unsafe { vaddq_s16(a, b) })
    }

    /// # Safety
    ///
    /// NEON must be enabled in the caller.
    #[inline(always)]
    unsafe fn sub(self, other: Self) -> Self {
        // SAFETY: NEON is enabled (caller contract).
        self.zip(other, |a, b| unsafe { vsubq_s16(a, b) })
    }

    /// # Safety
    ///
    /// NEON must be enabled in the caller.
    #[inline(always)]
    unsafe fn mul(self, other: Self) -> Self {
        // SAFETY: NEON is enabled (caller contract).
        self.zip(other, |a, b| unsafe { vmulq_s16(a, b) })
    }

    /// # Safety
    ///
    /// NEON must be enabled in the caller.
    #[inline(always)]
    unsafe fn min(self, other: Self) -> Self {
        // SAFETY: NEON is enabled (caller contract).
        self.zip(other, |a, b| unsafe { vminq_s16(a, b) })
    }

    /// # Safety
    ///
    /// NEON must be enabled in the caller.
    #[inline(always)]
    unsafe fn max(self, other: Self) -> Self {
        // SAFETY: NEON is enabled (caller contract).
        self.zip(other, |a, b| unsafe { vmaxq_s16(a, b) })
    }

    /// Each lane shifted left by `shift` (right, arithmetically, when
    /// negative).
    ///
    /// # Safety
    ///
    /// NEON must be enabled in the caller.
    #[inline(always)]
    unsafe fn shift(self, shift: i32) -> Self {
        // SAFETY: NEON is enabled (caller contract).
        unsafe { self.zip(Self::splat(shift), |a, b| vshlq_s16(a, b)) }
    }

    /// The low byte of each lane (values in `0..=255`), in lane order.
    ///
    /// # Safety
    ///
    /// NEON must be enabled in the caller.
    #[inline(always)]
    unsafe fn narrow(self) -> uint8x16_t {
        // SAFETY: NEON is enabled (caller contract).
        unsafe {
            vcombine_u8(
                vmovn_u16(vreinterpretq_u16_s16(self.low)),
                vmovn_u16(vreinterpretq_u16_s16(self.high)),
            )
        }
    }
}

/// The column and the row (within one step) of each lane for a block
/// `width` wide.
///
/// # Safety
///
/// NEON must be enabled in the caller.
#[inline(always)]
unsafe fn lane_grid(width: usize) -> (Lanes, Lanes) {
    const COLUMNS_4: [i16; LANES] = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3];
    const ROWS_4: [i16; LANES] = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3];
    const COLUMNS_8: [i16; LANES] = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7];
    const ROWS_8: [i16; LANES] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
    const COLUMNS_16: [i16; LANES] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    // SAFETY: NEON is enabled (caller contract).
    unsafe {
        match width {
            4 => (Lanes::load(&COLUMNS_4), Lanes::load(&ROWS_4)),
            8 => (Lanes::load(&COLUMNS_8), Lanes::load(&ROWS_8)),
            _ => (Lanes::load(&COLUMNS_16), Lanes::splat(0)),
        }
    }
}

/// A line's first [`ZONE_TABLE`] samples as a two-register `tbl` table.
#[derive(Clone, Copy)]
struct Table(uint8x16x2_t);

impl Table {
    /// # Safety
    ///
    /// NEON must be enabled in the caller.
    #[inline(always)]
    unsafe fn load(line: &[u8]) -> Self {
        let line = &line[..ZONE_TABLE];
        // SAFETY: NEON is enabled (caller contract); `line` holds the 32
        // bytes read.
        Self(unsafe { vld1q_u8_x2(line.as_ptr()) })
    }

    /// The blend of samples `index` and `index + 1` at `weight` (0..=31)
    /// in each lane, rounded: `(s0 (32 - w) + s1 w + 16) >> 5`, at most
    /// `255 · 32 + 16` before the shift (u16). Indices must lie in
    /// `0..ZONE_TABLE - 1`.
    ///
    /// # Safety
    ///
    /// NEON must be enabled in the caller.
    #[inline(always)]
    unsafe fn interpolate(self, index: Lanes, weight: Lanes) -> uint8x16_t {
        // SAFETY: NEON is enabled (caller contract).
        unsafe {
            let index = index.narrow();
            let s0 = vqtbl2q_u8(self.0, index);
            let s1 = vqtbl2q_u8(self.0, vaddq_u8(index, vdupq_n_u8(1)));
            let w1 = weight.narrow();
            let w0 = vsubq_u8(vdupq_n_u8(WEIGHT_SUM), w1);
            let low = vmlal_u8(
                vmull_u8(vget_low_u8(s0), vget_low_u8(w0)),
                vget_low_u8(s1),
                vget_low_u8(w1),
            );
            let high = vmlal_high_u8(vmull_high_u8(s0, w0), s1, w1);
            vrshrn_high_n_u16::<ROUND_BITS>(vrshrn_n_u16::<ROUND_BITS>(low), high)
        }
    }
}

/// The weight of each lane's 1/64 `position` on a line of resolution
/// `1 << upsample`: `((position << upsample) & 63) >> 1`.
///
/// # Safety
///
/// NEON must be enabled in the caller.
#[inline(always)]
unsafe fn weight(position: Lanes, upsample: u32) -> Lanes {
    // SAFETY: NEON is enabled (caller contract).
    unsafe {
        let scaled = position.shift(upsample as i32);
        let mask = vdupq_n_s16(POSITION_MASK);
        Lanes {
            low: vshrq_n_s16::<1>(vandq_s16(scaled.low, mask)),
            high: vshrq_n_s16::<1>(vandq_s16(scaled.high, mask)),
        }
    }
}

/// Each lane's whole-sample index of `position` on a line of resolution
/// `1 << upsample` (an arithmetic shift: zone 2 positions are negative).
///
/// # Safety
///
/// NEON must be enabled in the caller.
#[inline(always)]
unsafe fn base(position: Lanes, upsample: u32) -> Lanes {
    // SAFETY: NEON is enabled (caller contract).
    unsafe { position.shift(upsample as i32 - i32::from(POSITION_BITS)) }
}

/// The `W` samples of picture row `row`.
#[inline(always)]
fn row_of<const W: usize>(data: &mut [u8], stride: usize, row: usize) -> &mut [u8] {
    &mut data[row * stride..][..W]
}

/// Write the 16 results to rows `row..` of the picture `data` (rows
/// `stride` apart), `W` samples per row.
///
/// # Safety
///
/// NEON must be enabled in the caller.
#[inline(always)]
unsafe fn store<const W: usize>(values: uint8x16_t, data: &mut [u8], stride: usize, row: usize) {
    // SAFETY: NEON is enabled (caller contract); each store covers the
    // checked subslice it writes.
    unsafe {
        match W {
            16 => vst1q_u8(row_of::<W>(data, stride, row).as_mut_ptr(), values),
            8 => {
                vst1_u8(
                    row_of::<W>(data, stride, row).as_mut_ptr(),
                    vget_low_u8(values),
                );
                vst1_u8(
                    row_of::<W>(data, stride, row + 1).as_mut_ptr(),
                    vget_high_u8(values),
                );
            }
            _ => {
                let quads = vreinterpretq_u32_u8(values);
                let quads = [
                    vgetq_lane_u32::<0>(quads),
                    vgetq_lane_u32::<1>(quads),
                    vgetq_lane_u32::<2>(quads),
                    vgetq_lane_u32::<3>(quads),
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
/// The index bounds are [`super::zones_avx2::zones`]'s: every index and its
/// successor stay inside the 32-sample tables, and every intermediate fits
/// 16 bits.
///
/// # Panics
///
/// If the block does not fit or a line is shorter than [`ZONE_TABLE`].
///
/// # Safety
///
/// The CPU must support NEON.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn zones(block: &ZoneBlock<'_, u8>, dst: &mut PlaneViewMut<'_>) {
    let (w, h) = (dst.width(), dst.height());
    assert!(block.fits((w, h)) && h * w >= LANES && h.is_multiple_of(LANES / w.max(1)));
    // SAFETY: NEON is enabled (this function's target feature).
    unsafe {
        match w {
            4 => zones_w::<4>(block, dst),
            8 => zones_w::<8>(block, dst),
            16 => zones_w::<16>(block, dst),
            _ => unreachable!("AV1 transform widths are powers of two from 4"),
        }
    }
}

/// [`zones`] for blocks `W` wide: `LANES / W` rows per step.
///
/// # Safety
///
/// NEON must be enabled in the caller.
#[inline(always)]
unsafe fn zones_w<const W: usize>(block: &ZoneBlock<'_, u8>, dst: &mut PlaneViewMut<'_>) {
    let rows = LANES / W;
    let h = dst.height();
    let stride = dst.stride();
    let data = dst.data_mut();
    // SAFETY: NEON is enabled (caller contract); every memory access is a
    // checked slice.
    unsafe {
        let (column, row) = lane_grid(W);
        let one = Lanes::splat(1);
        match block.zone {
            Zone::Above {
                dx,
                upsample,
                max_base,
            } => {
                let line = Table::load(block.above);
                // Lane position (row + 1) dx; index (position >> fb) + (column << up).
                let mut position = row.add(one).mul(Lanes::splat(dx));
                let step = Lanes::splat(dx * rows as i32);
                let across = column.shift(upsample as i32);
                let last = Lanes::splat(max_base);
                for r in (0..h).step_by(rows) {
                    let index = base(position, upsample).add(across).min(last);
                    let values = line.interpolate(index, weight(position, upsample));
                    store::<W>(values, data, stride, r);
                    position = position.add(step);
                }
            }
            Zone::Left {
                dy,
                upsample,
                max_base,
            } => {
                let line = Table::load(block.left);
                // Lane position (column + 1) dy; index (position >> fb) + (row << up).
                let position = column.add(one).mul(Lanes::splat(dy));
                let weights = weight(position, upsample);
                let mut index = base(position, upsample).add(row.shift(upsample as i32));
                let step = Lanes::splat((rows as i32) << upsample);
                let last = Lanes::splat(max_base);
                for r in (0..h).step_by(rows) {
                    let values = line.interpolate(index.min(last), weights);
                    store::<W>(values, data, stride, r);
                    index = index.add(step);
                }
            }
            Zone::Both {
                dx,
                dy,
                upsample_above,
                upsample_left,
            } => {
                let (above, left) = (Table::load(block.above), Table::load(block.left));
                let margin = Lanes::splat(ZONE2_MARGIN as i32);
                let lowest = Lanes::splat(-(ZONE2_MARGIN as i32));
                let min_base_above = Lanes::splat(-(1 << upsample_above));
                // Lane x = (column << 6) - (row + 1) dx and y = (row << 6) - (column + 1) dy.
                let mut x = column
                    .shift(i32::from(POSITION_BITS))
                    .sub(row.add(one).mul(Lanes::splat(dx)));
                let mut y = row
                    .shift(i32::from(POSITION_BITS))
                    .sub(column.add(one).mul(Lanes::splat(dy)));
                // y moves by whole samples from row to row: one weight per lane.
                let left_weights = weight(y, upsample_left);
                let (x_step, y_step) = (
                    Lanes::splat(dx * rows as i32),
                    Lanes::splat((rows as i32) << POSITION_BITS),
                );
                for r in (0..h).step_by(rows) {
                    let base_x = base(x, upsample_above);
                    let from_left =
                        min_base_above.zip(base_x, |a, b| vreinterpretq_s16_u16(vcgtq_s16(a, b)));
                    let above_index = base_x.max(lowest).add(margin);
                    let left_index = base(y, upsample_left).max(lowest).add(margin);
                    let values = vbslq_u8(
                        from_left.narrow(),
                        left.interpolate(left_index, left_weights),
                        above.interpolate(above_index, weight(x, upsample_above)),
                    );
                    store::<W>(values, data, stride, r);
                    x = x.sub(x_step);
                    y = y.add(y_step);
                }
            }
        }
    }
}
