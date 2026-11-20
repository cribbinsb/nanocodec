//! NEON eight-lane 16-bit forward butterflies: the lane type of the exact
//! AV1/VP9 forward DCTs of 16x16 and larger (`narrow_forward`, the shared
//! driver and its range proofs), the in-register 8x8 forward, and the
//! approximate HEVC rows and columns. Multiplications widen to 32 bits
//! before rounding back to 16.

#![allow(clippy::inline_always)]

#[cfg(test)]
use super::generic;
use super::narrow_forward::{self, ForwardWords};
use super::{cospi, dct, forward_lanes::ForwardLanes};
use crate::simd::neon::I32x4;
use core::arch::aarch64::{
    int16x8_t, int32x4_t, uint16x8_t, vabsq_s16, vaddq_s16, vaddq_s32, vcombine_s16, vdupq_n_s16,
    vdupq_n_s32, vdupq_n_u16, vget_high_s16, vget_low_s16, vld1q_s16, vld1q_s32, vmaxq_s32,
    vmaxvq_u16, vminq_s32, vmlal_n_s16, vmovl_s16, vmovn_s32, vmull_n_s16, vmulq_n_s32, vqaddq_u16,
    vqmovn_s32, vqrdmulhq_n_s16, vreinterpretq_s16_s64, vreinterpretq_s32_s16,
    vreinterpretq_s64_s32, vreinterpretq_u16_s16, vrshlq_s32, vrshrn_high_n_s32, vrshrn_n_s32,
    vrshrq_n_s16, vrshrq_n_s32, vshlq_n_s16, vst1q_s16, vst1q_s32, vsubq_s16, vtrn1q_s16,
    vtrn1q_s32, vtrn1q_s64, vtrn2q_s16, vtrn2q_s32, vtrn2q_s64,
};
use core::mem::MaybeUninit;

#[derive(Clone, Copy)]
struct Column(int16x8_t);

impl ForwardLanes for Column {
    #[inline(always)]
    fn splat(value: i32) -> Self {
        // SAFETY: used only from the NEON forward entry point.
        unsafe { Self(vdupq_n_s16(value as i16)) }
    }
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        // SAFETY: NEON is enabled; the 8-bit column stages fit i16.
        unsafe { Self(vaddq_s16(self.0, other.0)) }
    }
    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        // SAFETY: NEON is enabled; the 8-bit column stages fit i16.
        unsafe { Self(vsubq_s16(self.0, other.0)) }
    }
    #[inline(always)]
    fn half_btf(w0: i32, a: Self, w1: i32, b: Self, bit: u32) -> Self {
        // SAFETY: NEON is enabled. Q13 weights fit i16; widened product
        // sums fit i32, and the rounded column result fits i16.
        unsafe {
            // The equal-weight rotations can add/subtract first. For these
            // low-bit-depth columns that intermediate fits i16 (checked by
            // the scalar stage tests). Q15 doubling-high multiplication then
            // performs the exact original Q12/Q13 rounding in one instruction.
            if (w0 == w1 || w0 == -w1) && w0.abs() < (1 << bit) {
                let sum = if w0 == w1 {
                    vaddq_s16(a.0, b.0)
                } else {
                    vsubq_s16(a.0, b.0)
                };
                return Self(vqrdmulhq_n_s16(sum, (w0 << (15 - bit)) as i16));
            }
            let low = vmlal_n_s16(
                vmull_n_s16(vget_low_s16(a.0), w0 as i16),
                vget_low_s16(b.0),
                w1 as i16,
            );
            let high = vmlal_n_s16(
                vmull_n_s16(vget_high_s16(a.0), w0 as i16),
                vget_high_s16(b.0),
                w1 as i16,
            );
            Self(round_narrow(low, high, bit))
        }
    }
}

impl ForwardWords for Column {
    const LANES: usize = 8;
    type Norm = uint16x8_t;
    type Wide = I32x4;

    #[inline(always)]
    fn load(src: &[i16]) -> Self {
        let src = &src[..8];
        // SAFETY: NEON kernels only; one 128-bit load of eight values.
        unsafe { Self(vld1q_s16(src.as_ptr())) }
    }

    #[inline(always)]
    fn store(self, dst: &mut [MaybeUninit<i16>]) {
        let dst = &mut dst[..8];
        // SAFETY: NEON kernels only; one 128-bit store of eight values.
        unsafe { vst1q_s16(dst.as_mut_ptr().cast(), self.0) }
    }

    #[inline(always)]
    fn shift_left_2(self) -> Self {
        // SAFETY: NEON kernels only; register arithmetic.
        unsafe { Self(vshlq_n_s16::<2>(self.0)) }
    }

    #[inline(always)]
    fn round_shift<const SHIFT: u32>(self) -> Self {
        // The rounding shift computes its offset without overflow. The
        // forwards shift by 2 or 4; `SHIFT` is a constant, so the match
        // folds away.
        // SAFETY: NEON kernels only; register arithmetic.
        unsafe {
            match SHIFT {
                2 => Self(vrshrq_n_s16::<2>(self.0)),
                4 => Self(vrshrq_n_s16::<4>(self.0)),
                _ => unreachable!("forward shifts are 2 or 4"),
            }
        }
    }

    #[inline(always)]
    fn transpose(tile: &mut [Self]) {
        let tile: &mut [Self; 8] = tile.try_into().unwrap();
        // SAFETY: NEON kernels only; register shuffles.
        *tile = unsafe { transpose(tile.map(|v| v.0)) }.map(Self);
    }

    #[inline(always)]
    fn norm_zero() -> uint16x8_t {
        // SAFETY: NEON kernels only.
        unsafe { vdupq_n_u16(0) }
    }

    #[inline(always)]
    fn accumulate_l1(self, norm: uint16x8_t) -> uint16x8_t {
        // SAFETY: NEON kernels only; `|i16::MIN|` reads correctly as
        // unsigned, and the sum saturates.
        unsafe { vqaddq_u16(norm, vreinterpretq_u16_s16(vabsq_s16(self.0))) }
    }

    #[inline(always)]
    fn within_limit(norm: uint16x8_t, limit: u16) -> bool {
        // SAFETY: NEON kernels only.
        unsafe { vmaxvq_u16(norm) <= limit }
    }
}

/// libaom `round_shift(.., bit)` of two `i32` halves, narrowed to one
/// `i16` vector (the caller's range keeps the results in 16 bits). The
/// forward cosine bits 10, 12 and 13 use one immediate rounding-narrow
/// instruction per half; `bit` is a constant once the graphs are inlined,
/// so the match folds away.
#[inline(always)]
fn round_narrow(low: int32x4_t, high: int32x4_t, bit: u32) -> int16x8_t {
    // SAFETY: NEON kernels only; register arithmetic.
    unsafe {
        match bit {
            10 => vrshrn_high_n_s32::<10>(vrshrn_n_s32::<10>(low), high),
            12 => vrshrn_high_n_s32::<12>(vrshrn_n_s32::<12>(low), high),
            13 => vrshrn_high_n_s32::<13>(vrshrn_n_s32::<13>(low), high),
            _ => {
                let shift = vdupq_n_s32(-(bit as i32));
                vcombine_s16(
                    vmovn_s32(vrshlq_s32(low, shift)),
                    vmovn_s32(vrshlq_s32(high, shift)),
                )
            }
        }
    }
}

/// Transpose one 8x8 i16 tile in registers.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn transpose(rows: [int16x8_t; 8]) -> [int16x8_t; 8] {
    let a = [
        vtrn1q_s16(rows[0], rows[1]),
        vtrn2q_s16(rows[0], rows[1]),
        vtrn1q_s16(rows[2], rows[3]),
        vtrn2q_s16(rows[2], rows[3]),
        vtrn1q_s16(rows[4], rows[5]),
        vtrn2q_s16(rows[4], rows[5]),
        vtrn1q_s16(rows[6], rows[7]),
        vtrn2q_s16(rows[6], rows[7]),
    ]
    .map(|row| vreinterpretq_s32_s16(row));
    let b = [
        vtrn1q_s32(a[0], a[2]),
        vtrn1q_s32(a[1], a[3]),
        vtrn2q_s32(a[0], a[2]),
        vtrn2q_s32(a[1], a[3]),
        vtrn1q_s32(a[4], a[6]),
        vtrn1q_s32(a[5], a[7]),
        vtrn2q_s32(a[4], a[6]),
        vtrn2q_s32(a[5], a[7]),
    ]
    .map(|row| vreinterpretq_s64_s32(row));
    [
        vtrn1q_s64(b[0], b[4]),
        vtrn1q_s64(b[1], b[5]),
        vtrn1q_s64(b[2], b[6]),
        vtrn1q_s64(b[3], b[7]),
        vtrn2q_s64(b[0], b[4]),
        vtrn2q_s64(b[1], b[5]),
        vtrn2q_s64(b[2], b[6]),
        vtrn2q_s64(b[3], b[7]),
    ]
    .map(|row| vreinterpretq_s16_s64(row))
}

/// Approximate HEVC row transforms in eight parallel 16-bit lanes.
/// # Safety
/// NEON is enabled; N is16/32, slices cover the block, residuals are8-bit differences.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn hevc_rows<const N: usize>(
    residual: &[i16],
    stride: usize,
    out: &mut [MaybeUninit<i32>],
) {
    assert!(matches!(N, 16 | 32));
    let mut input = [Column::splat(0); N];
    let mut output = input;
    for r0 in (0..N).step_by(8) {
        for c0 in (0..N).step_by(8) {
            let rows = core::array::from_fn(|i| {
                let row = &residual[(r0 + i) * stride + c0..][..8];
                // SAFETY: checked eight-sample slice.
                unsafe { vld1q_s16(row.as_ptr()) }
            });
            // SAFETY: NEON is enabled; transpose only uses registers.
            let tile = unsafe { transpose(rows) };
            for (i, row) in tile.into_iter().enumerate() {
                input[c0 + i] = Column(vshlq_n_s16::<2>(row));
            }
        }
        let mut dc = Column::splat(0);
        for value in &input {
            dc = dc.add(*value);
        }
        match N {
            16 => dct::fdct16::<Column>(
                input.as_slice().try_into().unwrap(),
                output.as_mut_slice().try_into().unwrap(),
                cospi(12),
                12,
            ),
            32 => dct::fdct32::<Column>(
                input.as_slice().try_into().unwrap(),
                output.as_mut_slice().try_into().unwrap(),
                cospi(12),
                12,
            ),
            _ => unreachable!(),
        }
        for k in (0..N).step_by(8) {
            // SAFETY: complete register tile and enabled NEON.
            let tile = unsafe { transpose(core::array::from_fn(|i| output[k + i].0)) };
            for (i, row) in tile.into_iter().enumerate() {
                let low = vmulq_n_s32(vmovl_s16(vget_low_s16(row)), 181);
                let high = vmulq_n_s32(vmovl_s16(vget_high_s16(row)), 181);
                let (low, high) = if N == 16 {
                    (vrshrq_n_s32::<6>(low), vrshrq_n_s32::<6>(high))
                } else {
                    (vrshrq_n_s32::<7>(low), vrshrq_n_s32::<7>(high))
                };
                let dest = &mut out[(r0 + i) * N + k..][..8];
                // SAFETY: two stores initialize all eight output coefficients.
                unsafe {
                    let ptr = dest.as_mut_ptr().cast::<i32>();
                    vst1q_s32(ptr, low);
                    vst1q_s32(ptr.add(4), high);
                }
            }
        }
        let mut dc_lanes = [0_i16; 8];
        // SAFETY: eight valid slots; lane sums fit i16 for8-bit inputs scaled4.
        unsafe { vst1q_s16(dc_lanes.as_mut_ptr(), dc.0) };
        for (i, value) in dc_lanes.into_iter().enumerate() {
            out[(r0 + i) * N].write(i32::from(value) << (5 - N.ilog2()));
        }
    }
}

/// Approximate HEVC column pass in eight 16-bit lanes (eight columns per
/// vector), with the results of the generic staged columns
/// (`codec_txfm::generic::staged_hevc_pass`): inputs clamped to
/// `STAGED_COLUMN_LIMIT` and scaled by the block's `2^input_scale`
/// (`generic::column_input_scale`, rounded) stay within the staged rows'
/// range, so every stage fits i16; outputs are scaled by 181, rounded by
/// `pass_shift + 1 + input_scale` and clamped to 16 bits; the DC row is
/// exact.
/// # Safety
/// NEON is enabled; N is 16/32; `m` and `out` hold N * N values.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn hevc_columns<const N: usize>(m: &[i32], pass_shift: u32, out: &mut [i16]) {
    use crate::codec_txfm::generic::{STAGED_COLUMN_LIMIT, column_input_scale};
    assert!(matches!(N, 16 | 32));
    let (m, out) = (&m[..N * N], &mut out[..N * N]);
    let input_scale = column_input_scale(m);
    let input_shift = vdupq_n_s32(input_scale);
    let (low_limit, high_limit) = (
        vdupq_n_s32(-STAGED_COLUMN_LIMIT),
        vdupq_n_s32(STAGED_COLUMN_LIMIT),
    );
    let output_shift = vdupq_n_s32(-(pass_shift as i32 + 1 + input_scale));
    let dc_shift = vdupq_n_s32(-(pass_shift as i32));
    let mut output = [Column::splat(0); N];
    for c0 in (0..N).step_by(8) {
        let mut dc = [vdupq_n_s32(0); 2];
        let input: [Column; N] = core::array::from_fn(|r| {
            let row = &m[r * N + c0..][..8];
            // SAFETY: `row` holds eight values; NEON is enabled.
            unsafe {
                let halves = [vld1q_s32(row.as_ptr()), vld1q_s32(row.as_ptr().add(4))];
                dc[0] = vaddq_s32(dc[0], halves[0]);
                dc[1] = vaddq_s32(dc[1], halves[1]);
                let clamped = halves.map(|h| vminq_s32(vmaxq_s32(h, low_limit), high_limit));
                // A rounding shift by `input_scale` (left when positive);
                // the result fits 16 bits.
                let scaled = clamped.map(|h| vrshlq_s32(h, input_shift));
                Column(vcombine_s16(vmovn_s32(scaled[0]), vmovn_s32(scaled[1])))
            }
        });
        match N {
            16 => dct::fdct16::<Column>(
                input.as_slice().try_into().unwrap(),
                output.as_mut_slice().try_into().unwrap(),
                cospi(12),
                12,
            ),
            _ => dct::fdct32::<Column>(
                input.as_slice().try_into().unwrap(),
                output.as_mut_slice().try_into().unwrap(),
                cospi(12),
                12,
            ),
        }
        for (k, coefficient) in output.iter().enumerate() {
            let dest = &mut out[k * N + c0..][..8];
            let halves = if k == 0 {
                // Exact DC: the mean of each column, as the exact pass.
                dc.map(|d| vrshlq_s32(vmulq_n_s32(d, 64), dc_shift))
            } else {
                [
                    vrshlq_s32(
                        vmulq_n_s32(vmovl_s16(vget_low_s16(coefficient.0)), 181),
                        output_shift,
                    ),
                    vrshlq_s32(
                        vmulq_n_s32(vmovl_s16(vget_high_s16(coefficient.0)), 181),
                        output_shift,
                    ),
                ]
            };
            // SAFETY: one store fills `dest`; the saturating narrow clamps
            // to 16 bits, as the exact pass does.
            unsafe {
                vst1q_s16(
                    dest.as_mut_ptr(),
                    vcombine_s16(vqmovn_s32(halves[0]), vqmovn_s32(halves[1])),
                );
            }
        }
    }
}

/// Computing only retained outputs saves 1.05% native encode CPU on the
/// 240-frame studio static case (three paired runs, identical stream/recon).
/// The full scalar graph remains the independent coefficient oracle.
#[cfg(test)]
#[target_feature(enable = "neon")]
unsafe fn forward_64_wide_reference(residual: &[i16], stride: usize, out: &mut [i16]) {
    const N: usize = 64;
    const CODED: usize = 32;
    let mut buf = [MaybeUninit::<i32>::uninit(); N * CODED];
    let mut input = [Column::splat(0); N];
    let mut output = input;
    for x in (0..N).step_by(8) {
        for (r, column) in input.iter_mut().enumerate() {
            let row = &residual[r * stride + x..][..8];
            // SAFETY: the checked slice contains eight residual samples.
            *column = Column(unsafe { vld1q_s16(row.as_ptr()) });
        }
        dct::fdct64_low32(&input, &mut output, cospi(13), 13);
        for k in (0..CODED).step_by(8) {
            let tile = core::array::from_fn(|i| vrshrq_n_s16::<2>(output[k + i].0));
            // SAFETY: NEON is enabled and the tile contains only registers.
            let tile = unsafe { transpose(tile) };
            for (i, row) in tile.into_iter().enumerate() {
                let dest = &mut buf[(x + i) * CODED + k..][..8];
                // SAFETY: dest has eight i32 slots; all are initialized by
                // these two widening stores. Every tile is written once.
                unsafe {
                    let ptr = dest.as_mut_ptr().cast::<i32>();
                    vst1q_s32(ptr, vmovl_s16(vget_low_s16(row)));
                    vst1q_s32(ptr.add(4), vmovl_s16(vget_high_s16(row)));
                }
            }
        }
    }
    // SAFETY: x covers all64 columns and k covers their32 kept outputs;
    // the tiled stores initialize every i32 in the inter-pass buffer.
    let transposed = unsafe { core::slice::from_raw_parts(buf.as_ptr().cast::<i32>(), N * CODED) };
    generic::forward_64_rows_low32::<I32x4>(transposed, out);
}

/// Low-bit-depth32-point columns: shift2, Q12 cosines, mid-shift4.
#[cfg(test)]
#[target_feature(enable = "neon")]
unsafe fn forward_32_wide_reference(residual: &[i16], stride: usize, out: &mut [i16]) {
    const N: usize = 32;
    let mut buf = [MaybeUninit::<i32>::uninit(); N * N];
    let mut input = [Column::splat(0); N];
    let mut output = input;
    for x in (0..N).step_by(8) {
        for (r, column) in input.iter_mut().enumerate() {
            let row = &residual[r * stride + x..][..8];
            // SAFETY: eight valid8-bit residuals, shifted values fit i16.
            *column = Column(unsafe { vshlq_n_s16::<2>(vld1q_s16(row.as_ptr())) });
        }
        dct::fdct32(&input, &mut output, cospi(12), 12);
        for k in (0..N).step_by(8) {
            let tile = core::array::from_fn(|i| vrshrq_n_s16::<4>(output[k + i].0));
            // SAFETY: NEON is enabled and tile contains only registers.
            let tile = unsafe { transpose(tile) };
            for (i, row) in tile.into_iter().enumerate() {
                let dest = &mut buf[(x + i) * N + k..][..8];
                // SAFETY: both widening stores initialize eight i32 slots.
                unsafe {
                    let ptr = dest.as_mut_ptr().cast::<i32>();
                    vst1q_s32(ptr, vmovl_s16(vget_low_s16(row)));
                    vst1q_s32(ptr.add(4), vmovl_s16(vget_high_s16(row)));
                }
            }
        }
    }
    // SAFETY: tiled stores cover all32x32 inter-pass slots exactly once.
    let transposed = unsafe { core::slice::from_raw_parts(buf.as_ptr().cast::<i32>(), N * N) };
    generic::forward_32_rows::<I32x4>(transposed, out);
}

/// Exact eight-lane AV1 16x16 forward: 16-bit columns need no guard for
/// 8-bit residuals, and the rows use the L1 guard (both proved in
/// `range_probe`).
#[target_feature(enable = "neon")]
pub(crate) unsafe fn forward_16(residual: &[i16], stride: usize, out: &mut [i16]) {
    // SAFETY: dispatcher guarantees NEON and complete8-bit residual windows.
    unsafe {
        narrow_forward::forward::<Column, 16, false, 16>(residual, stride, out);
    }
}

/// [`forward_16`] with raster coefficients (output tiles transposed in
/// registers).
#[target_feature(enable = "neon")]
pub(crate) unsafe fn forward_16_raster(residual: &[i16], stride: usize, out: &mut [i16]) {
    // SAFETY: as for `forward_16`.
    unsafe {
        narrow_forward::forward::<Column, 16, true, 16>(residual, stride, out);
    }
}

/// Eight-point DCT entirely in registers, including the inter-pass transpose.
/// The Q40 proof in `range_probe` bounds both passes including equal-weight
/// pre-adds: column inputs are at most 255 << 2, and shifted column outputs
/// fit the row graph's amplitude envelope. No runtime row guard is needed.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn forward_8(residual: &[i16], stride: usize, out: &mut [i16]) {
    // SAFETY: forwarded contract.
    unsafe { forward_8_layout::<false>(residual, stride, out) }
}

/// [`forward_8`] with raster (row-major) coefficients, the layout VP9 uses:
/// the output tile is transposed in registers before the stores.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn forward_8_raster(residual: &[i16], stride: usize, out: &mut [i16]) {
    // SAFETY: forwarded contract.
    unsafe { forward_8_layout::<true>(residual, stride, out) }
}

#[inline(always)]
unsafe fn forward_8_layout<const RASTER: bool>(residual: &[i16], stride: usize, out: &mut [i16]) {
    const SIDE: usize = 8;
    const INPUT_SHIFT: i32 = 2;
    const MID_SHIFT: i32 = 1;
    const COS_BIT: u32 = 13;
    let mut input = [Column::splat(0); SIDE];
    let mut output = input;
    // SAFETY: the dispatcher guarantees NEON and complete 8-bit residual
    // rows. Each checked slice covers one vector load or two half stores.
    unsafe {
        for (row, lane) in input.iter_mut().enumerate() {
            let src = &residual[row * stride..][..SIDE];
            *lane = Column(vshlq_n_s16::<INPUT_SHIFT>(vld1q_s16(src.as_ptr())));
        }
        dct::fdct8(&input, &mut output, cospi(COS_BIT), COS_BIT);
        let rows = transpose(output.map(|v| vrshrq_n_s16::<MID_SHIFT>(v.0)));
        input = rows.map(Column);
        dct::fdct8(&input, &mut output, cospi(COS_BIT), COS_BIT);
        // Lane `v` of `output[h]` is coefficient (v, h).
        let vectors = if RASTER {
            transpose(output.map(|v| v.0))
        } else {
            output.map(|v| v.0)
        };
        for (line, lane) in vectors.iter().enumerate() {
            let dst = &mut out[line * SIDE..][..SIDE];
            vst1q_s16(dst.as_mut_ptr(), *lane);
        }
    }
}

/// Exact guarded eight-lane AV1 forward rows; high-range batches retain
/// the wide graph. Guard accumulation shares the required row-input loads.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn forward_32(residual: &[i16], stride: usize, out: &mut [i16]) {
    // SAFETY: dispatcher guarantees NEON and complete8-bit residual windows.
    unsafe {
        narrow_forward::forward::<Column, 32, false, 32>(residual, stride, out);
    }
}

/// [`forward_32`] with raster coefficients (output tiles transposed in
/// registers).
#[target_feature(enable = "neon")]
pub(crate) unsafe fn forward_32_raster(residual: &[i16], stride: usize, out: &mut [i16]) {
    // SAFETY: as for `forward_32`.
    unsafe {
        narrow_forward::forward::<Column, 32, true, 32>(residual, stride, out);
    }
}

/// Exact guarded eight-lane AV1 forward rows; high-range batches retain
/// the wide graph. Guard accumulation shares the required row-input loads.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn forward_64(residual: &[i16], stride: usize, out: &mut [i16]) {
    // SAFETY: dispatcher guarantees NEON and complete8-bit residual windows.
    unsafe {
        narrow_forward::forward::<Column, 64, false, 32>(residual, stride, out);
    }
}

/// The low-frequency forward of an `N`-point block (`txfm::low_keep`):
/// the column pass keeps its `KEEP` lowest outputs, the row pass runs only
/// on those and keeps its `KEEP` lowest; the rest of the coded square is
/// zero. Equal to the full forward with the other coefficients zeroed
/// (`narrow_forward::forward`).
#[target_feature(enable = "neon")]
unsafe fn forward_low<const N: usize, const KEEP: usize, const RASTER: bool>(
    residual: &[i16],
    stride: usize,
    out: &mut [i16],
) {
    // SAFETY: dispatcher guarantees NEON and complete 8-bit residual windows.
    unsafe {
        narrow_forward::forward::<Column, N, RASTER, KEEP>(residual, stride, out);
    }
}

/// [`forward_low`] of a 16x16 block.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn forward_16_low(residual: &[i16], stride: usize, out: &mut [i16]) {
    // SAFETY: forwarded contract.
    unsafe { forward_low::<16, { super::low_keep(16) }, false>(residual, stride, out) }
}

/// [`forward_low`] of a 32x32 block.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn forward_32_low(residual: &[i16], stride: usize, out: &mut [i16]) {
    // SAFETY: forwarded contract.
    unsafe { forward_low::<32, { super::low_keep(32) }, false>(residual, stride, out) }
}

/// [`forward_low`] of a 16x16 block, raster coefficients.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn forward_16_low_raster(residual: &[i16], stride: usize, out: &mut [i16]) {
    // SAFETY: forwarded contract.
    unsafe { forward_low::<16, { super::low_keep(16) }, true>(residual, stride, out) }
}

/// [`forward_low`] of a 32x32 block, raster coefficients.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn forward_32_low_raster(residual: &[i16], stride: usize, out: &mut [i16]) {
    // SAFETY: forwarded contract.
    unsafe { forward_low::<32, { super::low_keep(32) }, true>(residual, stride, out) }
}

/// [`forward_low`] of a 64x64 block (in its coded 32x32 square).
#[target_feature(enable = "neon")]
pub(crate) unsafe fn forward_64_low(residual: &[i16], stride: usize, out: &mut [i16]) {
    // SAFETY: forwarded contract.
    unsafe { forward_low::<64, { super::low_keep(64) }, false>(residual, stride, out) }
}

/// The bounded raster 16-point forward (`narrow_forward::bounded`).
#[target_feature(enable = "neon")]
pub(crate) unsafe fn forward_16_bounded(
    residual: &[i16],
    stride: usize,
    zero_limit: u32,
    out: &mut [i16],
) -> bool {
    // SAFETY: forwarded contract.
    unsafe {
        narrow_forward::bounded::<Column, 16>(
            residual,
            stride,
            zero_limit,
            super::FORWARD_ROW_ROUNDING[0],
            out,
        )
    }
}

/// The bounded raster 32-point forward (`narrow_forward::bounded`).
#[target_feature(enable = "neon")]
pub(crate) unsafe fn forward_32_bounded(
    residual: &[i16],
    stride: usize,
    zero_limit: u32,
    out: &mut [i16],
) -> bool {
    // SAFETY: forwarded contract.
    unsafe {
        narrow_forward::bounded::<Column, 32>(
            residual,
            stride,
            zero_limit,
            super::FORWARD_ROW_ROUNDING[1],
            out,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guarded_narrow_rows_match_full_scalar_and_cover_fallback() {
        for n in [16, 32, 64] {
            for extra in [0, 5] {
                let stride = n + extra;
                for pattern in 0..7 {
                    let mut residual = vec![0i16; (n - 1) * stride + n + 1];
                    for r in 0..n {
                        for c in 0..n {
                            residual[1 + r * stride + c] = match pattern {
                                0 => 255,
                                1 => -255,
                                2 => {
                                    if (r + c) % 2 == 0 {
                                        255
                                    } else {
                                        -255
                                    }
                                }
                                3 => ((r * 73 + c * 101) % 511) as i16 - 255,
                                4 => ((r * 7 + c * 3) % 5) as i16 - 2,
                                5 => {
                                    if r < n / 2 {
                                        255
                                    } else {
                                        -255
                                    }
                                }
                                _ => {
                                    if c < n / 2 {
                                        255
                                    } else {
                                        -255
                                    }
                                }
                            };
                        }
                    }
                    let mut expected = [0; 1024];
                    let mut actual = [0; 1024];
                    // SAFETY: NEON is always present, windows are complete and residuals8-bit.
                    let batches = unsafe {
                        match n {
                            16 => {
                                generic::forward_16::<i32>(&residual[1..], stride, &mut expected);
                                narrow_forward::forward::<Column, 16, false, 16>(
                                    &residual[1..],
                                    stride,
                                    &mut actual,
                                )
                            }
                            32 => {
                                generic::forward_32::<i32>(&residual[1..], stride, &mut expected);
                                narrow_forward::forward::<Column, 32, false, 32>(
                                    &residual[1..],
                                    stride,
                                    &mut actual,
                                )
                            }
                            _ => {
                                generic::forward_64::<i32>(&residual[1..], stride, &mut expected);
                                narrow_forward::forward::<Column, 64, false, 32>(
                                    &residual[1..],
                                    stride,
                                    &mut actual,
                                )
                            }
                        }
                    };
                    assert_eq!(actual, expected, "N{n} extra{extra} pattern{pattern}");
                    let dsp = crate::Dsp::with_isa(crate::Isa::Neon).unwrap();
                    let mut dispatched = [0; 1024];
                    match crate::DctSize::from_side(n) {
                        Some(size) => {
                            dsp.av1_forward_dct(size, &residual[1..], stride, &mut dispatched);
                        }
                        None => dsp.av1_forward_dct_64(&residual[1..], stride, &mut dispatched),
                    }
                    assert_eq!(
                        dispatched, actual,
                        "production dispatch equals measured91 narrow-helper oracle"
                    );
                    let all_batches = narrow_forward::kept(n) as u32 / 8;
                    if pattern == 0 {
                        assert!(
                            batches < all_batches,
                            "dense DC must exercise wide fallback"
                        );
                    }
                    if pattern == 4 {
                        assert_eq!(
                            batches, all_batches,
                            "low energy exercises every narrow row batch"
                        );
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "bounded native kernel timing"]
    fn guarded_narrow_rows_native_screen() {
        for n in [32, 64] {
            for amplitude in [4, 32, 255, 0] {
                let residual: Vec<i16> = (0..n * n)
                    .map(|i| {
                        if amplitude == 0 {
                            255
                        } else {
                            ((i * 73 + i / 17) % (2 * amplitude + 1)) as i16 - amplitude as i16
                        }
                    })
                    .collect();
                let mut out = [0; 1024];
                for round in 0..3 {
                    for narrow in if round % 2 == 0 {
                        [false, true]
                    } else {
                        [true, false]
                    } {
                        let start = std::time::Instant::now();
                        let mut hits = 0;
                        for _ in 0..2000 {
                            // SAFETY: NEON present and complete8-bit residual/output windows.
                            unsafe {
                                let input = std::hint::black_box(&residual);
                                if narrow {
                                    hits += if n == 32 {
                                        narrow_forward::forward::<Column, 32, false, 32>(
                                            input, n, &mut out,
                                        )
                                    } else {
                                        narrow_forward::forward::<Column, 64, false, 32>(
                                            input, n, &mut out,
                                        )
                                    };
                                } else if n == 32 {
                                    forward_32_wide_reference(input, n, &mut out);
                                } else {
                                    forward_64_wide_reference(input, n, &mut out);
                                }
                            }
                            std::hint::black_box(&out);
                        }
                        println!(
                            "NARROW_ROW N{n} amplitude{amplitude} round{round} enabled{narrow} ns{} batches{hits}",
                            start.elapsed().as_nanos()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn lowbd_columns_match_scalar_on_extremes_and_unaligned_strides() {
        let stride = 69;
        let mut residual = vec![0_i16; stride * 64 + 1];
        let mut state = 0x84f7_u32;
        for case in 0..24 {
            for (i, sample) in residual.iter_mut().enumerate() {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                *sample = match case {
                    0 => 255,
                    1 => -255,
                    2 => {
                        if i % 2 == 0 {
                            255
                        } else {
                            -255
                        }
                    }
                    3 => {
                        if i / stride % 2 == 0 {
                            255
                        } else {
                            -255
                        }
                    }
                    _ => (state % 511) as i16 - 255,
                };
            }
            let mut actual = [0x1234; 1026];
            let mut expected = actual;
            // SAFETY: AArch64 has NEON, residuals are8-bit differences,
            // and both buffers include the full64x64 input/32x32 output.
            unsafe {
                forward_64(&residual[1..], stride, &mut actual[1..1025]);
                generic::forward_64::<i32>(&residual[1..], stride, &mut expected[1..1025]);
            }
            assert_eq!(actual, expected, "case {case}");
        }
    }
    #[test]
    fn lowbd_32_columns_match_scalar_on_extremes_and_unaligned_strides() {
        let stride = 37;
        let mut residual = vec![0_i16; stride * 32 + 1];
        let mut state = 0x84f7_u32;
        for case in 0..24 {
            for (i, sample) in residual.iter_mut().enumerate() {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                *sample = match case {
                    0 => 255,
                    1 => -255,
                    2 => {
                        if i % 2 == 0 {
                            255
                        } else {
                            -255
                        }
                    }
                    3 => {
                        if i / stride % 2 == 0 {
                            255
                        } else {
                            -255
                        }
                    }
                    _ => (state % 511) as i16 - 255,
                };
            }
            let mut actual = [0x1234; 1026];
            let mut expected = actual;
            // SAFETY: AArch64 has NEON, residuals are8-bit differences,
            // and both buffers include the full32x32 input/output.
            unsafe {
                forward_32(&residual[1..], stride, &mut actual[1..1025]);
                generic::forward_32::<i32>(&residual[1..], stride, &mut expected[1..1025]);
            }
            assert_eq!(actual, expected, "case {case}");
        }
    }
}
