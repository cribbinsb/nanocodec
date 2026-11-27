//! NEON H.264 luma prediction, including eight-wide I8MM blocks.
//!
//! The generic kernel (`generic::h264_luma`) widens every tap of the six-tap
//! filter to 16 bits separately: six widening loads and seven 16-bit
//! operations per eight outputs. Here a row of 16 outputs takes six
//! overlapping byte loads; the symmetric tap pairs are summed while widening
//! (`vaddl_u8`), so the filter is three widening adds and two multiply-adds
//! per eight outputs. Rounding and clipping use NEON's saturating rounding
//! narrows (`vqrshrun`), which compute `Clip1((x + 2^(n-1)) >> n)` exactly,
//! and the quarter-sample average is `vrhadd` (`(a + b + 1) >> 1`). The
//! arithmetic is that of H.264 §8.4.2.2.1 (eqs. 8-241..8-261), so output is
//! bit-identical to the scalar reference (checked by nc-checkasm).
//!
//! Window contract as for every H.264 kernel: `h + 5` rows of `w + 5`
//! samples from two up and two left of the block. No load leaves it: the
//! last of the six overlapping loads of a 16-output chunk ends at column
//! `c0 + 21 <= w + 5`.

use core::arch::aarch64::{
    int16x8_t, uint8x8_t, uint8x16_t, vaddl_u8, vaddq_s16, vcombine_u8, vcombine_u16, vdupq_n_s32,
    vdupq_n_u32, vext_u8, vget_high_s16, vget_high_u8, vget_low_s16, vget_low_u8, vgetq_lane_u32,
    vld1_u8, vld1q_s8, vld1q_u8, vmlal_n_s16, vmlaq_n_u16, vmlsl_n_s16, vmlsq_n_u16, vmovl_s16,
    vqmovn_u16, vqrshrun_n_s16, vqrshrun_n_s32, vreinterpretq_s16_u16, vreinterpretq_u8_u32,
    vreinterpretq_u32_u8, vrhaddq_u8, vst1_u8, vst1q_u8,
};

use super::horizontal_i8mm::{SAMPLES, matrix_sum};

/// Output columns per step: one 16-byte vector.
const CHUNK: usize = 16;

/// The unrounded six-tap sums (eq. 8-241) of 16 outputs from six vectors
/// holding taps 0..6 of each output, as low and high halves. Every
/// intermediate is exact modulo 2^16 and the result fits 16 bits
/// (|sum| < 2^14 for 8-bit samples). (No closures in these helpers: they
/// would not inherit the kernel's target features.)
#[target_feature(enable = "neon")]
#[inline]
fn six_tap(x: [uint8x16_t; 6]) -> (int16x8_t, int16x8_t) {
    let lo = taps_sum(
        [vget_low_u8(x[0]), vget_low_u8(x[1]), vget_low_u8(x[2])],
        [vget_low_u8(x[3]), vget_low_u8(x[4]), vget_low_u8(x[5])],
    );
    let hi = taps_sum(
        [vget_high_u8(x[0]), vget_high_u8(x[1]), vget_high_u8(x[2])],
        [vget_high_u8(x[3]), vget_high_u8(x[4]), vget_high_u8(x[5])],
    );
    (lo, hi)
}

/// `(x0 + x5) - 5·(x1 + x4) + 20·(x2 + x3)` of eight outputs, the symmetric
/// form of the six-tap filter, with each pair summed while widening.
#[target_feature(enable = "neon")]
#[inline]
fn taps_sum([x0, x1, x2]: [uint8x8_t; 3], [x3, x4, x5]: [uint8x8_t; 3]) -> int16x8_t {
    let outer = vaddl_u8(x0, x5);
    let inner = vaddl_u8(x1, x4);
    let centre = vaddl_u8(x2, x3);
    vreinterpretq_s16_u16(vmlsq_n_u16(vmlaq_n_u16(outer, centre, 20), inner, 5))
}

/// `Clip1((sum + 16) >> 5)` of 16 sums: half samples.
#[target_feature(enable = "neon")]
#[inline]
fn half((lo, hi): (int16x8_t, int16x8_t)) -> uint8x16_t {
    vcombine_u8(vqrshrun_n_s16::<5>(lo), vqrshrun_n_s16::<5>(hi))
}

/// Centre samples `j` (eq. 8-245) from six rows of unrounded horizontal
/// sums: the vertical six-tap in 32 bits, then `Clip1((x + 512) >> 10)`.
#[target_feature(enable = "neon")]
#[inline]
fn centre(sums: &[(int16x8_t, int16x8_t); 6]) -> uint8x16_t {
    let lo = centre_half([
        sums[0].0, sums[1].0, sums[2].0, sums[3].0, sums[4].0, sums[5].0,
    ]);
    let hi = centre_half([
        sums[0].1, sums[1].1, sums[2].1, sums[3].1, sums[4].1, sums[5].1,
    ]);
    vcombine_u8(lo, hi)
}

/// [`centre`] of eight outputs. The pair sums stay inside 16 bits (each
/// horizontal sum is in [-2550, 10710]), as in the generic kernel; the
/// weighted total needs 32.
#[target_feature(enable = "neon")]
#[inline]
fn centre_half(s: [int16x8_t; 6]) -> uint8x8_t {
    let outer = vaddq_s16(s[0], s[5]);
    let inner = vaddq_s16(s[1], s[4]);
    let middle = vaddq_s16(s[2], s[3]);
    let lo = vmlsl_n_s16(
        vmlal_n_s16(vmovl_s16(vget_low_s16(outer)), vget_low_s16(middle), 20),
        vget_low_s16(inner),
        5,
    );
    let hi = vmlsl_n_s16(
        vmlal_n_s16(vmovl_s16(vget_high_s16(outer)), vget_high_s16(middle), 20),
        vget_high_s16(inner),
        5,
    );
    vqmovn_u16(vcombine_u16(
        vqrshrun_n_s32::<10>(lo),
        vqrshrun_n_s32::<10>(hi),
    ))
}

/// Six overlapping loads at `p`: taps 0..6 of eight or 16 outputs.
///
/// # Safety
///
/// `p` must be valid for reading 13 bytes at W=8, otherwise 21 bytes.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn taps_at<const W: usize>(p: *const u8) -> [uint8x16_t; 6] {
    // SAFETY: the caller guarantees the complete overlapping loads.
    unsafe {
        [
            load_luma::<W>(p),
            load_luma::<W>(p.add(1)),
            load_luma::<W>(p.add(2)),
            load_luma::<W>(p.add(3)),
            load_luma::<W>(p.add(4)),
            load_luma::<W>(p.add(5)),
        ]
    }
}

/// The second load starts five bytes in, so all 21 bytes of the exact
/// window are readable. Its output starts three positions further in:
/// an out-of-range table index selects zero only for a zero-weight lane.
const RIGHT_SAMPLES: [[u8; 16]; 2] = {
    let mut masks = SAMPLES;
    let mut lane = 0;
    while lane < 16 {
        masks[0][lane] += 3;
        masks[1][lane] += 3;
        lane += 1;
    }
    masks
};

const MATRIX_TAPS: [i8; 16] = {
    let mut columns = [0; 16];
    let mut tap = 0;
    while tap < super::H264_TAPS.len() {
        columns[tap] = super::H264_TAPS[tap] as i8;
        columns[9 + tap] = columns[tap];
        tap += 1;
    }
    columns
};

/// Horizontal unrounded sums within the exact input window.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn horizontal_sums<const W: usize, const I8MM: bool>(
    row: *const u8,
) -> (int16x8_t, int16x8_t) {
    // SAFETY: the caller checked the input window and, when selected, I8MM.
    unsafe {
        if I8MM {
            let taps = vld1q_s8(MATRIX_TAPS.as_ptr());
            let bias = vdupq_n_s32(0);
            let left_pixels = if W == 8 {
                // Read exactly 13 source bytes. Table indices past byte 12
                // feed padding coefficients, which are zero.
                let tail = vld1_u8(row.add(5));
                vcombine_u8(vld1_u8(row), vext_u8::<3>(tail, tail))
            } else {
                vld1q_u8(row)
            };
            let left = matrix_sum(
                left_pixels,
                [vld1q_u8(SAMPLES[0].as_ptr()), vld1q_u8(SAMPLES[1].as_ptr())],
                taps,
                bias,
            );
            if W == 8 {
                (left, left)
            } else {
                (
                    left,
                    matrix_sum(
                        vld1q_u8(row.add(5)),
                        [
                            vld1q_u8(RIGHT_SAMPLES[0].as_ptr()),
                            vld1q_u8(RIGHT_SAMPLES[1].as_ptr()),
                        ],
                        taps,
                        bias,
                    ),
                )
            }
        } else {
            six_tap(taps_at::<W>(row))
        }
    }
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn load_luma<const W: usize>(src: *const u8) -> uint8x16_t {
    // SAFETY: the preceding window check covers four, eight or sixteen bytes.
    unsafe {
        if W == 4 {
            vreinterpretq_u8_u32(vdupq_n_u32(src.cast::<u32>().read_unaligned()))
        } else if W == 8 {
            let bytes = vld1_u8(src);
            vcombine_u8(bytes, bytes)
        } else {
            vld1q_u8(src)
        }
    }
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn store_luma<const W: usize>(dst: *mut u8, value: uint8x16_t) {
    // SAFETY: the preceding window check covers four, eight or sixteen bytes.
    unsafe {
        if W == 4 {
            let word = vgetq_lane_u32::<0>(vreinterpretq_u32_u8(value));
            dst.cast::<u32>().write_unaligned(word);
        } else if W == 8 {
            vst1_u8(dst, vget_low_u8(value));
        } else {
            vst1q_u8(dst, value);
        }
    }
}

/// Select the I8MM row on calls with horizontal filtering.
///
/// # Safety
///
/// I8MM must be available and the slices must satisfy the kernel contract.
pub(crate) unsafe fn h264_luma_i8mm<const W: usize>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    h: usize,
    frac_x: usize,
    frac_y: usize,
) {
    // SAFETY: the dispatch table checked I8MM. Both entries check the windows.
    unsafe {
        if frac_x == 0 && W == 8 {
            super::neon::h264_luma_w8(src, src_stride, dst, dst_stride, h, frac_x, frac_y);
        } else if frac_x == 0 {
            h264_luma::<W, false>(src, src_stride, dst, dst_stride, h, frac_x, frac_y);
        } else {
            h264_luma::<W, true>(src, src_stride, dst, dst_stride, h, frac_x, frac_y);
        }
    }
}

/// H.264 luma prediction at four, eight or a multiple of sixteen columns.
///
/// # Safety
///
/// NEON and, if selected, I8MM must be available. The slices satisfy the
/// kernel contract (checked once here).
#[inline(never)]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn h264_luma<const W: usize, const I8MM: bool>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    h: usize,
    frac_x: usize,
    frac_y: usize,
) {
    const { assert!((W == 4 && !I8MM) || W == 8 || W.is_multiple_of(CHUNK)) };
    super::generic::check_h264_window(src, src_stride, dst, dst_stride, (W, h, 5));
    let (s, d) = (src.as_ptr(), dst.as_mut_ptr());
    // SAFETY: every load below is inside rows 0..h + 5, columns 0..W + 5 of
    // the window and every store inside the h x W destination, which
    // `check_h264_window` verified.
    unsafe {
        match (frac_x, frac_y) {
            (0, 0) => {
                for r in 0..h {
                    core::ptr::copy_nonoverlapping(
                        s.add((r + 2) * src_stride + 2),
                        d.add(r * dst_stride),
                        W,
                    );
                }
            }
            (_, 0) => {
                for r in 0..h {
                    let row = s.add((r + 2) * src_stride);
                    for c0 in (0..W).step_by(W.min(CHUNK)) {
                        let b = half(horizontal_sums::<W, I8MM>(row.add(c0)));
                        let value = match frac_x {
                            1 => vrhaddq_u8(load_luma::<W>(row.add(c0 + 2)), b),
                            3 => vrhaddq_u8(load_luma::<W>(row.add(c0 + 3)), b),
                            _ => b,
                        };
                        store_luma::<W>(d.add(r * dst_stride + c0), value);
                    }
                }
            }
            (0, _) => {
                for c0 in (0..W).step_by(W.min(CHUNK)) {
                    let col = s.add(c0 + 2);
                    let mut rows = [load_luma::<W>(col); 6];
                    for (t, row) in rows.iter_mut().enumerate() {
                        *row = load_luma::<W>(col.add(t * src_stride));
                    }
                    for r in 0..h {
                        let hv = half(six_tap(rows));
                        let value = match frac_y {
                            1 => vrhaddq_u8(rows[2], hv),
                            3 => vrhaddq_u8(rows[3], hv),
                            _ => hv,
                        };
                        store_luma::<W>(d.add(r * dst_stride + c0), value);
                        if r + 1 < h {
                            rows.copy_within(1..6, 0);
                            rows[5] = load_luma::<W>(col.add((r + 6) * src_stride));
                        }
                    }
                }
            }
            _ => two_dimensional::<W, I8MM>(s, src_stride, d, dst_stride, h, frac_x, frac_y),
        }
    }
}

/// The two-dimensional positions: per 16-column chunk, a rolling window of
/// six rows of horizontal sums (centre `j`, and rounded the horizontal
/// halves `b` = row 2 and `s` = row 3) and, where needed, six rows of the
/// samples of the column the vertical half `h` (or `m`, one to the right)
/// filters, as in the generic kernel.
///
/// # Safety
///
/// As for [`h264_luma`], with the window checked.
#[target_feature(enable = "neon")]
unsafe fn two_dimensional<const W: usize, const I8MM: bool>(
    s: *const u8,
    src_stride: usize,
    d: *mut u8,
    dst_stride: usize,
    h: usize,
    frac_x: usize,
    frac_y: usize,
) {
    if frac_x != 2 && frac_y != 2 {
        // SAFETY: the same checked footprints cover both one-axis filters.
        unsafe {
            odd_quarter::<W, I8MM>(s, src_stride, d, dst_stride, h, frac_x >> 1, frac_y >> 1);
        }
        return;
    }
    let vertical_dx = match frac_x {
        1 => Some(0),
        3 => Some(1),
        _ => None,
    };
    // SAFETY: rows 0..h + 5, columns 0..W + 5 of the window and the h x W
    // destination, as the caller guarantees.
    unsafe {
        for c0 in (0..W).step_by(W.min(CHUNK)) {
            let col = s.add(c0 + 2 + vertical_dx.unwrap_or(0));
            let mut sums = [horizontal_sums::<W, I8MM>(s.add(c0)); 6];
            let mut samples = [load_luma::<W>(col); 6];
            for t in 0..6 {
                sums[t] = horizontal_sums::<W, I8MM>(s.add(t * src_stride + c0));
                if vertical_dx.is_some() {
                    samples[t] = load_luma::<W>(col.add(t * src_stride));
                }
            }
            for r in 0..h {
                let value = match (frac_x, frac_y) {
                    (2, 1) => vrhaddq_u8(half(sums[2]), centre(&sums)),
                    (2, 3) => vrhaddq_u8(centre(&sums), half(sums[3])),
                    (1 | 3, 2) => vrhaddq_u8(half(six_tap(samples)), centre(&sums)),
                    _ => centre(&sums),
                };
                store_luma::<W>(d.add(r * dst_stride + c0), value);
                if r + 1 < h {
                    sums.copy_within(1..6, 0);
                    sums[5] = horizontal_sums::<W, I8MM>(s.add((r + 6) * src_stride + c0));
                    if vertical_dx.is_some() {
                        samples.copy_within(1..6, 0);
                        samples[5] = load_luma::<W>(col.add((r + 6) * src_stride));
                    }
                }
            }
        }
    }
}

/// The positions with two odd quarters average one horizontal and one
/// vertical half sample. Only the centre positions use an unrounded
/// horizontal window, so these positions need one horizontal row per output.
#[inline(never)]
#[target_feature(enable = "neon")]
unsafe fn odd_quarter<const W: usize, const I8MM: bool>(
    src: *const u8,
    src_stride: usize,
    dst: *mut u8,
    dst_stride: usize,
    height: usize,
    dx: usize,
    dy: usize,
) {
    // SAFETY: the parent checked w + 5 bytes in each source row and all h + 5 rows.
    unsafe {
        for col in (0..W).step_by(W.min(CHUNK)) {
            let column = src.add(col + 2 + dx);
            let row = src.add((2 + dy) * src_stride + col);
            let mut vertical = [
                load_luma::<W>(column),
                load_luma::<W>(column.add(src_stride)),
                load_luma::<W>(column.add(2 * src_stride)),
                load_luma::<W>(column.add(3 * src_stride)),
                load_luma::<W>(column.add(4 * src_stride)),
                load_luma::<W>(column.add(5 * src_stride)),
            ];
            for line in 0..height {
                let horizontal = half(horizontal_sums::<W, I8MM>(row.add(line * src_stride)));
                let filtered = vrhaddq_u8(horizontal, half(six_tap(vertical)));
                store_luma::<W>(dst.add(line * dst_stride + col), filtered);
                if line + 1 < height {
                    super::slide(&mut vertical, 1, 5);
                    vertical[5] = load_luma::<W>(column.add((line + 6) * src_stride));
                }
            }
        }
    }
}
