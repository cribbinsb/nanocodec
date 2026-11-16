//! Scalar CDEF: the plain per-sample reference (and the kernels of builds
//! without SIMD), written directly from libaom `cdef_filter_block_internal`
//! and `cdef_find_dir_c` (`av1/common/cdef_block.c`, BSD-2-Clause, Alliance
//! for Open Media).

use nc_common::Sample;

use super::{CDEF_UNAVAILABLE, CdefArgs, CdefFns, DIRECTIONS, SEC_TAPS};

// The tables of SIMD builds use the 8-bit reference until their own
// kernels exist.
pub(crate) use self::interior::{block4_interior, block8_interior};

/// The scalar table.
pub(crate) fn cdef_fns() -> CdefFns {
    CdefFns {
        find_dir,
        block8,
        block4,
        block8_interior,
        block4_interior,
        block4_interior_pair: None,
    }
}

/// `1 / n` scaled by 840 for the direction search's line averages.
const DIV_TABLE: [i32; 9] = [0, 840, 420, 280, 210, 168, 140, 120, 105];

/// libaom `cdef_find_dir_c` on the 8x8 block of 8-bit samples at the start
/// of `src`.
pub(crate) fn find_dir(src: &[u8], stride: usize) -> (u8, i32) {
    find_dir_at(src, stride, 0)
}

/// libaom `cdef_find_dir_c` on samples `bits_above` bits deeper than 8
/// (`coeff_shift`: each sample is shifted down to 8 bits first). The index
/// arithmetic mirrors libaom line for line, so it stays checkable against
/// it.
#[allow(clippy::needless_range_loop)]
pub(crate) fn find_dir_at<T: Sample>(src: &[T], stride: usize, bits_above: u32) -> (u8, i32) {
    let mut partial = [[0i32; 15]; 8];
    for i in 0..8 {
        let row = &src[i * stride..i * stride + 8];
        for j in 0..8 {
            let v = (row[j].to_i32() >> bits_above) - 128;
            partial[0][i + j] += v;
            partial[1][i + j / 2] += v;
            partial[2][i] += v;
            partial[3][3 + i - j / 2] += v;
            partial[4][7 + i - j] += v;
            partial[5][3 - i / 2 + j] += v;
            partial[6][j] += v;
            partial[7][i / 2 + j] += v;
        }
    }
    let mut cost = [0i32; 8];
    for i in 0..8 {
        cost[2] += partial[2][i] * partial[2][i];
        cost[6] += partial[6][i] * partial[6][i];
    }
    cost[2] *= DIV_TABLE[8];
    cost[6] *= DIV_TABLE[8];
    for i in 0..7 {
        cost[0] += (partial[0][i] * partial[0][i] + partial[0][14 - i] * partial[0][14 - i])
            * DIV_TABLE[i + 1];
        cost[4] += (partial[4][i] * partial[4][i] + partial[4][14 - i] * partial[4][14 - i])
            * DIV_TABLE[i + 1];
    }
    cost[0] += partial[0][7] * partial[0][7] * DIV_TABLE[8];
    cost[4] += partial[4][7] * partial[4][7] * DIV_TABLE[8];
    for i in (1..8).step_by(2) {
        for j in 0..5 {
            cost[i] += partial[i][3 + j] * partial[i][3 + j];
        }
        cost[i] *= DIV_TABLE[8];
        for j in 0..3 {
            cost[i] += (partial[i][j] * partial[i][j] + partial[i][10 - j] * partial[i][10 - j])
                * DIV_TABLE[2 * j + 2];
        }
    }
    let (mut best_dir, mut best_cost) = (0, 0);
    for (d, &c) in cost.iter().enumerate() {
        if c > best_cost {
            best_cost = c;
            best_dir = d;
        }
    }
    let var = (best_cost - cost[(best_dir + 4) & 7]) >> 10;
    (best_dir as u8, var)
}

/// libaom `constrain` with a precomputed shift.
fn constrain(diff: i32, threshold: i32, shift: u32) -> i32 {
    if threshold == 0 {
        return 0;
    }
    let magnitude = diff.abs().min((threshold - (diff.abs() >> shift)).max(0));
    if diff < 0 { -magnitude } else { magnitude }
}

/// The filter of an `n`x`n` block from a padded 16-bit window, into
/// samples of any depth (the result lies between the taps' minimum and
/// maximum, or within the strengths of the centre, so no clip is needed).
#[allow(clippy::needless_range_loop)] // mirrors libaom's indexing
pub(crate) fn filter<T: Sample>(
    n: usize,
    src: &[i16],
    stride: usize,
    origin: usize,
    args: &CdefArgs,
    dst: &mut [T],
    dst_stride: usize,
) {
    let (use_pri, use_sec) = (args.pri != 0, args.sec != 0);
    let offset = |d: usize, k: usize| {
        let (dr, dc) = DIRECTIONS[d & 7][k];
        dr * stride as isize + dc
    };
    let pri_off = [offset(args.dir, 0), offset(args.dir, 1)];
    let sec_off = [
        [offset(args.dir + 2, 0), offset(args.dir + 6, 0)],
        [offset(args.dir + 2, 1), offset(args.dir + 6, 1)],
    ];
    for r in 0..n {
        for c in 0..n {
            let base = origin + r * stride + c;
            let x = i32::from(src[base]);
            let (mut sum, mut lo, mut hi) = (0i32, x, x);
            let mut take = |offset: isize, strength: i16, shift: u32, weight: i16| {
                for o in [offset, -offset] {
                    let p =
                        i32::from(src[base.checked_add_signed(o).expect("tap inside the source")]);
                    sum += i32::from(weight) * constrain(p - x, i32::from(strength), shift);
                    if p != i32::from(CDEF_UNAVAILABLE) {
                        lo = lo.min(p);
                        hi = hi.max(p);
                    }
                }
            };
            for k in 0..2 {
                if use_pri {
                    take(pri_off[k], args.pri, args.pri_shift, args.pri_taps[k]);
                }
                if use_sec {
                    for &offset in &sec_off[k] {
                        take(offset, args.sec, args.sec_shift, SEC_TAPS[k]);
                    }
                }
            }
            let mut y = x + ((8 + sum - i32::from(sum < 0)) >> 4);
            if use_pri && use_sec {
                y = y.clamp(lo, hi);
            }
            dst[r * dst_stride + c] = T::from_i32(y);
        }
    }
}

/// The filter of an `N`x`N` block whose taps all lie inside the plane
/// (`src` from the block's top-left sample `origin` at `stride`), samples
/// of any depth, written as whole-row operations per tap so that the ISA
/// instances vectorise it (`high_depth::vector`): every tap of a row is
/// one contiguous slice at a fixed offset. The same arithmetic as
/// [`filter`] without unavailable samples.
#[allow(clippy::inline_always, reason = "inlined into the ISA instances")]
#[allow(clippy::needless_range_loop)] // mirrors libaom's tap indexing
#[inline(always)]
pub(crate) fn interior_rows<S: Sample, const N: usize>(
    src: &[S],
    stride: usize,
    origin: usize,
    args: &CdefArgs,
    dst: &mut [S],
    dst_stride: usize,
) {
    let offset = |d: usize, k: usize| {
        let (dr, dc) = DIRECTIONS[d & 7][k];
        dr * stride as isize + dc
    };
    // (offset, strength, shift, weight) of every tap pair.
    let mut taps = [(0isize, 0i32, 0u32, 0i32); 6];
    let mut count = 0;
    for k in 0..2 {
        if args.pri != 0 {
            taps[count] = (
                offset(args.dir, k),
                i32::from(args.pri),
                args.pri_shift,
                i32::from(args.pri_taps[k]),
            );
            count += 1;
        }
        if args.sec != 0 {
            for d in [args.dir + 2, args.dir + 6] {
                taps[count] = (
                    offset(d, k),
                    i32::from(args.sec),
                    args.sec_shift,
                    i32::from(SEC_TAPS[k]),
                );
                count += 1;
            }
        }
    }
    let clip = args.pri != 0 && args.sec != 0;
    for r in 0..N {
        let base = origin + r * stride;
        let x: [i32; N] = core::array::from_fn(|c| src[base + c].to_i32());
        let (mut sum, mut lo, mut hi) = ([0i32; N], x, x);
        for &(offset, strength, shift, weight) in &taps[..count] {
            for o in [offset, -offset] {
                let start = base.checked_add_signed(o).expect("tap inside the plane");
                let p = &src[start..start + N];
                for c in 0..N {
                    let p = p[c].to_i32();
                    let diff = p - x[c];
                    let magnitude = diff.abs().min((strength - (diff.abs() >> shift)).max(0));
                    sum[c] += weight * if diff < 0 { -magnitude } else { magnitude };
                    lo[c] = lo[c].min(p);
                    hi[c] = hi[c].max(p);
                }
            }
        }
        for (c, out) in dst[r * dst_stride..][..N].iter_mut().enumerate() {
            let mut y = x[c] + ((8 + sum[c] - i32::from(sum[c] < 0)) >> 4);
            if clip {
                y = y.clamp(lo[c], hi[c]);
            }
            *out = S::from_i32(y);
        }
    }
}

/// 8x8 blocks.
fn block8(
    src: &[i16],
    stride: usize,
    origin: usize,
    args: &CdefArgs,
    dst: &mut [u8],
    dst_stride: usize,
) {
    filter(8, src, stride, origin, args, dst, dst_stride);
}

/// 4x4 blocks.
fn block4(
    src: &[i16],
    stride: usize,
    origin: usize,
    args: &CdefArgs,
    dst: &mut [u8],
    dst_stride: usize,
) {
    filter(4, src, stride, origin, args, dst, dst_stride);
}

/// The 8-bit references: the same filter on the unpadded plane, for blocks
/// whose taps all lie inside it (so no tap is unavailable and the clipping
/// range takes every tap).
mod interior {
    use super::{CdefArgs, DIRECTIONS, SEC_TAPS, constrain};

    #[allow(clippy::needless_range_loop)] // mirrors libaom's indexing
    fn filter(
        n: usize,
        src: &[u8],
        stride: usize,
        origin: usize,
        args: &CdefArgs,
        dst: &mut [u8],
        dst_stride: usize,
    ) {
        let (use_pri, use_sec) = (args.pri != 0, args.sec != 0);
        let offset = |d: usize, k: usize| {
            let (dr, dc) = DIRECTIONS[d & 7][k];
            dr * stride as isize + dc
        };
        let pri_off = [offset(args.dir, 0), offset(args.dir, 1)];
        let sec_off = [
            [offset(args.dir + 2, 0), offset(args.dir + 6, 0)],
            [offset(args.dir + 2, 1), offset(args.dir + 6, 1)],
        ];
        for r in 0..n {
            for c in 0..n {
                let base = origin + r * stride + c;
                let x = i32::from(src[base]);
                let (mut sum, mut lo, mut hi) = (0i32, x, x);
                let mut take = |offset: isize, strength: i16, shift: u32, weight: i16| {
                    for o in [offset, -offset] {
                        let p = i32::from(
                            src[base.checked_add_signed(o).expect("tap inside the plane")],
                        );
                        sum += i32::from(weight) * constrain(p - x, i32::from(strength), shift);
                        lo = lo.min(p);
                        hi = hi.max(p);
                    }
                };
                for k in 0..2 {
                    if use_pri {
                        take(pri_off[k], args.pri, args.pri_shift, args.pri_taps[k]);
                    }
                    if use_sec {
                        for &offset in &sec_off[k] {
                            take(offset, args.sec, args.sec_shift, SEC_TAPS[k]);
                        }
                    }
                }
                let mut y = x + ((8 + sum - i32::from(sum < 0)) >> 4);
                if use_pri && use_sec {
                    y = y.clamp(lo, hi);
                }
                dst[r * dst_stride + c] = y as u8;
            }
        }
    }

    /// 8x8 blocks.
    pub(crate) fn block8_interior(
        src: &[u8],
        stride: usize,
        origin: usize,
        args: &CdefArgs,
        dst: &mut [u8],
        dst_stride: usize,
    ) {
        filter(8, src, stride, origin, args, dst, dst_stride);
    }

    /// 4x4 blocks.
    pub(crate) fn block4_interior(
        src: &[u8],
        stride: usize,
        origin: usize,
        args: &CdefArgs,
        dst: &mut [u8],
        dst_stride: usize,
    ) {
        filter(4, src, stride, origin, args, dst, dst_stride);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn horizontal_stripes_are_direction_2() {
        let mut samples = [0u8; 64];
        for (i, s) in samples.iter_mut().enumerate() {
            *s = if (i / 8) % 2 == 0 { 200 } else { 50 };
        }
        assert_eq!(find_dir(&samples, 8).0, 2);
    }

    #[test]
    fn vertical_stripes_are_direction_6() {
        let mut samples = [0u8; 64];
        for (i, s) in samples.iter_mut().enumerate() {
            *s = if (i % 8) % 2 == 0 { 200 } else { 50 };
        }
        assert_eq!(find_dir(&samples, 8).0, 6);
    }
}
