//! The intra edge filter and edge upsampling (AV1 §7.11.2.9–12), applied to
//! an edge line before directional prediction. A line holds `AboveRow` or
//! `LeftCol` with index `i` at `i + LINE_ORIGIN`, so indices from -2 (the
//! upsampled corner) are plain array positions.

use nc_common::{BitDepth, Sample};

/// Array position of edge index 0.
pub(super) const LINE_ORIGIN: usize = 16;
/// Edge samples a line can hold: `w + h` of the largest block, the corner
/// and the upsampling margin.
pub(super) const LINE_LEN: usize = LINE_ORIGIN + super::EDGE_LEN + 1;

/// Taps of the three edge filter strengths (§7.11.2.12
/// `Intra_Edge_Kernel`).
const KERNELS: [[u16; 5]; 3] = [[0, 4, 8, 4, 0], [0, 5, 6, 5, 0], [2, 4, 4, 4, 2]];
/// The edge filter taps sum to `1 << FILTER_BITS`.
const FILTER_BITS: u32 = 4;
/// The corner filter (§7.11.2.4: `LeftCol[0]`, the corner, `AboveRow[0]`).
const CORNER_TAPS: [u16; 3] = [5, 6, 5];
/// Upsampling interpolates with taps (-1, 9, 9, -1) / 16 (§7.11.2.11).
const UPSAMPLE_OUTER: i32 = -1;
const UPSAMPLE_INNER: i32 = 9;
/// Upsampling applies only to deltas below this many degrees.
const UPSAMPLE_MAX_DELTA: u16 = 40;

/// The edge filter strength, 0 (off) to 3 (§7.11.2.9), for a block of
/// `width + height` samples whose prediction angle is `delta` degrees from
/// the edge's own direction.
pub(super) fn filter_strength(block_wh: usize, delta: i32, smooth: bool) -> usize {
    let d = delta.unsigned_abs();
    if smooth {
        match block_wh {
            0..=8 => usize::from(d >= 40) + usize::from(d >= 64),
            9..=16 => usize::from(d >= 20) + usize::from(d >= 48),
            17..=24 => 3 * usize::from(d >= 4),
            _ => 3 * usize::from(d >= 1),
        }
    } else {
        match block_wh {
            0..=8 => usize::from(d >= 56),
            9..=16 => usize::from(d >= 40),
            17..=24 => [8, 16, 32].iter().filter(|&&t| d >= t).count(),
            25..=32 => [1, 4, 32].iter().filter(|&&t| d >= t).count(),
            _ => 3 * usize::from(d >= 1),
        }
    }
}

/// Whether the edge is upsampled (§7.11.2.10 `useUpsample`): only small
/// blocks at angles near the edge's own direction.
pub(super) fn use_upsample(block_wh: usize, delta: i32, smooth: bool) -> bool {
    let d = delta.unsigned_abs();
    if d == 0 || d >= u32::from(UPSAMPLE_MAX_DELTA) {
        return false;
    }
    block_wh <= if smooth { 8 } else { 16 }
}

/// Filter edge indices `0..num_px - 1` directly from `(corner, src)` into `dst` (§7.11.2.12).
pub(super) fn filter_edge<T: Sample>(
    corner: T,
    src: &[T],
    dst: &mut [T],
    num_px: usize,
    strength: usize,
) {
    let count = num_px.saturating_sub(1);
    if strength == 0 || count == 0 {
        return;
    }
    let kernel = &KERNELS[strength - 1];
    let src = &src[..count];
    let corner_u16 = corner.to_i32() as u16;
    let sample_at = |idx: usize| -> u16 {
        if idx == 0 {
            corner_u16
        } else {
            src[idx - 1].to_i32() as u16
        }
    };
    for (m, out) in dst[..count].iter_mut().enumerate() {
        let i = m + 1;
        let sum: u16 = kernel
            .iter()
            .enumerate()
            .map(|(j, &tap)| tap * sample_at((i + j).saturating_sub(2).min(count)))
            .sum();
        *out = T::from_i32(i32::from((sum + (1 << (FILTER_BITS - 1))) >> FILTER_BITS));
    }
}

/// Filter edge indices `0..num_px - 1` of `line` (§7.11.2.12): sample
/// `i - 1` becomes the `strength` kernel over indices `i - 3 ..= i + 1`,
/// clamped to `-1 ..= num_px - 2`, of the unfiltered line.
pub(super) fn filter<T: Sample>(line: &mut [T; LINE_LEN], num_px: usize, strength: usize) {
    if strength == 0 {
        return;
    }
    let start = LINE_ORIGIN - 1;
    let mut edge = [T::default(); LINE_LEN];
    edge[..num_px].copy_from_slice(&line[start..start + num_px]);
    filter_edge(
        edge[0],
        &edge[1..num_px],
        &mut line[LINE_ORIGIN..LINE_ORIGIN + num_px - 1],
        num_px,
        strength,
    );
}

/// Filtered corner value from `(left0, corner, above0)` (§7.11.2.4).
#[inline]
pub(super) fn filtered_corner<T: Sample>(left0: T, corner: T, above0: T) -> T {
    let sample = |s: T| s.to_i32() as u16;
    let sum = CORNER_TAPS[0] * sample(left0)
        + CORNER_TAPS[1] * sample(corner)
        + CORNER_TAPS[2] * sample(above0);
    T::from_i32(i32::from((sum + (1 << (FILTER_BITS - 1))) >> FILTER_BITS))
}

/// The filtered corner shared by both lines (§7.11.2.4).
pub(super) fn filter_corner<T: Sample>(above: &mut [T; LINE_LEN], left: &mut [T; LINE_LEN]) {
    let corner = LINE_ORIGIN - 1;
    let value = filtered_corner(left[LINE_ORIGIN], above[corner], above[LINE_ORIGIN]);
    above[corner] = value;
    left[corner] = value;
}

/// Double the resolution of edge indices `-1..num_px` (§7.11.2.11): new
/// index `2i` is old index `i`, odd indices interpolate, index -2 repeats
/// the corner; interpolated samples clip to `depth`.
pub(super) fn upsample<T: Sample>(line: &mut [T; LINE_LEN], num_px: usize, depth: BitDepth) {
    // `dup[k]` is old index `k - 2`, with both ends repeated.
    let mut dup = [0i32; super::MAX_UPSAMPLE + 3];
    dup[0] = line[LINE_ORIGIN - 1].to_i32();
    for (k, value) in dup[1..=num_px + 1].iter_mut().enumerate() {
        *value = line[LINE_ORIGIN - 1 + k].to_i32();
    }
    dup[num_px + 2] = line[LINE_ORIGIN + num_px - 1].to_i32();
    line[LINE_ORIGIN - 2] = T::from_i32(dup[0]);
    for i in 0..num_px {
        let sum =
            UPSAMPLE_OUTER * (dup[i] + dup[i + 3]) + UPSAMPLE_INNER * (dup[i + 1] + dup[i + 2]);
        let value = (sum + (1 << (FILTER_BITS - 1))) >> FILTER_BITS;
        line[LINE_ORIGIN + 2 * i - 1] = depth.clip_to(value);
        line[LINE_ORIGIN + 2 * i] = T::from_i32(dup[i + 2]);
    }
}
