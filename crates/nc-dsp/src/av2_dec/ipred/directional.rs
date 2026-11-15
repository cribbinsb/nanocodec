//! Directional prediction in its three zones: z1 (angles below 90 degrees,
//! from the top and top-right edge), z2 (90 to 180, from top and left) and
//! z3 (above 180, from the left and bottom-left). Luma interpolates with
//! AV2's 4-tap filter, chroma bilinearly. Around them: the intra edge filter
//! (strength by block size, angle and smooth neighbours), multiple reference
//! lines (line `mrl_idx`, or with [`ANGLE_MULTI_MRL_FLAG`] the mean of that
//! line's and line 0's predictions) and intra bi-prediction, which blends z1
//! and z3 predictions of the opposite direction by position.
//!
//! Ported from dav2d `src/ipred_tmpl.c` (`ipred_z{1,2,3}_c`,
//! `get_filter_strength`, `filter_edge`, `ibp_blend`; commit 237229e,
//! BSD-2-Clause, see `docs/licenses/dav2d`). The angle derivatives and the
//! 4-tap filters are AVM's, shared with [`crate::intra_prediction::av2`].

use super::ibp::IBP_WEIGHTS;
use super::{
    ANGLE_HAS_LEFT_FLAG, ANGLE_HAS_TOP_FLAG, ANGLE_IBP_FLAG, ANGLE_IS_LUMA, ANGLE_MASK,
    ANGLE_MRL_IDX_SHIFT, ANGLE_MULTI_MRL_FLAG, ANGLE_SMOOTH_LEFT_EDGE_FLAG,
    ANGLE_SMOOTH_TOP_EDGE_FLAG, ANGLE_USE_EDGE_FILTER_FLAG, Block, SampleRef, TMP_LEN, TMP_STRIDE,
    clip_pixel, mrl_idx,
};
use crate::intra_prediction::av2::{DERIVATIVE, FOUR_TAP, FOUR_TAP_BITS, TWO_TAP_BITS};

/// Angle of pure vertical prediction (degrees).
const VERTICAL: i32 = 90;
/// Angle of pure horizontal prediction (degrees).
const HORIZONTAL: i32 = 180;
/// Angle opposite pure vertical (degrees), where z3 measures from.
const VERTICAL_DOWN: i32 = 270;
/// Fraction bits of a projected position.
const POSITION_BITS: u32 = 6;
/// One sample step in projected-position units.
const STEP: i32 = 1 << POSITION_BITS;
/// Longest edge run a zone reads: the reference line margin (at most 3),
/// one side, the other side and twice the margin again.
const EDGE_RUN: usize = 3 + 64 + 64 + 2 * 3;
/// Bits of the bi-prediction blend weights (they blend out of 128).
const IBP_BLEND_BITS: u32 = 7;
/// log2 of the bi-prediction weight grid's side (16).
const IBP_GRID_LOG2: u32 = 4;

/// The picture extent from the block (dav2d's `max_width`/`max_height`):
/// the edge filter leaves samples past the picture's edge unfiltered.
#[derive(Debug, Clone, Copy)]
pub(super) struct PictureLimits {
    pub(super) max_width: i32,
    pub(super) max_height: i32,
}

/// One zone's predictor, for the multi-line and bi-prediction passes.
type ZoneFn = fn(&mut Block<'_>, SampleRef<'_>, usize, usize, i32, PictureLimits);

/// Edge filter strengths by block size: for each largest `width + height`,
/// the (smallest angle distance, strength) steps, strongest first; below
/// the last step the filter is off.
type StrengthTable = &'static [(usize, &'static [(i32, usize)])];

/// Strengths next to smooth-predicted neighbours.
const SMOOTH_STRENGTH: StrengthTable = &[
    (8, &[(64, 2), (40, 1)]),
    (16, &[(48, 2), (20, 1)]),
    (24, &[(4, 3)]),
    (usize::MAX, &[(i32::MIN, 3)]),
];

/// Strengths next to other neighbours.
const SHARP_STRENGTH: StrengthTable = &[
    (8, &[(56, 1)]),
    (16, &[(40, 1)]),
    (24, &[(32, 3), (16, 2), (8, 1)]),
    (32, &[(32, 3), (4, 2), (i32::MIN, 1)]),
    (usize::MAX, &[(i32::MIN, 3)]),
];

/// Edge filter strength (0 = off, 1..=3 = kernel) for `wh` = width +
/// height and the angle's distance `d` from the edge's axis
/// (`get_filter_strength`); smooth-predicted neighbours filter harder.
fn filter_strength(wh: usize, d: i32, smooth: bool) -> usize {
    let table = if smooth {
        SMOOTH_STRENGTH
    } else {
        SHARP_STRENGTH
    };
    let (_, steps) = table
        .iter()
        .find(|&&(max_wh, _)| wh <= max_wh)
        .expect("last row is unbounded");
    steps
        .iter()
        .find(|&&(min_d, _)| d >= min_d)
        .map_or(0, |&(_, strength)| strength)
}

/// Smooths `input` into `out` with the 5-tap kernel of `strength`
/// (`filter_edge`): only positions in `lim_from..lim_to` are filtered (the
/// rest, past the picture, are copied), and reads clamp to `input`.
fn filter_edge(out: &mut [u8], lim_from: i32, lim_to: i32, input: &[u8], strength: usize) {
    const KERNELS: [[u16; 5]; 3] = [[0, 4, 8, 4, 0], [0, 5, 6, 5, 0], [2, 4, 4, 4, 2]];
    const KERNEL_BITS: u32 = 4;
    let kernel = &KERNELS[strength - 1];
    let last = input.len() as i32 - 1;
    let read = |i: i32| u16::from(input[i.clamp(0, last) as usize]);
    for (i, o) in out.iter_mut().enumerate() {
        let i = i as i32;
        *o = if i < lim_from || i >= lim_to {
            read(i) as u8
        } else {
            let sum: u16 = kernel.iter().zip(i - 2..).map(|(&k, j)| k * read(j)).sum();
            ((sum + (1 << (KERNEL_BITS - 1))) >> KERNEL_BITS) as u8
        };
    }
}

/// The edge `src` copied into `out`, through the edge filter when
/// `strength` is nonzero.
fn load_edge(out: &mut [u8], src: &[u8], strength: usize, lim_from: i32, lim_to: i32) {
    if strength > 0 {
        filter_edge(out, lim_from, lim_to, src, strength);
    } else {
        out.copy_from_slice(src);
    }
}

/// One interpolated sample between `taps[1]` and `taps[2]` at 1/32
/// position `shift`: 4-tap for luma, bilinear for chroma.
fn interpolate(luma: bool, taps: [u8; 4], shift: usize) -> u8 {
    if luma {
        let sum: i32 = FOUR_TAP[shift]
            .iter()
            .zip(taps)
            .map(|(&f, t)| f * i32::from(t))
            .sum();
        clip_pixel((sum + (1 << (FOUR_TAP_BITS - 1))) >> FOUR_TAP_BITS)
    } else {
        let full = 1 << TWO_TAP_BITS;
        let sum = (full - shift as i32) * i32::from(taps[1]) + shift as i32 * i32::from(taps[2]);
        clip_pixel((sum + (full >> 1)) >> TWO_TAP_BITS)
    }
}

/// The 1/32 interpolation position of a projected position.
fn shift_of(position: i32) -> usize {
    ((position & (STEP - 1)) >> 1) as usize
}

/// Multiple reference lines: the rounded mean of the prediction from line
/// `mrl_idx` and the one from line 0, which follows it in the edge buffer.
/// Neither pass filters edges or bi-predicts (dav2d drops those flags).
fn multi_line(
    zone: ZoneFn,
    dst: &mut Block<'_>,
    edge: SampleRef<'_>,
    width: usize,
    height: usize,
    angle: i32,
    mrl: usize,
    limits: PictureLimits,
) {
    let line0 = edge.shifted(((width + height) * 2 + mrl * 3 + 1) as isize);
    let mut tmp = [0; TMP_LEN];
    let mut far = Block::new(&mut tmp, 0, TMP_STRIDE);
    let far_flags = angle | ((mrl as i32) << ANGLE_MRL_IDX_SHIFT) | ANGLE_IS_LUMA;
    zone(&mut far, edge, width, height, far_flags, limits);
    zone(dst, line0, width, height, angle | ANGLE_IS_LUMA, limits);
    for y in 0..height {
        let far = &tmp[y * TMP_STRIDE..][..width];
        for (out, &f) in dst.row(y, width).iter_mut().zip(far) {
            *out = ((u16::from(f) + u16::from(*out) + 1) >> 1) as u8;
        }
    }
}

/// Intra bi-prediction: blends `dst` with the opposite-direction prediction
/// `tmp` by the position weights of `mode` (`ibp_blend`); `transposed`
/// swaps the weights' axes for z3.
fn ibp_blend(
    dst: &mut Block<'_>,
    tmp: &[u8],
    width: usize,
    height: usize,
    transposed: bool,
    mode: usize,
) {
    let weights = &IBP_WEIGHTS[mode];
    let x_shift = width >> (IBP_GRID_LOG2 + 1);
    let y_shift = height >> (IBP_GRID_LOG2 + 1);
    let full = 1u16 << IBP_BLEND_BITS;
    for y in 0..height {
        let wy = y >> y_shift;
        let other = &tmp[y * TMP_STRIDE..][..width];
        for (x, (out, &o)) in dst.row(y, width).iter_mut().zip(other).enumerate() {
            let wx = x >> x_shift;
            let weight = u16::from(if transposed {
                weights[wx][wy]
            } else {
                weights[wy][wx]
            });
            *out = ((u16::from(o) * (full - weight) + u16::from(*out) * weight + (full >> 1))
                >> IBP_BLEND_BITS) as u8;
        }
    }
}

/// The angle argument's flags apart from the angle and bi-prediction, which
/// the bi-prediction pass passes on.
fn pass_flags(angle: i32) -> i32 {
    angle & !(ANGLE_MASK | ANGLE_IBP_FLAG)
}

/// Bi-prediction weight set of a z1 angle (dav2d's observed table: 84 -> 0,
/// 73/67/61 -> 1..3, 51/45/39 -> 4..6, wide angles 6).
fn ibp_mode_z1(angle: i32) -> usize {
    (10 - (angle >> 3)).min(6) as usize
}

/// Bi-prediction weight set of a z3 angle (186 -> 0, 197/203/209 -> 1..3,
/// 219/225/231 -> 4..6, wide angles 6).
fn ibp_mode_z3(angle: i32) -> usize {
    ((angle - 183) >> 3).min(6) as usize
}

/// Zone 1: projects each row onto the top edge.
pub(super) fn z1(
    dst: &mut Block<'_>,
    edge: SampleRef<'_>,
    width: usize,
    height: usize,
    flags: i32,
    limits: PictureLimits,
) {
    let angle = flags & ANGLE_MASK;
    let mrl = mrl_idx(flags);
    debug_assert!(angle < VERTICAL);
    if flags & ANGLE_MULTI_MRL_FLAG != 0 {
        multi_line(z1, dst, edge, width, height, angle, mrl, limits);
        return;
    }
    let luma = flags & ANGLE_IS_LUMA != 0;
    let dx = DERIVATIVE[angle as usize];
    let max_base_x = (width + height + 2 * mrl) as i32 - 1;

    // filt: one sample of left padding, the corner, the line margin, the
    // top and top-right runs, two of right padding; `top` is the first top
    // sample.
    let mut filt = [0u8; 2 + EDGE_RUN + 2];
    let top = 2 + mrl;
    let use_filter = flags & (ANGLE_USE_EDGE_FILTER_FLAG | ANGLE_HAS_TOP_FLAG)
        == ANGLE_USE_EDGE_FILTER_FLAG | ANGLE_HAS_TOP_FLAG
        && mrl == 0;
    let strength = if use_filter {
        filter_strength(
            width + height,
            VERTICAL - angle,
            flags & ANGLE_SMOOTH_TOP_EDGE_FLAG != 0,
        )
    } else {
        0
    };
    let sz = 1 + mrl + width + height + 2 * mrl;
    let lim_to = sz as i32 + limits.max_width - width as i32;
    load_edge(&mut filt[1..=sz], edge.slice(0, sz), strength, 1, lim_to);
    filt[0] = filt[1];
    filt[sz + 1] = filt[sz];
    filt[sz + 2] = filt[sz];

    let last = filt[top + max_base_x as usize];
    for y in 0..height {
        let xpos = dx * (1 + mrl as i32 + y as i32);
        let base = xpos >> POSITION_BITS;
        let row = dst.row(y, width);
        if base > max_base_x {
            row.fill(last);
            continue;
        }
        let shift = shift_of(xpos);
        let reach = ((max_base_x - base + 1) as usize).min(width);
        for (x, out) in row[..reach].iter_mut().enumerate() {
            let at = top + base as usize + x;
            *out = interpolate(
                luma,
                [filt[at - 1], filt[at], filt[at + 1], filt[at + 2]],
                shift,
            );
        }
        row[reach..].fill(last);
    }

    if flags & ANGLE_IBP_FLAG != 0 {
        let mut tmp = [0; TMP_LEN];
        let opposite = (HORIZONTAL + angle) | pass_flags(flags);
        z3(
            &mut Block::new(&mut tmp, 0, TMP_STRIDE),
            edge,
            width,
            height,
            opposite,
            limits,
        );
        ibp_blend(dst, &tmp, width, height, false, ibp_mode_z1(angle));
    }
}

/// Zone 2: projects each sample onto the top edge, or the left edge where
/// the projection passes the corner.
pub(super) fn z2(
    dst: &mut Block<'_>,
    edge: SampleRef<'_>,
    width: usize,
    height: usize,
    flags: i32,
    limits: PictureLimits,
) {
    let angle = flags & ANGLE_MASK;
    let mrl = mrl_idx(flags);
    debug_assert!(angle > VERTICAL && angle < HORIZONTAL);
    if flags & ANGLE_MULTI_MRL_FLAG != 0 {
        multi_line(z2, dst, edge, width, height, angle, mrl, limits);
        return;
    }
    let luma = flags & ANGLE_IS_LUMA != 0;
    let filter = flags & ANGLE_USE_EDGE_FILTER_FLAG != 0 && mrl == 0;
    let dy = DERIVATIVE[(angle - VERTICAL) as usize];
    let dx = DERIVATIVE[(HORIZONTAL - angle) as usize];
    let wh = width + height;

    // filt: padding, the corner, the line margin, the top run, padding;
    // `filt[top + 1]` is the corner of line mrl_idx.
    let mut filt = [0u8; 1 + 1 + 3 + 64 + 1];
    let top = mrl;
    let strength_t = if filter && flags & ANGLE_HAS_TOP_FLAG != 0 {
        filter_strength(
            wh,
            angle - VERTICAL,
            flags & ANGLE_SMOOTH_TOP_EDGE_FLAG != 0,
        )
    } else {
        0
    };
    let sz_t = 1 + width + mrl;
    let lim_to = sz_t as i32 + limits.max_width - width as i32;
    load_edge(
        &mut filt[1..=sz_t],
        edge.slice(0, sz_t),
        strength_t,
        1,
        lim_to,
    );
    filt[0] = filt[1];
    filt[sz_t + 1] = filt[sz_t];

    // filt2: padding, the left run bottom to top, the line margin and the
    // corner, padding; `filt2[left - (k + 2)]` is left sample k.
    let mut filt2 = [0u8; 1 + 1 + 3 + 64 + 1];
    let left = height + 2;
    let strength_l = if filter && flags & ANGLE_HAS_LEFT_FLAG != 0 {
        filter_strength(
            wh,
            HORIZONTAL - angle,
            flags & ANGLE_SMOOTH_LEFT_EDGE_FLAG != 0,
        )
    } else {
        0
    };
    let sz_l = 1 + height + mrl;
    let lim_from = height as i32 - limits.max_height;
    let src = edge.slice(-((height + mrl) as isize), sz_l);
    load_edge(
        &mut filt2[1..=sz_l],
        src,
        strength_l,
        lim_from,
        sz_l as i32 - 1,
    );
    filt2[sz_l + 1] = filt2[sz_l];
    filt2[0] = filt2[1];

    let margin = 1 + mrl as i32;
    for y in 0..height {
        let row = dst.row(y, width);
        let mut xpos = -(y as i32 + margin) * dx;
        for (x, out) in row.iter_mut().enumerate() {
            *out = if xpos < -STEP * margin {
                let ypos = ((y as i32) << POSITION_BITS) - (x as i32 + margin) * dy;
                let base_y = ypos >> POSITION_BITS;
                debug_assert!(base_y >= -margin);
                let at = (left as i32 - base_y) as usize;
                interpolate(
                    luma,
                    [filt2[at - 1], filt2[at - 2], filt2[at - 3], filt2[at - 4]],
                    shift_of(ypos),
                )
            } else {
                // The first tap is the sample before line mrl_idx's corner
                // (`filt[0]`) at the steepest reach.
                let first = (top as i32 + (xpos >> POSITION_BITS) + 1) as usize;
                interpolate(
                    luma,
                    [
                        filt[first],
                        filt[first + 1],
                        filt[first + 2],
                        filt[first + 3],
                    ],
                    shift_of(xpos),
                )
            };
            xpos += STEP;
        }
    }
}

/// Zone 3: projects each column onto the left edge.
pub(super) fn z3(
    dst: &mut Block<'_>,
    edge: SampleRef<'_>,
    width: usize,
    height: usize,
    flags: i32,
    limits: PictureLimits,
) {
    let angle = flags & ANGLE_MASK;
    let mrl = mrl_idx(flags);
    debug_assert!(angle > HORIZONTAL);
    if flags & ANGLE_MULTI_MRL_FLAG != 0 {
        multi_line(z3, dst, edge, width, height, angle, mrl, limits);
        return;
    }
    let luma = flags & ANGLE_IS_LUMA != 0;
    let dy = DERIVATIVE[(VERTICAL_DOWN - angle) as usize];
    let max_base_y = (width + height + 2 * mrl) as i32 - 1;

    // filt: two samples of padding, the bottom-left and left runs bottom to
    // top, the line margin and the corner, one of padding; `filt[left - k]`
    // is left sample k.
    let mut filt = [0u8; 2 + EDGE_RUN + 2];
    let left = 1 + width + height + 2 * mrl;
    let use_filter = flags & (ANGLE_USE_EDGE_FILTER_FLAG | ANGLE_HAS_LEFT_FLAG)
        == ANGLE_USE_EDGE_FILTER_FLAG | ANGLE_HAS_LEFT_FLAG
        && mrl == 0;
    let strength = if use_filter {
        filter_strength(
            width + height,
            angle - HORIZONTAL,
            flags & ANGLE_SMOOTH_LEFT_EDGE_FLAG != 0,
        )
    } else {
        0
    };
    let sz = 1 + mrl + width + height + 2 * mrl;
    let lim_from = height as i32 - limits.max_height;
    let src = edge.slice(1 - sz as isize, sz);
    load_edge(&mut filt[2..2 + sz], src, strength, lim_from, sz as i32 - 1);
    filt[0] = filt[2];
    filt[1] = filt[2];
    filt[sz + 2] = filt[sz + 1];

    let last = filt[left - max_base_y as usize];
    for x in 0..width {
        let ypos = dy * (1 + mrl as i32 + x as i32);
        let shift = shift_of(ypos);
        let first = ypos >> POSITION_BITS;
        for y in 0..height {
            let base = first + y as i32;
            let value = if base <= max_base_y {
                let at = (left as i32 - base) as usize;
                interpolate(
                    luma,
                    [filt[at + 1], filt[at], filt[at - 1], filt[at - 2]],
                    shift,
                )
            } else {
                last
            };
            dst.row(y, width)[x] = value;
        }
    }

    if flags & ANGLE_IBP_FLAG != 0 {
        let mut tmp = [0; TMP_LEN];
        let opposite = (angle - HORIZONTAL) | pass_flags(flags);
        z1(
            &mut Block::new(&mut tmp, 0, TMP_STRIDE),
            edge,
            width,
            height,
            opposite,
            limits,
        );
        ibp_blend(dst, &tmp, width, height, true, ibp_mode_z3(angle));
    }
}
