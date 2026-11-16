//! `residual_coding()` (H.265 §7.3.8.11) of Main-profile streams: last
//! position, coded sub-block and significance flags, greater-1/2 flags,
//! Rice/Exp-Golomb remainders (§9.3.3.11), sign data hiding and
//! `transform_skip_flag`.
//!
//! The context selection of `sig_coeff_flag` and `coded_sub_block_flag` is
//! the encoder's (`crate::residual`), which follows §9.3.4.2.4–5, tabulated
//! once by scan index within a sub-block (`crate::residual::sig_contexts`),
//! so the significance loop is one table load and one bin per position. Significant positions are recorded without a branch on the
//! flag, and the greater-1 context state advances by table.

use super::cabac::Engine;
use super::ctx::{self, Contexts};
use crate::residual::{SIGN_HIDING_THRESHOLD, neighbour_coded, sig_contexts, sig_pattern};
use crate::scan::{SUB_BLOCK_COEFFS, Scan, ScanKind};
use nc_common::decode::DecodeError;
use nc_dsp::Pixel;

// The shared significance table carries the encoder's chroma offset.
const _: () = assert!(ctx::SIG_CHROMA == crate::contexts::SIG_CHROMA_OFFSET);

/// Smallest value of each `last_sig_coeff_x/y_prefix` group: the last
/// position before its suffix is added (§7.4.9.11).
const MIN_IN_GROUP: [u32; 10] = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24];
/// Greater-1 flags coded per sub-block.
const MAX_GREATER1: usize = 8;
/// Largest Rice parameter (§9.3.3.11).
const MAX_RICE: u32 = 4;
/// Prefix length below which the remainder is plain Rice coded.
const RICE_PREFIX: u32 = 3;
/// Longest Exp-Golomb suffix that keeps a level within 16 bits plus margin.
const MAX_SUFFIX: u32 = 28;
/// `CoeffMinY` magnitude for 8-bit video (§7.4.9.11).
const MAX_MAGNITUDE: u32 = 1 << 15;
/// Transform skip is allowed for 4x4 blocks only in version 1 (§7.3.8.11).
const TRANSFORM_SKIP_LOG2: u8 = 2;

/// `greater1Ctx` after a `coeff_abs_level_greater1_flag` of 0 (index 0) or
/// 1 (index 1), by its value before (§9.3.4.2.6): a 1 ends the run at 0
/// for the rest of the sub-block, a 0 counts up to 3 unless already 0.
const NEXT_GREATER1: [[u8; 4]; 2] = [[0, 2, 3, 3], [0; 4]];
/// No `coeff_abs_level_greater1_flag` of the sub-block was 1 (yet).
const NO_GREATER1: usize = usize::MAX;

/// What `residual_coding()` needs besides the scan.
#[derive(Debug, Clone, Copy)]
pub(super) struct Options {
    pub luma: bool,
    /// `sign_data_hiding_enabled_flag && !cu_transquant_bypass_flag`.
    pub sign_hiding: bool,
    /// `transform_skip_enabled_flag && !cu_transquant_bypass_flag`.
    pub transform_skip: bool,
}

/// Parse one transform block's levels into `levels` (raster, all zero on
/// entry) and list the non-zero positions in `positions`. Returns
/// `transform_skip_flag`.
///
/// The parser is generic over the sample type `T` only so that each
/// instance of the block decoder (`u8`, `u16`) calls its own copy: with one
/// copy and two call sites LLVM stopped inlining it into the 8-bit block
/// decoder, +0.85 % decode instructions on a 720p stream of ours
/// (callgrind, 2026-11-15).
pub(super) fn read<T: Pixel>(
    engine: &mut Engine<'_>,
    ctx: &mut Contexts,
    scan: &Scan,
    options: Options,
    levels: &mut [i16],
    positions: &mut Vec<u16>,
) -> Result<bool, DecodeError> {
    // The engine works on a local copy, written back at the end: its
    // range, window and counters then stay in registers through the
    // block's bins instead of going through memory at each one.
    let mut local = engine.clone();
    let skip = read_levels::<T>(&mut local, ctx, scan, options, levels, positions)?;
    *engine = local;
    Ok(skip)
}

/// [`read`] on the local engine.
fn read_levels<T: Pixel>(
    e: &mut Engine<'_>,
    ctx: &mut Contexts,
    scan: &Scan,
    options: Options,
    levels: &mut [i16],
    positions: &mut Vec<u16>,
) -> Result<bool, DecodeError> {
    let luma = options.luma;
    let skip = options.transform_skip
        && scan.log2 == TRANSFORM_SKIP_LOG2
        && e.bin(ctx.at(ctx::TRANSFORM_SKIP, usize::from(!luma)));
    let side = 1usize << (scan.log2 - 2);
    let last = last_position::<T>(e, ctx, scan, luma)?;
    let last_group = last / SUB_BLOCK_COEFFS;
    let sig_cases = sig_contexts(scan, luma);
    let mut coded = [false; 64];
    let mut greater1_state = 1;
    for group in (0..=last_group).rev() {
        let (xs, ys) = scan.sub_blocks[group];
        let flag = if group == last_group || group == 0 {
            true
        } else {
            let base = if luma { 0 } else { ctx::CODED_SUB_BLOCK_CHROMA };
            let inc = base + usize::from(neighbour_coded(&coded, side, xs, ys));
            e.bin(ctx.at(ctx::CODED_SUB_BLOCK, inc))
        };
        coded[ys * side + xs] = flag;
        if !flag {
            continue;
        }
        let pattern = sig_pattern(&coded, side, xs, ys).unwrap_or(0);
        let sig_contexts = &sig_cases[usize::from(xs + ys == 0)][pattern];
        // Significant scan indices within the sub-block, highest first.
        let mut found = [0u8; SUB_BLOCK_COEFFS];
        let mut count = 0;
        let top = if group == last_group {
            found[0] = (last % SUB_BLOCK_COEFFS) as u8;
            count = 1;
            last % SUB_BLOCK_COEFFS
        } else {
            SUB_BLOCK_COEFFS
        };
        for index in (1..top).rev() {
            let significant = e.bin(ctx.at(ctx::SIG_COEFF, usize::from(sig_contexts[index])));
            // Written unconditionally, kept only when significant: no
            // branch on a flag the predictor cannot guess.
            found[count] = index as u8;
            count += usize::from(significant);
        }
        if top > 0 {
            // The sub-block's DC is inferred significant when nothing else
            // in a coded (non-first, non-last) sub-block was (§7.4.9.11).
            let inferred = group > 0 && group < last_group && count == 0;
            let significant =
                inferred || e.bin(ctx.at(ctx::SIG_COEFF, usize::from(sig_contexts[0])));
            found[count] = 0;
            count += usize::from(significant);
        }
        if count == 0 {
            continue;
        }
        let found = &found[..count];
        let hidden =
            options.sign_hiding && usize::from(found[0] - found[count - 1]) > SIGN_HIDING_THRESHOLD;
        let first = group * SUB_BLOCK_COEFFS;
        sub_block_levels::<T, _>(
            e,
            ctx,
            found,
            group,
            luma,
            hidden,
            &mut greater1_state,
            |index, level| {
                let raster = scan.positions[first + usize::from(index)];
                // In -2^15..2^15 - 1: `sub_block_levels` rejects the rest.
                levels[raster] = level as i16;
                positions.push(raster as u16);
            },
        )?;
    }
    Ok(skip)
}

/// `last_sig_coeff_{x,y}_{prefix,suffix}` as a scan position.
#[allow(
    clippy::extra_unused_type_parameters,
    reason = "one copy per decoder instance, see `read`"
)]
fn last_position<T: Pixel>(
    e: &mut Engine<'_>,
    ctx: &mut Contexts,
    scan: &Scan,
    luma: bool,
) -> Result<usize, DecodeError> {
    let log2 = u32::from(scan.log2);
    let size = 1usize << log2;
    // §9.3.4.2.3: ctxOffset and ctxShift.
    let (offset, shift) = if luma {
        (
            (3 * (log2 - 2) + ((log2 - 1) >> 2)) as usize,
            (log2 + 1) >> 2,
        )
    } else {
        (ctx::LAST_CHROMA, log2 - 2)
    };
    let mut prefix = [0u32; 2];
    for (axis, element) in [ctx::LAST_X, ctx::LAST_Y].into_iter().enumerate() {
        while prefix[axis] < 2 * log2 - 1
            && e.bin(ctx.at(element, offset + (prefix[axis] >> shift) as usize))
        {
            prefix[axis] += 1;
        }
    }
    let mut xy = [0usize; 2];
    for axis in 0..2 {
        let g = prefix[axis];
        let suffix = if g > 3 {
            e.bypass_bits((g >> 1) - 1)
        } else {
            0
        };
        xy[axis] = (MIN_IN_GROUP[g as usize] + suffix) as usize;
        if xy[axis] >= size {
            return Err(DecodeError::Malformed("last coefficient position"));
        }
    }
    if scan.kind == ScanKind::Vertical {
        xy.swap(0, 1);
    }
    // Both coordinates are below `size`, so the raster index is in range.
    Ok(usize::from(scan.index_of[xy[1] * size + xy[0]]))
}

/// `coeff_abs_level_remaining` with Rice parameter `rice` (§9.3.3.11): a
/// unary prefix, then the suffix bits in one bypass read.
#[allow(
    clippy::extra_unused_type_parameters,
    reason = "one copy per decoder instance, see `read`"
)]
fn remaining<T: Pixel>(e: &mut Engine<'_>, rice: u32) -> Result<u32, DecodeError> {
    let prefix = e.bypass_unary(RICE_PREFIX + MAX_SUFFIX + 1);
    if prefix > RICE_PREFIX + MAX_SUFFIX {
        return Err(DecodeError::Malformed("coefficient remainder prefix"));
    }
    if prefix < RICE_PREFIX {
        Ok((prefix << rice) + e.bypass_bits(rice))
    } else {
        let length = rice + prefix - RICE_PREFIX;
        if length > MAX_SUFFIX {
            return Err(DecodeError::Malformed("coefficient remainder suffix"));
        }
        Ok((2 << rice) + (1 << length) + e.bypass_bits(length))
    }
}

/// The levels of one sub-block's significant coefficients (scan indices
/// `found` within the sub-block, highest first): greater-1/2 flags, signs,
/// remainders.
#[allow(clippy::too_many_arguments)]
fn sub_block_levels<T: Pixel, F: FnMut(u8, i32)>(
    e: &mut Engine<'_>,
    ctx: &mut Contexts,
    found: &[u8],
    group: usize,
    luma: bool,
    hidden: bool,
    greater1_state: &mut usize,
    mut store: F,
) -> Result<(), DecodeError> {
    // ctxSet (§9.3.4.2.6): 2 for luma sub-blocks other than the first,
    // plus one when the previous sub-block ended with a greater-1 flag.
    let set = usize::from(group > 0 && luma) * 2 + usize::from(*greater1_state == 0);
    let g1_base = if luma { 0 } else { ctx::GREATER1_CHROMA } + 4 * set;
    let mut magnitude = [1u32; SUB_BLOCK_COEFFS];
    let mut first_greater1 = NO_GREATER1;
    let mut state = 1;
    for (i, m) in magnitude[..found.len().min(MAX_GREATER1)]
        .iter_mut()
        .enumerate()
    {
        let greater1 = e.bin(ctx.at(ctx::GREATER1, g1_base + usize::from(state)));
        *m += u32::from(greater1);
        if greater1 && first_greater1 == NO_GREATER1 {
            first_greater1 = i;
        }
        state = NEXT_GREATER1[usize::from(greater1)][usize::from(state)];
    }
    *greater1_state = usize::from(state);
    if first_greater1 != NO_GREATER1 {
        let base = if luma { 0 } else { ctx::GREATER2_CHROMA };
        magnitude[first_greater1] += u32::from(e.bin(ctx.at(ctx::GREATER2, base + set)));
    }
    let signed = found.len() - usize::from(hidden);
    let signs = e.bypass_bits(signed as u32);
    if first_greater1 != NO_GREATER1 || found.len() > MAX_GREATER1 {
        let mut rice = 0;
        for (i, m) in magnitude[..found.len()].iter_mut().enumerate() {
            // baseLevel at which a remainder follows (§7.3.8.11).
            let threshold = if i >= MAX_GREATER1 {
                1
            } else if i == first_greater1 {
                3
            } else {
                2
            };
            if *m == threshold {
                *m += remaining::<T>(e, rice)?;
                if *m > 3 << rice {
                    rice = (rice + 1).min(MAX_RICE);
                }
            }
        }
    }
    let mut sum = 0u32;
    for (i, &index) in found.iter().enumerate() {
        let m = magnitude[i];
        if m > MAX_MAGNITUDE {
            return Err(DecodeError::Malformed("coefficient magnitude"));
        }
        sum += m;
        let negative = if i < signed {
            (signs >> (signed - 1 - i)) & 1
        } else {
            // The hidden sign: odd sum of magnitudes means negative.
            sum & 1
        };
        // Two's complement negation by mask: no branch on the sign.
        let mask = negative.wrapping_neg() as i32;
        let level = (m as i32 ^ mask) - mask;
        if level == MAX_MAGNITUDE as i32 {
            return Err(DecodeError::Malformed("coefficient magnitude"));
        }
        store(index, level);
    }
    Ok(())
}
