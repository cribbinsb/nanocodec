//! Dependent quantisation (H.266 §7.3.11.11 `QState`, §8.7.3): two scalar
//! quantisers, Q0 reconstructing at even multiples of half the QP + 1 step
//! and Q1 at odd ones (and zero), chosen per coefficient by a four-state
//! machine driven by the parity of the levels coded before it. The encoder
//! picks the levels with a Viterbi search over the four states along the
//! reverse scan, as `VVenC`'s `DepQuant` (BSD-3-Clause-Clear,
//! `THIRD_PARTY.md`), reimplemented with a simpler rate model:
//!
//! - every position's contexts and Rice parameter come from the template of
//!   the nearest Q0 levels (not of each survivor's own levels), and only
//!   the significance context set follows the state;
//! - the coded-group flags and the context-coded bin budget are not
//!   priced; the block's last position is searched (a path may start at
//!   any position, priced by its coordinates).
//!
//! The levels are coded exactly (`coeff::walk`); only the search's choice
//! is approximate.
use crate::coeff::{
    TEMPLATE, last_coordinate_bits, next_state, pass_one_magnitude, remainder_bits, rice_parameter,
    template_contexts,
};
use crate::coeff_contexts::CoeffContexts;
use nc_dsp::VvcQuantizer;

/// Quantiser states (§7.3.11.11).
const STATES: usize = 4;
/// [`Step::from`] of a path's first (last coded) position.
const UNSTARTED: u8 = STATES as u8;
/// Magnitude the first pass's flags imply at most (`abs_remainder` above).
const PASS_ONE_LIMIT: u32 = 4;
/// The QP whose quantisation step is one sample (`2^((qp − 4) / 6)`).
const QSTEP_UNIT_QP: f32 = 4.;
/// Largest transform block.
const MAX_SAMPLES: usize = 32 * 32;
/// Largest level: its reconstruction index `2k` (`2k − 1` in states 2 and 3)
/// must stay within `CoeffMax` = 32767 (§7.3.11.11, §7.4.11.11). A full-scale
/// 64x64 DC at QP 0 would otherwise reach `k` ≈ 23263, which a decoder rejects.
const MAX_LEVEL: u16 = i16::MAX as u16 / 2;

/// What the search of one block needs besides its coefficients.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DepQuant<'a> {
    pub(crate) contexts: &'a CoeffContexts,
    /// Channel: 0 luma, 1 chroma.
    pub(crate) ch: usize,
    /// λ in sample SSE per bit.
    pub(crate) lambda: f32,
}

/// A survivor's step into a state: the state it came from and its level.
#[derive(Debug, Clone, Copy, Default)]
struct Step {
    from: u8,
    /// At most [`MAX_LEVEL`].
    level: u16,
}

/// The reconstruction of level `k` in quantiser state `state`, in steps of
/// QP + 1 (§8.7.3: index 2k, or 2k − 1 in states 2 and 3, halved).
fn reconstruction(state: u8, k: u32) -> f32 {
    if k == 0 {
        0.
    } else if state < 2 {
        k as f32
    } else {
        k as f32 - 0.5
    }
}

/// The candidate levels of a coefficient `magnitude` steps (of QP + 1)
/// large in quantiser state `state`: zero and the two levels around it,
/// at most [`MAX_LEVEL`].
fn candidates(state: u8, magnitude: f32) -> [u16; 3] {
    // A float-to-integer `as` saturates; the clamp below caps it anyway.
    let below = if state < 2 {
        magnitude.floor()
    } else {
        (magnitude + 0.5).floor()
    } as u16;
    let below = below.clamp(1, MAX_LEVEL - 1);
    [0, below, below + 1]
}

/// Levels of `coefficients` (raster order: the coded region of a transform
/// block of `sides`, `transform_tree::coded_samples`) chosen by the
/// four-state search into `levels`; `quantizer` is the block's at `qp`.
/// Returns whether any level is nonzero.
#[allow(clippy::too_many_arguments, reason = "one block's complete inputs")]
pub(crate) fn quantize(
    search: &DepQuant<'_>,
    coefficients: &[i16],
    quantizer: &VvcQuantizer,
    qp: u32,
    levels: &mut [i32],
    sides: (usize, usize),
) -> bool {
    let coded = crate::transform_tree::MAX_CODED_SIDE;
    let (width, height) = (sides.0.min(coded), sides.1.min(coded));
    let samples = width * height;
    let scan = crate::scan::rect_positions(width, height);
    // Magnitudes in steps of QP + 1, 2^(1/6) larger than QP's.
    let shrink = 2f32.powf(-1. / 6.);
    let magnitude = |p: usize| quantizer.magnitude_in_steps(coefficients[p]) as f32 * shrink;
    // The nearest Q0 levels: the context template and the last position.
    // Rounded magnitudes stay below 2^15 (32767-sample coefficients at
    // the finest step); `as` saturates in any case.
    let mut nearest = [0u16; MAX_SAMPLES];
    for (p, level) in nearest[..samples].iter_mut().enumerate() {
        *level = (magnitude(p) + 0.5) as u16;
    }
    levels[..samples].fill(0);
    let Some(reach) = scan.iter().rposition(|&p| nearest[p] != 0) else {
        return false;
    };
    let step = 2f32.powf((qp as f32 + 1. - QSTEP_UNIT_QP) / 6.);
    let weight = step * step;
    let contexts = search.contexts;
    let ch = search.ch;
    // The states, and a path that has not started (every level so far
    // zero): it may make any position the block's last.
    let mut cost = [f32::INFINITY; STATES];
    let mut unstarted = 0f32;
    let mut steps = vec![[Step::default(); STATES]; reach + 1];
    for sp in (0..=reach).rev() {
        let pos = scan[sp];
        let m = magnitude(pos);
        let (pass_one, count, whole, diag) = template(&nearest, pos, (width, height));
        let (sig_ctx, level_ctx) = template_contexts(pass_one, count, diag, ch);
        let rice = rice_parameter(whole, PASS_ONE_LIMIT);
        let level_bits = |k: u32, ctx: usize| {
            let mut bits = contexts.gt1[ch][ctx].bits(k > 1) + 1.; // and the sign
            if k > 1 {
                bits +=
                    contexts.parity[ch][ctx].bits(k & 1 != 0) + contexts.gt2[ch][ctx].bits(k > 3);
            }
            if k >= PASS_ONE_LIMIT {
                bits += remainder_bits((k - PASS_ONE_LIMIT) >> 1, rice);
            }
            bits
        };
        let mut next = [f32::INFINITY; STATES];
        let mut offer = |to: usize, total: f32, step: Step, next: &mut [f32; STATES]| {
            if total < next[to] {
                next[to] = total;
                steps[sp][to] = step;
            }
        };
        // Start here: the last position (its significance inferred, the
        // level contexts' first set), from state 0.
        let (x, y) = (pos % width, pos / width);
        let last_bits = last_coordinate_bits(&contexts.last_x[ch], x, sides.0, ch)
            + last_coordinate_bits(&contexts.last_y[ch], y, sides.1, ch);
        for k in candidates(0, m).into_iter().filter(|&k| k > 0) {
            let (level, k) = (k, u32::from(k));
            let error = m - reconstruction(0, k);
            let total =
                unstarted + error * error * weight + search.lambda * (last_bits + level_bits(k, 0));
            let step = Step {
                from: UNSTARTED,
                level,
            };
            offer(usize::from(next_state(0, k)), total, step, &mut next);
        }
        for state in 0..STATES as u8 {
            let from = cost[usize::from(state)];
            if from.is_infinite() {
                continue;
            }
            let sig = contexts.sig_context(ch, state, sig_ctx);
            for k in candidates(state, m) {
                let (level, k) = (k, u32::from(k));
                let error = m - reconstruction(state, k);
                let bits = if k == 0 {
                    sig.bits(false)
                } else {
                    sig.bits(true) + level_bits(k, level_ctx)
                };
                let total = from + error * error * weight + search.lambda * bits;
                let step = Step { from: state, level };
                offer(usize::from(next_state(state, k)), total, step, &mut next);
            }
        }
        unstarted += m * m * weight;
        cost = next;
    }
    backtrack(&steps, &cost, unstarted, scan, coefficients, levels)
}

/// The levels of the cheapest path (`cost` per final state, `steps` per
/// position) into `levels`, signed as `coefficients`; false when the
/// all-zero block (`unstarted`) is cheaper.
fn backtrack(
    steps: &[[Step; STATES]],
    cost: &[f32; STATES],
    unstarted: f32,
    scan: &[usize],
    coefficients: &[i16],
    levels: &mut [i32],
) -> bool {
    let Some(mut state) = (0..STATES)
        .filter(|&s| cost[s] < unstarted)
        .min_by(|&a, &b| cost[a].total_cmp(&cost[b]))
        .map(|s| s as u8)
    else {
        return false;
    };
    for (sp, row) in steps.iter().enumerate() {
        let Step { from, level } = row[usize::from(state)];
        let pos = scan[sp];
        let sign = if coefficients[pos] < 0 { -1 } else { 1 };
        levels[pos] = i32::from(level) * sign;
        if from == UNSTARTED {
            return true;
        }
        state = from;
    }
    unreachable!("every path starts at a last position")
}

/// The context template of position `pos` (§9.3.4.2.8) over the nearest
/// levels: first-pass magnitude sum, nonzero count, whole sum, diagonal.
fn template(
    nearest: &[u16],
    pos: usize,
    (width, height): (usize, usize),
) -> (u32, u32, u32, usize) {
    let (x, y) = (pos % width, pos / width);
    let (mut pass_one, mut count, mut whole) = (0, 0, 0);
    for (dx, dy) in TEMPLATE {
        if x + dx < width && y + dy < height {
            let a = u32::from(nearest[(y + dy) * width + x + dx]);
            pass_one += pass_one_magnitude(a);
            count += u32::from(a != 0);
            whole += a;
        }
    }
    (pass_one, count, whole, x + y)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A full-scale DC (the forward transform's 8-bit peak, 32640) of a
    /// 64x64 block at QP 0 must still code an index `2k − (QState > 1)`
    /// within `CoeffMax` (§7.4.11.11), which the decoder enforces.
    #[test]
    fn full_scale_dc_at_qp_zero_stays_within_coeff_max() {
        let contexts = CoeffContexts::new(0, false);
        let search = DepQuant {
            contexts: &contexts,
            ch: 0,
            lambda: 0.,
        };
        let sides = (64, 64);
        let quantizer = VvcQuantizer::rect(0, 6, 6);
        let mut coefficients = [0i16; MAX_SAMPLES];
        coefficients[0] = 32640;
        let mut levels = [0i32; MAX_SAMPLES];
        assert!(quantize(
            &search,
            &coefficients,
            &quantizer,
            0,
            &mut levels,
            sides
        ));
        assert!(
            i16::try_from(2 * levels[0].unsigned_abs()).is_ok(),
            "level {}",
            levels[0]
        );
    }
}
