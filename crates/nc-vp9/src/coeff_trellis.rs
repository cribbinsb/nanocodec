//! The VP9 token rate model of the shared forward trellis
//! ([`nc_encoder::ForwardCoeffRateModel`]) and the trellis quantiser built
//! on it.
//!
//! VP9 codes tokens in scan order (VP9 §9.3; libvpx `vp9_tokenize.c`).
//! Each token's probabilities come from its band and from the energy class
//! of two neighbours earlier in the scan; `more_coefs` (the end-of-block
//! decision) is coded before a token only after a nonzero token or at the
//! start, so a block can end only after a nonzero level. The state the
//! trellis carries is therefore the energy classes of the last
//! [`WINDOW`] scan positions (every scan's neighbours lie within it,
//! tested) and whether `more_coefs` is coded next. It is exact for every
//! token's probabilities; libvpx's own `vp9_optimize_b` prices the
//! neighbours from the unoptimised levels instead.

use nc_encoder::{Candidates, ForwardCoeffRateModel, ForwardTrellis};

use super::{
    COL_SCAN_4X4_INVERSE, COL_SCAN_8X8_INVERSE, COL_SCAN_16X16_INVERSE, DEFAULT_SCAN_4X4_INVERSE,
    DEFAULT_SCAN_8X8_INVERSE, DEFAULT_SCAN_16X16_INVERSE, DEFAULT_SCAN_32X32_INVERSE, Model,
    ROW_SCAN_4X4_INVERSE, ROW_SCAN_8X8_INVERSE, ROW_SCAN_16X16_INVERSE, ScanOrder, band, energy,
    scan_order, write_magnitude,
};
use crate::tables::{self, BandProbs};
use crate::transform::{Steps, TxSize, TxType};
use crate::tree::bit_cost;

/// Scan positions whose energy classes the state keeps: context
/// neighbours reach back at most 87 positions (32x32 default scan; 29 at
/// 16x16, 12 at 8x8, 6 at 4x4).
const WINDOW: usize = 128;
/// Probability of the equiprobable sign bit.
const SIGN_PROBABILITY: u8 = 128;

/// The energy classes of the last [`WINDOW`] scan positions (position `c`
/// in slot `c % WINDOW`), and whether `more_coefs` is coded next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TokenState {
    energies: [u8; WINDOW],
    more_coefs: bool,
}

/// The scan positions of each position's two context neighbours (the
/// scan's neighbour list through its inverse), so a context needs no
/// inverse lookups.
const fn neighbour_scans<const M: usize, const N: usize>(
    neighbors: &[u16; M],
    inverse: &[u16; N],
) -> [u16; M] {
    let mut out = [0; M];
    let mut i = 0;
    while i < M {
        out[i] = inverse[neighbors[i] as usize];
        i += 1;
    }
    out
}

const DEFAULT_4X4_NEIGHBOURS: [u16; 34] = neighbour_scans(
    &tables::DEFAULT_SCAN_4X4_NEIGHBORS,
    &DEFAULT_SCAN_4X4_INVERSE,
);
const ROW_4X4_NEIGHBOURS: [u16; 34] =
    neighbour_scans(&tables::ROW_SCAN_4X4_NEIGHBORS, &ROW_SCAN_4X4_INVERSE);
const COL_4X4_NEIGHBOURS: [u16; 34] =
    neighbour_scans(&tables::COL_SCAN_4X4_NEIGHBORS, &COL_SCAN_4X4_INVERSE);
const DEFAULT_8X8_NEIGHBOURS: [u16; 130] = neighbour_scans(
    &tables::DEFAULT_SCAN_8X8_NEIGHBORS,
    &DEFAULT_SCAN_8X8_INVERSE,
);
const ROW_8X8_NEIGHBOURS: [u16; 130] =
    neighbour_scans(&tables::ROW_SCAN_8X8_NEIGHBORS, &ROW_SCAN_8X8_INVERSE);
const COL_8X8_NEIGHBOURS: [u16; 130] =
    neighbour_scans(&tables::COL_SCAN_8X8_NEIGHBORS, &COL_SCAN_8X8_INVERSE);
const DEFAULT_16X16_NEIGHBOURS: [u16; 514] = neighbour_scans(
    &tables::DEFAULT_SCAN_16X16_NEIGHBORS,
    &DEFAULT_SCAN_16X16_INVERSE,
);
const ROW_16X16_NEIGHBOURS: [u16; 514] =
    neighbour_scans(&tables::ROW_SCAN_16X16_NEIGHBORS, &ROW_SCAN_16X16_INVERSE);
const COL_16X16_NEIGHBOURS: [u16; 514] =
    neighbour_scans(&tables::COL_SCAN_16X16_NEIGHBORS, &COL_SCAN_16X16_INVERSE);
const DEFAULT_32X32_NEIGHBOURS: [u16; 2050] = neighbour_scans(
    &tables::DEFAULT_SCAN_32X32_NEIGHBORS,
    &DEFAULT_SCAN_32X32_INVERSE,
);

/// [`neighbour_scans`] of the scan `scan_order` picks for `tx` and `ty`.
fn neighbour_scan(tx: TxSize, ty: TxType) -> &'static [u16] {
    match (tx, ty) {
        (TxSize::Tx4, TxType::AdstDct) => &ROW_4X4_NEIGHBOURS,
        (TxSize::Tx4, TxType::DctAdst) => &COL_4X4_NEIGHBOURS,
        (TxSize::Tx4, _) => &DEFAULT_4X4_NEIGHBOURS,
        (TxSize::Tx8, TxType::AdstDct) => &ROW_8X8_NEIGHBOURS,
        (TxSize::Tx8, TxType::DctAdst) => &COL_8X8_NEIGHBOURS,
        (TxSize::Tx8, _) => &DEFAULT_8X8_NEIGHBOURS,
        (TxSize::Tx16, TxType::AdstDct) => &ROW_16X16_NEIGHBOURS,
        (TxSize::Tx16, TxType::DctAdst) => &COL_16X16_NEIGHBOURS,
        (TxSize::Tx16, _) => &DEFAULT_16X16_NEIGHBOURS,
        (TxSize::Tx32, _) => &DEFAULT_32X32_NEIGHBOURS,
    }
}

/// Prices one transform block's tokens.
pub(crate) struct RateModel<'a> {
    probs: &'a BandProbs,
    tx: TxSize,
    order: ScanOrder,
    /// Scan positions of each position's context neighbours.
    neighbours: &'static [u16],
    /// Context of the first token, from the above and left blocks.
    initial: usize,
}

impl<'a> RateModel<'a> {
    pub(crate) fn new(model: Model, probs: &'a BandProbs, initial: usize) -> Self {
        Self {
            probs,
            tx: model.tx,
            order: scan_order(model.tx, model.ty),
            neighbours: neighbour_scan(model.tx, model.ty),
            initial,
        }
    }

    /// The probabilities of the token at scan position `c` (band and
    /// neighbour context, VP9 §9.3.1).
    fn probs_at(&self, state: &TokenState, c: usize) -> &[u8; 3] {
        let ctx = if c == 0 {
            self.initial
        } else {
            let energy = |k: usize| state.energies[usize::from(self.neighbours[k]) % WINDOW];
            let (a, b) = (energy(2 * c), energy(2 * c + 1));
            usize::from((1 + a + b) >> 1)
        };
        &self.probs[band(self.tx, c)][ctx]
    }
}

/// Bits of a magnitude above one: the Pareto tail tokens and their extra
/// bits (`write_magnitude`).
fn magnitude_bits(magnitude: i32, pivot: u8) -> f32 {
    let mut bits = 0.0;
    write_magnitude(
        &mut |bit, prob| bits += bit_cost(bit, prob),
        magnitude,
        pivot,
    );
    bits
}

impl ForwardCoeffRateModel for RateModel<'_> {
    type State = TokenState;

    fn initial_state(&self) -> TokenState {
        TokenState {
            energies: [0; WINDOW],
            more_coefs: true,
        }
    }

    /// The EOB token (`more_coefs` = 0) where it is coded; nothing after
    /// the last scan position.
    fn end_rate(&mut self, state: &TokenState, scan_pos: usize) -> Option<f32> {
        if scan_pos == self.order.scan.len() {
            return Some(0.0);
        }
        state
            .more_coefs
            .then(|| bit_cost(false, self.probs_at(state, scan_pos)[0]))
    }

    fn level_rates(
        &mut self,
        state: &TokenState,
        scan_pos: usize,
        levels: &[u32],
        bits: &mut [f32],
    ) {
        let probs = self.probs_at(state, scan_pos);
        let more = if state.more_coefs {
            bit_cost(true, probs[0])
        } else {
            0.0
        };
        for (&level, bits) in levels.iter().zip(bits) {
            *bits = more
                + if level == 0 {
                    bit_cost(false, probs[1])
                } else {
                    let tail = if level > 1 {
                        magnitude_bits(level as i32, probs[2])
                    } else {
                        0.0
                    };
                    bit_cost(true, probs[1])
                        + bit_cost(level > 1, probs[2])
                        + tail
                        + bit_cost(false, SIGN_PROBABILITY)
                };
        }
    }

    fn advance(&self, state: &mut TokenState, scan_pos: usize, level: u32) {
        state.energies[scan_pos % WINDOW] = energy(level as i32);
        state.more_coefs = level != 0;
    }
}

/// Trellis quantisation of VP9 transform blocks, with reusable buffers.
#[derive(Debug, Default)]
pub(crate) struct TrellisQuantizer {
    search: ForwardTrellis<TokenState>,
    candidates: Vec<Candidates>,
    chosen: Vec<u32>,
}

impl TrellisQuantizer {
    /// Choose the levels of a block from its raster `coefficients` by
    /// squared error plus `lambda` (sample SSE per bit) times `model`'s
    /// rate, keeping at most `max_states` survivors per scan position. On
    /// entry `levels` holds the plain quantiser's raster levels (rounding
    /// below half a step), some nonzero; the search spans their end of
    /// block, with candidates around each level rounded to the nearest.
    /// Returns whether any level is left.
    pub(crate) fn quantize(
        &mut self,
        model: &mut RateModel<'_>,
        coefficients: &[i32],
        steps: Steps,
        (lambda, max_states): (f32, usize),
        levels: &mut [i32],
    ) -> bool {
        let tx = model.tx;
        let shift = tx.dequant_shift();
        // Coefficients are 8x orthonormal (32x32: 4x, `coefficient_sse`).
        let scale_log2 = if tx == TxSize::Tx32 { 2 } else { 3 };
        let to_sse = 1.0 / f64::from(1_u32 << (2 * scale_log2));
        let eob = model.order.end_of_block(levels);
        let scan = &model.order.scan[..eob];
        self.candidates.clear();
        self.candidates.extend(scan.iter().map(|&p| {
            let p = usize::from(p);
            let step = i64::from(steps.step(p));
            let magnitude = i64::from(coefficients[p].unsigned_abs());
            let plain = levels[p].unsigned_abs();
            let past_half = 2 * (magnitude << shift) >= (2 * i64::from(plain) + 1) * step;
            let nearest = if past_half && (plain as i32) < super::MAX_LEVEL {
                plain + 1
            } else {
                plain
            };
            Candidates::around(nearest, |level| {
                let error = (magnitude - ((i64::from(level) * step) >> shift)) as f64;
                (error * error * to_sse) as f32
            })
        }));
        self.chosen.resize(eob, 0);
        self.search.quantize(
            model,
            &self.candidates,
            lambda,
            max_states,
            &mut self.chosen,
        );
        let mut any = false;
        for (&p, &level) in scan.iter().zip(&self.chosen) {
            let p = usize::from(p);
            let level = level as i32;
            levels[p] = if coefficients[p] < 0 { -level } else { level };
            any |= level != 0;
        }
        any
    }
}

#[cfg(test)]
mod tests {
    use super::super::cost;
    use super::*;
    use crate::coeff::PlaneType;
    use crate::probs::FrameContext;

    /// The state window holds every context neighbour.
    #[test]
    fn neighbours_lie_within_the_window() {
        for tx in [TxSize::Tx4, TxSize::Tx8, TxSize::Tx16, TxSize::Tx32] {
            for ty in [
                TxType::DctDct,
                TxType::AdstDct,
                TxType::DctAdst,
                TxType::AdstAdst,
            ] {
                let order = scan_order(tx, ty);
                for c in 1..order.scan.len() {
                    for k in 0..2 {
                        let n = usize::from(order.inverse[usize::from(order.neighbors[2 * c + k])]);
                        assert!(n < c && c - n < WINDOW, "{tx:?} {ty:?} c {c} neighbour {n}");
                    }
                }
            }
        }
    }

    /// Summed token rates equal the coder's estimate (`coeff::cost`).
    #[test]
    fn rates_add_up_to_the_coded_cost() {
        let context = FrameContext::default();
        let mut seed = 5_u32;
        let mut next = || {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
            seed >> 16
        };
        for tx in [TxSize::Tx4, TxSize::Tx8, TxSize::Tx16, TxSize::Tx32] {
            let model = Model {
                tx,
                ty: TxType::DctDct,
                plane: PlaneType::Y,
                inter: true,
            };
            let probs = &context.coef[tx as usize][0][1];
            let order = scan_order(tx, TxType::DctDct);
            for initial in 0..3 {
                let mut levels = vec![0_i32; tx.area()];
                for &p in &order.scan[..tx.area().min(40)] {
                    let v = match next() % 8 {
                        0..=3 => 0,
                        4 | 5 => 1,
                        6 => 2 + (next() % 70) as i32,
                        _ => 2,
                    };
                    levels[usize::from(p)] = if next() % 2 == 0 { v } else { -v };
                }
                let expected = cost(&levels, model, probs, initial);
                let mut rates = RateModel::new(model, probs, initial);
                let eob = order.end_of_block(&levels);
                let mut state = rates.initial_state();
                let mut bits = 0.0;
                for c in 0..eob {
                    let level = levels[usize::from(order.scan[c])].unsigned_abs();
                    let mut b = [0.0];
                    rates.level_rates(&state, c, &[level], &mut b);
                    bits += b[0];
                    rates.advance(&mut state, c, level);
                }
                bits += rates.end_rate(&state, eob).expect("ends after a nonzero");
                assert!(
                    (bits - expected).abs() < 0.01,
                    "{tx:?}: {bits} vs {expected}"
                );
            }
        }
    }
}
