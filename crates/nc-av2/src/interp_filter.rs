//! Interpolation filter signalling: the frame's filter mode (AVM v1.0.0
//! `read_frame_interp_filter`) and, in a switchable frame, each inter
//! block's `switchable_interp` symbol (`read_mb_interp_filter`), coded right
//! after the block's motion vector, with its neighbour context
//! (`av2_get_pred_context_switchable_interp`).
//!
//! Of the subset's inter blocks, `GLOBALMV` ones code no symbol and take
//! `EIGHTTAP_REGULAR` (`av2_is_interp_needed` exempts non-translational
//! global motion, which includes the identity at 8×8 and above;
//! `set_default_interp_filters` with `av2_unswitchable_filter`). The other
//! exemptions (skip mode, `WARPMV`, optical-flow and refined-MV compound
//! modes, warped motion, TIP references) are outside the subset. With a
//! single reference (`num_total_refs = 1`) and AVM's one filter for both
//! directions, the context offset is always zero, so only contexts 0..3
//! occur.
use crate::cdf_models::TileCdfs;
use crate::entropy::{Reader, Writer};
use crate::mv_refs::InterMode;
use crate::prediction::InterpFilter;
use nc_common::decode::{DecodeError, Result};
use nc_encoder::symbol_cost::symbol_bits;

/// How a frame's inter blocks get their interpolation filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameFilter {
    /// Every block uses this filter; blocks code none.
    Fixed(InterpFilter),
    /// Each block codes one of [`InterpFilter::SWITCHABLE`] (AVM `SWITCHABLE`).
    Switchable,
}

/// Whether a block coded with `mode` in a switchable frame codes its filter
/// (module docs); otherwise its filter is [`UNCODED`].
pub(crate) fn is_coded(mode: InterMode) -> bool {
    mode != InterMode::Global
}

/// The filter of a block that codes none in a switchable frame.
pub(crate) const UNCODED: InterpFilter = InterpFilter::Regular;

/// Symbols of the switchable alphabet (AVM `SWITCHABLE_FILTERS`).
const SYMBOLS: usize = InterpFilter::SWITCHABLE.len();
/// `switchable_interp` contexts of single-reference blocks.
pub(crate) const CONTEXTS: usize = SYMBOLS + 1;
/// The neighbour filter type of a missing, intra or other-reference
/// neighbour, and the context of disagreeing neighbours (AVM
/// `SWITCHABLE_FILTERS` used as a filter type).
const NO_FILTER: usize = SYMBOLS;

/// The `switchable_interp` context from the filters of a block's first two
/// spatial neighbours (AVM `xd->neighbors`): `None` for a missing or intra
/// neighbour. Agreeing neighbours, or one alone, give their filter; two
/// different filters give [`NO_FILTER`].
pub(crate) fn context(neighbours: [Option<InterpFilter>; 2]) -> usize {
    let [first, second] = neighbours.map(|filter| filter.map_or(NO_FILTER, symbol));
    if first == second || second == NO_FILTER {
        first
    } else if first == NO_FILTER {
        second
    } else {
        NO_FILTER
    }
}

/// The symbol of a switchable filter.
fn symbol(filter: InterpFilter) -> usize {
    InterpFilter::SWITCHABLE
        .iter()
        .position(|&f| f == filter)
        .expect("only switchable filters are coded per block")
}

/// The per-block filter symbol's fixed probabilities, used when the tile
/// does not adapt (AVM `default_switchable_interp_cdf`, which does not
/// depend on the quantizer).
#[derive(Debug, Clone, Copy)]
pub(crate) struct FilterSymbols {
    fixed: [[u16; SYMBOLS]; CONTEXTS],
}

/// Any qindex: the filter defaults are not quantizer-dependent.
const ANY_QINDEX: u8 = 0;

impl FilterSymbols {
    pub fn new() -> Self {
        let defaults = TileCdfs::new(ANY_QINDEX).switchable_interp;
        Self {
            fixed: defaults.map(|cdf| *cdf.inverse()),
        }
    }

    /// Code `filter` (one of [`InterpFilter::SWITCHABLE`]) in `context`.
    /// # Errors
    /// Propagates entropy-coder errors.
    pub fn write(&self, writer: &mut Writer, context: usize, filter: InterpFilter) -> Result<()> {
        writer.model(symbol(filter), &self.fixed[context], |m| {
            &mut m.switchable_interp[context]
        })
    }

    /// Read a block's filter in `context`.
    /// # Errors
    /// Rejects truncated or malformed data.
    pub fn read(&self, reader: &mut Reader<'_>, context: usize) -> Result<InterpFilter> {
        let value = reader.model(&self.fixed[context], |m| &mut m.switchable_interp[context])?;
        InterpFilter::SWITCHABLE
            .get(value)
            .copied()
            .ok_or(DecodeError::Malformed("AV2 interpolation filter symbol"))
    }

    /// Approximate bits of each switchable filter in `context` under the
    /// writer's current probabilities.
    pub fn bits(&self, writer: &Writer, context: usize) -> [f32; SYMBOLS] {
        let inverse = writer.current(&self.fixed[context], |m| &m.switchable_interp[context]);
        std::array::from_fn(|s| symbol_bits(&inverse, s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contexts_follow_the_neighbour_rule() {
        let (r, s, h) = (
            Some(InterpFilter::Regular),
            Some(InterpFilter::Smooth),
            Some(InterpFilter::Sharp),
        );
        assert_eq!(context([None, None]), NO_FILTER);
        assert_eq!(context([s, None]), 1);
        assert_eq!(context([None, h]), 2);
        assert_eq!(context([r, r]), 0);
        assert_eq!(context([s, s]), 1);
        assert_eq!(context([r, h]), NO_FILTER);
    }

    #[test]
    fn symbols_round_trip_in_every_context() {
        let symbols = FilterSymbols::new();
        for adaptive in [false, true] {
            let mut writer = Writer::new();
            if adaptive {
                writer.enable_adaptation(100);
            }
            let sequence: Vec<(usize, InterpFilter)> = (0..40)
                .map(|i| {
                    (
                        i % CONTEXTS,
                        InterpFilter::SWITCHABLE[(i * 7 / 3) % SYMBOLS],
                    )
                })
                .collect();
            for &(context, filter) in &sequence {
                symbols.write(&mut writer, context, filter).unwrap();
            }
            let data = writer.finish();
            let mut reader = Reader::new(&data).unwrap();
            if adaptive {
                reader.enable_adaptation(100);
            }
            for &(context, filter) in &sequence {
                assert_eq!(symbols.read(&mut reader, context).unwrap(), filter);
            }
        }
    }
}
