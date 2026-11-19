//! Bits of the `sao()` syntax, whose binarisation HEVC (§7.3.8.3, §9.3.3,
//! Table 9-4) and VVC (H.266 §7.3.11.3, Table 127) share:
//! - `sao_merge_left_flag`, `sao_merge_up_flag`: one context-coded bin each
//!   (one shared context);
//! - `sao_type_idx_luma` / `_chroma`: truncated rice, cMax 2; the first bin
//!   (on / off) context-coded (one shared context), the second (edge or
//!   band) bypass;
//! - four `sao_offset_abs`: truncated unary, cMax 7, bypass;
//! - band offset: a bypass sign per non-zero offset and the 5-bit
//!   `sao_band_position`; edge offset: the 2-bit `sao_eo_class` (luma, and
//!   once for Cb and Cr).
//!
//! Only the two context-coded bins depend on the coder state; the codec
//! supplies their models ([`SaoContexts`]). They are used by SAO syntax
//! alone, so following the decisions CTB by CTB gives exactly the states
//! the coder will have when it writes them.

use super::{ComponentSao, CtbSao, SaoKind, Signal, offsets::magnitude_bits};

/// Bits of `sao_band_position` (fixed length, bypass).
const BAND_POSITION_BITS: u32 = 5;
/// Bits of `sao_eo_class_luma` / `sao_eo_class_chroma` (fixed length).
const EO_CLASS_BITS: u32 = 2;
/// The bypass bin of `sao_type_idx` (edge or band).
const TYPE_SECOND_BIN: u32 = 1;

/// The codec's two SAO context models: their cost estimates and their
/// adaptation (HEVC §9.3.4.3.2, H.266 §9.3.4.3.2).
pub trait SaoContexts: Clone {
    /// Estimated bits of a merge flag `bin`.
    fn merge_bits(&self, bin: bool) -> f32;
    /// Estimated bits of the first `sao_type_idx` bin (`bin`: SAO on).
    fn type_bits(&self, bin: bool) -> f32;
    /// Adapt the merge context to a coded `bin`.
    fn code_merge(&mut self, bin: bool);
    /// Adapt the type context to a coded first bin `bin`.
    fn code_type(&mut self, bin: bool);
}

/// Code the merge flags of `signal` for a CTB with a left and/or upper
/// neighbour (`neighbours`) into `ctx`, returning their bits.
pub(super) fn code_merges<C: SaoContexts>(
    ctx: &mut C,
    signal: Signal,
    (left, up): (bool, bool),
) -> f64 {
    let mut bits = 0.0;
    let mut flag = |ctx: &mut C, bin: bool| {
        bits += f64::from(ctx.merge_bits(bin));
        ctx.code_merge(bin);
    };
    if left {
        flag(ctx, signal == Signal::MergeLeft);
        if signal == Signal::MergeLeft {
            return bits;
        }
    }
    if up {
        flag(ctx, signal == Signal::MergeUp);
    }
    bits
}

/// Bypass bits of `params` after the first `sao_type_idx` bin, for a
/// component of `planes` planes (1 luma, 2 chroma).
pub(super) fn offset_bits(params: &ComponentSao, planes: usize) -> f64 {
    let band = matches!(params.kind, SaoKind::Band(_));
    let per_plane = |offsets: &[i8; 4]| -> u32 {
        let magnitudes: u32 = offsets
            .iter()
            .map(|&o| magnitude_bits(i32::from(o), band))
            .sum();
        magnitudes + if band { BAND_POSITION_BITS } else { 0 }
    };
    let planes: u32 = params.offsets[..planes].iter().map(per_plane).sum();
    let class = if band { 0 } else { EO_CLASS_BITS };
    f64::from(TYPE_SECOND_BIN + planes + class)
}

/// Code a CTB's whole `sao()` syntax into `contexts` (the coder adapts the
/// same way when it writes it), returning its bits. `channels` are the
/// slice's luma and chroma SAO flags.
pub fn commit<C: SaoContexts>(
    contexts: &mut C,
    params: &CtbSao,
    signal: Signal,
    neighbours: (bool, bool),
    channels: [bool; 2],
) -> f64 {
    let mut bits = code_merges(contexts, signal, neighbours);
    if signal != Signal::Explicit {
        return bits;
    }
    for (channel, planes) in [(0, 1), (1, 2)] {
        if !channels[channel] {
            continue;
        }
        let component = params.channel(channel);
        bits += f64::from(contexts.type_bits(component.is_some()));
        contexts.code_type(component.is_some());
        if let Some(component) = component {
            bits += offset_bits(&component, planes);
        }
    }
    bits
}
