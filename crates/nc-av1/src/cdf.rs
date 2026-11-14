//! Adaptive CDFs (AV1 §8.2.6) and the per-frame CDF context.
//!
//! Every symbol is coded with a CDF that adapts towards the symbols seen
//! (`disable_cdf_update = 0`): after each symbol its CDF moves by a rate
//! that starts fast and slows as the CDF's counter grows (libaom
//! `update_cdf`, `aom_dsp/prob.h`, BSD-2-Clause, Alliance for Open Media).
//! [`CdfContext`] owns one copy of every CDF the encoder writes; a frame
//! starts from the defaults (key frames) or from the context its reference
//! saved at the end of its tile (inter frames, `load_cdfs`), see
//! [`crate::backend`].
//!
//! CDFs are stored in libaom's inverse convention (`32768 − cdf`, last
//! symbol's entry 0), as the default tables are.

use crate::coeff::CoeffCdfs;
use crate::mvcoding::MvCdfs;
use crate::tables::{
    ANGLE_DELTA_CDF, CFL_ALPHA_CDF, CFL_SIGN_CDF, COMP_INTER_CDF, COMP_REF_TYPE_CDF, DELTA_Q_CDF,
    DRL_CDF, FILTER_INTRA_CDF, FILTER_INTRA_MODE_CDF, INTER_COMPOUND_MODE_CDF, INTRA_INTER_CDF,
    KF_Y_MODE_CDF, NEWMV_CDF, PARTITION_CDF, REFMV_CDF, SINGLE_REF_CDF, SKIP_CDF,
    SPATIAL_PRED_SEG_TREE_CDF, TX_SIZE_CDF, TXFM_PARTITION_CDF, UNI_COMP_REF_CDF, UV_MODE_CDF,
    WIENER_RESTORE_CDF, Y_MODE_CDF, ZEROMV_CDF,
};
use nc_encoder::symbol_cost::{SymbolCosts, symbol_bits};

/// Probability scale: CDFs are 15-bit fixed point (AV1 §4.10.7).
pub const PROB_TOP: u16 = 1 << 15;
/// The adaptation counter saturates here (AV1 §8.2.6).
const COUNT_MAX: u8 = 32;
/// The counter's two slow-down steps (above 15, and at 32) are its value
/// shifted right by this: the counter never exceeds [`COUNT_MAX`].
const COUNT_STEP_SHIFT: u32 = 4;
/// Adaptation rate before the counter and alphabet-size terms.
const RATE_BASE: u32 = 3;
/// Cap on the alphabet-size term of the rate (`Min(FloorLog2(N), 2)`).
const RATE_ALPHABET_MAX: u32 = 2;

/// One adaptive CDF of up to `N` symbols, with its counter (the spec's
/// extra last array entry) and a cache of its symbol costs.
///
/// Rate estimation prices far more symbols than are coded, so the cost of
/// every symbol is computed once per CDF state ([`SymbolCosts`], shared with
/// the other backends) and dropped whenever the CDF adapts. Equality ignores
/// the cache.
#[derive(Debug, Clone, Copy)]
pub struct Cdf<const N: usize> {
    /// Inverse CDF; entries past the alphabet in use are zero padding.
    pub icdf: [u16; N],
    count: u8,
    costs: SymbolCosts<N>,
}

impl<const N: usize> PartialEq for Cdf<N> {
    fn eq(&self, other: &Self) -> bool {
        self.icdf == other.icdf && self.count == other.count
    }
}

impl<const N: usize> Eq for Cdf<N> {}

impl<const N: usize> Cdf<N> {
    /// A fresh CDF (counter 0) with inverse CDF `icdf`.
    #[must_use]
    pub const fn new(icdf: [u16; N]) -> Self {
        Self {
            icdf,
            count: 0,
            costs: SymbolCosts::new(),
        }
    }

    /// Cost of coding `symbol` with this CDF as it stands, in 1/512 bit
    /// ([`nc_encoder::symbol_cost::COST_UNITS_PER_BIT`]): [`Cdf::bits`]
    /// rounded, computed once per CDF state.
    pub fn cost_q9(&mut self, symbol: usize) -> u32 {
        self.costs.cost_q9(&self.icdf, symbol)
    }

    /// Move the CDF towards `symbol` of an `nsyms`-symbol alphabet (AV1
    /// §8.2.6, the update at the end of `read_symbol`).
    #[inline]
    pub fn adapt(&mut self, symbol: usize, nsyms: usize) {
        adapt(&mut self.icdf, &mut self.count, symbol, nsyms);
        self.costs.invalidate();
    }

    /// Cost in bits of coding `symbol` with this CDF as it stands.
    #[must_use]
    pub fn bits(&self, symbol: usize) -> f32 {
        symbol_bits(&self.icdf, symbol)
    }
}

/// Move inverse CDF `icdf` with adaptation counter `count` towards
/// `symbol` of an `nsyms`-symbol alphabet (AV1 §8.2.6): the update of the
/// encoder's [`Cdf`] and of the decoder's own CDF type.
#[inline]
pub(crate) fn adapt(icdf: &mut [u16], count: &mut u8, symbol: usize, nsyms: usize) {
    let rate = adaptation_rate(*count, nsyms);
    // In the inverse convention the entries below the coded symbol move
    // towards 32768 and the others towards 0.
    for (i, p) in icdf[..nsyms - 1].iter_mut().enumerate() {
        if i < symbol {
            *p += (PROB_TOP - *p) >> rate;
        } else {
            *p -= *p >> rate;
        }
    }
    advance_counter(count);
}

/// The adaptation rate of a CDF with counter `count` over an `nsyms`-symbol
/// alphabet (AV1 §8.2.6: `3 + (count > 15) + (count > 31) + Min(FloorLog2(N), 2)`).
#[inline]
pub(crate) fn adaptation_rate(count: u8, nsyms: usize) -> u32 {
    debug_assert!(count <= COUNT_MAX);
    RATE_BASE + (u32::from(count) >> COUNT_STEP_SHIFT) + nsyms.ilog2().min(RATE_ALPHABET_MAX)
}

/// Count one more adaptation, saturating at the counter's maximum.
#[inline]
pub(crate) fn advance_counter(count: &mut u8) {
    *count += u8::from(*count < COUNT_MAX);
}

/// Clears adaptation counters (`av1_reset_cdf_symbol_counters`), for CDFs
/// and nested arrays of them.
pub trait ResetCounters {
    /// Set every counter to 0.
    fn reset_counters(&mut self);
}

impl<const N: usize> ResetCounters for Cdf<N> {
    fn reset_counters(&mut self) {
        self.count = 0;
    }
}

impl<T: ResetCounters, const M: usize> ResetCounters for [T; M] {
    fn reset_counters(&mut self) {
        self.iter_mut().for_each(ResetCounters::reset_counters);
    }
}

/// Builds adaptive CDFs from a (nested) default table.
pub trait FromDefaults {
    /// The same nesting with each inverse CDF made a [`Cdf`].
    type Cdfs;
    /// Fresh CDFs with these defaults.
    fn cdfs(&self) -> Self::Cdfs;
}

impl<const N: usize> FromDefaults for [u16; N] {
    type Cdfs = Cdf<N>;
    fn cdfs(&self) -> Cdf<N> {
        Cdf::new(*self)
    }
}

impl<T: FromDefaults, const M: usize> FromDefaults for [T; M] {
    type Cdfs = [T::Cdfs; M];
    fn cdfs(&self) -> Self::Cdfs {
        std::array::from_fn(|i| self[i].cdfs())
    }
}

/// Every CDF the encoder codes with (AV1 `init_non_coeff_cdfs` and
/// `init_coeff_cdfs`, restricted to the syntax this encoder writes).
#[derive(Debug, Clone)]
pub struct CdfContext {
    /// `partition`, by size and neighbour context.
    pub partition: [Cdf<10>; 20],
    /// `skip`, by neighbour context.
    pub skip: [Cdf<2>; 3],
    /// `intra_frame_y_mode`, by above and left mode context.
    pub kf_y_mode: [[Cdf<13>; 5]; 5],
    /// `angle_delta_y`, by directional mode.
    pub angle_delta: [Cdf<7>; 8],
    /// `use_filter_intra`, by block size (`MiSize`).
    pub filter_intra: [Cdf<2>; 22],
    /// `filter_intra_mode`.
    pub filter_intra_mode: Cdf<5>,
    /// `y_mode` of intra blocks in inter frames, by size group.
    pub y_mode: [Cdf<13>; 4],
    /// `uv_mode`, by `CfL` allowed and luma mode.
    pub uv_mode: [[Cdf<14>; 13]; 2],
    /// `CfL` joint sign.
    pub cfl_sign: Cdf<8>,
    /// Nonzero `CfL` slope magnitudes, conditioned on both signs.
    pub cfl_alpha: [Cdf<16>; 6],
    /// `is_inter`, by neighbour context.
    pub intra_inter: [Cdf<2>; 4],
    /// `comp_mode`, by neighbour context (`AV1 §5.11.25`).
    pub comp_mode: [Cdf<2>; 5],
    /// `comp_ref_type`, by neighbour context (`AV1 §5.11.25`).
    pub comp_ref_type: [Cdf<2>; 5],
    /// `uni_comp_ref` .. `uni_comp_ref_p2`, by neighbour context (`AV1 §5.11.25`).
    pub uni_comp_ref: [[Cdf<2>; 3]; 3],
    /// `single_ref_p1` .. `single_ref_p6`, by neighbour context.
    pub single_ref: [[Cdf<2>; 6]; 3],
    /// `compound_mode`, by `CompoundModeContext` (`AV1 §5.11.26`).
    pub compound_mode: [Cdf<8>; 8],
    /// `new_mv`, by `NewMvContext`.
    pub new_mv: [Cdf<2>; 6],
    /// `zero_mv`, by `ZeroMvContext`.
    pub zero_mv: [Cdf<2>; 2],
    /// `ref_mv`, by `RefMvContext`.
    pub ref_mv: [Cdf<2>; 6],
    /// `drl_mode`, by `DrlCtxStack` context.
    pub drl: [Cdf<2>; 3],
    /// `txfm_split` of inter transform trees, by `txfm_partition_context`.
    pub txfm_split: [Cdf<2>; 21],
    /// Intra `tx_depth`, by size category and neighbour context.
    pub tx_depth: [[Cdf<3>; 3]; 4],
    /// `delta_q_abs` of superblock quantizer deltas.
    pub delta_q: Cdf<4>,
    /// `segment_id` coded against its spatial prediction, by context.
    pub segment_id: [Cdf<8>; 3],
    /// `use_wiener` of loop restoration units (AV1 §5.11.58).
    pub use_wiener: Cdf<2>,
    /// Motion-vector difference CDFs (`MvCtx` 0).
    pub mv: MvCdfs,
    /// Transform type and coefficient CDFs.
    pub coeff: CoeffCdfs,
}

impl CdfContext {
    /// The default CDFs of a frame whose `base_q_idx` selects coefficient
    /// set `q_ctx` (AV1 §7.20 `init_non_coeff_cdfs`, `init_coeff_cdfs`).
    #[must_use]
    pub fn new(q_ctx: usize) -> Self {
        Self {
            partition: PARTITION_CDF.cdfs(),
            skip: SKIP_CDF.cdfs(),
            kf_y_mode: KF_Y_MODE_CDF.cdfs(),
            angle_delta: ANGLE_DELTA_CDF.cdfs(),
            filter_intra: FILTER_INTRA_CDF.cdfs(),
            filter_intra_mode: Cdf::new(FILTER_INTRA_MODE_CDF),
            y_mode: Y_MODE_CDF.cdfs(),
            uv_mode: UV_MODE_CDF.cdfs(),
            cfl_sign: Cdf::new(CFL_SIGN_CDF),
            cfl_alpha: CFL_ALPHA_CDF.cdfs(),
            intra_inter: INTRA_INTER_CDF.cdfs(),
            comp_mode: COMP_INTER_CDF.cdfs(),
            comp_ref_type: COMP_REF_TYPE_CDF.cdfs(),
            uni_comp_ref: UNI_COMP_REF_CDF.cdfs(),
            single_ref: SINGLE_REF_CDF.cdfs(),
            compound_mode: INTER_COMPOUND_MODE_CDF.cdfs(),
            new_mv: NEWMV_CDF.cdfs(),
            zero_mv: ZEROMV_CDF.cdfs(),
            ref_mv: REFMV_CDF.cdfs(),
            drl: DRL_CDF.cdfs(),
            txfm_split: TXFM_PARTITION_CDF.cdfs(),
            tx_depth: TX_SIZE_CDF.cdfs(),
            delta_q: Cdf::new(DELTA_Q_CDF),
            segment_id: SPATIAL_PRED_SEG_TREE_CDF.cdfs(),
            use_wiener: Cdf::new(WIENER_RESTORE_CDF),
            mv: MvCdfs::default(),
            coeff: CoeffCdfs::new(q_ctx),
        }
    }

    /// The context as a reference frame saves it at the end of its tile
    /// (`disable_frame_end_update_cdf = 0`, one tile so
    /// `context_update_tile_id = 0`): the final CDFs with every counter
    /// cleared (AV1 §7.20 `save_cdfs`; libaom `av1_reset_cdf_symbol_counters`
    /// after the frame-end update).
    #[must_use]
    pub fn saved(&self) -> Self {
        let mut saved = self.clone();
        saved.reset_counters();
        saved
    }
}

impl ResetCounters for CdfContext {
    fn reset_counters(&mut self) {
        self.partition.reset_counters();
        self.skip.reset_counters();
        self.kf_y_mode.reset_counters();
        self.angle_delta.reset_counters();
        self.filter_intra.reset_counters();
        self.filter_intra_mode.reset_counters();
        self.y_mode.reset_counters();
        self.uv_mode.reset_counters();
        self.cfl_sign.reset_counters();
        self.cfl_alpha.reset_counters();
        self.intra_inter.reset_counters();
        self.comp_mode.reset_counters();
        self.comp_ref_type.reset_counters();
        self.uni_comp_ref.reset_counters();
        self.single_ref.reset_counters();
        self.compound_mode.reset_counters();
        self.new_mv.reset_counters();
        self.zero_mv.reset_counters();
        self.ref_mv.reset_counters();
        self.drl.reset_counters();
        self.txfm_split.reset_counters();
        self.tx_depth.reset_counters();
        self.delta_q.reset_counters();
        self.segment_id.reset_counters();
        self.use_wiener.reset_counters();
        self.mv.reset_counters();
        self.coeff.reset_counters();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptation_follows_libaom() {
        // libaom update_cdf on a 4-symbol CDF, coding symbol 2 with count 0:
        // rate = 3 + 0 + 0 + 2 = 5.
        let mut cdf = Cdf::new([24576u16, 16384, 8192, 0]);
        cdf.adapt(2, 4);
        assert_eq!(cdf.icdf, [24576 + 256, 16384 + 512, 8192 - 256, 0]);
    }

    #[test]
    fn repeated_symbol_becomes_cheap_and_counter_saturates() {
        let mut cdf = Cdf::new([16384u16, 0]);
        let before = cdf.bits(1);
        for _ in 0..100 {
            cdf.adapt(1, 2);
        }
        assert!(cdf.bits(1) < before / 4.0);
        assert_eq!(cdf.count, COUNT_MAX);
        cdf.reset_counters();
        assert_eq!(cdf.count, 0);
    }
}
