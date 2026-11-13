//! Mutable per-tile AV2 models for optional within-tile adaptation.
//!
//! Models preserve AVM identities even for equal defaults. Reset for each
//! tile/frame while cross-frame context is disabled. RD trials price and adapt
//! a scratch copy of these models (`entropy::Writer` estimates), so rejected
//! trials never touch live state. Chroma DC signs in the current subset are bypass:
//! the separate sign plane defaults are retained for explicit future use.
//! TCQ's second-quantizer base models are a separate [`TcqCdfs`], built
//! only by TCQ frames: kept among the hot models they cost the default
//! encoder about 1% Jetson cycles.
use crate::entropy::Cdf;
#[path = "cdf_defaults.rs"]
mod defaults;
#[derive(Clone, Copy)]
struct Initial<const N: usize> {
    inverse: [u16; N],
    rates: [u8; 3],
}
impl<const N: usize> Initial<N> {
    fn build(self) -> Cdf<N> {
        Cdf::new(self.inverse, self.rates).expect("independently verified AVM defaults")
    }
}
/// The coefficient CDF quantizer category of a frame base qindex (AVM `get_q_ctx`).
fn category(qindex: u8) -> usize {
    if qindex <= 90 {
        0
    } else if qindex <= 140 {
        1
    } else if qindex <= 190 {
        2
    } else {
        3
    }
}
/// Probability models for the coded AV2 subset (coefficients TX4..TX64, modes, motion).
/// Axes are named in field documentation; there is no pointer/value deduplication.
/// A clone is one ~8 KiB block copy (plus TCQ's models where built).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileCdfs {
    /// AVM `av2_default_txb_skip_cdfs[q][a][b][c]` (q selected at construction; a/b are array axes).
    pub skip: [[[Cdf<2>; 10]; 5]; 2],
    /// AVM `av2_default_v_txb_skip_cdfs[q][a]` (q selected at construction; a/b are array axes).
    pub v_skip: [Cdf<2>; 12],
    /// AVM `av2_default_eob_multi16_cdfs[q][a]` (q selected at construction; a/b are array axes).
    pub eob16: [Cdf<5>; 3],
    /// AVM `av2_default_eob_multi64_cdfs[q][a]` (q selected at construction; a/b are array axes).
    pub eob64: [Cdf<7>; 3],
    /// AVM `av2_default_eob_multi256_cdfs[q][a]` (q selected at construction; a/b are array axes).
    pub eob256: [Cdf<8>; 3],
    /// AVM `av2_default_eob_multi1024_cdfs[q][a]` (q selected at construction; a/b are array axes).
    pub eob1024: [Cdf<8>; 3],
    /// AVM `av2_default_eob_extra_cdfs[q]` (q selected at construction; a/b are array axes).
    pub extra: Cdf<2>,
    /// AVM `av2_default_dc_sign_cdfs[q][a][0][b]` (q selected at construction; a/b are array axes).
    pub sign: [[Cdf<2>; 3]; 2],
    /// AVM `av2_default_dc_sign_cdfs[q][0][1][a]` (q selected at construction; a/b are array axes).
    pub sign_hidden: [Cdf<2>; 3],
    /// AVM `av2_default_coeff_base_ph_cdfs[q][a]` (q selected at construction; a/b are array axes).
    pub base_ph: [Cdf<4>; 5],
    /// AVM `av2_default_coeff_base_eob_multi_cdfs[q][a][b]` (q selected at construction; a/b are array axes).
    pub base_eob: [[Cdf<3>; 4]; 5],
    /// AVM `av2_default_coeff_base_lf_eob_multi_cdfs[q][a][b]` (q selected at construction; a/b are array axes).
    pub base_lf_eob: [[Cdf<5>; 4]; 5],
    /// AVM `av2_default_coeff_base_eob_multi_uv_cdfs[q][a]` (q selected at construction; a/b are array axes).
    pub uv_base_eob: [Cdf<3>; 4],
    /// AVM `av2_default_coeff_base_lf_eob_multi_uv_cdfs[q][a]` (q selected at construction; a/b are array axes).
    pub uv_base_lf_eob: [Cdf<5>; 4],
    /// AVM `av2_default_coeff_base_multi_cdfs[q][a][b][0]` (q selected at construction; a/b are array axes).
    pub base: [[Cdf<4>; 20]; 5],
    /// AVM `av2_default_coeff_base_lf_multi_cdfs[q][a][b][0]` (q selected at construction; a/b are array axes).
    pub base_lf: [[Cdf<6>; 33]; 5],
    /// AVM `av2_default_coeff_base_multi_uv_cdfs[q][a]` (q selected at construction; a/b are array axes).
    pub uv_base: [Cdf<4>; 12],
    /// AVM `av2_default_coeff_base_lf_multi_uv_cdfs[q][a]` (q selected at construction; a/b are array axes).
    pub uv_base_lf: [Cdf<6>; 12],
    /// AVM `av2_default_coeff_lps_multi_cdfs[q][a]` (q selected at construction; a/b are array axes).
    pub br: [Cdf<4>; 7],
    /// AVM `av2_default_coeff_lps_lf_multi_cdfs[q][a]` (q selected at construction; a/b are array axes).
    pub br_lf: [Cdf<4>; 14],
    /// AVM `av2_default_coeff_lps_multi_uv_cdfs[q][a]` (q selected at construction; a/b are array axes).
    pub uv_br: [Cdf<4>; 4],
    /// AVM `default_inter_ext_tx_cdf[3][a][b]` (q selected at construction; a/b are array axes).
    pub inter_tx: [[Cdf<2>; 4]; 3],
    /// AVM `default_inter_tx_type_set_cdf[a][b][c]` (q selected at construction; a/b are array axes).
    pub inter_tx_set: [[[Cdf<2>; 4]; 3]; 2],
    /// AVM `default_inter_tx_type_idx_cdf[a][b]` (q selected at construction; a/b are array axes).
    pub inter_tx_idx: [[Cdf<8>; 3]; 2],
    /// AVM `default_inter_tx_type_offset_1_cdf[a]` (q selected at construction; a/b are array axes).
    pub inter_tx_offset1: [Cdf<8>; 3],
    /// AVM `default_inter_tx_type_offset_2_cdf[a]` (q selected at construction; a/b are array axes).
    pub inter_tx_offset2: [Cdf<4>; 3],
    /// AVM `default_intra_ext_tx_cdf[2][a]` (q selected at construction; a/b are array axes).
    pub intra_tx: [Cdf<2>; 4],
    /// AVM `default_intra_ext_tx_cdf[1][a]` (q selected at construction; a/b are array axes).
    pub intra_tx_full: [Cdf<7>; 4],
    /// AVM `default_txfm_do_partition_cdf[0][a][b]` (q selected at construction; a/b are array axes).
    pub tx_split: [[Cdf<2>; 9]; 2],
    /// AVM `default_txfm_4way_partition_type_reduced_cdf[0][a][b]` (q selected at construction; a/b are array axes).
    pub tx_partition: [[Cdf<7>; 14]; 2],
    /// AVM `default_do_split_cdf[0][a]` (q selected at construction; a/b are array axes).
    pub split: [Cdf<2>; 16],
    /// AVM `default_rect_type_cdf[0][a]` (q selected at construction; a/b are array axes).
    pub rect: [Cdf<2>; 28],
    /// AVM `default_y_mode_set_cdf` (q selected at construction; a/b are array axes).
    pub mode_set: Cdf<4>,
    /// AVM `default_y_mode_idx_cdf[a]` (q selected at construction; a/b are array axes).
    pub y_mode: [Cdf<8>; 3],
    /// AVM `default_y_mode_idx_offset_cdf[a]` (q selected at construction; a/b are array axes).
    pub y_mode_offset: [Cdf<6>; 3],
    /// AVM `default_uv_mode_cdf[a]` (q selected at construction; a/b are array axes).
    pub uv_mode: [Cdf<8>; 2],
    /// AVM `default_dpcm_cdf` (q selected at construction; a/b are array axes).
    pub dpcm_y: Cdf<2>,
    /// AVM `default_dpcm_uv_cdf` (q selected at construction; a/b are array axes).
    pub dpcm_uv: Cdf<2>,
    /// AVM `default_lossless_tx_size_cdf[a][b]` (q selected at construction; a/b are array axes).
    pub lossless_size: [[Cdf<2>; 2]; 4],
    /// AVM `default_lossless_inter_tx_type_cdf` (q selected at construction; a/b are array axes).
    pub lossless_inter_tx: Cdf<2>,
    /// AVM `default_skip_txfm_cdfs[a]` (q selected at construction; a/b are array axes).
    pub block_skip: [Cdf<2>; 6],
    /// AVM `default_skip_mode_cdfs[a]` (q selected at construction; a/b are array axes).
    pub skip_mode: [Cdf<2>; 3],
    /// AVM `default_skip_drl_cdf[a]` (q selected at construction; a/b are array axes).
    pub skip_drl: [Cdf<2>; 3],
    /// AVM `default_intra_inter_cdf[a]` (q selected at construction; a/b are array axes).
    pub is_inter: [Cdf<2>; 4],
    /// AVM `default_inter_single_mode_cdf[a]` (q selected at construction; a/b are array axes).
    pub inter_mode: [Cdf<3>; 5],
    /// AVM `default_drl_cdf[a][b]` (q selected at construction; a/b are array axes).
    pub drl: [[Cdf<2>; 5]; 3],
    /// AVM `default_nmv_context.joint_shell_set_cdf` (q selected at construction; a/b are array axes).
    pub mv_set: Cdf<2>,
    /// AVM `default_nmv_context.joint_shell_class_cdf_0[MV_PRECISION_QTR_PEL]` (q selected at construction; a/b are array axes).
    pub mv_class_low: Cdf<8>,
    /// AVM `default_nmv_context.joint_shell_class_cdf_1[MV_PRECISION_QTR_PEL]` (q selected at construction; a/b are array axes).
    pub mv_class_high: Cdf<8>,
    /// AVM `default_nmv_context.shell_offset_low_class_cdf[a]` (q selected at construction; a/b are array axes).
    pub mv_low_offset: [Cdf<2>; 2],
    /// AVM `default_nmv_context.shell_offset_class2_cdf` (q selected at construction; a/b are array axes).
    pub mv_class2_offset: Cdf<2>,
    /// AVM `default_nmv_context.shell_offset_other_class_cdf[0][a]` (q selected at construction; a/b are array axes).
    pub mv_offset: [Cdf<2>; 16],
    /// AVM `default_nmv_context.col_mv_greater_flags_cdf[a]` (q selected at construction; a/b are array axes).
    pub mv_col_greater: [Cdf<2>; 2],
    /// AVM `default_nmv_context.col_mv_index_cdf[a]` (q selected at construction; a/b are array axes).
    pub mv_col_index: [Cdf<2>; 4],
    /// AVM `default_cdef_strength_index0_cdf[a]` (q selected at construction; a/b are array axes).
    pub cdef_index0: [Cdf<2>; 4],
    /// AVM `default_ccso_cdf[a][b]` (q selected at construction; a/b are array axes).
    pub ccso: [[Cdf<2>; 4]; 3],
    /// AVM `default_delta_q_cdf` (q selected at construction; a/b are array axes).
    pub delta_q: Cdf<8>,
    /// AVM `default_switchable_interp_cdf[a]` (q selected at construction; a/b are array axes).
    pub switchable_interp: [Cdf<3>; 4],
    /// AVM `default_cdef_cdf[0]` (q selected at construction; a/b are array axes).
    pub cdef_index3: Cdf<2>,
    /// AVM `default_cdef_cdf[1]` (q selected at construction; a/b are array axes).
    pub cdef_index4: Cdf<3>,
    /// AVM `default_cdef_cdf[2]` (q selected at construction; a/b are array axes).
    pub cdef_index5: Cdf<4>,
    /// AVM `default_cdef_cdf[3]` (q selected at construction; a/b are array axes).
    pub cdef_index6: Cdf<5>,
    /// AVM `default_cdef_cdf[4]` (q selected at construction; a/b are array axes).
    pub cdef_index7: Cdf<6>,
    /// AVM `default_cdef_cdf[5]` (q selected at construction; a/b are array axes).
    pub cdef_index8: Cdf<7>,
    /// TCQ's second-quantizer base models, built on first use ([`Self::tcq`]).
    tcq: Option<Box<TcqCdfs>>,
    /// The quantizer category these models started from, which TCQ's take too.
    category: u8,
}
/// The luma base models of TCQ's second quantizer (states 2, 3, 6 and 7).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TcqCdfs {
    /// AVM `av2_default_coeff_base_multi_cdfs[q][a][b][1]` (q selected at construction; a/b are array axes).
    pub base: [[Cdf<4>; 20]; 5],
    /// AVM `av2_default_coeff_base_lf_multi_cdfs[q][a][b][1]` (q selected at construction; a/b are array axes).
    pub base_lf: [[Cdf<6>; 33]; 5],
}
impl TcqCdfs {
    /// Fresh normative defaults of quantizer category `q`.
    fn new(q: usize) -> Self {
        Self {
            base: defaults::BASE_TCQ[q].map(|value| value.map(Initial::build)),
            base_lf: defaults::BASE_LF_TCQ[q].map(|value| value.map(Initial::build)),
        }
    }
    /// [`TileCdfs::restart_adaptation`] of these models.
    fn restart_adaptation(&mut self) {
        for v0 in &mut self.base {
            for v1 in v0 {
                v1.restart_adaptation();
            }
        }
        for v0 in &mut self.base_lf {
            for v1 in v0 {
                v1.restart_adaptation();
            }
        }
    }
}
impl TileCdfs {
    /// Fresh normative defaults for one tile at the frame base quantizer.
    #[must_use]
    pub fn new(qindex: u8) -> Self {
        let q = category(qindex);
        Self {
            skip: defaults::SKIP[q].map(|value| value.map(|value| value.map(Initial::build))),
            v_skip: defaults::V_SKIP[q].map(Initial::build),
            eob16: defaults::EOB16[q].map(Initial::build),
            eob64: defaults::EOB64[q].map(Initial::build),
            eob256: defaults::EOB256[q].map(Initial::build),
            eob1024: defaults::EOB1024[q].map(Initial::build),
            extra: defaults::EXTRA[q].build(),
            sign: defaults::SIGN[q].map(|value| value.map(Initial::build)),
            sign_hidden: defaults::SIGN_HIDDEN[q].map(Initial::build),
            base_ph: defaults::BASE_PH[q].map(Initial::build),
            base_eob: defaults::BASE_EOB[q].map(|value| value.map(Initial::build)),
            base_lf_eob: defaults::BASE_LF_EOB[q].map(|value| value.map(Initial::build)),
            uv_base_eob: defaults::UV_BASE_EOB[q].map(Initial::build),
            uv_base_lf_eob: defaults::UV_BASE_LF_EOB[q].map(Initial::build),
            base: defaults::BASE[q].map(|value| value.map(Initial::build)),
            base_lf: defaults::BASE_LF[q].map(|value| value.map(Initial::build)),
            uv_base: defaults::UV_BASE[q].map(Initial::build),
            uv_base_lf: defaults::UV_BASE_LF[q].map(Initial::build),
            br: defaults::BR[q].map(Initial::build),
            br_lf: defaults::BR_LF[q].map(Initial::build),
            uv_br: defaults::UV_BR[q].map(Initial::build),
            inter_tx: defaults::INTER_TX.map(|value| value.map(Initial::build)),
            inter_tx_set: defaults::INTER_TX_SET
                .map(|value| value.map(|value| value.map(Initial::build))),
            inter_tx_idx: defaults::INTER_TX_IDX.map(|value| value.map(Initial::build)),
            inter_tx_offset1: defaults::INTER_TX_OFFSET1.map(Initial::build),
            inter_tx_offset2: defaults::INTER_TX_OFFSET2.map(Initial::build),
            intra_tx: defaults::INTRA_TX.map(Initial::build),
            intra_tx_full: defaults::INTRA_TX_FULL.map(Initial::build),
            tx_split: defaults::TX_SPLIT.map(|value| value.map(Initial::build)),
            tx_partition: defaults::TX_PARTITION.map(|value| value.map(Initial::build)),
            split: defaults::SPLIT.map(Initial::build),
            rect: defaults::RECT.map(Initial::build),
            mode_set: defaults::MODE_SET.build(),
            y_mode: defaults::Y_MODE.map(Initial::build),
            y_mode_offset: defaults::Y_MODE_OFFSET.map(Initial::build),
            uv_mode: defaults::UV_MODE.map(Initial::build),
            dpcm_y: defaults::DPCM_Y.build(),
            dpcm_uv: defaults::DPCM_UV.build(),
            lossless_size: defaults::LOSSLESS_SIZE.map(|value| value.map(Initial::build)),
            lossless_inter_tx: defaults::LOSSLESS_INTER_TX.build(),
            block_skip: defaults::BLOCK_SKIP.map(Initial::build),
            skip_mode: defaults::SKIP_MODE.map(Initial::build),
            skip_drl: defaults::SKIP_DRL.map(Initial::build),
            is_inter: defaults::IS_INTER.map(Initial::build),
            inter_mode: defaults::INTER_MODE.map(Initial::build),
            drl: defaults::DRL.map(|value| value.map(Initial::build)),
            mv_set: defaults::MV_SET.build(),
            mv_class_low: defaults::MV_CLASS_LOW.build(),
            mv_class_high: defaults::MV_CLASS_HIGH.build(),
            mv_low_offset: defaults::MV_LOW_OFFSET.map(Initial::build),
            mv_class2_offset: defaults::MV_CLASS2_OFFSET.build(),
            mv_offset: defaults::MV_OFFSET.map(Initial::build),
            mv_col_greater: defaults::MV_COL_GREATER.map(Initial::build),
            mv_col_index: defaults::MV_COL_INDEX.map(Initial::build),
            cdef_index0: defaults::CDEF_INDEX0.map(Initial::build),
            ccso: defaults::CCSO.map(|value| value.map(Initial::build)),
            delta_q: defaults::DELTA_Q.build(),
            switchable_interp: defaults::SWITCHABLE_INTERP.map(Initial::build),
            cdef_index3: defaults::CDEF_INDEX3.build(),
            cdef_index4: defaults::CDEF_INDEX4.build(),
            cdef_index5: defaults::CDEF_INDEX5.build(),
            cdef_index6: defaults::CDEF_INDEX6.build(),
            cdef_index7: defaults::CDEF_INDEX7.build(),
            cdef_index8: defaults::CDEF_INDEX8.build(),
            tcq: None,
            category: q as u8,
        }
    }
    /// TCQ's models, at their defaults (of the category these models started
    /// from, as AVM's frame contexts carry them) until a TCQ frame adapts them.
    pub fn tcq(&mut self) -> &mut TcqCdfs {
        let q = usize::from(self.category);
        self.tcq.get_or_insert_with(|| Box::new(TcqCdfs::new(q)))
    }
    /// TCQ's models where a TCQ frame built them ([`Self::tcq`]).
    #[must_use]
    pub fn built_tcq(&self) -> Option<&TcqCdfs> {
        self.tcq.as_deref()
    }
    /// Move every model back along its adaptation-rate schedule, keeping
    /// its probabilities: AVM `av2_reset_cdf_symbol_counters` on a
    /// frame's final models before later frames inherit them. TCQ's
    /// unbuilt defaults have nothing to move.
    #[allow(clippy::too_many_lines)]
    pub fn restart_adaptation(&mut self) {
        for v0 in &mut self.skip {
            for v1 in v0 {
                for v2 in v1 {
                    v2.restart_adaptation();
                }
            }
        }
        for v0 in &mut self.v_skip {
            v0.restart_adaptation();
        }
        for v0 in &mut self.eob16 {
            v0.restart_adaptation();
        }
        for v0 in &mut self.eob64 {
            v0.restart_adaptation();
        }
        for v0 in &mut self.eob256 {
            v0.restart_adaptation();
        }
        for v0 in &mut self.eob1024 {
            v0.restart_adaptation();
        }
        self.extra.restart_adaptation();
        for v0 in &mut self.sign {
            for v1 in v0 {
                v1.restart_adaptation();
            }
        }
        for v0 in &mut self.sign_hidden {
            v0.restart_adaptation();
        }
        for v0 in &mut self.base_ph {
            v0.restart_adaptation();
        }
        for v0 in &mut self.base_eob {
            for v1 in v0 {
                v1.restart_adaptation();
            }
        }
        for v0 in &mut self.base_lf_eob {
            for v1 in v0 {
                v1.restart_adaptation();
            }
        }
        for v0 in &mut self.uv_base_eob {
            v0.restart_adaptation();
        }
        for v0 in &mut self.uv_base_lf_eob {
            v0.restart_adaptation();
        }
        for v0 in &mut self.base {
            for v1 in v0 {
                v1.restart_adaptation();
            }
        }
        for v0 in &mut self.base_lf {
            for v1 in v0 {
                v1.restart_adaptation();
            }
        }
        for v0 in &mut self.uv_base {
            v0.restart_adaptation();
        }
        for v0 in &mut self.uv_base_lf {
            v0.restart_adaptation();
        }
        for v0 in &mut self.br {
            v0.restart_adaptation();
        }
        for v0 in &mut self.br_lf {
            v0.restart_adaptation();
        }
        for v0 in &mut self.uv_br {
            v0.restart_adaptation();
        }
        for v0 in &mut self.inter_tx {
            for v1 in v0 {
                v1.restart_adaptation();
            }
        }
        for v0 in &mut self.inter_tx_set {
            for v1 in v0 {
                for v2 in v1 {
                    v2.restart_adaptation();
                }
            }
        }
        for v0 in &mut self.inter_tx_idx {
            for v1 in v0 {
                v1.restart_adaptation();
            }
        }
        for v0 in &mut self.inter_tx_offset1 {
            v0.restart_adaptation();
        }
        for v0 in &mut self.inter_tx_offset2 {
            v0.restart_adaptation();
        }
        for v0 in &mut self.intra_tx {
            v0.restart_adaptation();
        }
        for v0 in &mut self.intra_tx_full {
            v0.restart_adaptation();
        }
        for v0 in &mut self.tx_split {
            for v1 in v0 {
                v1.restart_adaptation();
            }
        }
        for v0 in &mut self.tx_partition {
            for v1 in v0 {
                v1.restart_adaptation();
            }
        }
        for v0 in &mut self.split {
            v0.restart_adaptation();
        }
        for v0 in &mut self.rect {
            v0.restart_adaptation();
        }
        self.mode_set.restart_adaptation();
        for v0 in &mut self.y_mode {
            v0.restart_adaptation();
        }
        for v0 in &mut self.y_mode_offset {
            v0.restart_adaptation();
        }
        for v0 in &mut self.uv_mode {
            v0.restart_adaptation();
        }
        self.dpcm_y.restart_adaptation();
        self.dpcm_uv.restart_adaptation();
        for v0 in &mut self.lossless_size {
            for v1 in v0 {
                v1.restart_adaptation();
            }
        }
        self.lossless_inter_tx.restart_adaptation();
        for v0 in &mut self.block_skip {
            v0.restart_adaptation();
        }
        for v0 in &mut self.skip_mode {
            v0.restart_adaptation();
        }
        for v0 in &mut self.skip_drl {
            v0.restart_adaptation();
        }
        for v0 in &mut self.is_inter {
            v0.restart_adaptation();
        }
        for v0 in &mut self.inter_mode {
            v0.restart_adaptation();
        }
        for v0 in &mut self.drl {
            for v1 in v0 {
                v1.restart_adaptation();
            }
        }
        self.mv_set.restart_adaptation();
        self.mv_class_low.restart_adaptation();
        self.mv_class_high.restart_adaptation();
        for v0 in &mut self.mv_low_offset {
            v0.restart_adaptation();
        }
        self.mv_class2_offset.restart_adaptation();
        for v0 in &mut self.mv_offset {
            v0.restart_adaptation();
        }
        for v0 in &mut self.mv_col_greater {
            v0.restart_adaptation();
        }
        for v0 in &mut self.mv_col_index {
            v0.restart_adaptation();
        }
        for v0 in &mut self.cdef_index0 {
            v0.restart_adaptation();
        }
        for v0 in &mut self.ccso {
            for v1 in v0 {
                v1.restart_adaptation();
            }
        }
        self.delta_q.restart_adaptation();
        for v0 in &mut self.switchable_interp {
            v0.restart_adaptation();
        }
        self.cdef_index3.restart_adaptation();
        self.cdef_index4.restart_adaptation();
        self.cdef_index5.restart_adaptation();
        self.cdef_index6.restart_adaptation();
        self.cdef_index7.restart_adaptation();
        self.cdef_index8.restart_adaptation();
        if let Some(tcq) = &mut self.tcq {
            tcq.restart_adaptation();
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::Writer;
    fn trace<const N: usize>(cdf: &mut Cdf<N>, hash: &mut u64) {
        let mut writer = Writer::new();
        for step in 0..65 {
            writer.symbol((step * 7 + step / 3) % N, cdf).unwrap();
            for &value in cdf.inverse() {
                *hash = (*hash ^ u64::from(value)).wrapping_mul(1_099_511_628_211);
            }
        }
    }
    #[test]
    #[allow(clippy::too_many_lines)]
    fn original_c_update_traces_and_reset() {
        for (q, expected) in [0, 91, 141, 191].into_iter().zip(defaults::TRACE_HASHES) {
            let mut models = TileCdfs::new(q);
            let snapshot = models.clone();
            let mut hash = 14_695_981_039_346_656_037;
            for v0 in &mut models.skip {
                for v1 in v0 {
                    for v2 in v1 {
                        trace(v2, &mut hash);
                    }
                }
            }
            for v0 in &mut models.v_skip {
                trace(v0, &mut hash);
            }
            for v0 in &mut models.eob16 {
                trace(v0, &mut hash);
            }
            for v0 in &mut models.eob64 {
                trace(v0, &mut hash);
            }
            for v0 in &mut models.eob256 {
                trace(v0, &mut hash);
            }
            for v0 in &mut models.eob1024 {
                trace(v0, &mut hash);
            }
            trace(&mut models.extra, &mut hash);
            for v0 in &mut models.sign {
                for v1 in v0 {
                    trace(v1, &mut hash);
                }
            }
            for v0 in &mut models.sign_hidden {
                trace(v0, &mut hash);
            }
            for v0 in &mut models.base_ph {
                trace(v0, &mut hash);
            }
            for v0 in &mut models.base_eob {
                for v1 in v0 {
                    trace(v1, &mut hash);
                }
            }
            for v0 in &mut models.base_lf_eob {
                for v1 in v0 {
                    trace(v1, &mut hash);
                }
            }
            for v0 in &mut models.uv_base_eob {
                trace(v0, &mut hash);
            }
            for v0 in &mut models.uv_base_lf_eob {
                trace(v0, &mut hash);
            }
            for v0 in &mut models.base {
                for v1 in v0 {
                    trace(v1, &mut hash);
                }
            }
            for v0 in &mut models.base_lf {
                for v1 in v0 {
                    trace(v1, &mut hash);
                }
            }
            for v0 in &mut models.uv_base {
                trace(v0, &mut hash);
            }
            for v0 in &mut models.uv_base_lf {
                trace(v0, &mut hash);
            }
            for v0 in &mut models.br {
                trace(v0, &mut hash);
            }
            for v0 in &mut models.br_lf {
                trace(v0, &mut hash);
            }
            for v0 in &mut models.uv_br {
                trace(v0, &mut hash);
            }
            for v0 in &mut models.inter_tx {
                for v1 in v0 {
                    trace(v1, &mut hash);
                }
            }
            for v0 in &mut models.inter_tx_set {
                for v1 in v0 {
                    for v2 in v1 {
                        trace(v2, &mut hash);
                    }
                }
            }
            for v0 in &mut models.inter_tx_idx {
                for v1 in v0 {
                    trace(v1, &mut hash);
                }
            }
            for v0 in &mut models.inter_tx_offset1 {
                trace(v0, &mut hash);
            }
            for v0 in &mut models.inter_tx_offset2 {
                trace(v0, &mut hash);
            }
            for v0 in &mut models.intra_tx {
                trace(v0, &mut hash);
            }
            for v0 in &mut models.intra_tx_full {
                trace(v0, &mut hash);
            }
            for v0 in &mut models.tx_split {
                for v1 in v0 {
                    trace(v1, &mut hash);
                }
            }
            for v0 in &mut models.tx_partition {
                for v1 in v0 {
                    trace(v1, &mut hash);
                }
            }
            for v0 in &mut models.split {
                trace(v0, &mut hash);
            }
            for v0 in &mut models.rect {
                trace(v0, &mut hash);
            }
            trace(&mut models.mode_set, &mut hash);
            for v0 in &mut models.y_mode {
                trace(v0, &mut hash);
            }
            for v0 in &mut models.y_mode_offset {
                trace(v0, &mut hash);
            }
            for v0 in &mut models.uv_mode {
                trace(v0, &mut hash);
            }
            trace(&mut models.dpcm_y, &mut hash);
            trace(&mut models.dpcm_uv, &mut hash);
            for v0 in &mut models.lossless_size {
                for v1 in v0 {
                    trace(v1, &mut hash);
                }
            }
            trace(&mut models.lossless_inter_tx, &mut hash);
            for v0 in &mut models.block_skip {
                trace(v0, &mut hash);
            }
            for v0 in &mut models.skip_mode {
                trace(v0, &mut hash);
            }
            for v0 in &mut models.skip_drl {
                trace(v0, &mut hash);
            }
            for v0 in &mut models.is_inter {
                trace(v0, &mut hash);
            }
            for v0 in &mut models.inter_mode {
                trace(v0, &mut hash);
            }
            for v0 in &mut models.drl {
                for v1 in v0 {
                    trace(v1, &mut hash);
                }
            }
            trace(&mut models.mv_set, &mut hash);
            trace(&mut models.mv_class_low, &mut hash);
            trace(&mut models.mv_class_high, &mut hash);
            for v0 in &mut models.mv_low_offset {
                trace(v0, &mut hash);
            }
            trace(&mut models.mv_class2_offset, &mut hash);
            for v0 in &mut models.mv_offset {
                trace(v0, &mut hash);
            }
            for v0 in &mut models.mv_col_greater {
                trace(v0, &mut hash);
            }
            for v0 in &mut models.mv_col_index {
                trace(v0, &mut hash);
            }
            for v0 in &mut models.cdef_index0 {
                trace(v0, &mut hash);
            }
            for v0 in &mut models.ccso {
                for v1 in v0 {
                    trace(v1, &mut hash);
                }
            }
            trace(&mut models.delta_q, &mut hash);
            for v0 in &mut models.switchable_interp {
                trace(v0, &mut hash);
            }
            trace(&mut models.cdef_index3, &mut hash);
            trace(&mut models.cdef_index4, &mut hash);
            trace(&mut models.cdef_index5, &mut hash);
            trace(&mut models.cdef_index6, &mut hash);
            trace(&mut models.cdef_index7, &mut hash);
            trace(&mut models.cdef_index8, &mut hash);
            let tcq = models.tcq();
            for v0 in &mut tcq.base {
                for v1 in v0 {
                    trace(v1, &mut hash);
                }
            }
            for v0 in &mut tcq.base_lf {
                for v1 in v0 {
                    trace(v1, &mut hash);
                }
            }
            assert_eq!(hash, expected);
            assert_ne!(models, snapshot);
            assert_eq!(TileCdfs::new(q), snapshot);
            let mut restarted = models.clone();
            restarted.restart_adaptation();
            assert_ne!(restarted, models);
            assert_eq!(restarted.drl[0][0].inverse(), models.drl[0][0].inverse());
            models = snapshot.clone();
            assert_eq!(models, snapshot);
        }
    }
    #[test]
    fn equal_defaults_have_independent_state() {
        let mut models = TileCdfs::new(100);
        let uv = models.dpcm_uv;
        let sign_uv = models.sign[1];
        Writer::new().symbol(0, &mut models.dpcm_y).unwrap();
        Writer::new().symbol(1, &mut models.sign[0][0]).unwrap();
        assert_eq!(models.dpcm_uv, uv);
        assert_eq!(models.sign[1], sign_uv);
        assert_ne!(models.dpcm_y.inverse(), models.dpcm_uv.inverse());
    }
    #[test]
    fn tcq_models_start_at_their_categorys_defaults() {
        for q in [0, 91, 141, 191] {
            let mut models = TileCdfs::new(q);
            assert!(models.built_tcq().is_none());
            let built = *models.tcq();
            assert_eq!(built, TcqCdfs::new(category(q)));
            assert_eq!(models.built_tcq(), Some(&built));
        }
    }
    #[test]
    fn quantizer_category_boundaries() {
        for (a, b) in [(0, 90), (91, 140), (141, 190), (191, 255)] {
            assert_eq!(TileCdfs::new(a), TileCdfs::new(b));
        }
        for (a, b) in [(90, 91), (140, 141), (190, 191)] {
            assert_ne!(TileCdfs::new(a), TileCdfs::new(b));
        }
    }
}
