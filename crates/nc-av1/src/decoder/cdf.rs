//! The decoder's CDF set (AV1 §7.20 `init_non_coeff_cdfs`,
//! `init_coeff_cdfs`, `load_cdfs`, `save_cdfs`): every CDF the block layer
//! reads, as a tile starts with it and a reference slot keeps it. Unlike
//! the encoder's [`crate::cdf::CdfContext`] it holds the coefficient CDFs
//! of every transform size.
use super::coefficients::CoefficientCdfs;
use crate::cdf::ResetCounters;
use crate::decoder_range::{Cdf, DecoderDefaults};
use crate::mvcoding::DecoderMvCdfs;
use crate::tables::{
    ANGLE_DELTA_CDF, CFL_ALPHA_CDF, CFL_SIGN_CDF, COMP_BWDREF_CDF, COMP_GROUP_IDX_CDF,
    COMP_INTER_CDF, COMP_REF_CDF, COMP_REF_TYPE_CDF, COMPOUND_IDX_CDF, COMPOUND_TYPE_CDF,
    DELTA_Q_CDF, DRL_CDF, FILTER_INTRA_CDF, FILTER_INTRA_MODE_CDF, INTER_COMPOUND_MODE_CDF,
    INTERINTRA_CDF, INTERINTRA_MODE_CDF, INTRA_INTER_CDF, INTRABC_CDF, KF_Y_MODE_CDF,
    MOTION_MODE_CDF, NEWMV_CDF, OBMC_CDF, PALETTE_UV_COLOR_INDEX_CDF, PALETTE_UV_MODE_CDF,
    PALETTE_UV_SIZE_CDF, PALETTE_Y_COLOR_INDEX_CDF, PALETTE_Y_MODE_CDF, PALETTE_Y_SIZE_CDF,
    PARTITION_CDF, REFMV_CDF, SEGMENT_PRED_CDF, SGRPROJ_RESTORE_CDF, SINGLE_REF_CDF, SKIP_CDF,
    SKIP_MODE_CDF, SPATIAL_PRED_SEG_TREE_CDF, SWITCHABLE_INTERP_CDF, SWITCHABLE_RESTORE_CDF,
    TX_SIZE_CDF, TXFM_PARTITION_CDF, UNI_COMP_REF_CDF, UV_MODE_CDF, WEDGE_IDX_CDF,
    WEDGE_INTERINTRA_CDF, WIENER_RESTORE_CDF, Y_MODE_CDF, ZEROMV_CDF,
};

/// Every CDF of a tile.
#[derive(Debug, Clone)]
pub(super) struct Cdfs {
    /// `partition`, by size and neighbour context.
    pub partition: [Cdf<10>; 20],
    /// `skip`, by neighbour context.
    pub skip: [Cdf<2>; 3],
    /// `intra_frame_y_mode`, by above and left mode context.
    pub kf_y_mode: [[Cdf<13>; 5]; 5],
    /// `angle_delta_y` and `angle_delta_uv`, by directional mode.
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
    /// `single_ref_p1` .. `single_ref_p6`, by neighbour context.
    pub single_ref: [[Cdf<2>; 6]; 3],
    /// `skip_mode`, by neighbour context.
    pub skip_mode: [Cdf<2>; 3],
    /// `comp_mode` and `comp_ref_type`, by neighbour context.
    pub comp_mode: [Cdf<2>; 5],
    pub comp_ref_type: [Cdf<2>; 5],
    /// `uni_comp_ref`, `uni_comp_ref_p1`, `uni_comp_ref_p2`, by neighbour
    /// context.
    pub uni_comp_ref: [[Cdf<2>; 3]; 3],
    /// `comp_ref`, `comp_ref_p1`, `comp_ref_p2`, by neighbour context.
    pub comp_ref: [[Cdf<2>; 3]; 3],
    /// `comp_bwdref`, `comp_bwdref_p1`, by neighbour context.
    pub comp_bwdref: [[Cdf<2>; 2]; 3],
    /// `compound_mode`, by its context.
    pub compound_mode: [Cdf<8>; 8],
    /// `comp_group_idx` and `compound_idx`, by neighbour context.
    pub comp_group_idx: [Cdf<2>; 6],
    pub compound_idx: [Cdf<2>; 6],
    /// `compound_type` (wedge or difference-weighted), by block size.
    pub compound_type: [Cdf<2>; 22],
    /// `new_mv`, by `NewMvContext`.
    pub new_mv: [Cdf<2>; 6],
    /// `zero_mv`, by `ZeroMvContext`.
    pub zero_mv: [Cdf<2>; 2],
    /// `ref_mv`, by `RefMvContext`.
    pub ref_mv: [Cdf<2>; 6],
    /// `drl_mode`, by `DrlCtxStack` context.
    pub drl: [Cdf<2>; 3],
    /// `txfm_split`, by `txfm_partition_context`.
    pub txfm_split: [Cdf<2>; 21],
    /// `tx_depth`, by size category and neighbour context.
    pub tx_depth: [[Cdf<3>; 3]; 4],
    /// Motion-vector difference CDFs (`MvCtx` 0).
    pub mv: DecoderMvCdfs,
    /// `use_wiener`, `use_sgrproj` and `restoration_type` of restoration
    /// units.
    pub use_wiener: Cdf<2>,
    pub use_sgrproj: Cdf<2>,
    pub restoration_type: Cdf<3>,
    /// `interp_filter`, by filter context.
    pub interp_filter: [Cdf<3>; 16],
    /// `interintra` and `interintra_mode`, by size group.
    pub interintra: [Cdf<2>; 4],
    pub interintra_mode: [Cdf<4>; 4],
    /// `wedge_interintra` and `wedge_index`, by block size.
    pub wedge_interintra: [Cdf<2>; 22],
    pub wedge_index: [Cdf<16>; 22],
    /// `use_obmc` and `motion_mode`, by block size.
    pub obmc: [Cdf<2>; 22],
    pub motion_mode: [Cdf<3>; 22],
    /// `delta_q_abs`.
    pub delta_q: Cdf<4>,
    /// `segment_id` coded against its spatial prediction, by context.
    pub segment_id: [Cdf<8>; 3],
    /// `seg_id_predicted`, by the above and left blocks' flags.
    pub segment_predicted: [Cdf<2>; 3],
    /// `has_palette_y`, by block-size and neighbour context;
    /// `has_palette_uv`, by whether luma has a palette.
    pub palette_y_mode: [[Cdf<2>; 3]; 7],
    pub palette_uv_mode: [Cdf<2>; 2],
    /// `palette_size_y_minus_2`, `palette_size_uv_minus_2`, by block-size
    /// context.
    pub palette_y_size: [Cdf<7>; 7],
    pub palette_uv_size: [Cdf<7>; 7],
    /// `palette_color_idx_y`, `palette_color_idx_uv`, by palette size and
    /// colour context.
    pub palette_y_color_index: [[Cdf<8>; 5]; 7],
    pub palette_uv_color_index: [[Cdf<8>; 5]; 7],
    /// `use_intrabc`.
    pub intrabc: Cdf<2>,
    /// Intra block copy vector differences (`MvCtx` `MV_INTRABC_CONTEXT`).
    pub dv: DecoderMvCdfs,
    /// Transform-type and coefficient CDFs.
    pub coefficients: CoefficientCdfs,
}

impl Cdfs {
    /// The defaults of a frame whose `base_q_idx` selects coefficient set
    /// `q_ctx`.
    pub(super) fn new(q_ctx: usize) -> Self {
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
            single_ref: SINGLE_REF_CDF.cdfs(),
            skip_mode: SKIP_MODE_CDF.cdfs(),
            comp_mode: COMP_INTER_CDF.cdfs(),
            comp_ref_type: COMP_REF_TYPE_CDF.cdfs(),
            uni_comp_ref: UNI_COMP_REF_CDF.cdfs(),
            comp_ref: COMP_REF_CDF.cdfs(),
            comp_bwdref: COMP_BWDREF_CDF.cdfs(),
            compound_mode: INTER_COMPOUND_MODE_CDF.cdfs(),
            comp_group_idx: COMP_GROUP_IDX_CDF.cdfs(),
            compound_idx: COMPOUND_IDX_CDF.cdfs(),
            compound_type: COMPOUND_TYPE_CDF.cdfs(),
            new_mv: NEWMV_CDF.cdfs(),
            zero_mv: ZEROMV_CDF.cdfs(),
            ref_mv: REFMV_CDF.cdfs(),
            drl: DRL_CDF.cdfs(),
            txfm_split: TXFM_PARTITION_CDF.cdfs(),
            tx_depth: TX_SIZE_CDF.cdfs(),
            mv: DecoderMvCdfs::default(),
            use_wiener: Cdf::new(WIENER_RESTORE_CDF),
            use_sgrproj: Cdf::new(SGRPROJ_RESTORE_CDF),
            restoration_type: Cdf::new(SWITCHABLE_RESTORE_CDF),
            interp_filter: SWITCHABLE_INTERP_CDF.cdfs(),
            interintra: INTERINTRA_CDF.cdfs(),
            interintra_mode: INTERINTRA_MODE_CDF.cdfs(),
            wedge_interintra: WEDGE_INTERINTRA_CDF.cdfs(),
            wedge_index: WEDGE_IDX_CDF.cdfs(),
            obmc: OBMC_CDF.cdfs(),
            motion_mode: MOTION_MODE_CDF.cdfs(),
            delta_q: Cdf::new(DELTA_Q_CDF),
            segment_id: SPATIAL_PRED_SEG_TREE_CDF.cdfs(),
            segment_predicted: SEGMENT_PRED_CDF.cdfs(),
            palette_y_mode: PALETTE_Y_MODE_CDF.cdfs(),
            palette_uv_mode: PALETTE_UV_MODE_CDF.cdfs(),
            palette_y_size: PALETTE_Y_SIZE_CDF.cdfs(),
            palette_uv_size: PALETTE_UV_SIZE_CDF.cdfs(),
            palette_y_color_index: PALETTE_Y_COLOR_INDEX_CDF.cdfs(),
            palette_uv_color_index: PALETTE_UV_COLOR_INDEX_CDF.cdfs(),
            intrabc: Cdf::new(INTRABC_CDF),
            dv: DecoderMvCdfs::default(),
            coefficients: CoefficientCdfs::new(q_ctx),
        }
    }

    /// The set as a reference slot keeps it after the frame-end update
    /// (§7.20 `save_cdfs`): the final CDFs (of `context_update_tile_id`)
    /// with every counter cleared.
    pub(super) fn saved(&self) -> Self {
        let mut saved = self.clone();
        saved.reset_counters();
        saved
    }
}

impl ResetCounters for Cdfs {
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
        self.single_ref.reset_counters();
        self.skip_mode.reset_counters();
        self.comp_mode.reset_counters();
        self.comp_ref_type.reset_counters();
        self.uni_comp_ref.reset_counters();
        self.comp_ref.reset_counters();
        self.comp_bwdref.reset_counters();
        self.compound_mode.reset_counters();
        self.comp_group_idx.reset_counters();
        self.compound_idx.reset_counters();
        self.compound_type.reset_counters();
        self.new_mv.reset_counters();
        self.zero_mv.reset_counters();
        self.ref_mv.reset_counters();
        self.drl.reset_counters();
        self.txfm_split.reset_counters();
        self.tx_depth.reset_counters();
        self.mv.reset_counters();
        self.use_wiener.reset_counters();
        self.use_sgrproj.reset_counters();
        self.restoration_type.reset_counters();
        self.interp_filter.reset_counters();
        self.interintra.reset_counters();
        self.interintra_mode.reset_counters();
        self.wedge_interintra.reset_counters();
        self.wedge_index.reset_counters();
        self.obmc.reset_counters();
        self.motion_mode.reset_counters();
        self.delta_q.reset_counters();
        self.segment_id.reset_counters();
        self.segment_predicted.reset_counters();
        self.palette_y_mode.reset_counters();
        self.palette_uv_mode.reset_counters();
        self.palette_y_size.reset_counters();
        self.palette_uv_size.reset_counters();
        self.palette_y_color_index.reset_counters();
        self.palette_uv_color_index.reset_counters();
        self.intrabc.reset_counters();
        self.dv.reset_counters();
        self.coefficients.reset_counters();
    }
}
