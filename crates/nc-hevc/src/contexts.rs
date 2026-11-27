//! The CABAC context variables used by I, P and B slices, with their
//! initial values (HEVC §9.3.2.2, Tables 9-5 to 9-37). `cabac_init_flag` is
//! never set, so I slices use `initType` 0, P slices 1 and B slices 2.
//!
//! The initial values are the decoder's table (`dec/ctx.rs`), one table for
//! both directions.

use crate::cabac::Context;
pub(crate) use crate::dec::ctx::INTER_PRED_IDC_LAST;
use crate::dec::ctx::{self as table, Element};

/// Slice type, which selects the initialisation table (`initType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceKind {
    /// I slice: `initType` 0.
    Intra,
    /// P slice: `initType` 1.
    Predicted,
    /// B slice whose two reference lists hold the same past pictures
    /// (generalised B, low-delay bi-prediction): `initType` 2.
    Bi,
}

impl SliceKind {
    /// Whether the slice codes inter prediction (P or B).
    #[must_use]
    pub fn is_inter(self) -> bool {
        self != Self::Intra
    }

    /// `initType` (§9.3.2.2, eq. 9-7 with `cabac_init_flag` 0).
    fn init_type(self) -> usize {
        match self {
            Self::Intra => 0,
            Self::Predicted => 1,
            Self::Bi => 2,
        }
    }
}

/// Number of luma `sig_coeff_flag` contexts; chroma contexts follow them.
pub const SIG_CHROMA_OFFSET: usize = 27;
/// Number of luma `last_sig_coeff_*_prefix` contexts.
pub const LAST_CHROMA_OFFSET: usize = 15;
/// Number of luma `coeff_abs_level_greater1_flag` contexts.
pub const GREATER1_CHROMA_OFFSET: usize = 16;
/// Number of luma `coeff_abs_level_greater2_flag` contexts.
pub const GREATER2_CHROMA_OFFSET: usize = 4;
/// Number of luma `coded_sub_block_flag` contexts.
pub const CODED_SUB_BLOCK_CHROMA_OFFSET: usize = 2;

/// Every context variable of an I, P or B slice segment.
#[derive(Debug, Clone)]
pub struct Contexts {
    /// `sao_merge_left_flag` and `sao_merge_up_flag` (shared).
    pub sao_merge_flag: Context,
    /// `sao_type_idx_luma` / `sao_type_idx_chroma` bin 0 (shared).
    pub sao_type_idx: Context,
    /// `split_cu_flag`, by `ctxInc` 0..2.
    pub split_cu_flag: [Context; 3],
    /// `cu_skip_flag`, by `ctxInc` 0..2 (P slices).
    pub cu_skip_flag: [Context; 3],
    /// `pred_mode_flag` (P slices).
    pub pred_mode_flag: Context,
    /// `part_mode` bin 0.
    pub part_mode: Context,
    /// `prev_intra_luma_pred_flag`.
    pub prev_intra_luma_pred_flag: Context,
    /// `intra_chroma_pred_mode` bin 0.
    pub intra_chroma_pred_mode: Context,
    /// `merge_flag`.
    pub merge_flag: Context,
    /// `merge_idx` bin 0.
    pub merge_idx: Context,
    /// `inter_pred_idc` (B slices): bin 0 by `CtDepth`, then the second
    /// bin's context ([`INTER_PRED_IDC_LAST`]).
    pub inter_pred_idc: [Context; INTER_PRED_IDC_CONTEXTS],
    /// `ref_idx_l0` / `ref_idx_l1` bin 0.
    pub ref_idx: Context,
    /// `mvp_l0_flag` / `mvp_l1_flag`.
    pub mvp_flag: Context,
    /// `rqt_root_cbf`.
    pub rqt_root_cbf: Context,
    /// `cu_qp_delta_abs` prefix bins, by `ctxInc` (bin 0, bins 1 to 4).
    pub cu_qp_delta_abs: [Context; 2],
    /// `abs_mvd_greater0_flag` (both components).
    pub abs_mvd_greater0: Context,
    /// `abs_mvd_greater1_flag` (both components).
    pub abs_mvd_greater1: Context,
    /// `split_transform_flag`, by `5 - log2TrafoSize`.
    pub transform_split: [Context; 3],
    /// `cbf_luma`, by `ctxInc` (`trafoDepth == 0`).
    pub cbf_luma: [Context; 2],
    /// `cbf_cb` and `cbf_cr` (shared), by `trafoDepth`.
    pub cbf_chroma: [Context; 4],
    /// `last_sig_coeff_x_prefix`.
    pub last_x_prefix: [Context; 18],
    /// `last_sig_coeff_y_prefix`.
    pub last_y_prefix: [Context; 18],
    /// `coded_sub_block_flag`.
    pub coded_sub_block_flag: [Context; 4],
    /// `sig_coeff_flag`.
    pub sig_coeff_flag: [Context; 42],
    /// `coeff_abs_level_greater1_flag`.
    pub greater1_flag: [Context; 24],
    /// `coeff_abs_level_greater2_flag`.
    pub greater2_flag: [Context; 6],
}

impl Contexts {
    /// All contexts initialised for a slice of `kind` at `slice_qp`.
    #[must_use]
    pub fn for_slice(kind: SliceKind, slice_qp: i32) -> Self {
        let t = kind.init_type();
        // Elements absent from I slices start there from their P-slice
        // values: nothing codes them, but key-frame CUs are priced as P ones
        // (`backend/coded_snapshot.rs`) and these states are what that
        // pricing reads.
        let inter = match kind {
            SliceKind::Intra => SliceKind::Predicted.init_type(),
            SliceKind::Predicted | SliceKind::Bi => t,
        };
        let one = |e: Element| Context::new(table::initial_values(e, t)[0], slice_qp);
        let inter_one = |e: Element| Context::new(table::initial_values(e, inter)[0], slice_qp);
        Self {
            sao_merge_flag: one(table::SAO_MERGE),
            sao_type_idx: one(table::SAO_TYPE),
            split_cu_flag: init(table::initial_values(table::SPLIT_CU, t), slice_qp),
            cu_skip_flag: init(table::initial_values(table::SKIP, inter), slice_qp),
            pred_mode_flag: inter_one(table::PRED_MODE),
            part_mode: one(table::PART_MODE),
            prev_intra_luma_pred_flag: one(table::PREV_INTRA_LUMA),
            intra_chroma_pred_mode: one(table::CHROMA_PRED),
            merge_flag: inter_one(table::MERGE_FLAG),
            merge_idx: inter_one(table::MERGE_IDX),
            inter_pred_idc: init(
                table::initial_values(table::INTER_PRED_IDC, inter),
                slice_qp,
            ),
            ref_idx: inter_one(table::REF_IDX),
            mvp_flag: inter_one(table::MVP_FLAG),
            rqt_root_cbf: inter_one(table::RQT_ROOT_CBF),
            cu_qp_delta_abs: init(table::initial_values(table::CU_QP_DELTA, t), slice_qp),
            abs_mvd_greater0: inter_one(table::ABS_MVD_GT0),
            abs_mvd_greater1: inter_one(table::ABS_MVD_GT1),
            transform_split: init(table::initial_values(table::SPLIT_TRANSFORM, t), slice_qp),
            cbf_luma: init(table::initial_values(table::CBF_LUMA, t), slice_qp),
            cbf_chroma: init(table::initial_values(table::CBF_CHROMA, t), slice_qp),
            last_x_prefix: init(table::initial_values(table::LAST_X, t), slice_qp),
            last_y_prefix: init(table::initial_values(table::LAST_Y, t), slice_qp),
            coded_sub_block_flag: init(table::initial_values(table::CODED_SUB_BLOCK, t), slice_qp),
            sig_coeff_flag: init(table::initial_values(table::SIG_COEFF, t), slice_qp),
            greater1_flag: init(table::initial_values(table::GREATER1, t), slice_qp),
            greater2_flag: init(table::initial_values(table::GREATER2, t), slice_qp),
        }
    }
}

/// `inter_pred_idc` contexts: four `CtDepth` contexts and the second bin's.
pub const INTER_PRED_IDC_CONTEXTS: usize = 5;

/// The first `N` contexts of an element's initial values (Main 4:2:0 uses
/// fewer `cbf_cb`/`cbf_cr` contexts than the table lists).
fn init<const N: usize>(values: &[u8], slice_qp: i32) -> [Context; N] {
    std::array::from_fn(|i| Context::new(values[i], slice_qp))
}
