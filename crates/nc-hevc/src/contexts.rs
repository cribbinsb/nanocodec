//! The CABAC context variables used by I and P slices, with their initial
//! values (HEVC §9.3.2.2, Tables 9-5 to 9-37). `cabac_init_flag` is never
//! set, so I slices use `initType` 0 and P slices `initType` 1.
//!
//! Initial values as tabulated in kvazaar's `context.c` (BSD-3-Clause, see
//! `THIRD_PARTY.md`), which follows the specification.

use crate::cabac::Context;

/// Slice type, which selects the initialisation table (`initType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceKind {
    /// I slice: `initType` 0.
    Intra,
    /// P slice: `initType` 1.
    Predicted,
}

/// Initial values per `initType` (index 0: I slice, 1: P slice).
struct InitTable {
    sao_merge_flag: u8,
    sao_type_idx: [u8; 2],
    split_cu_flag: [[u8; 3]; 2],
    cu_skip_flag: [u8; 3],
    pred_mode_flag: u8,
    part_mode: [u8; 2],
    prev_intra_luma_pred_flag: [u8; 2],
    intra_chroma_pred_mode: [u8; 2],
    merge_flag: u8,
    merge_idx: u8,
    mvp_flag: u8,
    rqt_root_cbf: u8,
    cu_qp_delta_abs: u8,
    abs_mvd_greater0: u8,
    abs_mvd_greater1: u8,
    transform_split: [[u8; 3]; 2],
    cbf_luma: [[u8; 2]; 2],
    cbf_chroma: [[u8; 4]; 2],
    last_sig_coeff_prefix: [[u8; 18]; 2],
    coded_sub_block_flag: [[u8; 4]; 2],
    sig_coeff_flag: [[u8; 42]; 2],
    greater1_flag: [[u8; 24]; 2],
    greater2_flag: [[u8; 6]; 2],
}

/// Values of syntax elements that exist only in P slices are listed once
/// (their `initType` 1 values).
#[rustfmt::skip]
const INIT: InitTable = InitTable {
    sao_merge_flag: 153,
    sao_type_idx: [200, 185],
    split_cu_flag: [[139, 141, 157], [107, 139, 126]],
    cu_skip_flag: [197, 185, 201],
    pred_mode_flag: 149,
    part_mode: [184, 154],
    prev_intra_luma_pred_flag: [184, 154],
    intra_chroma_pred_mode: [63, 152],
    merge_flag: 110,
    merge_idx: 122,
    mvp_flag: 168,
    rqt_root_cbf: 79,
    // Table 9-24: the same value for both contexts and every initType.
    cu_qp_delta_abs: 154,
    abs_mvd_greater0: 140,
    abs_mvd_greater1: 198,
    transform_split: [[153, 138, 138], [124, 138, 94]],
    cbf_luma: [[111, 141], [153, 111]],
    cbf_chroma: [[94, 138, 182, 154], [149, 107, 167, 154]],
    // 15 luma contexts, then 3 chroma contexts.
    last_sig_coeff_prefix: [
        [110, 110, 124, 125, 140, 153, 125, 127, 140, 109, 111, 143, 127, 111, 79, 108, 123, 63],
        [125, 110, 94, 110, 95, 79, 125, 111, 110, 78, 110, 111, 111, 95, 94, 108, 123, 108],
    ],
    // 2 luma, then 2 chroma.
    coded_sub_block_flag: [[91, 171, 134, 141], [121, 140, 61, 154]],
    // 27 luma, then 15 chroma.
    sig_coeff_flag: [
        [111, 111, 125, 110, 110, 94, 124, 108, 124, 107, 125, 141, 179, 153, 125, 107, 125, 141,
         179, 153, 125, 107, 125, 141, 179, 153, 125, 140, 139, 182, 182, 152, 136, 152, 136, 153,
         136, 139, 111, 136, 139, 111],
        [155, 154, 139, 153, 139, 123, 123, 63, 153, 166, 183, 140, 136, 153, 154, 166, 183, 140,
         136, 153, 154, 166, 183, 140, 136, 153, 154, 170, 153, 123, 123, 107, 121, 107, 121, 167,
         151, 183, 140, 151, 183, 140],
    ],
    // 16 luma, then 8 chroma.
    greater1_flag: [
        [140, 92, 137, 138, 140, 152, 138, 139, 153, 74, 149, 92, 139, 107, 122, 152, 140, 179,
         166, 182, 140, 227, 122, 197],
        [154, 196, 196, 167, 154, 152, 167, 182, 182, 134, 149, 136, 153, 121, 136, 137, 169, 194,
         166, 167, 154, 167, 137, 182],
    ],
    // 4 luma, then 2 chroma.
    greater2_flag: [[138, 153, 136, 167, 152, 152], [107, 167, 91, 122, 107, 167]],
};

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

/// Every context variable of an I or P slice segment.
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
    /// `mvp_l0_flag`.
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
        let t = match kind {
            SliceKind::Intra => 0,
            SliceKind::Predicted => 1,
        };
        let one = |v: u8| Context::new(v, slice_qp);
        Self {
            sao_merge_flag: one(INIT.sao_merge_flag),
            sao_type_idx: one(INIT.sao_type_idx[t]),
            split_cu_flag: init(&INIT.split_cu_flag[t], slice_qp),
            cu_skip_flag: init(&INIT.cu_skip_flag, slice_qp),
            pred_mode_flag: one(INIT.pred_mode_flag),
            part_mode: one(INIT.part_mode[t]),
            prev_intra_luma_pred_flag: one(INIT.prev_intra_luma_pred_flag[t]),
            intra_chroma_pred_mode: one(INIT.intra_chroma_pred_mode[t]),
            merge_flag: one(INIT.merge_flag),
            merge_idx: one(INIT.merge_idx),
            mvp_flag: one(INIT.mvp_flag),
            rqt_root_cbf: one(INIT.rqt_root_cbf),
            cu_qp_delta_abs: [one(INIT.cu_qp_delta_abs); 2],
            abs_mvd_greater0: one(INIT.abs_mvd_greater0),
            abs_mvd_greater1: one(INIT.abs_mvd_greater1),
            transform_split: init(&INIT.transform_split[t], slice_qp),
            cbf_luma: init(&INIT.cbf_luma[t], slice_qp),
            cbf_chroma: init(&INIT.cbf_chroma[t], slice_qp),
            last_x_prefix: init(&INIT.last_sig_coeff_prefix[t], slice_qp),
            last_y_prefix: init(&INIT.last_sig_coeff_prefix[t], slice_qp),
            coded_sub_block_flag: init(&INIT.coded_sub_block_flag[t], slice_qp),
            sig_coeff_flag: init(&INIT.sig_coeff_flag[t], slice_qp),
            greater1_flag: init(&INIT.greater1_flag[t], slice_qp),
            greater2_flag: init(&INIT.greater2_flag[t], slice_qp),
        }
    }
}

/// Initialise a table of contexts.
fn init<const N: usize>(values: &[u8; N], slice_qp: i32) -> [Context; N] {
    values.map(|v| Context::new(v, slice_qp))
}
