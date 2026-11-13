//! AV1 constant tables.
//!
//! The large tables (default CDFs, quantiser lookups, scans, coefficient
//! context offsets) are generated from libaom by
//! `tools/codegen/gen_av1_tables.py`; small structural tables live here.

pub mod generated;

pub use generated::*;

/// Number of coefficient CDF sets, selected by `base_q_idx` (AV1 §7.20,
/// `get_qctx` in the spec's `init_coeff_cdfs`).
const Q_CTX_THRESHOLDS: [u32; 3] = [20, 60, 120];

/// Which of the four default coefficient-CDF sets a frame starts from.
#[must_use]
pub fn coeff_cdf_q_ctx(base_q_idx: u32) -> usize {
    Q_CTX_THRESHOLDS
        .iter()
        .take_while(|&&t| base_q_idx > t)
        .count()
}

/// Partition types (AV1 §6.10.4), in symbol order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Partition {
    /// No split.
    None = 0,
    /// Two horizontal halves.
    Horz = 1,
    /// Two vertical halves.
    Vert = 2,
    /// Four quadrants.
    Split = 3,
    /// Top half split, bottom whole.
    HorzA = 4,
    /// Top whole, bottom half split.
    HorzB = 5,
    /// Left half split, right whole.
    VertA = 6,
    /// Left whole, right half split.
    VertB = 7,
    /// Four horizontal strips.
    Horz4 = 8,
    /// Four vertical strips.
    Vert4 = 9,
}

/// `DC_PRED` in the luma intra mode alphabet (AV1 §6.10.22).
pub const DC_PRED: usize = 0;
/// `UV_DC_PRED` in the chroma intra mode alphabet.
pub const UV_DC_PRED: usize = 0;
/// Luma intra mode symbols.
pub const INTRA_MODES: usize = 13;
/// Chroma intra mode symbols when `CfL` is allowed (adds `UV_CFL_PRED`).
pub const UV_INTRA_MODES_CFL_ALLOWED: usize = 14;
/// `CFL_ALLOWED` index into the chroma mode CDFs.
pub const CFL_ALLOWED: usize = 1;
/// `CFL_NOT_ALLOWED` index (blocks larger than 32x32).
pub const CFL_NOT_ALLOWED: usize = 0;
/// Chroma intra mode symbols without `UV_CFL_PRED`.
pub const UV_INTRA_MODES_CFL_NOT_ALLOWED: usize = 13;
