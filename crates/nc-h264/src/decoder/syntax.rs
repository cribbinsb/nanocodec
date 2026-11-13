//! The macroblock-layer syntax elements (§7.3.5) as one interface over the
//! two entropy coders, CAVLC (`cavlc.rs`) and CABAC (`cabac_syntax.rs`).
//! The macroblock layer (`macroblock.rs`) is written once against it.
//!
//! Both coders select contexts / tables from neighbouring blocks; the
//! methods get the available neighbours ([`Neighbours`]) and the current
//! macroblock's state as decoded so far, and record in it what later
//! blocks read (`total_coeff`, coded flags, motion vector differences).

use super::Result;
use super::mbstate::{MbState, Neighbours};
use super::slice::SliceType;

/// Most coefficients of one residual block (an 8x8 luma block).
pub(super) const MAX_COEFFS: usize = 64;

/// The non-zero coefficient levels of one residual block, by scan
/// position (counted from the block's first coded coefficient).
#[derive(Debug, Clone)]
pub(super) struct Coeffs {
    pub count: usize,
    pub scan: [u8; MAX_COEFFS],
    pub level: [i32; MAX_COEFFS],
}

impl Coeffs {
    pub const fn new() -> Self {
        Self {
            count: 0,
            scan: [0; MAX_COEFFS],
            level: [0; MAX_COEFFS],
        }
    }

    pub fn push(&mut self, scan: usize, level: i32) {
        self.scan[self.count] = scan as u8;
        self.level[self.count] = level;
        self.count += 1;
    }

    pub fn iter(&self) -> impl Iterator<Item = (usize, i32)> + '_ {
        self.scan[..self.count]
            .iter()
            .zip(&self.level[..self.count])
            .map(|(&s, &l)| (usize::from(s), l))
    }
}

/// A residual block of a 4x4 luma or chroma block (§7.3.5.3.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Block4x4 {
    /// `Intra16x16ACLevel` of the block at (`x4`, `y4`): 15 coefficients.
    LumaAc(usize, usize),
    /// `LumaLevel4x4`: 16 coefficients.
    Luma(usize, usize),
    /// `ChromaACLevel` of component `comp` (0 Cb, 1 Cr), block 0..=3.
    ChromaAc(usize, usize),
}

/// The macroblock-layer syntax of one entropy coder.
pub(super) trait MbSyntax {
    /// Bits consumed so far (analysis: differences give a macroblock's
    /// cost; restarts after `I_PCM` under CABAC).
    #[cfg(feature = "analysis")]
    fn bits_consumed(&self) -> usize;
    /// `mb_type`: Table 7-11 numbering in I slices; in P slices Table 7-13
    /// numbering, with intra types as 5 + their Table 7-11 number.
    fn mb_type(&mut self, slice: SliceType, nb: &Neighbours<'_>) -> Result<u32>;
    /// `sub_mb_type` of a P macroblock (Table 7-17).
    fn sub_mb_type(&mut self) -> Result<u8>;
    fn transform_size_8x8_flag(&mut self, nb: &Neighbours<'_>) -> Result<bool>;
    /// `prev_intra4x4_pred_mode_flag` / `rem_intra4x4_pred_mode` (or the
    /// 8x8 ones): `None` when the predicted mode is used.
    fn intra_pred_mode(&mut self) -> Result<Option<u8>>;
    fn intra_chroma_pred_mode(&mut self, nb: &Neighbours<'_>) -> Result<u8>;
    /// `ref_idx_l0` of the partition whose first 4x4 block is (`x4`, `y4`),
    /// with `num_ref_idx_active` references (more than one).
    fn ref_idx(
        &mut self,
        nb: &Neighbours<'_>,
        cur: &MbState,
        x4: usize,
        y4: usize,
        num_ref_idx_active: usize,
    ) -> Result<u8>;
    /// `mvd_l0` of that partition (both components).
    fn mvd(&mut self, nb: &Neighbours<'_>, cur: &MbState, x4: usize, y4: usize)
    -> Result<[i32; 2]>;
    /// `coded_block_pattern` (both parts) of a non-`Intra_16x16` macroblock.
    fn coded_block_pattern(
        &mut self,
        nb: &Neighbours<'_>,
        cur: &MbState,
        monochrome: bool,
    ) -> Result<u8>;
    fn mb_qp_delta(&mut self) -> Result<i32>;
    /// A macroblock without `mb_qp_delta` (CABAC context of the next one).
    fn no_qp_delta(&mut self) {}
    /// `Intra16x16DCLevel`.
    fn luma_dc(&mut self, nb: &Neighbours<'_>, cur: &mut MbState, out: &mut Coeffs) -> Result<()>;
    fn block_4x4(
        &mut self,
        nb: &Neighbours<'_>,
        cur: &mut MbState,
        block: Block4x4,
        out: &mut Coeffs,
    ) -> Result<()>;
    /// `LumaLevel8x8` of 8x8 block `b8` (64 coefficients, zig-zag order).
    fn luma_8x8(
        &mut self,
        nb: &Neighbours<'_>,
        cur: &mut MbState,
        b8: usize,
        out: &mut Coeffs,
    ) -> Result<()>;
    /// `ChromaDCLevel` of component `comp`.
    fn chroma_dc(
        &mut self,
        nb: &Neighbours<'_>,
        cur: &mut MbState,
        comp: usize,
        out: &mut Coeffs,
    ) -> Result<()>;
    /// The `pcm_sample_luma` and `pcm_sample_chroma` bytes.
    fn pcm_samples(&mut self, out: &mut [u8]) -> Result<()>;
}
