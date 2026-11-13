//! Intra block copy (AV1 §5.11.7 `use_intrabc`, §5.11.26 `assign_mv`,
//! §7.11.3 prediction from the current frame): a block of an intra frame
//! that allows it predicts from already decoded samples of the same frame,
//! displaced by a whole-sample vector (`Mv[0]`, the "DV") coded against a
//! reference DV from the MV stack of `INTRA_FRAME`. Such blocks are
//! decoded as inter blocks (transform sizes, types, contexts, the motion
//! store their neighbours' stacks read) with these modes and this
//! prediction; the frame's loop filters are off.
//!
//! The reference DV follows libaom `read_intrabc_info`,
//! `av1_find_best_ref_mvs` and `av1_find_ref_dv`
//! (`av1/decoder/decodemv.c`, `av1/common/mvref_common.{c,h}`;
//! BSD-2-Clause, see `THIRD_PARTY.md`).
use super::super::block_info::InterFlags;
use super::super::motion::{INTRA_FRAME, InterMode, NONE_FRAME};
use super::super::mv_stack::MvStack;
use super::inter::{InterModes, MotionMode};
use super::inter_pred::{PlaneReference, gather, window_position};
use super::{BlockPos, FrameDecoder, MI_SIZE};
use crate::backend::Av1Mv;
use crate::decoder_range::SymbolReader;
use crate::mvcoding::MvPrecision;
use nc_common::PlaneIndex;
use nc_common::decode::DecodeError;
use nc_dsp::av1_mc::filter::{AV1_FILTER_TAPS, Av1Filter, Av1Filters};

/// `INTRABC_DELAY_PIXELS`: the default DV of a block in a tile's first
/// superblock row points this far left beyond the superblock.
const INTRABC_DELAY_PIXELS: i32 = 256;
/// Eighths per sample: DVs are whole samples in 1/8 units.
const SUBPEL_PER_SAMPLE: i16 = 8;

impl FrameDecoder<'_> {
    /// The modes of an intra block copy block from `stack`, the MV stack
    /// of `INTRA_FRAME`: its DV, the reference DV (the nearest or near
    /// candidate at quarter precision, else a default a superblock up or
    /// left, then whole samples) plus a whole-sample difference coded with
    /// the DV CDFs; bilinear filters, simple translation.
    // Out of line: our own streams have no intra block copy.
    #[cold]
    #[inline(never)]
    pub(super) fn intrabc_modes(
        &mut self,
        r: &mut SymbolReader<'_>,
        block: BlockPos,
        stack: &MvStack,
    ) -> Result<InterModes, DecodeError> {
        let nearest = quarter_precision(stack.vector(0, 0));
        let near = quarter_precision(stack.vector(1, 0));
        let mut reference = if nearest == Av1Mv::ZERO {
            near
        } else {
            nearest
        };
        if reference == Av1Mv::ZERO {
            reference = self.default_dv(block);
        }
        let whole = |v: i16| v.div_euclid(SUBPEL_PER_SAMPLE) * SUBPEL_PER_SAMPLE;
        let reference = Av1Mv {
            row: whole(reference.row),
            col: whole(reference.col),
        };
        let dv = self.cdfs.dv.read(r, reference, MvPrecision::Integer)?;
        Ok(InterModes {
            refs: [INTRA_FRAME, NONE_FRAME],
            mode: InterMode::NewMv,
            mv: [dv, Av1Mv::ZERO],
            filters: Av1Filters::both(Av1Filter::Bilinear),
            motion: MotionMode::Simple,
            interintra: None,
            compound: None,
            flags: InterFlags::SINGLE,
        })
    }

    /// The DV of a block with no candidates: a superblock up, or in a
    /// tile's first superblock row a superblock and `INTRABC_DELAY_PIXELS`
    /// left (libaom `av1_find_ref_dv`).
    fn default_dv(&self, block: BlockPos) -> Av1Mv {
        let sb_mi = self.superblock.width_mi();
        let sb = (sb_mi * MI_SIZE) as i32;
        let eighths = |samples: i32| (samples * i32::from(SUBPEL_PER_SAMPLE)) as i16;
        if block.mi_row < self.tile.row_start + sb_mi {
            Av1Mv {
                row: 0,
                col: eighths(-sb - INTRABC_DELAY_PIXELS),
            }
        } else {
            Av1Mv {
                row: eighths(-sb),
                col: 0,
            }
        }
    }

    /// Predict every plane of the block from the frame's own decoded
    /// samples displaced by `dv`, with the bilinear filter (chroma lands
    /// on half samples); the chroma of a 4-sample side comes whole from
    /// this block's DV (libaom: no sub-8x8 split for intra block copy).
    #[cold]
    #[inline(never)]
    pub(super) fn predict_intrabc(
        &mut self,
        block: BlockPos,
        dv: Av1Mv,
    ) -> Result<(), DecodeError> {
        let planes: &[PlaneIndex] = if block.has_chroma() {
            &PlaneIndex::ALL
        } else {
            &[PlaneIndex::Y]
        };
        let filters = Av1Filters::both(Av1Filter::Bilinear);
        for &plane in planes {
            let ss = usize::from(plane != PlaneIndex::Y);
            let (origin, size) = Self::plane_block(block, plane);
            let ((x0, y0), axes) = window_position(origin, dv, (ss == 1, ss == 1));
            let span = (size.0 + AV1_FILTER_TAPS - 1, size.1 + AV1_FILTER_TAPS - 1);
            let (width, height) = (self.header.size.width, self.header.size.height);
            // Copied out first: the window and the block are in the same
            // picture.
            let source = PlaneReference {
                plane: self.recon.plane(plane),
                size: ((width + ss) >> ss, (height + ss) >> ss),
                scale: None,
            };
            let window = gather(source, &mut self.maps.prediction.window, (x0, y0), span);
            let mut dst = self
                .recon
                .plane_mut(plane)
                .region_mut(origin.0 as isize, origin.1 as isize, size.0, size.1)
                .ok_or(DecodeError::Malformed("AV1 prediction rectangle"))?;
            self.dsp.av1_predict(filters, &window, size, axes, &mut dst);
        }
        Ok(())
    }
}

/// `lower_mv_precision` without high precision (libaom
/// `av1_find_best_ref_mvs` with `allow_hp` 0): odd components step toward
/// zero.
fn quarter_precision(mv: Av1Mv) -> Av1Mv {
    let even = |v: i16| if v & 1 == 0 { v } else { v - v.signum() };
    Av1Mv {
        row: even(mv.row),
        col: even(mv.col),
    }
}
