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
//! Whole-pel prediction uses the in-bounds copy path of dav1d 1.5.4
//! `src/recon_tmpl.c` (`mc`, BSD-2-Clause); clipping and fractional
//! prediction still use the same clamped-window path.
use super::super::block_info::InterFlags;
use super::super::motion::{INTRA_FRAME, InterMode, NONE_FRAME};
use super::super::mv_stack::MvStack;
use super::inter::{InterModes, MotionMode};
use super::inter_pred::{PlaneReference, gather, window_position};
use super::{BlockPos, FrameDecoder, MI_SIZE};
use crate::backend::Av1Mv;
use crate::decoder_range::SymbolReader;
use crate::mvcoding::MvPrecision;
use nc_common::decode::DecodeError;
use nc_common::{Plane, PlaneIndex, Sample};
use nc_dsp::Pixel;
use nc_dsp::av1_mc::filter::{AV1_FILTER_TAPS, Av1Filter, Av1Filters};

/// `INTRABC_DELAY_PIXELS`: the default DV of a block in a tile's first
/// superblock row points this far left beyond the superblock.
const INTRABC_DELAY_PIXELS: i32 = 256;
/// Eighths per sample: DVs are whole samples in 1/8 units.
const SUBPEL_PER_SAMPLE: i16 = 8;

impl<T: Pixel> FrameDecoder<'_, T> {
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
        let kernels = self.kernels();
        for &plane in planes {
            let (ss_x, ss_y) = self.subsampling(plane);
            let (origin, size) = Self::plane_block(block, plane);
            let decoded = decoded_size((self.header.width, self.header.height), (ss_x, ss_y));
            let pel = (SUBPEL_PER_SAMPLE << ss_x, SUBPEL_PER_SAMPLE << ss_y);
            if dv.col % pel.0 == 0 && dv.row % pel.1 == 0 {
                let source = (
                    origin.0 as isize + isize::from(dv.col / pel.0),
                    origin.1 as isize + isize::from(dv.row / pel.1),
                );
                if copy_intrabc(self.recon.plane_mut(plane), (decoded, size), source, origin) {
                    continue;
                }
            }
            let ((x0, y0), axes) = window_position(origin, dv, (ss_x == 1, ss_y == 1));
            let span = (size.0 + AV1_FILTER_TAPS - 1, size.1 + AV1_FILTER_TAPS - 1);
            // Copied out first: the window and the block are in the same
            // picture.
            let source = PlaneReference {
                plane: self.recon.plane(plane),
                size: decoded,
                scale: None,
            };
            let window = gather(source, &mut self.maps.prediction.window, (x0, y0), span);
            let mut dst = self
                .recon
                .plane_mut(plane)
                .region_mut(origin.0 as isize, origin.1 as isize, size.0, size.1)
                .ok_or(DecodeError::Malformed("AV1 prediction rectangle"))?;
            kernels.av1_predict(filters, &window, size, axes, &mut dst);
        }
        Ok(())
    }
}

/// The part of a plane an intra block copy reads unclamped: every decoded
/// sample, including the 8x8 units beyond the frame's right and bottom
/// edges (`coded`: the 8-aligned luma size; chroma subsampled by
/// `(ss_x, ss_y)`).
/// libaom's `extend_mc_border` skips intra block copy and dav1d's `mc`
/// bounds it by the 8-aligned `bw`/`bh`, so a DV reaching past the frame
/// edge reads those decoded samples; only samples past them are the
/// clamped edge.
fn decoded_size((width, height): (usize, usize), (ss_x, ss_y): (usize, usize)) -> (usize, usize) {
    (width >> ss_x, height >> ss_y)
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

/// Copy a full-pel rectangle from the current picture. Choose the row
/// direction as for memmove so an overlapping rectangle still reads
/// the original samples, just like the gathered prediction.
fn copy_intrabc<T: Sample>(
    plane: &mut Plane<T>,
    ((decoded_width, decoded_height), (width, height)): ((usize, usize), (usize, usize)),
    (source_x, source_y): (isize, isize),
    (dest_x, dest_y): (usize, usize),
) -> bool {
    let (Ok(source_x), Ok(source_y)) = (usize::try_from(source_x), usize::try_from(source_y))
    else {
        return false;
    };
    if source_x + width > decoded_width
        || source_y + height > decoded_height
        || dest_x + width > plane.width()
        || dest_y + height > plane.height()
    {
        return false;
    }

    let mut view = plane.view_mut();
    let stride = view.stride();
    let samples = view.data_mut();
    let mut copy = |row| {
        let source = (source_y + row) * stride + source_x;
        let dest = (dest_y + row) * stride + dest_x;
        samples.copy_within(source..source + width, dest);
    };
    if dest_y > source_y {
        (0..height).rev().for_each(&mut copy);
    } else {
        (0..height).for_each(copy);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copies_reach_the_decoded_units_beyond_the_frame_edge() {
        // A 140-wide frame codes 144 columns. Argon test1583 (and 20 more
        // streams) copy from columns 140..143, which the decoder had
        // clamped to column 139.
        let coded = (144, 16);
        assert_eq!(decoded_size(coded, (0, 0)), (144, 16));
        assert_eq!(decoded_size(coded, (1, 1)), (72, 8));
        assert_eq!(decoded_size(coded, (1, 0)), (72, 16));
        let mut plane = Plane::<u8>::new(144, 16, 0).unwrap();
        for (x, sample) in plane.view_mut().data_mut()[..144].iter_mut().enumerate() {
            *sample = x as u8;
        }
        assert!(copy_intrabc(&mut plane, (coded, (8, 1)), (136, 0), (0, 8)));
        let view = plane.view();
        let start = 8 * view.stride();
        assert_eq!(
            &view.data()[start..start + 8],
            &[136, 137, 138, 139, 140, 141, 142, 143]
        );
    }
}
