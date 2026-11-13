//! The zero-residual skip of still blocks, shared by the AV1 and HEVC
//! backends (`av1.still_skip`, `hevc.still_skip`).
//!
//! On a block the stillness detector marks still (`nc_analysis::still`), the
//! zero vector is the natural prediction, and on static content most of a P
//! frame's time goes into coding residuals that quantise to nothing
//! (`docs/handoff/fast-levers.md`: AV1 `code_plane` 43%, HEVC
//! `code_residual` 38% of static P frames). When the zero-vector residual of
//! every plane passes the pre-transform zero test
//! ([`residual_quantises_to_zero`]) against the backend's own quantisers, the
//! block is coded as the zero vector with no residual, skipping mode
//! decision, transforms and quantisation. H.264's `try_static_skip` does the
//! same with its own 4x4 test.
//!
//! The zero vector is a whole-sample copy of the co-located reference
//! samples, so no interpolation is needed for the test or the
//! reconstruction.

use nc_common::{Picture, PlaneIndex, PlaneView};
use nc_dsp::Dsp;

use crate::partition::Block;
use crate::zero_residual::{ZeroLimits, residual_quantises_to_zero};

/// Smallest block side the test covers: its 4:2:0 chroma side must be a
/// multiple of the zero test's 8x8 sub-blocks.
pub const MIN_STILL_SKIP_SIZE: usize = 16;

/// The zero test's AC energy allowance: exactly the squared limits (the
/// limits themselves scale through [`PlaneZeroLimits::scaled`]).
const TEXTURE_SCALE: f32 = 1.0;

/// Zero limits of the luma and chroma quantisers of a frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlaneZeroLimits {
    /// Luma.
    pub luma: ZeroLimits,
    /// Both chroma planes.
    pub chroma: ZeroLimits,
}

impl PlaneZeroLimits {
    /// Every limit multiplied by `scale`: above 1 the test no longer proves
    /// a zero residual but accepts residuals the codec would code with a few
    /// small levels (H.264's `static_skip_*_scale` do the same).
    #[must_use]
    pub fn scaled(self, scale: f32) -> Self {
        let scale_one = |l: ZeroLimits| ZeroLimits {
            dc: l.dc * scale,
            ac: l.ac * scale,
        };
        Self {
            luma: scale_one(self.luma),
            chroma: scale_one(self.chroma),
        }
    }
}

/// True when the zero-vector residual (`source` − co-located `reference`)
/// of every plane of `block` would quantise to all zeros by
/// [`residual_quantises_to_zero`] (texture scale 1). Blocks smaller than
/// [`MIN_STILL_SKIP_SIZE`] never pass.
#[must_use]
pub fn zero_vector_residual_is_zero(
    dsp: &Dsp,
    block: &Block,
    source: &Picture,
    reference: &Picture,
    limits: PlaneZeroLimits,
) -> bool {
    if block.size() < MIN_STILL_SKIP_SIZE {
        return false;
    }
    PlaneIndex::ALL.into_iter().all(|plane| {
        let (src, pred) = (
            co_located(source, plane, block),
            co_located(reference, plane, block),
        );
        let limits = if plane == PlaneIndex::Y {
            limits.luma
        } else {
            limits.chroma
        };
        residual_quantises_to_zero(dsp, src.width(), &src, &pred, limits, TEXTURE_SCALE)
    })
}

/// Write the zero-vector prediction of `block` (the co-located `reference`
/// samples, every plane) into `recon`: the reconstruction of a block coded
/// with no residual.
pub fn copy_zero_vector_prediction(block: &Block, reference: &Picture, recon: &mut Picture) {
    for plane in PlaneIndex::ALL {
        let shift = usize::from(plane != PlaneIndex::Y);
        let n = block.size() >> shift;
        recon
            .plane_mut(plane)
            .region_mut(
                (block.x >> shift) as isize,
                (block.y >> shift) as isize,
                n,
                n,
            )
            .expect("block inside the reconstruction")
            .copy_from(&co_located(reference, plane, block));
    }
}

/// `block`'s samples of `plane` in `picture` (4:2:0).
fn co_located<'a>(picture: &'a Picture, plane: PlaneIndex, block: &Block) -> PlaneView<'a> {
    let shift = usize::from(plane != PlaneIndex::Y);
    let n = block.size() >> shift;
    picture
        .plane(plane)
        .region(
            (block.x >> shift) as isize,
            (block.y >> shift) as isize,
            n,
            n,
        )
        .expect("block inside the picture")
}

#[cfg(test)]
mod tests {
    use super::*;

    // An offset of 1 gives an orthonormal DC of 16 on a 16x16 luma block
    // and 8 on its 8x8 chroma.
    const LIMITS: PlaneZeroLimits = PlaneZeroLimits {
        luma: ZeroLimits { dc: 20.0, ac: 8.0 },
        chroma: ZeroLimits { dc: 20.0, ac: 8.0 },
    };

    fn picture(value: u8) -> Picture {
        let mut p = Picture::new(64, 64, 0).unwrap();
        for plane in PlaneIndex::ALL {
            p.plane_mut(plane).view_mut().fill(value);
        }
        p
    }

    #[test]
    fn small_changes_pass_and_the_prediction_is_copied() {
        let (source, reference) = (picture(101), picture(100));
        let block = Block::square(16, 16, 4);
        assert!(zero_vector_residual_is_zero(
            &Dsp::new(),
            &block,
            &source,
            &reference,
            LIMITS
        ));
        let mut recon = picture(0);
        copy_zero_vector_prediction(&block, &reference, &mut recon);
        assert_eq!(recon.plane(PlaneIndex::Y).view().row(16)[16], 100);
        assert_eq!(recon.plane(PlaneIndex::U).view().row(8)[8], 100);
        assert_eq!(recon.plane(PlaneIndex::Y).view().row(15)[16], 0);
    }

    #[test]
    fn a_chroma_change_or_a_small_block_fails() {
        let reference = picture(100);
        let mut source = picture(100);
        source.plane_mut(PlaneIndex::V).view_mut().fill(140);
        let block = Block::square(0, 0, 5);
        let dsp = Dsp::new();
        assert!(!zero_vector_residual_is_zero(
            &dsp, &block, &source, &reference, LIMITS
        ));
        let small = Block::square(0, 0, 3);
        assert!(!zero_vector_residual_is_zero(
            &dsp, &small, &reference, &reference, LIMITS
        ));
    }
}
