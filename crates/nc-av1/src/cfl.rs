//! AV1 4:2:0 chroma from reconstructed luma (AV1 §7.11.5).
//!
//! Subsampling and signed rounding follow libaom `av1/common/cfl.c` and
//! `cfl.h` (BSD-2-Clause, Copyright (c) 2016, Alliance for Open Media;
//! see `THIRD_PARTY.md`). Luma blocks up to 32x32: the squares, and the
//! 2:1 halves of key frames' `PARTITION_HORZ` / `VERT`.

use nc_common::{Picture, PlaneIndex, PlaneViewMut};
use nc_dsp::Dsp;
use nc_dsp::cfl_sse::predict_sample;
use nc_encoder::Block;

use crate::predict::dc_value;
use nc_common::BitDepth;
use nc_dsp::intra_prediction::av1_block::{BlockEdges, EdgeAvail};

const MAX_CHROMA_SIZE: usize = 16;
const MAX_CHROMA_AREA: usize = MAX_CHROMA_SIZE * MAX_CHROMA_SIZE;
const SIGNS: usize = 3;

/// Reconstructed luma AC in Q3, and the selected per-plane Q3 slopes.
#[derive(Debug)]
pub struct ChromaFromLuma {
    ac: [i16; MAX_CHROMA_AREA],
    dc: [u8; 2],
    /// Signed Q3 slopes for U and V; zero means DC for that plane.
    pub alpha: [i8; 2],
    /// Exact prediction-SSE improvement of the winning slopes over DC.
    pub improvement: u64,
    /// The chroma block's width and height.
    size: (usize, usize),
}

impl ChromaFromLuma {
    /// Estimate each slope by prediction SSE. The caller accounts for mode
    /// and slope syntax before selecting this candidate over DC.
    /// Every candidate's SSE is evaluated by the fused rectangular kernel
    /// (`Dsp::cfl_best_alpha`) in exact `0, +1, -1, ..., +16, -16` order
    /// with strict `<` tie-breaking.
    #[must_use]
    pub fn search(dsp: &Dsp, block: &Block, source: &Picture, recon: &Picture) -> Self {
        let (w, h) = (block.width() / 2, block.height() / 2);
        assert!(
            [w, h]
                .iter()
                .all(|side| [4, 8, MAX_CHROMA_SIZE].contains(side))
        );
        let (x, y) = (block.x / 2, block.y / 2);
        let luma = recon.plane(PlaneIndex::Y).view();
        let mut result = Self {
            ac: [0; MAX_CHROMA_AREA],
            dc: [0; 2],
            alpha: [0; 2],
            improvement: 0,
            size: (w, h),
        };
        let area = w * h;
        for (r, row) in result.ac[..area].chunks_exact_mut(w).enumerate() {
            let top = luma.row(block.y + 2 * r);
            let bot = luma.row(block.y + 2 * r + 1);
            for (c, value) in row.iter_mut().enumerate() {
                let col = block.x + 2 * c;
                *value = 2
                    * (i16::from(top[col])
                        + i16::from(top[col + 1])
                        + i16::from(bot[col])
                        + i16::from(bot[col + 1]));
            }
        }
        let ac = &mut result.ac[..area];
        let sum: i32 = ac.iter().map(|&v| i32::from(v)).sum();
        let mean = (sum + (area / 2) as i32) / area as i32;
        for value in ac.iter_mut() {
            *value -= mean as i16;
        }
        for (p, plane) in [PlaneIndex::U, PlaneIndex::V].into_iter().enumerate() {
            result.dc[p] = chroma_dc(recon, plane, (x, y), (w, h));
            let src = source
                .plane(plane)
                .region(x as isize, y as isize, w, h)
                .expect("chroma block inside the source");
            let (dc_sse, best_sse, best_alpha) =
                dsp.cfl_best_alpha(&src, &result.ac[..area], result.dc[p]);
            result.improvement += u64::from(dc_sse - best_sse);
            result.alpha[p] = best_alpha;
        }
        result
    }

    /// Prediction SSE for one slope (plane 0 is U, plane 1 is V).
    #[cfg(test)]
    #[must_use]
    pub fn sse(&self, dsp: &Dsp, source: &Picture, block: &Block, plane: usize, alpha: i8) -> u64 {
        let index = [PlaneIndex::U, PlaneIndex::V][plane];
        let (w, h) = self.size;
        let src = source
            .plane(index)
            .region((block.x / 2) as isize, (block.y / 2) as isize, w, h)
            .expect("chroma block inside the source");
        u64::from(dsp.cfl_sse(&src, &self.ac[..w * h], self.dc[plane], alpha))
    }

    /// Add scaled luma AC to the chroma DC predictor.
    pub fn predict(&self, plane: usize, dst: &mut PlaneViewMut<'_>) {
        let (w, h) = self.size;
        for (r, row) in self.ac[..w * h].chunks_exact(w).enumerate() {
            let out = &mut dst.row_mut(r)[..w];
            for (pixel, &ac) in out.iter_mut().zip(row) {
                *pixel = predict_sample(self.dc[plane], ac, self.alpha[plane]);
            }
        }
    }

    /// `CfL` excludes the zero/zero joint sign (AV1 §5.11.45).
    #[must_use]
    pub fn joint_sign(&self) -> usize {
        sign(self.alpha[0]) * SIGNS + sign(self.alpha[1]) - 1
    }

    /// Magnitude CDF context for a plane whose slope is nonzero.
    #[must_use]
    pub fn alpha_context(&self, plane: usize) -> usize {
        let own = sign(self.alpha[plane]);
        let other = sign(self.alpha[1 - plane]);
        own * SIGNS + other - SIGNS
    }
}

/// The DC prediction of chroma `plane`'s `w` x `h` block at (`x`, `y`)
/// (AV1 §7.11.2.5), from which `CfL` predicts: the squares' kernel, the
/// any-shape edges' for a rectangle.
fn chroma_dc(
    recon: &Picture,
    plane: PlaneIndex,
    (x, y): (usize, usize),
    (w, h): (usize, usize),
) -> u8 {
    let samples = recon.plane(plane);
    if w == h {
        return dc_value(samples, x, y, w, y > 0, x > 0);
    }
    let avail = EdgeAvail {
        have_above: y > 0,
        have_left: x > 0,
        have_above_right: false,
        have_below_left: false,
    };
    BlockEdges::read(samples, (x, y), (w, h), avail, BitDepth::EIGHT).dc_value()
}

fn sign(alpha: i8) -> usize {
    match alpha.cmp(&0) {
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Less => 1,
        std::cmp::Ordering::Greater => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nc_dsp::cfl_sse::MAX_ALPHA;

    #[test]
    fn subsampling_mean_and_opposite_slopes() {
        let mut recon = Picture::new(8, 8, 0).unwrap();
        let mut source = Picture::new(8, 8, 0).unwrap();
        for y in 0..8 {
            for x in 0..8 {
                recon.plane_mut(PlaneIndex::Y).view_mut().row_mut(y)[x] = 96 + 16 * (x / 2) as u8;
            }
        }
        for y in 0..4 {
            for x in 0..4 {
                source.plane_mut(PlaneIndex::U).view_mut().row_mut(y)[x] = 104 + 16 * x as u8;
                source.plane_mut(PlaneIndex::V).view_mut().row_mut(y)[x] = 152 - 16 * x as u8;
            }
        }
        let block = Block::square(0, 0, 3);
        let cfl = ChromaFromLuma::search(&Dsp::new(), &block, &source, &recon);
        assert_eq!(&cfl.ac[..4], &[-192, -64, 64, 192]);
        assert_eq!(cfl.alpha, [8, -8]);
        assert_eq!(cfl.joint_sign(), 6);
        assert_eq!(cfl.alpha_context(0), 4);
        assert_eq!(cfl.alpha_context(1), 2);
        assert_eq!(cfl.sse(&Dsp::new(), &source, &block, 0, 8), 0);
        assert_eq!(cfl.sse(&Dsp::new(), &source, &block, 1, -8), 0);
    }

    #[test]
    fn search_matches_the_exhaustive_order_and_exact_improvement() {
        for log2_size in [3, 4, 5] {
            for (x, y) in [(0, 0), (32, 32)] {
                let mut state = 0x3141_5926_u32;
                for pattern in 0..24 {
                    let mut source = Picture::new(64, 64, 0).unwrap();
                    let mut recon = source.clone();
                    for plane in PlaneIndex::ALL {
                        let mut dst = recon.plane_mut(plane).view_mut();
                        let mut src = source.plane_mut(plane).view_mut();
                        for row in 0..dst.height() {
                            for col in 0..dst.width() {
                                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                                dst.row_mut(row)[col] = (state >> 24) as u8;
                                src.row_mut(row)[col] = match pattern {
                                    0 => 128,
                                    1 => 0,
                                    2 => 255,
                                    _ => (state >> 16) as u8,
                                };
                            }
                        }
                    }
                    let block = Block::square(x, y, log2_size);
                    let bounded = ChromaFromLuma::search(&Dsp::new(), &block, &source, &recon);
                    let mut improvement = 0;
                    for plane in 0..2 {
                        let mut best = (u64::MAX, 0);
                        for alpha in std::iter::once(0).chain((1..=MAX_ALPHA).flat_map(|a| [a, -a]))
                        {
                            let sse = bounded.sse(&Dsp::new(), &source, &block, plane, alpha);
                            if sse < best.0 {
                                best = (sse, alpha);
                            }
                        }
                        assert_eq!(bounded.alpha[plane], best.1);
                        improvement += bounded.sse(&Dsp::new(), &source, &block, plane, 0) - best.0;
                    }
                    assert_eq!(bounded.improvement, improvement);
                }
            }
        }
    }

    #[test]
    fn signed_rounding_and_clipping() {
        assert_eq!(predict_sample(128, 31, 1), 128);
        assert_eq!(predict_sample(128, 32, 1), 129);
        assert_eq!(predict_sample(128, 32, -1), 127);
        assert_eq!(predict_sample(128, -32, -1), 129);
        assert_eq!(predict_sample(250, 2040, 16), 255);
        assert_eq!(predict_sample(5, 2040, -16), 0);
    }
}
