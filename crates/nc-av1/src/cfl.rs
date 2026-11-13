//! AV1 4:2:0 chroma from reconstructed luma (AV1 §7.11.5).
//!
//! Subsampling and signed rounding follow libaom `av1/common/cfl.c` and
//! `cfl.h` (BSD-2-Clause, Copyright (c) 2016, Alliance for Open Media;
//! see `THIRD_PARTY.md`). Only square luma blocks up to 32 are supported.

use nc_common::{Picture, PlaneIndex, PlaneViewMut};
use nc_encoder::Block;

use crate::predict::dc_value;

const MAX_CHROMA_SIZE: usize = 16;
const MAX_CHROMA_AREA: usize = MAX_CHROMA_SIZE * MAX_CHROMA_SIZE;
/// Product of Q3 luma AC and a Q3 slope has six fractional bits.
const SCALE_SHIFT: u32 = 6;
const SCALE_HALF: i32 = 1 << (SCALE_SHIFT - 1);
const SIGNS: usize = 3;
const ALPHA_MAX: i8 = 16;

/// Reconstructed luma AC in Q3, and the selected per-plane Q3 slopes.
#[derive(Debug)]
pub struct ChromaFromLuma {
    ac: [i16; MAX_CHROMA_AREA],
    dc: [u8; 2],
    /// Signed Q3 slopes for U and V; zero means DC for that plane.
    pub alpha: [i8; 2],
    /// Exact prediction-SSE improvement of the winning slopes over DC.
    pub improvement: u64,
    size: usize,
}

impl ChromaFromLuma {
    /// Estimate each slope by prediction SSE. The caller accounts for mode
    /// and slope syntax before selecting this candidate over DC.
    /// Bounded SSE and cached improvement reduce enabled key instructions by
    /// 42.8% on the two legal TRAIN cost points (GOP300 -1.04%, GOP60 -4.78%).
    /// Search order and strict ties are unchanged: bit-exact, zero BD effect.
    #[must_use]
    pub fn search(block: &Block, source: &Picture, recon: &Picture) -> Self {
        let n = block.size() / 2;
        assert!([4, 8, MAX_CHROMA_SIZE].contains(&n));
        let (x, y) = (block.x / 2, block.y / 2);
        let luma = recon.plane(PlaneIndex::Y).view();
        let mut result = Self {
            ac: [0; MAX_CHROMA_AREA],
            dc: [0; 2],
            alpha: [0; 2],
            improvement: 0,
            size: n,
        };
        let ac = &mut result.ac[..n * n];
        for (r, row) in ac.chunks_exact_mut(n).enumerate() {
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
        let sum: i32 = ac.iter().map(|&v| i32::from(v)).sum();
        let mean = (sum + (n * n / 2) as i32) / (n * n) as i32;
        for value in ac {
            *value -= mean as i16;
        }
        for (p, plane) in [PlaneIndex::U, PlaneIndex::V].into_iter().enumerate() {
            result.dc[p] = dc_value(recon.plane(plane), x, y, n, y > 0, x > 0);
            // DC is first; ties keep its cheaper zero slope. Every partial
            // SSE is a lower bound, so a candidate cannot win once >=best.
            let dc_sse = result.sse(source, block, p, 0);
            let mut best = (dc_sse, 0);
            for alpha in (1..=ALPHA_MAX).flat_map(|a| [a, -a]) {
                if best.0 == 0 {
                    break;
                }
                let sse = result.sse_bounded(source, block, p, alpha, best.0);
                if sse < best.0 {
                    best = (sse, alpha);
                }
            }
            result.improvement += dc_sse - best.0;
            result.alpha[p] = best.1;
        }
        result
    }

    /// Prediction SSE for one slope (plane 0 is U, plane 1 is V).
    #[must_use]
    pub fn sse(&self, source: &Picture, block: &Block, plane: usize, alpha: i8) -> u64 {
        self.sse_bounded(source, block, plane, alpha, u64::MAX)
    }

    #[inline]
    fn sse_bounded(
        &self,
        source: &Picture,
        block: &Block,
        plane: usize,
        alpha: i8,
        limit: u64,
    ) -> u64 {
        let index = [PlaneIndex::U, PlaneIndex::V][plane];
        let src = source.plane(index).view();
        let mut sum = 0;
        for (r, row) in self.ac[..self.size * self.size]
            .chunks_exact(self.size)
            .enumerate()
        {
            for (c, &ac) in row.iter().enumerate() {
                let error = i32::from(src.row(block.y / 2 + r)[block.x / 2 + c])
                    - i32::from(sample(self.dc[plane], ac, alpha));
                sum += (error * error) as u64;
            }
            if sum >= limit {
                return sum;
            }
        }
        sum
    }

    /// Add scaled luma AC to the chroma DC predictor.
    pub fn predict(&self, plane: usize, dst: &mut PlaneViewMut<'_>) {
        for (r, row) in self.ac[..self.size * self.size]
            .chunks_exact(self.size)
            .enumerate()
        {
            for (pixel, &ac) in dst.row_mut(r).iter_mut().zip(row) {
                *pixel = sample(self.dc[plane], ac, self.alpha[plane]);
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

fn sign(alpha: i8) -> usize {
    match alpha.cmp(&0) {
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Less => 1,
        std::cmp::Ordering::Greater => 2,
    }
}

fn sample(dc: u8, ac: i16, alpha: i8) -> u8 {
    let product = i32::from(ac) * i32::from(alpha);
    let magnitude = (product.abs() + SCALE_HALF) >> SCALE_SHIFT;
    let scaled = if product < 0 { -magnitude } else { magnitude };
    (i32::from(dc) + scaled).clamp(0, 255) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let cfl = ChromaFromLuma::search(&block, &source, &recon);
        assert_eq!(&cfl.ac[..4], &[-192, -64, 64, 192]);
        assert_eq!(cfl.alpha, [8, -8]);
        assert_eq!(cfl.joint_sign(), 6);
        assert_eq!(cfl.alpha_context(0), 4);
        assert_eq!(cfl.alpha_context(1), 2);
        assert_eq!(cfl.sse(&source, &block, 0, 8), 0);
        assert_eq!(cfl.sse(&source, &block, 1, -8), 0);
    }

    #[test]
    fn bounded_search_matches_full_exhaustive_order_and_exact_improvement() {
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
                    let bounded = ChromaFromLuma::search(&block, &source, &recon);
                    let mut improvement = 0;
                    for plane in 0..2 {
                        let mut best = (u64::MAX, 0);
                        for alpha in std::iter::once(0).chain((1..=ALPHA_MAX).flat_map(|a| [a, -a]))
                        {
                            let sse = bounded.sse(&source, &block, plane, alpha);
                            if sse < best.0 {
                                best = (sse, alpha);
                            }
                        }
                        assert_eq!(bounded.alpha[plane], best.1);
                        improvement += bounded.sse(&source, &block, plane, 0) - best.0;
                    }
                    assert_eq!(bounded.improvement, improvement);
                }
            }
        }
    }

    #[test]
    fn signed_rounding_and_clipping() {
        assert_eq!(sample(128, 31, 1), 128);
        assert_eq!(sample(128, 32, 1), 129);
        assert_eq!(sample(128, 32, -1), 127);
        assert_eq!(sample(128, -32, -1), 129);
        assert_eq!(sample(250, 2040, 16), 255);
        assert_eq!(sample(5, 2040, -16), 0);
    }
}
