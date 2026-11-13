//! H.266 §8.4.5.2 prediction of one intra sub-partition (ISP: a coding
//! unit's luma coded as two or four rectangular transform blocks, each
//! predicted from the reconstruction of the one before). Differences from
//! the square path: rectangular planar/DC/PDPC (§8.4.5.2.11-12, §8.4.5.2.15),
//! unfiltered references (§8.4.5.2.8 filterFlag 0), the cubic filter for
//! every fractional direction (§8.4.5.2.13 filterFlag 0), and wide-angle
//! remapping by the coding unit's shape (§8.4.5.2.7 nCbW / nCbH).
//!
//! This module holds the planar and DC part; the angular directions and the
//! entry point are `rectangular.rs` ([`crate::Dsp::vvc_intra_predict_isp`]).
//!
//! Scalar only: ISP blocks are a small share of the intra work. Equations
//! follow `VVdeC` `IntraPrediction` (BSD-3-Clause-Clear; `THIRD_PARTY.md`).

/// Planar (`planar`) or DC, then PDPC (§8.4.5.2.15, nScale from both sides).
// Keep the existing ISP call inlined when the general decoder also uses
// these equations; sharing must not change the own-stream prediction path.
#[allow(clippy::inline_always)]
#[inline(always)]
pub(super) fn smooth(
    (width, height): (usize, usize),
    planar: bool,
    top: &[i32],
    left: &[i32],
    out: &mut [u8],
) {
    let (log2_w, log2_h) = (width.ilog2(), height.ilog2());
    // §8.4.5.2.12: a rectangle averages its longer side only.
    let dc = match width.cmp(&height) {
        std::cmp::Ordering::Equal => {
            (top[..width].iter().chain(&left[..height]).sum::<i32>() + width as i32) >> (log2_w + 1)
        }
        std::cmp::Ordering::Greater => {
            (top[..width].iter().sum::<i32>() + (width as i32 >> 1)) >> log2_w
        }
        std::cmp::Ordering::Less => {
            (left[..height].iter().sum::<i32>() + (height as i32 >> 1)) >> log2_h
        }
    };
    let scale = (log2_w + log2_h - 2) >> 2;
    let (top_right, bottom_left) = (top[width], left[height]);
    for y in 0..height {
        let wt = 32 >> ((2 * y) >> scale).min(31);
        for x in 0..width {
            let wl = 32 >> ((2 * x) >> scale).min(31);
            let value = if planar {
                let vertical =
                    ((height - 1 - y) as i32 * top[x] + (y + 1) as i32 * bottom_left) << log2_w;
                let horizontal =
                    ((width - 1 - x) as i32 * left[y] + (x + 1) as i32 * top_right) << log2_h;
                (vertical + horizontal + (width * height) as i32) >> (log2_w + log2_h + 1)
            } else {
                dc
            };
            out[y * width + x] =
                (value + ((wl * (left[y] - value) + wt * (top[x] - value) + 32) >> 6)) as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    /// A square sub-partition predicts as the square kernel does with
    /// unfiltered references, except the size-dependent Gaussian choice:
    /// at 4x4 every non-diagonal direction uses the cubic filter on both.
    #[test]
    fn a_square_block_matches_the_square_kernel() {
        let dsp = crate::Dsp::new();
        let top: Vec<i32> = (0..64).map(|i| i * 37 % 251).collect();
        let left: Vec<i32> = (0..64).map(|i| i * 53 % 239).collect();
        let corner = 77;
        for direction in 0..=66u8 {
            let mut square = [0u8; 16];
            let mut isp = [0u8; 16];
            let mode = match direction {
                0 => crate::VvcIntraMode::Planar,
                1 => crate::VvcIntraMode::Dc,
                18 => crate::VvcIntraMode::Horizontal,
                50 => crate::VvcIntraMode::Vertical,
                _ => crate::VvcIntraMode::Angular {
                    direction,
                    luma: true,
                },
            };
            super::super::scalar(4, mode, &top, &left, corner, false, &mut square);
            let (top, left) = (&top[..8], &left[..8]);
            dsp.vvc_intra_predict_isp((4, 4), (4, 4), direction, top, left, corner, &mut isp);
            assert_eq!(square, isp, "direction {direction}");
        }
    }

    #[test]
    fn rectangles_average_their_longer_side() {
        let dsp = crate::Dsp::new();
        let top = [100; 64];
        let left = [20; 64];
        let mut out = [0u8; 32 * 8];
        dsp.vvc_intra_predict_isp((32, 32), (32, 8), 1, &top, &left, 100, &mut out);
        // Far from the edges PDPC no longer pulls towards the left samples.
        assert_eq!(out[7 * 32 + 31], 100);
        dsp.vvc_intra_predict_isp((32, 32), (8, 32), 1, &top, &left, 100, &mut out);
        assert_eq!(out[31 * 8 + 7], 20);
    }
}
