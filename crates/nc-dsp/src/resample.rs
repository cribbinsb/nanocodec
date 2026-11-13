//! Encoder-side picture resampling: a separable Lanczos-3 polyphase filter
//! that scales one plane to any size (the source of a frame coded at a
//! reduced size: AV1 superres, resolution switching, spatial scalability).
//!
//! Not normative: decoders never resample with it. Each plane is mapped
//! centre to centre (`in = (out + 0.5) * in_size / out_size - 0.5`), the
//! mapping the AV1 superres upscaler (§7.16) assumes for every plane. When
//! downscaling, the kernel is stretched by the ratio so it also low-passes
//! (no aliasing); upscaling uses the plain Lanczos-3 kernel. Taps are
//! fixed-point (`WEIGHT_BITS`) and normalised to sum exactly to one, so a
//! flat plane stays flat.
//!
//! [`Dsp::resample_plane`](crate::Dsp::resample_plane) and
//! [`Dsp::resample_plane_with`](crate::Dsp::resample_plane_with) dispatch
//! bit-identical AVX2, NEON and WASM SIMD128 kernels verified against the
//! scalar reference by `nc-checkasm`.

pub(crate) mod kernels;

use std::f64::consts::PI;

use nc_common::{PlaneView, PlaneViewMut};

/// Lanczos lobes on each side of the centre.
const LOBES: f64 = 3.0;
/// Fixed-point precision of the filter weights.
pub(crate) const WEIGHT_BITS: u32 = 14;
pub(crate) const WEIGHT_ONE: i32 = 1 << WEIGHT_BITS;

/// The taps of one output sample along one direction: the first input
/// sample (possibly outside the plane: positions are clamped to its edge)
/// and its weights.
#[derive(Debug, Clone)]
struct Taps {
    first: isize,
    weights: Vec<i32>,
}

/// Precomputed SIMD-friendly horizontal polyphase layout: input rows are
/// edge-extended by `left_pad` and `right_pad`, and each output column's
/// weights are zero-padded to `groups` 8-tap chunks (`[i16; 8]`).
#[derive(Debug, Clone)]
pub(crate) struct HorizontalPlan {
    pub(crate) left_pad: usize,
    pub(crate) right_pad: usize,
    pub(crate) groups: usize,
    pub(crate) starts: Vec<usize>,
    pub(crate) padded_weights: Vec<[i16; 8]>,
    scalar: Vec<Taps>,
}

/// Precomputed vertical polyphase taps for one output row: clamped row pairs
/// `(row_a, weight_a, row_b, weight_b)`.
#[derive(Debug, Clone)]
pub(crate) struct VerticalTap {
    pub(crate) pairs: Vec<(usize, i16, usize, i16)>,
    scalar: Taps,
}

/// Precomputed Lanczos-3 polyphase filter tables and reusable intermediate
/// scratch buffers for scaling planes from `(in_w, in_h)` to `(out_w, out_h)`
/// across many frames with zero per-call heap allocations.
#[derive(Debug, Clone)]
pub struct PlaneResampler {
    in_size: (usize, usize),
    out_size: (usize, usize),
    pub(crate) horizontal: Option<HorizontalPlan>,
    pub(crate) vertical: Option<Vec<VerticalTap>>,
    pub(crate) scratch_rows: Vec<u8>,
    pub(crate) scratch_padded: Vec<u8>,
}

impl PlaneResampler {
    /// Precompute Lanczos-3 polyphase taps and reusable scratch buffers from
    /// `in_size` to `out_size`.
    ///
    /// # Panics
    ///
    /// If any width or height is zero.
    #[must_use]
    pub fn new((in_w, in_h): (usize, usize), (out_w, out_h): (usize, usize)) -> Self {
        assert!(
            in_w > 0 && in_h > 0 && out_w > 0 && out_h > 0,
            "PlaneResampler dimensions ({in_w}x{in_h} -> {out_w}x{out_h}) must be non-zero"
        );
        let horizontal = (in_w != out_w).then(|| build_horizontal(in_w, out_w));
        let vertical = (in_h != out_h).then(|| build_vertical(in_h, out_h, out_w));
        let scratch_rows = if vertical.is_some() {
            vec![0u8; out_w * in_h]
        } else {
            Vec::new()
        };
        let scratch_padded = match &horizontal {
            Some(h) => vec![0u8; h.left_pad + in_w + h.right_pad],
            None => Vec::new(),
        };
        Self {
            in_size: (in_w, in_h),
            out_size: (out_w, out_h),
            horizontal,
            vertical,
            scratch_rows,
            scratch_padded,
        }
    }

    /// Input dimensions `(width, height)` this plan expects.
    #[must_use]
    pub const fn in_size(&self) -> (usize, usize) {
        self.in_size
    }

    /// Output dimensions `(width, height)` this plan produces.
    #[must_use]
    pub const fn out_size(&self) -> (usize, usize) {
        self.out_size
    }
}

fn build_horizontal(in_w: usize, out_w: usize) -> HorizontalPlan {
    let scalar = direction_taps(in_w, out_w);
    let max_len = scalar.iter().map(|t| t.weights.len()).max().unwrap_or(0);
    let groups = max_len.div_ceil(8).max(1);
    let stride = groups * 8;
    let left_pad = scalar
        .iter()
        .map(|t| (-t.first).max(0) as usize)
        .max()
        .unwrap_or(0);
    let starts: Vec<usize> = scalar
        .iter()
        .map(|t| (t.first + left_pad as isize) as usize)
        .collect();
    let max_end = starts.iter().map(|&s| s + stride).max().unwrap_or(0);
    let right_pad = max_end.saturating_sub(left_pad + in_w);
    let mut padded_weights = vec![[0i16; 8]; out_w * groups];
    for (x, t) in scalar.iter().enumerate() {
        let chunks = &mut padded_weights[x * groups..(x + 1) * groups];
        for (k, &w) in t.weights.iter().enumerate() {
            chunks[k / 8][k % 8] = i16::try_from(w).expect("Lanczos-3 Q14 tap fits in i16");
        }
    }
    HorizontalPlan {
        left_pad,
        right_pad,
        groups,
        starts,
        padded_weights,
        scalar,
    }
}

fn build_vertical(in_h: usize, out_h: usize, out_w: usize) -> Vec<VerticalTap> {
    let last = in_h as isize - 1;
    direction_taps(in_h, out_h)
        .into_iter()
        .map(|scalar| {
            let len = scalar.weights.len();
            let mut pairs = Vec::with_capacity(len.div_ceil(2));
            for k in (0..len).step_by(2) {
                let ra = (scalar.first + k as isize).clamp(0, last) as usize;
                let wa = i16::try_from(scalar.weights[k]).expect("Lanczos-3 Q14 tap fits in i16");
                let (rb, wb) = if k + 1 < len {
                    let r = (scalar.first + (k + 1) as isize).clamp(0, last) as usize;
                    let w = i16::try_from(scalar.weights[k + 1])
                        .expect("Lanczos-3 Q14 tap fits in i16");
                    (r, w)
                } else {
                    (0, 0)
                };
                pairs.push((ra * out_w, wa, rb * out_w, wb));
            }
            VerticalTap { pairs, scalar }
        })
        .collect()
}

/// `sinc(x) * sinc(x / LOBES)` for `|x| < LOBES`.
fn lanczos(x: f64) -> f64 {
    if x.abs() < 1e-9 {
        return 1.0;
    }
    if x.abs() >= LOBES {
        return 0.0;
    }
    let px = PI * x;
    LOBES * px.sin() * (px / LOBES).sin() / (px * px)
}

/// The taps of every output sample of a direction scaled from `input` to
/// `output` samples.
fn direction_taps(input: usize, output: usize) -> Vec<Taps> {
    let ratio = input as f64 / output as f64;
    // Stretch the kernel when downscaling so that it band-limits to the
    // output's Nyquist frequency.
    let stretch = ratio.max(1.0);
    let reach = LOBES * stretch;
    (0..output)
        .map(|out| {
            let centre = (out as f64 + 0.5) * ratio - 0.5;
            let first = (centre - reach).ceil() as isize;
            let last = (centre + reach).floor() as isize;
            let real: Vec<f64> = (first..=last)
                .map(|i| lanczos((i as f64 - centre) / stretch))
                .collect();
            let total: f64 = real.iter().sum();
            let mut weights: Vec<i32> = real
                .iter()
                .map(|w| (w / total * f64::from(WEIGHT_ONE)).round() as i32)
                .collect();
            // Put the rounding remainder on the centre tap: weights sum to one.
            let remainder = WEIGHT_ONE - weights.iter().sum::<i32>();
            let centre_tap = (centre.round() as isize - first).clamp(0, weights.len() as isize - 1);
            weights[centre_tap as usize] += remainder;
            Taps { first, weights }
        })
        .collect()
}

/// Apply `taps` to the samples `sample(i)` (`i` clamped to `0..len`).
fn filter(taps: &Taps, len: usize, sample: impl Fn(usize) -> i32) -> i32 {
    let last = len as isize - 1;
    taps.weights
        .iter()
        .enumerate()
        .map(|(k, &w)| w * sample((taps.first + k as isize).clamp(0, last) as usize))
        .sum()
}

/// Round a fixed-point filter sum back to a sample.
pub(crate) fn to_sample(sum: i32) -> u8 {
    ((sum + (WEIGHT_ONE >> 1)) >> WEIGHT_BITS).clamp(0, 255) as u8
}

/// Scalar reference execution of a precomputed [`PlaneResampler`].
pub(crate) fn resample_scalar_with(
    plan: &mut PlaneResampler,
    src: &PlaneView<'_>,
    dst: &mut PlaneViewMut<'_>,
) {
    let (in_w, in_h) = plan.in_size;
    let (out_w, out_h) = plan.out_size;
    let Some(vert) = &plan.vertical else {
        for y in 0..out_h {
            let src_row = &src.row(y)[..in_w];
            let dst_row = &mut dst.row_mut(y)[..out_w];
            match &plan.horizontal {
                Some(horiz) => {
                    for (out, t) in dst_row.iter_mut().zip(&horiz.scalar) {
                        *out = to_sample(filter(t, in_w, |i| i32::from(src_row[i])));
                    }
                }
                None => dst_row.copy_from_slice(src_row),
            }
        }
        return;
    };
    let rows = &mut plan.scratch_rows[..out_w * in_h];
    for (y, row) in rows.chunks_exact_mut(out_w).enumerate() {
        let src_row = &src.row(y)[..in_w];
        match &plan.horizontal {
            Some(horiz) => {
                for (out, t) in row.iter_mut().zip(&horiz.scalar) {
                    *out = to_sample(filter(t, in_w, |i| i32::from(src_row[i])));
                }
            }
            None => row.copy_from_slice(src_row),
        }
    }
    for (y, vtap) in vert.iter().enumerate().take(out_h) {
        let dst_row = &mut dst.row_mut(y)[..out_w];
        let t = &vtap.scalar;
        for (x, out) in dst_row.iter_mut().enumerate() {
            *out = to_sample(filter(t, in_h, |i| i32::from(rows[i * out_w + x])));
        }
    }
}

/// Resample all of `src` into all of `dst` (horizontal pass, then
/// vertical) via the scalar reference implementation. A direction whose size
/// does not change is copied through.
pub fn resample_plane(src: &PlaneView<'_>, dst: &mut PlaneViewMut<'_>) {
    let mut plan = PlaneResampler::new((src.width(), src.height()), (dst.width(), dst.height()));
    resample_scalar_with(&mut plan, src, dst);
}

#[cfg(test)]
mod tests {
    use super::*;
    use nc_common::Plane;

    fn plane(width: usize, height: usize, value: impl Fn(usize, usize) -> u8) -> Plane {
        let mut p = Plane::new(width, height, 0).unwrap();
        let mut view = p.view_mut();
        for y in 0..height {
            for (x, s) in view.row_mut(y).iter_mut().enumerate().take(width) {
                *s = value(x, y);
            }
        }
        p
    }

    #[test]
    fn weights_sum_to_one() {
        for (input, output) in [(1280, 640), (1280, 854), (720, 360), (360, 720), (16, 16)] {
            for taps in direction_taps(input, output) {
                assert_eq!(taps.weights.iter().sum::<i32>(), WEIGHT_ONE);
            }
        }
    }

    #[test]
    fn flat_planes_stay_flat() {
        let src = plane(64, 32, |_, _| 77);
        let mut dst = Plane::new(40, 18, 0).unwrap();
        resample_plane(&src.view(), &mut dst.view_mut());
        assert!(dst.view().rows().all(|r| r.iter().all(|&s| s == 77)));
    }

    #[test]
    fn same_size_copies() {
        let src = plane(24, 10, |x, y| (x * 7 + y * 3) as u8);
        let mut dst = Plane::new(24, 10, 0).unwrap();
        resample_plane(&src.view(), &mut dst.view_mut());
        assert!(
            src.view()
                .rows()
                .zip(dst.view().rows())
                .all(|(a, b)| a == b)
        );
    }

    #[test]
    fn halving_a_ramp_keeps_the_ramp() {
        // A linear ramp resampled centre to centre stays the same ramp.
        let src = plane(128, 4, |x, _| (x + 40) as u8);
        let mut dst = Plane::new(64, 4, 0).unwrap();
        resample_plane(&src.view(), &mut dst.view_mut());
        for x in 8..56 {
            let expected = 2.0 * x as f64 + 0.5 + 40.0;
            let got = f64::from(dst.view().row(1)[x]);
            assert!((got - expected).abs() <= 1.0, "x {x}: {got} vs {expected}");
        }
    }
}
