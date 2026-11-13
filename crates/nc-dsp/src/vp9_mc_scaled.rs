//! VP9 prediction from a reference of another size (VP9 §8.5.2.3; libvpx
//! `vpx_scaled_2d`, BSD-3-Clause, see `THIRD_PARTY.md`): a horizontal pass
//! at `x_step` sixteenths per output sample into a temporary block, then a
//! vertical pass at `y_step`, each rounded and clipped to 8 bits.
//!
//! The scalar oracle is [`crate::vp9_convolve_scaled`]. This driver composes
//! the unscaled SIMD row kernels ([`crate::Dsp::vp9_convolve_into`]) instead
//! of new kernels: a pass at step 8 (a reference half the frame's size, as
//! every 2:1 spatial layer predicts from its base) is two interleaved passes
//! at step 16, one per phase, and a vertical pass at any step is one unscaled
//! vertical row per output row (its phase is constant along the row). Only a
//! horizontal pass at another step stays scalar.

use crate::vp9_mc::{VP9_POSITIONS, VP9_TAPS, VP9_TAPS_BEFORE, Vp9Filter};

const SUBPEL_BITS: usize = 4;
const SUBPEL_MASK: usize = VP9_POSITIONS - 1;
/// The step of a reference half the frame's size (`x_step_q4` 8).
const HALF_SIZE_STEP: usize = VP9_POSITIONS / 2;
/// Rows of the horizontal pass at the largest step (libvpx's 135).
const MAX_ROWS: usize =
    (((64 - 1) * crate::VP9_MAX_SCALED_STEP + SUBPEL_MASK) >> SUBPEL_BITS) + VP9_TAPS;
/// Stride of the temporary block: 64 samples plus the columns an unscaled
/// vertical pass reads around them.
const TEMP_STRIDE: usize = 64 + VP9_TAPS;
/// Rows filtered per horizontal call ([`crate::Dsp::vp9_convolve_into`]
/// takes at most 64).
const MAX_CALL_ROWS: usize = 64;

/// Working memory of [`crate::Dsp::vp9_convolve_scaled_into`], kept by the
/// caller across predictions (zeroing it per call cost more than the filter).
#[derive(Debug)]
pub struct Vp9ScaledScratch {
    /// The horizontal pass, `TEMP_STRIDE` apart, columns from `VP9_TAPS_BEFORE`.
    temp: Vec<u8>,
    /// The two phases of a half-size horizontal pass, before interleaving.
    halves: [Vec<u8>; 2],
}

impl Default for Vp9ScaledScratch {
    fn default() -> Self {
        Self {
            temp: vec![0; MAX_ROWS * TEMP_STRIDE],
            halves: [vec![0; MAX_ROWS * 32], vec![0; MAX_ROWS * 32]],
        }
    }
}

/// One phase class of a half-size pass: its first output's integer sample
/// (from the footprint's start) and its phase.
#[derive(Clone, Copy)]
struct Class {
    start: usize,
    phase: usize,
}

/// The two classes (even and odd outputs) of a pass at step 8 from `q4`.
fn half_size_classes(q4: usize) -> [Class; 2] {
    [q4, q4 + HALF_SIZE_STEP].map(|q| Class {
        start: q >> SUBPEL_BITS,
        phase: q & SUBPEL_MASK,
    })
}

impl crate::Dsp {
    /// Predict a `width` x `height` block (each at most 64, width a
    /// multiple of 4) from a scaled reference, writing it into `dst` (rows
    /// `dst_stride` apart), or averaging it with what is there
    /// (`(a + b + 1) >> 1`, `vpx_scaled_avg_2d`) when `average`. `x` and `y`
    /// are the start phase (below 16) and the step (1..=32) in sixteenths.
    ///
    /// `window` holds the filter footprint with three extra rows above and
    /// four below (rows `stride` apart): its row 3, column 0 is the top-left
    /// sample of the footprint of [`crate::vp9_convolve_scaled`], whose output
    /// this equals bit for bit.
    ///
    /// # Panics
    /// If a size, phase or step is out of range, or `window` or `dst` is too
    /// small.
    #[allow(
        clippy::too_many_arguments,
        reason = "the scaled prediction's full description"
    )]
    pub fn vp9_convolve_scaled_into(
        &self,
        kind: Vp9Filter,
        (window, stride): (&[u8], usize),
        (x0, x_step): (usize, usize),
        (y0, y_step): (usize, usize),
        (width, height): (usize, usize),
        (dst, dst_stride): (&mut [u8], usize),
        average: bool,
        scratch: &mut Vp9ScaledScratch,
    ) {
        assert!((1..=64).contains(&width) && width.is_multiple_of(4) && (1..=64).contains(&height));
        assert!(
            (1..=crate::VP9_MAX_SCALED_STEP).contains(&x_step)
                && (1..=crate::VP9_MAX_SCALED_STEP).contains(&y_step)
        );
        assert!(x0 < VP9_POSITIONS && y0 < VP9_POSITIONS);
        let rows = (((height - 1) * y_step + y0) >> SUBPEL_BITS) + VP9_TAPS;
        let columns = (((width - 1) * x_step + x0) >> SUBPEL_BITS) + VP9_TAPS;
        crate::mc::generic::check_plane(window, stride, columns, rows + VP9_TAPS - 1);
        if x_step == HALF_SIZE_STEP {
            self.horizontal_half_size(kind, (window, stride), x0, (width, rows), scratch);
        } else {
            horizontal_scalar(
                kind,
                (window, stride),
                (x0, x_step),
                (width, rows),
                &mut scratch.temp,
            );
        }
        let temp = &scratch.temp[..rows * TEMP_STRIDE];
        if y_step == HALF_SIZE_STEP {
            for (parity, class) in half_size_classes(y0).into_iter().enumerate() {
                let count = (height + 1 - parity) / 2;
                if count == 0 {
                    continue;
                }
                let dst = &mut dst[parity * dst_stride..];
                let source = &temp[class.start * TEMP_STRIDE..];
                self.vertical(
                    kind,
                    source,
                    class.phase,
                    (width, count),
                    (dst, 2 * dst_stride),
                    average,
                );
            }
        } else {
            for (y, line) in (0..height).zip(dst.chunks_mut(dst_stride)) {
                let q4 = y0 + y * y_step;
                let source = &temp[(q4 >> SUBPEL_BITS) * TEMP_STRIDE..];
                self.vertical(
                    kind,
                    source,
                    q4 & SUBPEL_MASK,
                    (width, 1),
                    (line, dst_stride),
                    average,
                );
            }
        }
    }

    /// The vertical pass at `phase` of `rows` x `width` outputs from `source`
    /// (the temporary block from the first row of their footprint).
    fn vertical(
        &self,
        kind: Vp9Filter,
        source: &[u8],
        phase: usize,
        size: (usize, usize),
        (dst, dst_stride): (&mut [u8], usize),
        average: bool,
    ) {
        if average {
            self.vp9_convolve_avg_into(
                kind,
                source,
                TEMP_STRIDE,
                (0, phase),
                size,
                dst,
                dst_stride,
            );
        } else {
            self.vp9_convolve_into(kind, source, TEMP_STRIDE, (0, phase), size, dst, dst_stride);
        }
    }

    /// The horizontal pass at step 8 into `scratch.temp`: the even and odd
    /// outputs are unscaled passes, each at one phase, interleaved.
    fn horizontal_half_size(
        &self,
        kind: Vp9Filter,
        (window, stride): (&[u8], usize),
        x0: usize,
        (width, rows): (usize, usize),
        scratch: &mut Vp9ScaledScratch,
    ) {
        let half = width / 2;
        for (class, out) in half_size_classes(x0).into_iter().zip(&mut scratch.halves) {
            for first in (0..rows).step_by(MAX_CALL_ROWS) {
                let count = (rows - first).min(MAX_CALL_ROWS);
                // Window row `first` is three rows above output row `first`,
                // where an unscaled pass at vertical phase 0 starts reading.
                self.vp9_convolve_into(
                    kind,
                    &window[first * stride + class.start..],
                    stride,
                    (class.phase, 0),
                    (half, count),
                    &mut out[first * half..],
                    half,
                );
            }
        }
        let [even, odd] = &scratch.halves;
        for ((line, even), odd) in scratch
            .temp
            .as_chunks_mut::<TEMP_STRIDE>()
            .0
            .iter_mut()
            .zip(even.chunks_exact(half))
            .zip(odd.chunks_exact(half))
            .take(rows)
        {
            for (pair, (&a, &b)) in line[VP9_TAPS_BEFORE..VP9_TAPS_BEFORE + width]
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
                .zip(even.iter().zip(odd))
            {
                pair[0] = a;
                pair[1] = b;
            }
        }
    }
}

/// The horizontal pass at any step, as the scalar oracle computes it.
fn horizontal_scalar(
    kind: Vp9Filter,
    (window, stride): (&[u8], usize),
    (x0, x_step): (usize, usize),
    (width, rows): (usize, usize),
    temp: &mut [u8],
) {
    let kernel = kind.kernel();
    for (row, line) in temp
        .as_chunks_mut::<TEMP_STRIDE>()
        .0
        .iter_mut()
        .take(rows)
        .enumerate()
    {
        let source = &window[(row + VP9_TAPS_BEFORE) * stride..];
        for (x, out) in line[VP9_TAPS_BEFORE..VP9_TAPS_BEFORE + width]
            .iter_mut()
            .enumerate()
        {
            let q4 = x0 + x * x_step;
            *out =
                crate::vp9_mc::filter(&source[q4 >> SUBPEL_BITS..], &kernel[q4 & SUBPEL_MASK], 1);
        }
    }
}
