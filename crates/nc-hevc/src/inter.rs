//! HEVC inter prediction (HEVC §8.5.3.3): fractional sample interpolation,
//! with default (uni-directional) weighted sample prediction or as the
//! 14-bit samples that weighted sample prediction combines.
//!
//! Luma uses the 8-tap filters of Table 8-12 at quarter-sample positions,
//! chroma the 4-tap filters of Table 8-13 at eighth-sample positions. The
//! arithmetic is nc-dsp's [`Dsp::predict`] kernel (`Filter::HevcLuma`,
//! `Filter::HevcChroma`); this module only places the reference window.
//!
//! Reference samples outside the picture are clamped to its edge (the
//! `Clip3` in equations 8-228 and 8-229). Reference planes have borders
//! extended by edge replication, which is the same thing, so a window that
//! stays inside the padded area is used in place; one that leaves it (a
//! vector pointing far outside the picture) is gathered with explicit
//! clamping first.

use nc_common::{Plane, PlaneView, PlaneViewMut};
use nc_dsp::{BlockSize, Dsp, Filter};
use nc_encoder::Mv;

/// Predict the `w`x`h` luma block at `(x, y)` from `reference` displaced by
/// `mv` (quarter samples), writing `w * h` samples row-major into `out`.
pub fn predict_luma(
    dsp: &Dsp,
    reference: &Plane,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    mv: Mv,
    out: &mut [u8],
) {
    predict(dsp, Filter::HevcLuma, 2, reference, x, y, w, h, mv, out);
}

/// Predict the `w`x`h` chroma block at chroma position `(x, y)`: the luma
/// vector `mv` in quarter luma samples is an eighth-sample chroma vector
/// for 4:2:0 (§8.5.3.2.10).
pub fn predict_chroma(
    dsp: &Dsp,
    reference: &Plane,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    mv: Mv,
    out: &mut [u8],
) {
    predict(dsp, Filter::HevcChroma, 3, reference, x, y, w, h, mv, out);
}

/// Shared body: `frac_bits` fractional bits per vector component.
fn predict(
    dsp: &Dsp,
    filter: Filter,
    frac_bits: u32,
    reference: &Plane,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    mv: Mv,
    out: &mut [u8],
) {
    let mut dst = PlaneViewMut::new(out, w, w, h).expect("output covers the block");
    predict_view(dsp, filter, frac_bits, reference, (x, y), mv, &mut dst);
}

/// Predict a block of luma (`chroma` false) or chroma straight into `dst`
/// (its size is the block's, a power of two from 4 to 64 per side), as
/// [`predict_luma`] / [`predict_chroma`] do into a buffer.
pub fn predict_into(
    dsp: &Dsp,
    reference: &Plane,
    (x, y): (usize, usize),
    mv: Mv,
    chroma: bool,
    dst: &mut PlaneViewMut<'_>,
) {
    let (filter, frac_bits) = if chroma {
        (Filter::HevcChroma, 3)
    } else {
        (Filter::HevcLuma, 2)
    };
    predict_view(dsp, filter, frac_bits, reference, (x, y), mv, dst);
}

/// Place the reference window of the block at `(x, y)` and run the kernel
/// into `dst`.
fn predict_view(
    dsp: &Dsp,
    filter: Filter,
    frac_bits: u32,
    reference: &Plane,
    (x, y): (usize, usize),
    mv: Mv,
    dst: &mut PlaneViewMut<'_>,
) {
    let (w, h) = (dst.width(), dst.height());
    let size = BlockSize::new(w, h).expect("prediction blocks are 4..64 square");
    with_window(
        filter,
        frac_bits,
        reference,
        (x, y),
        (w, h),
        mv,
        |window, (fx, fy)| {
            dsp.predict(filter, size, window, fx, fy, dst);
        },
    );
}

/// The centered prediction samples `predSamplesLX - 8192` (HEVC §8.5.3.3.3) of the
/// `w`x`h` luma (`chroma` false) or chroma block at `(x, y)` displaced by
/// `mv`, row-major at stride `w` in `out` (grown as needed; at least the
/// block rounded up to kernel sizes), for weighted sample prediction.
pub fn predict_samples(
    dsp: &Dsp,
    reference: &Plane,
    (x, y): (usize, usize),
    (w, h): (usize, usize),
    mv: Mv,
    chroma: bool,
    out: &mut Vec<i16>,
) {
    let (filter, frac_bits) = if chroma {
        (Filter::HevcChroma, 3)
    } else {
        (Filter::HevcLuma, 2)
    };
    // Odd block sides (chroma 2 and 6, AMP 12, 24 and 48) run the next
    // kernel size up: interpolation is per sample, so the block's samples
    // are the same and the extra ones are dropped.
    let (kw, kh) = (kernel_side(w), kernel_side(h));
    if out.len() < kw * kh {
        out.resize(kw * kh, 0);
    }
    let size = BlockSize::new(kw, kh).expect("kernel sides are 4..64");
    with_window(
        filter,
        frac_bits,
        reference,
        (x, y),
        (kw, kh),
        mv,
        |window, frac| {
            dsp.predict_compound(filter, size, window, frac, out, kw);
        },
    );
    if kw != w {
        for row in 1..h {
            out.copy_within(row * kw..row * kw + w, row * w);
        }
    }
}

/// Default-weight bi-prediction (HEVC eq. 8-251) of a block of luma
/// (`chroma` false) or chroma straight into `dst` (its size is the
/// block's, a power of two from 4 to 64 per side): the second list's
/// prediction from `reference` displaced by `mv`, averaged with the first
/// list's centred prediction `first` ([`predict_samples`], stride the
/// block width).
pub fn predict_bi_into(
    dsp: &Dsp,
    reference: &Plane,
    (x, y): (usize, usize),
    mv: Mv,
    chroma: bool,
    first: &[i16],
    dst: &mut PlaneViewMut<'_>,
) {
    let (filter, frac_bits) = if chroma {
        (Filter::HevcChroma, 3)
    } else {
        (Filter::HevcLuma, 2)
    };
    let (w, h) = (dst.width(), dst.height());
    let size = BlockSize::new(w, h).expect("bi-prediction blocks are 4..64 square");
    with_window(
        filter,
        frac_bits,
        reference,
        (x, y),
        (w, h),
        mv,
        |window, frac| {
            dsp.hevc_predict_bi(filter, size, window, frac, (first, w), dst);
        },
    );
}

/// The smallest kernel block side (4, 8, 16, 32 or 64) covering `n`.
fn kernel_side(n: usize) -> usize {
    n.next_power_of_two().max(4)
}

/// Run `f` on the reference window of the `w`x`h` block at `(x, y)`
/// displaced by `mv` (`frac_bits` fractional bits), with the vector's
/// sub-sample offsets.
fn with_window(
    filter: Filter,
    frac_bits: u32,
    reference: &Plane,
    (x, y): (usize, usize),
    (w, h): (usize, usize),
    mv: Mv,
    run: impl FnOnce(&PlaneView<'_>, (usize, usize)),
) {
    let (int_x, frac_x) = split(i32::from(mv.x), frac_bits);
    let (int_y, frac_y) = split(i32::from(mv.y), frac_bits);
    let before = filter.before() as i32;
    let (wx, wy) = (x as i32 + int_x - before, y as i32 + int_y - before);
    let (ww, wh) = (w + filter.taps() - 1, h + filter.taps() - 1);
    if let Some(window) = reference.region(wx as isize, wy as isize, ww, wh) {
        run(&window, (frac_x, frac_y));
    } else {
        let mut storage = [0; MAX_WINDOW];
        let samples = gather(reference, (wx, wy), (ww, wh), &mut storage);
        let window = PlaneView::new(samples, ww, ww, wh).expect("gathered window");
        run(&window, (frac_x, frac_y));
    }
}

/// Integer and fractional parts of a vector component with `frac_bits`
/// fractional bits.
fn split(v: i32, frac_bits: u32) -> (i32, usize) {
    (v >> frac_bits, (v & ((1 << frac_bits) - 1)) as usize)
}

/// The `w`x`h` reference window at `(x, y)` with coordinates clamped to the
/// picture (§8.5.3.3.3.1, equations 8-228/8-229).
fn gather<'a>(
    reference: &Plane,
    (x, y): (i32, i32),
    (w, h): (usize, usize),
    window: &'a mut [u8; MAX_WINDOW],
) -> &'a [u8] {
    let view = reference.view();
    let max_x = reference.width() as i32 - 1;
    let max_y = reference.height() as i32 - 1;
    let window = &mut window[..w * h];
    for (r, dst) in window.chunks_exact_mut(w).enumerate() {
        let row = view.row((y + r as i32).clamp(0, max_y) as usize);
        for (c, d) in dst.iter_mut().enumerate() {
            *d = row[(x + c as i32).clamp(0, max_x) as usize];
        }
    }
    window
}

/// Samples of the largest reference window: a 64x64 block and the 8-tap
/// filter's 7 extra rows and columns.
const MAX_WINDOW: usize = (64 + 7) * (64 + 7);

#[cfg(test)]
mod tests {
    use super::*;

    // An independent, specification-literal implementation (this module's
    // previous scalar code, verified bit-exact against HM), used as the
    // oracle for the nc-dsp kernels.
    /// Luma interpolation filter coefficients `fL[xFrac]` (Table 8-12).
    const LUMA_FILTER: [[i32; 8]; 4] = [
        [0, 0, 0, 64, 0, 0, 0, 0],
        [-1, 4, -10, 58, 17, -5, 1, 0],
        [-1, 4, -11, 40, 40, -11, 4, -1],
        [0, 1, -5, 17, 58, -10, 4, -1],
    ];
    /// Chroma interpolation filter coefficients `fC[xFrac]` (Table 8-13).
    const CHROMA_FILTER: [[i32; 4]; 8] = [
        [0, 64, 0, 0],
        [-2, 58, 10, -2],
        [-4, 54, 16, -2],
        [-6, 46, 28, -4],
        [-4, 36, 36, -4],
        [-4, 28, 46, -6],
        [-2, 16, 54, -4],
        [-2, 10, 58, -2],
    ];
    /// Taps before the integer position: 3 for luma, 1 for chroma.
    const LUMA_TAPS_BEFORE: usize = 3;
    const CHROMA_TAPS_BEFORE: usize = 1;
    /// `shift2` of §8.5.3.3.3 (8-bit).
    const SHIFT2: u32 = 6;
    /// `shift3`: integer samples are scaled up by this to the intermediate
    /// 14-bit precision.
    const SHIFT3: u32 = 6;
    /// Uni-prediction output shift and offset (§8.5.3.3.4.2, `shift1 = 14 −
    /// BitDepth`).
    const OUTPUT_SHIFT: u32 = 6;
    const OUTPUT_OFFSET: i32 = 1 << (OUTPUT_SHIFT - 1);

    /// Separable `TAPS`-tap interpolation of a `w`x`h` block whose integer
    /// position is `(x0, y0)` (§8.5.3.3.3.1 / .2), with the specification's
    /// four cases: integer, horizontal only, vertical only, and both (a
    /// horizontal pass into 14-bit intermediates, then a vertical pass).
    #[allow(clippy::too_many_arguments)]
    fn interpolate<const TAPS: usize>(
        reference: &Plane,
        x0: i32,
        y0: i32,
        w: usize,
        h: usize,
        fx: &[i32; TAPS],
        fy: &[i32; TAPS],
        before: usize,
        filter_x: bool,
        filter_y: bool,
        out: &mut [u8],
    ) {
        let span = w + TAPS - 1;
        let mut storage = [0; MAX_WINDOW];
        let window = gather(
            reference,
            (x0 - before as i32, y0 - before as i32),
            (span, h + TAPS - 1),
            &mut storage,
        );
        let at = |r: usize, c: usize| i32::from(window[r * span + c]);
        let horizontal =
            |r: usize, c: usize| -> i32 { (0..TAPS).map(|i| fx[i] * at(r, c + i)).sum() };
        let vertical =
            |r: usize, c: usize| -> i32 { (0..TAPS).map(|i| fy[i] * at(r + i, c + before)).sum() };

        let mut put = |r: usize, c: usize, v: i32| {
            out[r * w + c] = ((v + OUTPUT_OFFSET) >> OUTPUT_SHIFT).clamp(0, 255) as u8;
        };
        match (filter_x, filter_y) {
            (false, false) => {
                for r in 0..h {
                    for c in 0..w {
                        put(r, c, at(r + before, c + before) << SHIFT3);
                    }
                }
            }
            (true, false) => {
                for r in 0..h {
                    for c in 0..w {
                        put(r, c, horizontal(r + before, c));
                    }
                }
            }
            (false, true) => {
                for r in 0..h {
                    for c in 0..w {
                        put(r, c, vertical(r, c));
                    }
                }
            }
            (true, true) => {
                let rows = h + TAPS - 1;
                let tmp: Vec<i32> = (0..rows * w).map(|i| horizontal(i / w, i % w)).collect();
                for r in 0..h {
                    for c in 0..w {
                        let v: i32 = (0..TAPS).map(|i| fy[i] * tmp[(r + i) * w + c]).sum();
                        put(r, c, v >> SHIFT2);
                    }
                }
            }
        }
    }

    /// Oracle prediction with the same clamped window as `predict`.
    fn oracle(luma: bool, plane: &Plane, x: usize, y: usize, n: usize, mv: Mv) -> Vec<u8> {
        let mut out = vec![0u8; n * n];
        if luma {
            let (ix, fx) = split(i32::from(mv.x), 2);
            let (iy, fy) = split(i32::from(mv.y), 2);
            interpolate::<8>(
                plane,
                x as i32 + ix,
                y as i32 + iy,
                n,
                n,
                &LUMA_FILTER[fx],
                &LUMA_FILTER[fy],
                LUMA_TAPS_BEFORE,
                fx != 0,
                fy != 0,
                &mut out,
            );
        } else {
            let (ix, fx) = split(i32::from(mv.x), 3);
            let (iy, fy) = split(i32::from(mv.y), 3);
            interpolate::<4>(
                plane,
                x as i32 + ix,
                y as i32 + iy,
                n,
                n,
                &CHROMA_FILTER[fx],
                &CHROMA_FILTER[fy],
                CHROMA_TAPS_BEFORE,
                fx != 0,
                fy != 0,
                &mut out,
            );
        }
        out
    }

    fn noise_plane(width: usize, height: usize, border: usize) -> Plane {
        let mut p = Plane::new(width, height, border).unwrap();
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut v = p.view_mut();
        for y in 0..height {
            for s in v.row_mut(y) {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                *s = (seed >> 56) as u8;
            }
        }
        p.extend_borders();
        p
    }

    #[test]
    fn kernels_match_the_specification_at_every_position() {
        let dsp = Dsp::new();
        let plane = noise_plane(96, 80, 32);
        for n in [4usize, 8, 16, 32, 64] {
            for frac in 0..16i16 {
                for luma in [true, false] {
                    // Inside the picture, inside the border, and far outside
                    // (the gathered fallback).
                    for (x, y, base) in [(16, 8, 0i16), (0, 0, -40), (8, 4, -400)] {
                        let mv = Mv {
                            x: base + frac % 4 + 4 * (frac / 4),
                            y: base - frac,
                        };
                        if x + n > plane.width() || y + n > plane.height() {
                            continue;
                        }
                        let mut got = vec![0u8; n * n];
                        if luma {
                            predict_luma(&dsp, &plane, x, y, n, n, mv, &mut got);
                        } else {
                            predict_chroma(&dsp, &plane, x, y, n, n, mv, &mut got);
                        }
                        assert_eq!(
                            got,
                            oracle(luma, &plane, x, y, n, mv),
                            "luma {luma} n {n} mv {mv:?}"
                        );
                    }
                }
            }
        }
    }
}
