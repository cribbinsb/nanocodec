//! The browser demo's pixel kernels against the scalar ones: I420 to RGBA
//! (canvas path), RGBA to I420 (encoder input) and plane SSE (PSNR), with
//! odd sizes, strides and extreme samples.
use crate::check::{CheckReport, MAX_EXTRA_STRIDE};
use crate::rng::Rng;
use nc_common::{PlaneView, PlaneViewMut};
use nc_dsp::{DisplayMatrix, Dsp};

pub(crate) fn check(
    candidate: &Dsp,
    reference: &Dsp,
    iterations: usize,
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    check_i420_to_rgba_every_sample(candidate, reference, report);
    check_rgba_to_i420_every_colour(candidate, reference, report);
    for _ in 0..iterations {
        check_i420_to_rgba(candidate, reference, rng, report);
        check_rgba_to_i420(candidate, reference, rng, report);
        check_sse(candidate, reference, rng, report);
        check_reconstruction(candidate, reference, rng, report);
    }
}

/// Largest I420 to RGBA width tried: several 16-pixel iterations plus every
/// tail length, odd widths included.
const MAX_RGBA_WIDTH: usize = 70;
/// Largest I420 to RGBA height tried (odd heights share the last chroma row).
const MAX_RGBA_HEIGHT: usize = 5;

/// A random sample, a quarter of them at the 0 and 255 extremes, which
/// drive every channel into both clamps.
fn extreme_byte(rng: &mut Rng) -> u8 {
    match rng.below(8) {
        0 => 0,
        1 => 255,
        _ => rng.byte(),
    }
}

fn check_i420_to_rgba(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    let (w, h) = (
        1 + rng.below(MAX_RGBA_WIDTH),
        1 + rng.below(MAX_RGBA_HEIGHT),
    );
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let stride = w + rng.below(MAX_EXTRA_STRIDE + 1);
    let chroma_stride = cw + rng.below(MAX_EXTRA_STRIDE + 1);
    let mut plane = |len: usize| (0..len).map(|_| extreme_byte(rng)).collect::<Vec<u8>>();
    let (luma, cb, cr) = (
        plane((h - 1) * stride + w),
        plane((ch - 1) * chroma_stride + cw),
        plane((ch - 1) * chroma_stride + cw),
    );
    let view = |data, stride, w, h| PlaneView::new(data, stride, w, h).expect("buffer fits");
    let (luma, cb, cr) = (
        view(&luma, stride, w, h),
        view(&cb, chroma_stride, cw, ch),
        view(&cr, chroma_stride, cw, ch),
    );
    let matrix = DisplayMatrix::ALL[rng.below(DisplayMatrix::ALL.len())];
    let run = |dsp: &Dsp| {
        let mut rgba = vec![0xA5u8; 4 * w * h];
        dsp.i420_to_rgba(matrix, &luma, &cb, &cr, &mut rgba);
        rgba
    };
    report.cases += 1;
    if run(candidate) != run(reference) {
        report.failures.push(format!(
            "{} i420_to_rgba {matrix:?} {w}x{h} stride={stride} chroma_stride={chroma_stride}: outputs differ",
            candidate.isa()
        ));
    }
}

/// Every (Y, Cb, Cr) of every matrix through the row-pair kernels: one
/// 512x256 picture per matrix and Cr value, whose chroma columns hold every Cb and whose luma gives each
/// 2x2 group's four pixels, over the picture's rows, every Y.
fn check_i420_to_rgba_every_sample(candidate: &Dsp, reference: &Dsp, report: &mut CheckReport) {
    const SAMPLES: usize = 256;
    let (w, h) = (2 * SAMPLES, SAMPLES);
    let (cw, ch) = (w / 2, h / 2);
    // Pixel (x, y) of a column pair gets Y = y + 128 * (x & 1) (mod 256):
    // rows 0..256 cover every Y for both pixels of each chroma column.
    let luma: Vec<u8> = (0..w * h)
        .map(|i| ((i / w + SAMPLES / 2 * (i % 2)) % SAMPLES) as u8)
        .collect();
    let cb: Vec<u8> = (0..cw * ch).map(|i| (i % cw) as u8).collect();
    let view = |data, w, h| PlaneView::new(data, w, w, h).expect("buffer fits");
    let luma = view(&luma, w, h);
    let cb = view(&cb, cw, ch);
    let (mut expected, mut actual) = (vec![0u8; 4 * w * h], vec![0u8; 4 * w * h]);
    for matrix in DisplayMatrix::ALL {
        for cr_value in 0..=u8::MAX {
            let cr = vec![cr_value; cw * ch];
            let cr = PlaneView::new(&cr, cw, cw, ch).expect("buffer fits");
            reference.i420_to_rgba(matrix, &luma, &cb, &cr, &mut expected);
            candidate.i420_to_rgba(matrix, &luma, &cb, &cr, &mut actual);
            report.cases += 1;
            if expected != actual {
                report.failures.push(format!(
                    "{} i420_to_rgba {matrix:?} every sample, Cr={cr_value}: outputs differ",
                    candidate.isa()
                ));
            }
        }
    }
}

/// Every (R, G, B) through the forward kernels once: a 4096x4096 picture
/// whose pixel `i` has colour `i` (alpha varies too), so every luma sum
/// and a wide spread of 2x2 chroma means are compared.
fn check_rgba_to_i420_every_colour(candidate: &Dsp, reference: &Dsp, report: &mut CheckReport) {
    const SIDE: usize = 1 << 12;
    let rgba: Vec<u8> = (0..SIDE * SIDE)
        .flat_map(|i| {
            let [r, g, b, _] = (i as u32).to_le_bytes();
            [r, g, b, (i % 251) as u8]
        })
        .collect();
    let half = SIDE / 2;
    let run = |dsp: &Dsp| {
        let (mut luma, mut cb, mut cr) = (
            vec![0u8; SIDE * SIDE],
            vec![0u8; half * half],
            vec![0u8; half * half],
        );
        dsp.rgba_to_i420(
            &rgba,
            &mut PlaneViewMut::new(&mut luma, SIDE, SIDE, SIDE).expect("fits"),
            &mut PlaneViewMut::new(&mut cb, half, half, half).expect("fits"),
            &mut PlaneViewMut::new(&mut cr, half, half, half).expect("fits"),
        );
        (luma, cb, cr)
    };
    report.cases += 1;
    if run(candidate) != run(reference) {
        report.failures.push(format!(
            "{} rgba_to_i420 every colour: outputs differ",
            candidate.isa()
        ));
    }
}

fn check_rgba_to_i420(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    let (w, h) = (
        1 + rng.below(MAX_RGBA_WIDTH),
        1 + rng.below(MAX_RGBA_HEIGHT),
    );
    let (cw, ch) = (w / 2, h / 2);
    let stride = w + rng.below(MAX_EXTRA_STRIDE + 1);
    let chroma_stride = cw.max(1) + rng.below(MAX_EXTRA_STRIDE + 1);
    let rgba: Vec<u8> = (0..4 * w * h).map(|_| extreme_byte(rng)).collect();
    let run = |dsp: &Dsp| {
        // 0xA5 shows writes outside the converted samples (and between rows).
        let mut luma = vec![0xA5u8; (h - 1) * stride + w];
        let chroma_len = ch.saturating_sub(1) * chroma_stride + cw;
        let (mut cb, mut cr) = (vec![0xA5u8; chroma_len], vec![0xA5u8; chroma_len]);
        let view = |data, stride, w, h| PlaneViewMut::new(data, stride, w, h).expect("fits");
        dsp.rgba_to_i420(
            &rgba,
            &mut view(&mut luma, stride, w, h),
            &mut view(&mut cb, chroma_stride, cw, ch),
            &mut view(&mut cr, chroma_stride, cw, ch),
        );
        (luma, cb, cr)
    };
    report.cases += 1;
    if run(candidate) != run(reference) {
        report.failures.push(format!(
            "{} rgba_to_i420 {w}x{h} stride={stride} chroma_stride={chroma_stride}: outputs differ",
            candidate.isa()
        ));
    }
}

/// Largest SSE plane width tried: several 16-sample iterations and every tail.
const MAX_SSE_WIDTH: usize = 80;
/// Largest SSE plane height tried.
const MAX_SSE_HEIGHT: usize = 4;

fn check_sse(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    let (w, h) = (1 + rng.below(MAX_SSE_WIDTH), 1 + rng.below(MAX_SSE_HEIGHT));
    let stride = w + rng.below(MAX_EXTRA_STRIDE + 1);
    let mut plane = || {
        (0..(h - 1) * stride + w)
            .map(|_| extreme_byte(rng))
            .collect::<Vec<u8>>()
    };
    let (a, b) = (plane(), plane());
    let view = |data| PlaneView::new(data, stride, w, h).expect("buffer fits");
    let (a, b) = (view(&a), view(&b));
    report.cases += 1;
    if candidate.sse(&a, &b) != reference.sse(&a, &b) {
        report.failures.push(format!(
            "{} sse {w}x{h} stride={stride}: sums differ",
            candidate.isa()
        ));
    }
}

/// Full i32 extremes, asymmetric strides, untouched padding, and both DC/sample
/// inputs exercise the saturating narrow/add identities used by every ISA.
fn check_reconstruction(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    use nc_dsp::{BlockSize, Residual32};
    for width in [4, 8, 16, 32, 64] {
        let height = 4 << rng.below(5);
        let size = BlockSize::new(width, height).unwrap();
        let ps = width + 3;
        let ds = width + 7;
        let prediction: Vec<_> = (0..ps * height).map(|_| extreme_byte(rng)).collect();
        let values = [
            i32::MIN,
            -32769,
            -32768,
            -256,
            -255,
            -1,
            0,
            1,
            255,
            256,
            32767,
            32768,
            i32::MAX,
        ];
        let residual: Vec<_> = (0..width * height)
            .map(|_| values[rng.below(values.len())])
            .collect();
        let input = PlaneView::new(&prediction, ps, width, height).unwrap();
        for residual in [
            Residual32::Samples(&residual),
            Residual32::Constant(values[rng.below(values.len())]),
        ] {
            let run = |dsp: &Dsp| {
                let mut output = vec![0xa5; 1 + ds * height];
                let mut view = PlaneViewMut::new(&mut output[1..], ds, width, height).unwrap();
                dsp.reconstruct_i32(size, &input, residual, &mut view);
                output
            };
            report.cases += 1;
            if run(candidate) != run(reference) {
                report.failures.push(format!(
                    "{} reconstruct_i32 {width}x{height}: outputs differ",
                    candidate.isa()
                ));
            }
        }
    }
}
