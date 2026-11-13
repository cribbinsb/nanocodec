//! Bit-exactness checks of the HEVC, H.264 and AV2 transforms and quantisers
//! against scalar, on random, extreme and sparse inputs.

use nc_dsp::{
    Av2TxSize, Dsp, H264Quantizer, HevcForwardMode, HevcForwardStaging, HevcQuantizer,
    HevcTransform,
};

use nc_common::PlaneViewMut;

use crate::check::{CheckReport, MAX_EXTRA_STRIDE};
use crate::check_codec::record;
use crate::rng::Rng;

pub(crate) fn check_codec_txfm(
    candidate: &Dsp,
    reference: &Dsp,
    iterations: usize,
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    // Cheap kernels with many parameters (QPs, sizes): more cases.
    for _ in 0..8 * iterations {
        for transform in HevcTransform::ALL {
            check_hevc_transform(candidate, reference, transform, rng, report);
            check_hevc_inverse_add(candidate, reference, transform, rng, report);
        }
        check_hevc_quant(candidate, reference, rng, report);
        check_h264(candidate, reference, rng, report);
        check_nonzero_mask(candidate, reference, rng, report);
    }
    for _ in 0..2 * iterations {
        for size in Av2TxSize::ALL {
            check_av2_transform(candidate, reference, size, rng, report);
            check_av2_inverse_corner(candidate, reference, size, rng, report);
        }
    }
}

/// Most levels `nonzero_mask` takes (an 8x8 block).
const MAX_MASK_LEVELS: usize = 64;

/// `nonzero_mask` of every length up to 64 (whole SIMD groups and every
/// tail), on random, extreme, small and sparse levels.
fn check_nonzero_mask(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    let len = rng.below(MAX_MASK_LEVELS + 1);
    let levels = coefficients(len, rng.below(4), rng);
    let (got, want) = (
        candidate.nonzero_mask(&levels),
        reference.nonzero_mask(&levels),
    );
    record(report, candidate.isa(), got == want, || {
        format!("nonzero_mask len {len}: got {got:#x}, want {want:#x} ({levels:?})")
    });
}

/// A residual of 8-bit sample differences in one of four patterns: random,
/// random ±255, a ±255 checkerboard, small noise.
fn residual(len: usize, stride: usize, mode: usize, rng: &mut Rng) -> Vec<i16> {
    (0..len)
        .map(|i| match mode {
            0 => rng.below(511) as i16 - 255,
            1 => {
                if rng.below(2) == 0 {
                    255
                } else {
                    -255
                }
            }
            2 => {
                if ((i % stride) + i / stride).is_multiple_of(2) {
                    255
                } else {
                    -255
                }
            }
            _ => rng.below(21) as i16 - 10,
        })
        .collect()
}

/// A ±255 residual with the signs of one 2-D DCT basis function (random
/// horizontal and vertical frequencies): it drives that coefficient, and
/// every row output of its frequency, to its largest magnitude.
fn basis_signs(n: usize, stride: usize, rng: &mut Rng) -> Vec<i16> {
    let sign = |k: usize, i: usize| {
        let phase = std::f64::consts::PI * ((2 * i + 1) * k) as f64 / (2 * n) as f64;
        if phase.cos() >= 0.0 { 1 } else { -1 }
    };
    let (u, v) = (rng.below(n), rng.below(n));
    (0..(n - 1) * stride + n)
        .map(|i| 255 * sign(u, i % stride) * sign(v, i / stride))
        .collect()
}

/// Coefficients within 16 bits: random, extreme, small or sparse.
fn coefficients(len: usize, mode: usize, rng: &mut Rng) -> Vec<i32> {
    (0..len)
        .map(|_| match mode {
            0 => rng.below(1 << 16) as i32 - (1 << 15),
            1 => {
                if rng.below(2) == 0 {
                    32767
                } else {
                    -32768
                }
            }
            2 => rng.below(401) as i32 - 200,
            _ => {
                if rng.below(8) == 0 {
                    rng.below(20001) as i32 - 10000
                } else {
                    0
                }
            }
        })
        .collect()
}

fn check_hevc_transform(
    candidate: &Dsp,
    reference: &Dsp,
    transform: HevcTransform,
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    let n = transform.side();
    let isa = candidate.isa();
    let stride = n + rng.below(MAX_EXTRA_STRIDE + 1);
    let mode = rng.below(4);
    let input = residual((n - 1) * stride + n, stride, mode, rng);
    let forward = |dsp: &Dsp| {
        let mut out = vec![0i32; n * n];
        dsp.hevc_forward(transform, &input, stride, &mut out);
        out
    };
    let same = forward(candidate) == forward(reference);
    record(report, isa, same, || {
        format!("hevc_forward {transform:?} mode={mode} stride={stride}")
    });
    if matches!(n, 16 | 32) {
        check_hevc_forward_modes(candidate, reference, transform, &input, stride, rng, report);
        // The largest row outputs: kernels that carry them in 16 bits must
        // not wrap or saturate.
        let peak = basis_signs(n, stride, rng);
        check_hevc_forward_modes(candidate, reference, transform, &peak, stride, rng, report);
    }
    let coeffs = coefficients(n * n, mode, rng);
    let inverse = |dsp: &Dsp| {
        let mut out = vec![0i32; n * n];
        dsp.hevc_inverse(transform, &coeffs, &mut out);
        out
    };
    let same = inverse(candidate) == inverse(reference);
    record(report, isa, same, || {
        format!("hevc_inverse {transform:?} mode={mode}")
    });
    // Encoder blocks are nonzero in a top-left corner only (inverses may
    // skip the zero rest): every extent, including an all-zero block.
    let (rows, columns) = (rng.below(n + 1), rng.below(n + 1));
    let mut corner = coeffs;
    for (i, c) in corner.iter_mut().enumerate() {
        if i / n >= rows || i % n >= columns {
            *c = 0;
        }
    }
    let inverse = |dsp: &Dsp| {
        let mut out = vec![0i32; n * n];
        dsp.hevc_inverse(transform, &corner, &mut out);
        out
    };
    let same = inverse(candidate) == inverse(reference);
    record(report, isa, same, || {
        format!("hevc_inverse {transform:?} mode={mode} corner {rows}x{columns}")
    });
}

/// Fused coefficient input against the independent full scalar inverse followed by
/// pixel addition. Padding, clipping and all-zero/DC/low-corner paths matter.
fn check_hevc_inverse_add(
    candidate: &Dsp,
    reference: &Dsp,
    transform: HevcTransform,
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    let n = transform.side();
    let stride = n + 3;
    let mode = rng.below(5);
    for (rows, columns) in [
        (0, 0),
        (1, 1),
        (4, 4),
        (8, 8),
        (n, n),
        (rng.below(n + 1), rng.below(n + 1)),
    ] {
        let (rows, columns) = (rows.min(n), columns.min(n));
        let mut coeffs = if mode == 4 {
            (0..n * n)
                .map(|i| {
                    if i.is_multiple_of(2) {
                        i32::MIN
                    } else {
                        i32::MAX
                    }
                })
                .collect()
        } else {
            coefficients(n * n, mode, rng)
        };
        for (i, c) in coeffs.iter_mut().enumerate() {
            if i / n >= rows || i % n >= columns {
                *c = 0;
            }
        }
        let mut residual = vec![0; n * n];
        reference.hevc_inverse(transform, &coeffs, &mut residual);
        let mut expected = vec![0; stride * (n + 2) + 1];
        rng.fill(&mut expected);
        let prediction = expected.clone();
        for y in 0..n {
            for x in 0..n {
                let at = 1 + y * stride + x;
                expected[at] = (i32::from(expected[at]) + residual[y * n + x]).clamp(0, 255) as u8;
            }
        }
        // The corner itself (as a decoder passes it) and an over-estimate.
        for extent in [(rows, columns), (n, n)] {
            let mut actual = prediction.clone();
            candidate.hevc_inverse_add(
                transform,
                &coeffs,
                extent,
                &mut PlaneViewMut::new(&mut actual[1..], stride, n, n).unwrap(),
            );
            record(report, candidate.isa(), expected == actual, || {
                format!(
                    "hevc_inverse_add {transform:?} mode={mode} corner={rows}x{columns} extent={extent:?}"
                )
            });
        }
    }
}

/// The encoder-side 16/32-point forwards (`hevc_forward_staged_unless_zero`)
/// in a random [`HevcForwardMode`] and proven-zero limit: the same
/// decision and coefficients as scalar.
fn check_hevc_forward_modes(
    candidate: &Dsp,
    reference: &Dsp,
    transform: HevcTransform,
    input: &[i16],
    stride: usize,
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    const STAGINGS: [HevcForwardStaging; 3] = [
        HevcForwardStaging::Exact,
        HevcForwardStaging::Rows,
        HevcForwardStaging::RowsAndColumns,
    ];
    const LIMITS: [u32; 5] = [0, 7, 31, 127, 511];
    let n = transform.side();
    let forward_mode = HevcForwardMode {
        staging: STAGINGS[rng.below(STAGINGS.len())],
        zero_columns: rng.below(2) == 0,
        low_frequencies: rng.below(2) == 0,
    };
    if forward_mode == HevcForwardMode::EXACT {
        return;
    }
    let limit = LIMITS[rng.below(LIMITS.len())];
    let forward = |dsp: &Dsp| {
        let mut out = vec![i32::MIN; n * n];
        let coded = dsp.hevc_forward_staged_unless_zero(
            transform,
            input,
            stride,
            limit,
            forward_mode,
            &mut out,
        );
        (coded, out)
    };
    let same = forward(candidate) == forward(reference);
    record(report, candidate.isa(), same, || {
        format!(
            "hevc_forward_staged_unless_zero {transform:?} {forward_mode:?} limit={limit} stride={stride}"
        )
    });
}

fn check_av2_transform(
    candidate: &Dsp,
    reference: &Dsp,
    size: Av2TxSize,
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    let (n, coded) = (size.side(), size.coded_area());
    let isa = candidate.isa();
    let stride = n + rng.below(MAX_EXTRA_STRIDE + 1);
    let mode = rng.below(4);
    let input = residual((n - 1) * stride + n, stride, mode, rng);
    let forward = |dsp: &Dsp| {
        let mut out = vec![0i32; coded];
        dsp.av2_forward(size, &input, stride, &mut out);
        out
    };
    let same = forward(candidate) == forward(reference);
    record(report, isa, same, || {
        format!("av2_forward {size:?} mode={mode} stride={stride}")
    });
    // Beyond 16 bits too: the inverse clamps its input (AVM `clamp_buf`).
    let coeffs: Vec<i32> = coefficients(coded, mode, rng)
        .into_iter()
        .map(|c| if mode == 1 { c * 4 } else { c })
        .collect();
    let inverse = |dsp: &Dsp| {
        let mut out = vec![0i32; n * n];
        dsp.av2_inverse(size, &coeffs, &mut out);
        out
    };
    let same = inverse(candidate) == inverse(reference);
    record(report, isa, same, || {
        format!("av2_inverse {size:?} mode={mode}")
    });
}

fn check_hevc_quant(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    let isa = candidate.isa();
    let qp = rng.below(52) as u8;
    let log2 = 2 + rng.below(4) as u8;
    let intra = rng.below(2) == 0;
    let q = HevcQuantizer::new(qp, log2, intra);
    let n = 1usize << log2;
    // Quantiser inputs: anything the forward transform can produce.
    let mode = rng.below(4);
    let coeffs: Vec<i32> = coefficients(n * n, mode, rng)
        .into_iter()
        .map(|c| if mode == 1 && c < 0 { -32768 } else { c })
        .collect();
    let quantize = |dsp: &Dsp| {
        let mut levels = coeffs.clone();
        let nonzero = dsp.hevc_quantize(&mut levels, &q);
        (levels, nonzero)
    };
    let same = quantize(candidate) == quantize(reference);
    record(report, isa, same, || {
        format!("hevc_quantize qp={qp} log2={log2} intra={intra} mode={mode}")
    });
    // Levels of any i32 value: the dequantiser is exact for all of them.
    let levels: Vec<i32> = (0..n * n)
        .map(|_| match rng.below(3) {
            0 => rng.below(2001) as i32 - 1000,
            1 => rng.next_u64() as i32,
            _ => rng.below(1 << 18) as i32 - (1 << 17),
        })
        .collect();
    let dequantize = |dsp: &Dsp| {
        let mut out = vec![0i32; n * n];
        dsp.hevc_dequantize(&levels, &q, &mut out);
        out
    };
    let same = dequantize(candidate) == dequantize(reference);
    record(report, isa, same, || {
        format!("hevc_dequantize qp={qp} log2={log2}")
    });
}

fn blocks<const N: usize>(count: usize, mode: usize, rng: &mut Rng) -> Vec<[i32; N]> {
    let flat = coefficients(count * N, mode, rng);
    flat.as_chunks::<N>().0.to_vec()
}

fn check_h264_mb(
    candidate: &Dsp,
    reference: &Dsp,
    qp: u8,
    mode: usize,
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    let isa = candidate.isa();
    let stride_mb = 16 + rng.below(MAX_EXTRA_STRIDE + 1);
    let input_mb = residual(15 * stride_mb + 16, stride_mb, mode, rng);
    let zero_mask = rng.below(16) as u8;
    let q_mb = H264Quantizer::with_rounding_divisor(qp, 2 + rng.below(15) as u8);
    let mut fused = [[99; 64]; 4];
    let fused_counts =
        candidate.h264_forward_quantize_mb(&input_mb, stride_mb, zero_mask, &q_mb, &mut fused);
    let mut separate = [[0; 64]; 4];
    let mut separate_counts = [0; 4];
    for (i, block) in separate.iter_mut().enumerate() {
        if zero_mask & (1 << i) == 0 {
            let start = (i / 2) * 8 * stride_mb + (i % 2) * 8;
            reference.h264_forward_8x8(&input_mb[start..], stride_mb, block);
            separate_counts[i] = reference.h264_quantize_8x8(block, &q_mb);
        }
    }
    record(
        report,
        isa,
        fused == separate && fused_counts == separate_counts,
        || {
            format!(
                "h264_forward_quantize_mb qp={qp} zero={zero_mask} mode={mode} stride={stride_mb}"
            )
        },
    );
}

fn check_h264(candidate: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    let isa = candidate.isa();
    let mode = rng.below(4);
    // A region of up to 4x4 blocks at a random stride.
    let (bw, bh) = (1 + rng.below(4), 1 + rng.below(4));
    let stride = 4 * bw + rng.below(MAX_EXTRA_STRIDE + 1);
    let input = residual((4 * bh - 1) * stride + 4 * bw, stride, mode, rng);
    let forward = |dsp: &Dsp| {
        let mut out = vec![[0i32; 16]; bw * bh];
        dsp.h264_forward_4x4(&input, stride, bw, &mut out);
        out
    };
    let same = forward(candidate) == forward(reference);
    record(report, isa, same, || {
        format!("h264_forward_4x4 {bw}x{bh} blocks mode={mode} stride={stride}")
    });

    let coeffs = blocks::<16>(bw * bh, mode, rng);
    let inverse = |dsp: &Dsp| {
        let mut out = vec![[0i32; 16]; coeffs.len()];
        dsp.h264_inverse_4x4(&coeffs, &mut out);
        let hadamard: Vec<[i32; 16]> = coeffs.iter().map(|c| dsp.h264_hadamard_4x4(c)).collect();
        (out, hadamard)
    };
    let same = inverse(candidate) == inverse(reference);
    record(report, isa, same, || {
        format!("h264_inverse_4x4 / hadamard mode={mode}")
    });

    let stride8 = 8 + rng.below(MAX_EXTRA_STRIDE + 1);
    let input8 = residual(7 * stride8 + 8, stride8, mode, rng);
    let coeffs8 = blocks::<64>(1, mode, rng)[0];
    let eight = |dsp: &Dsp| {
        let (mut f, mut i) = ([0i32; 64], [0i32; 64]);
        dsp.h264_forward_8x8(&input8, stride8, &mut f);
        dsp.h264_inverse_8x8(&coeffs8, &mut i);
        (f, i)
    };
    let same = eight(candidate) == eight(reference);
    record(report, isa, same, || {
        format!("h264_forward_8x8 / inverse_8x8 mode={mode} stride={stride8}")
    });

    let qp = rng.below(52) as u8;
    let intra = rng.below(2) == 0;
    let q = H264Quantizer::new(qp, intra);
    check_h264_mb(candidate, reference, qp, mode, rng, report);
    // Quantiser inputs within transform output range (|c| <= 2^15).
    let to_quantize = blocks::<16>(bw * bh, mode, rng);
    let dc: Vec<i32> = coefficients(if rng.below(2) == 0 { 4 } else { 16 }, mode, rng);
    let to_quantize8 = blocks::<64>(1, mode, rng)[0];
    let quantize = |dsp: &Dsp| {
        let mut a = to_quantize.clone();
        let mut d = dc.clone();
        let mut e = to_quantize8;
        let counts = (
            dsp.h264_quantize_4x4(&mut a, &q),
            dsp.h264_quantize_dc(&mut d, &q),
            dsp.h264_quantize_8x8(&mut e, &q),
        );
        (a, d, e, counts)
    };
    let same = quantize(candidate) == quantize(reference);
    record(report, isa, same, || {
        format!("h264_quantize qp={qp} intra={intra} mode={mode}")
    });

    // Levels as a conformant stream carries them (scaled values fit 16 bits).
    let levels = blocks::<16>(bw * bh, 2, rng);
    let dc_levels = blocks::<16>(1, 2, rng)[0];
    let chroma_dc = [dc_levels[0], dc_levels[1], dc_levels[2], dc_levels[3]];
    let levels8 = blocks::<64>(1, 2, rng)[0];
    let dequantize = |dsp: &Dsp| {
        let mut out = vec![[0i32; 16]; levels.len()];
        dsp.h264_dequantize_4x4(&levels, &q, &mut out);
        let mut out8 = [0i32; 64];
        dsp.h264_dequantize_8x8(&levels8, &q, &mut out8);
        (
            out,
            out8,
            dsp.h264_dequantize_luma_dc(&dc_levels, &q),
            dsp.h264_dequantize_chroma_dc(&chroma_dc, &q),
        )
    };
    let same = dequantize(candidate) == dequantize(reference);
    record(report, isa, same, || format!("h264_dequantize qp={qp}"));
    check_h264_add_inverse(candidate, reference, (&q, qp), rng, report);

    let stride_c = 8 + rng.below(MAX_EXTRA_STRIDE + 1);
    let input_c = residual(7 * stride_c + 8, stride_c, mode, rng);
    let chroma = |dsp: &Dsp| {
        let mut ac = [[99; 16]; 4];
        let (dc, count) = dsp.h264_forward_quantize_chroma(&input_c, stride_c, &q, &mut ac);
        (ac, dc, count)
    };
    let same = chroma(candidate) == chroma(reference);
    record(report, isa, same, || {
        format!("h264_forward_quantize_chroma qp={qp} intra={intra} mode={mode} stride={stride_c}")
    });
}

/// The fused scaling, inverse and add of H.264 reconstruction, on a
/// region of random prediction samples at a random stride, with blocks
/// made empty or DC-only at random (the kernels' shortcuts).
fn check_h264_add_inverse(
    candidate: &Dsp,
    reference: &Dsp,
    (q, qp): (&H264Quantizer, u8),
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    // Levels whose scaled values fit 16 bits, as in a conformant stream
    // (§8.5.12.1): the largest flat scale (8x8: 16 · 32) at `qP / 6`.
    const MAX_SCALED: i32 = 1 << 15;
    const MAX_SCALE: i32 = 16 * 32;
    let max_level = (MAX_SCALED * 16 / (MAX_SCALE << (qp / 6))).max(1);
    let isa = candidate.isa();
    let (bw, bh) = (1 + rng.below(4), 1 + rng.below(4));
    let stride = 4 * bw.max(2) + rng.below(MAX_EXTRA_STRIDE + 1);
    let rows = 4 * bh.max(2);
    let pred: Vec<u8> = (0..stride * rows).map(|_| rng.below(256) as u8).collect();
    let sparse = |block: &mut [i32], rng: &mut Rng| {
        for level in block.iter_mut() {
            *level = (*level).clamp(-max_level, max_level);
        }
        match rng.below(3) {
            0 => block.fill(0),
            1 => block[1..].fill(0),
            _ => {}
        }
    };
    let mut levels = blocks::<16>(bw * bh, 2, rng);
    for block in &mut levels {
        sparse(block, rng);
    }
    let dc: Vec<i32> = coefficients(bw * bh, 2, rng)
        .into_iter()
        .map(|d| d.clamp(-MAX_SCALED, MAX_SCALED - 1))
        .collect();
    let with_dc = rng.below(2) == 0;
    let mut levels8 = blocks::<64>(1, 2, rng)[0];
    sparse(&mut levels8, rng);
    let add = |dsp: &Dsp| {
        let mut four = pred.clone();
        let mut view = PlaneViewMut::new(&mut four, stride, 4 * bw, 4 * bh).expect("region");
        dsp.h264_add_inverse_4x4(&levels, with_dc.then_some(&dc[..]), q, bw, &mut view);
        let mut eight = pred.clone();
        let mut view = PlaneViewMut::new(&mut eight, stride, 8, 8).expect("region");
        dsp.h264_add_inverse_8x8(&levels8, q, &mut view);
        (four, eight)
    };
    let same = add(candidate) == add(reference);
    record(report, isa, same, || {
        format!("h264_add_inverse {bw}x{bh} blocks dc={with_dc} stride={stride}")
    });
    // The decoder's form: coefficients already scaled (with any scaling
    // matrix) into the 16-bit range of a conformant stream.
    let scaled = |block: &mut [i32], rng: &mut Rng| {
        for c in block.iter_mut() {
            *c = (*c).clamp(-MAX_SCALED, MAX_SCALED - 1);
        }
        match rng.below(3) {
            0 => block.fill(0),
            1 => block[1..].fill(0),
            _ => {}
        }
    };
    let mut coeffs = blocks::<16>(bw * bh, 2, rng);
    for block in &mut coeffs {
        scaled(block, rng);
    }
    let mut coeffs8 = blocks::<64>(1, 2, rng)[0];
    scaled(&mut coeffs8, rng);
    let add_scaled = |dsp: &Dsp| {
        let mut four = pred.clone();
        let mut view = PlaneViewMut::new(&mut four, stride, 4 * bw, 4 * bh).expect("region");
        dsp.h264_add_scaled_4x4(&coeffs, bw, &mut view);
        let mut eight = pred.clone();
        let mut view = PlaneViewMut::new(&mut eight, stride, 8, 8).expect("region");
        dsp.h264_add_scaled_8x8(&coeffs8, &mut view);
        (four, eight)
    };
    let same = add_scaled(candidate) == add_scaled(reference);
    record(report, isa, same, || {
        format!("h264_add_scaled {bw}x{bh} blocks stride={stride}")
    });
    // The macroblock form with the caller's coded / AC masks: blocks
    // outside `coded` hold garbage the kernel must not read, blocks outside
    // `ac` only a DC.
    let mut mb = [[0i32; 16]; 16];
    for (b, block) in mb.iter_mut().enumerate() {
        *block = coeffs[b % coeffs.len()];
    }
    let (coded, ac) = (rng.below(1 << 16) as u16, rng.below(1 << 16) as u16);
    for (b, block) in mb.iter_mut().enumerate() {
        if (coded >> b) & 1 == 1 && (ac >> b) & 1 == 0 {
            block[1..].fill(0);
        }
    }
    let stride_mb = 16 + rng.below(MAX_EXTRA_STRIDE + 1);
    let pred_mb: Vec<u8> = (0..stride_mb * 16).map(|_| rng.below(256) as u8).collect();
    let add_mb = |dsp: &Dsp| {
        let mut out = pred_mb.clone();
        let mut view = PlaneViewMut::new(&mut out, stride_mb, 16, 16).expect("region");
        dsp.h264_add_scaled_mb(&mb, coded, ac, &mut view);
        out
    };
    let same = add_mb(candidate) == add_mb(reference);
    record(report, isa, same, || {
        format!("h264_add_scaled_mb coded={coded:#06x} ac={ac:#06x} stride={stride_mb}")
    });
}

/// The AV2 inverse DCT of coefficients confined to a random top-left corner
/// (as an end of block bounds them), told the corner: the kernel may skip
/// the zero rows, and must still equal the reference's full inverse. Values
/// outside the square it may read (`Av2TxSize::inverse_read_side`) are
/// garbage, which it must ignore.
fn check_av2_inverse_corner(
    candidate: &Dsp,
    reference: &Dsp,
    size: Av2TxSize,
    rng: &mut Rng,
    report: &mut CheckReport,
) {
    let (n, coded) = (size.side(), size.coded_area());
    let isa = candidate.isa();
    let mode = rng.below(4);
    let coeffs = coefficients(coded, mode, rng);
    let side = size.coded_side();
    let corner = rng.below(side + 1);
    let cornered: Vec<i32> = coeffs
        .iter()
        .enumerate()
        .map(|(i, &c)| {
            if i / side < corner && i % side < corner {
                c
            } else {
                0
            }
        })
        .collect();
    let read = size.inverse_read_side(corner);
    let unread: Vec<i32> = cornered
        .iter()
        .zip(&coeffs)
        .enumerate()
        .map(|(i, (&c, &garbage))| {
            if i / side < read && i % side < read {
                c
            } else {
                garbage | 1
            }
        })
        .collect();
    let mut partial = vec![0i32; n * n];
    candidate.av2_inverse_corner(size, &unread, corner, &mut partial);
    let mut full = vec![0i32; n * n];
    reference.av2_inverse(size, &cornered, &mut full);
    record(report, isa, partial == full, || {
        format!("av2_inverse_corner {size:?} mode={mode} corner={corner}")
    });
}
