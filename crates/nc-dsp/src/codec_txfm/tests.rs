//! The scalar kernels against literal ports of the backends' scalar code
//! (`nc-hevc` `transform.rs` / `quant.rs`, `nc-h264` `transform.rs`), and
//! the best ISA against scalar. `nc-checkasm` repeats the ISA comparison on
//! many more inputs.

use super::{
    DCT32, DST4, H264Quantizer, HevcForwardMode, HevcForwardStaging, HevcQuantizer, HevcTransform,
    QUANT_MF_8X8_POS,
};
use crate::{Dsp, Isa};

/// xorshift64 values in `-amplitude..=amplitude`.
struct Rng(u64);

impl Rng {
    fn next(&mut self, amplitude: i32) -> i32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % (2 * amplitude as u64 + 1)) as i32 - amplitude
    }
}

fn dsps() -> [Dsp; 2] {
    [Dsp::with_isa(Isa::Scalar).unwrap(), Dsp::new()]
}

// ----------------------------------------------------- HEVC reference

fn basis(transform: HevcTransform, k: usize, n: usize) -> i64 {
    match transform {
        HevcTransform::Dst4 => i64::from(DST4[k][n]),
        _ => i64::from(DCT32[k << (5 - transform.log2())][n]),
    }
}

fn round_shift_i64(value: i64, shift: u32) -> i64 {
    (value + (1 << (shift - 1))) >> shift
}

/// `nc-hevc` `transform::forward`, as a plain matrix product in 64 bits.
fn hevc_forward_ref(t: HevcTransform, residual: &[i16]) -> Vec<i32> {
    let n = t.side();
    let log2 = u32::from(t.log2());
    let mut tmp = vec![0i64; n * n];
    for y in 0..n {
        for k in 0..n {
            let v: i64 = (0..n)
                .map(|i| basis(t, k, i) * i64::from(residual[y * n + i]))
                .sum();
            tmp[y * n + k] = round_shift_i64(v, log2 - 1);
        }
    }
    let mut out = vec![0i32; n * n];
    for x in 0..n {
        for k in 0..n {
            let v: i64 = (0..n).map(|y| basis(t, k, y) * tmp[y * n + x]).sum();
            out[k * n + x] = round_shift_i64(v, log2 + 6).clamp(-32768, 32767) as i32;
        }
    }
    out
}

/// `nc-hevc` `transform::inverse`.
#[allow(clippy::many_single_char_names)]
fn hevc_inverse_ref(t: HevcTransform, d: &[i32]) -> Vec<i32> {
    let n = t.side();
    let mut g = vec![0i64; n * n];
    for x in 0..n {
        for y in 0..n {
            let v: i64 = (0..n)
                .map(|k| basis(t, k, y) * i64::from(d[k * n + x]))
                .sum();
            g[y * n + x] = round_shift_i64(v, 7).clamp(-32768, 32767);
        }
    }
    let mut out = vec![0i32; n * n];
    for y in 0..n {
        for x in 0..n {
            let v: i64 = (0..n).map(|k| basis(t, k, x) * g[y * n + k]).sum();
            out[y * n + x] = round_shift_i64(v, 12) as i32;
        }
    }
    out
}

/// `nc-hevc` `quant::quantize`.
fn hevc_quantize_ref(coeffs: &mut [i32], qp: i32, log2: u8, intra: bool) {
    const SCALE: [i64; 6] = [26214, 23302, 20560, 18396, 16384, 14564];
    let qbits = 14 + qp / 6 + (15 - 8 - i32::from(log2));
    let offset = (if intra { 171i64 } else { 85 }) << (qbits - 9);
    for c in coeffs {
        let magnitude = (i64::from(c.unsigned_abs()) * SCALE[(qp % 6) as usize] + offset) >> qbits;
        let level = magnitude.min(32767) as i32;
        *c = if *c < 0 { -level } else { level };
    }
}

/// `nc-hevc` `quant::dequantize`.
fn hevc_dequantize_ref(levels: &[i32], qp: i32, log2: u8) -> Vec<i32> {
    const LEVEL_SCALE: [i64; 6] = [40, 45, 51, 57, 64, 72];
    let bd_shift = 8 + i32::from(log2) - 5;
    let scale = 16 * LEVEL_SCALE[(qp % 6) as usize];
    levels
        .iter()
        .map(|&l| {
            let v = ((i64::from(l) * scale) << (qp / 6)) + (1i64 << (bd_shift - 1));
            (v >> bd_shift).clamp(-32768, 32767) as i32
        })
        .collect()
}

#[test]
fn hevc_transforms_match_the_backend() {
    let mut rng = Rng(0x1234_5678_9abc_def1);
    for dsp in dsps() {
        for t in HevcTransform::ALL {
            let n = t.side();
            for amplitude in [0, 1, 20, 255] {
                for _ in 0..20 {
                    let residual: Vec<i16> =
                        (0..n * n).map(|_| rng.next(amplitude) as i16).collect();
                    let mut got = vec![0; n * n];
                    dsp.hevc_forward(t, &residual, n, &mut got);
                    assert_eq!(got, hevc_forward_ref(t, &residual), "{t:?} forward");
                }
            }
            for amplitude in [1, 300, 32768] {
                for _ in 0..20 {
                    let d: Vec<i32> = (0..n * n).map(|_| rng.next(amplitude)).collect();
                    let d: Vec<i32> = d.iter().map(|&v| v.clamp(-32768, 32767)).collect();
                    let mut got = vec![0; n * n];
                    dsp.hevc_inverse(t, &d, &mut got);
                    assert_eq!(got, hevc_inverse_ref(t, &d), "{t:?} inverse");
                }
            }
        }
    }
}

#[test]
fn hevc_quantisers_match_the_backend() {
    let mut rng = Rng(0x0bad_cafe_dead_beef);
    for dsp in dsps() {
        for qp in [0u8, 4, 22, 37, 51] {
            for log2 in 2..=5u8 {
                let n = 1 << log2;
                for intra in [false, true] {
                    let q = HevcQuantizer::new(qp, log2, intra);
                    let coeffs: Vec<i32> = (0..n * n).map(|_| rng.next(32768)).collect();
                    let mut got = coeffs.clone();
                    let nonzero = dsp.hevc_quantize(&mut got, &q);
                    let mut separate = vec![i32::MIN; n * n + 3];
                    let separate_nonzero = dsp.hevc_quantize_from(&coeffs, &q, &mut separate);
                    assert_eq!(&separate[..n * n], got);
                    assert_eq!(&separate[n * n..], &[i32::MIN; 3]);
                    assert_eq!(separate_nonzero, nonzero);
                    let mut want = coeffs.clone();
                    hevc_quantize_ref(&mut want, i32::from(qp), log2, intra);
                    assert_eq!(got, want, "qp {qp} log2 {log2}");
                    assert_eq!(nonzero as usize, want.iter().filter(|&&l| l != 0).count());
                }
                let q = HevcQuantizer::new(qp, log2, false);
                for amplitude in [3, 4000, 70000, i32::MAX / 2] {
                    let levels: Vec<i32> = (0..n * n).map(|_| rng.next(amplitude)).collect();
                    let mut got = vec![0; n * n];
                    dsp.hevc_dequantize(&levels, &q, &mut got);
                    assert_eq!(
                        got,
                        hevc_dequantize_ref(&levels, i32::from(qp), log2),
                        "qp {qp} log2 {log2} amplitude {amplitude}"
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------- H.264 reference

/// `nc-h264` `transform::forward_4x4`.
fn h264_forward_4x4_ref(block: &[i32; 16]) -> [i32; 16] {
    let mut tmp = [0i32; 16];
    for r in 0..4 {
        let [a, b, c, d] = [
            block[r * 4],
            block[r * 4 + 1],
            block[r * 4 + 2],
            block[r * 4 + 3],
        ];
        let (s03, d03, s12, d12) = (a + d, a - d, b + c, b - c);
        tmp[r * 4] = s03 + s12;
        tmp[r * 4 + 1] = 2 * d03 + d12;
        tmp[r * 4 + 2] = s03 - s12;
        tmp[r * 4 + 3] = d03 - 2 * d12;
    }
    let mut out = [0i32; 16];
    for c in 0..4 {
        let [a, b, cc, d] = [tmp[c], tmp[4 + c], tmp[8 + c], tmp[12 + c]];
        let (s03, d03, s12, d12) = (a + d, a - d, b + cc, b - cc);
        out[c] = s03 + s12;
        out[4 + c] = 2 * d03 + d12;
        out[8 + c] = s03 - s12;
        out[12 + c] = d03 - 2 * d12;
    }
    out
}

/// `nc-h264` `transform::inverse_4x4`.
fn h264_inverse_4x4_ref(d: &[i32; 16]) -> [i32; 16] {
    let mut f = [0i32; 16];
    for i in 0..4 {
        let row = &d[i * 4..i * 4 + 4];
        let e0 = row[0] + row[2];
        let e1 = row[0] - row[2];
        let e2 = (row[1] >> 1) - row[3];
        let e3 = row[1] + (row[3] >> 1);
        f[i * 4] = e0 + e3;
        f[i * 4 + 1] = e1 + e2;
        f[i * 4 + 2] = e1 - e2;
        f[i * 4 + 3] = e0 - e3;
    }
    let mut r = [0i32; 16];
    for j in 0..4 {
        let g0 = f[j] + f[8 + j];
        let g1 = f[j] - f[8 + j];
        let g2 = (f[4 + j] >> 1) - f[12 + j];
        let g3 = f[4 + j] + (f[12 + j] >> 1);
        r[j] = (g0 + g3 + 32) >> 6;
        r[4 + j] = (g1 + g2 + 32) >> 6;
        r[8 + j] = (g1 - g2 + 32) >> 6;
        r[12 + j] = (g0 - g3 + 32) >> 6;
    }
    r
}

/// `nc-h264` `transform::hadamard_4x4`.
fn hadamard_ref(c: &[i32; 16]) -> [i32; 16] {
    let mut tmp = [0i32; 16];
    for r in 0..4 {
        let [a, b, cc, d] = [c[r * 4], c[r * 4 + 1], c[r * 4 + 2], c[r * 4 + 3]];
        tmp[r * 4] = a + b + cc + d;
        tmp[r * 4 + 1] = a + b - cc - d;
        tmp[r * 4 + 2] = a - b - cc + d;
        tmp[r * 4 + 3] = a - b + cc - d;
    }
    let mut out = [0i32; 16];
    for col in 0..4 {
        let [a, b, cc, d] = [tmp[col], tmp[4 + col], tmp[8 + col], tmp[12 + col]];
        out[col] = a + b + cc + d;
        out[4 + col] = a + b - cc - d;
        out[8 + col] = a - b - cc + d;
        out[12 + col] = a - b + cc - d;
    }
    out
}

const NORM_ADJUST: [[i32; 3]; 6] = [
    [10, 16, 13],
    [11, 18, 14],
    [13, 20, 16],
    [14, 23, 18],
    [16, 25, 20],
    [18, 29, 23],
];
const QUANT_MF: [[i32; 3]; 6] = [
    [13107, 5243, 8066],
    [11916, 4660, 7490],
    [10082, 4194, 6554],
    [9362, 3647, 5825],
    [8192, 3355, 5243],
    [7282, 2893, 4559],
];

fn class(pos: usize) -> usize {
    match ((pos / 4) % 2, (pos % 4) % 2) {
        (0, 0) => 0,
        (1, 1) => 1,
        _ => 2,
    }
}

/// `nc-h264` `Quantizer::quantize` / `quantize_dc` (`dc`: one more bit,
/// position 0's multiplier).
fn h264_quantize_ref(coeff: i32, pos: usize, qp: u8, intra: bool, dc: bool) -> i32 {
    let (per, rem) = (i32::from(qp / 6), usize::from(qp % 6));
    let qbits = 15 + per + i32::from(dc);
    let offset = (1 << qbits) / if intra { 3 } else { 6 };
    let mf = QUANT_MF[rem][if dc { 0 } else { class(pos) }];
    let magnitude = (coeff.abs() * mf + offset) >> qbits;
    if coeff < 0 { -magnitude } else { magnitude }
}

/// `nc-h264` `Quantizer::dequantize_4x4` (without the DC replacement).
fn h264_dequantize_ref(levels: &[i32; 16], qp: u8) -> [i32; 16] {
    let (per, rem) = (i32::from(qp / 6), usize::from(qp % 6));
    core::array::from_fn(|pos| {
        let scaled = levels[pos] * 16 * NORM_ADJUST[rem][class(pos)];
        if per >= 4 {
            scaled << (per - 4)
        } else {
            (scaled + (1 << (3 - per))) >> (4 - per)
        }
    })
}

/// `nc-h264` `Quantizer::dequantize_luma_dc`.
fn h264_dequantize_luma_dc_ref(levels: &[i32; 16], qp: u8) -> [i32; 16] {
    let (per, rem) = (i32::from(qp / 6), usize::from(qp % 6));
    let scale = 16 * NORM_ADJUST[rem][0];
    hadamard_ref(levels).map(|v| {
        if per >= 6 {
            (v * scale) << (per - 6)
        } else {
            (v * scale + (1 << (5 - per))) >> (6 - per)
        }
    })
}

#[test]
fn h264_4x4_transforms_match_the_backend() {
    let mut rng = Rng(0x5eed_5eed_1234_4321);
    for dsp in dsps() {
        // A 16x8 region: 4 blocks wide, 2 high, at stride 20.
        let stride = 20;
        let residual: Vec<i16> = (0..8 * stride).map(|_| rng.next(255) as i16).collect();
        let mut got = [[0; 16]; 8];
        dsp.h264_forward_4x4(&residual, stride, 4, &mut got);
        for (b, block) in got.iter().enumerate() {
            let (bx, by) = (b % 4, b / 4);
            let input: [i32; 16] = core::array::from_fn(|i| {
                i32::from(residual[(by * 4 + i / 4) * stride + bx * 4 + i % 4])
            });
            assert_eq!(*block, h264_forward_4x4_ref(&input), "forward block {b}");
        }
        for amplitude in [1, 100, 30000] {
            let coeffs: Vec<[i32; 16]> = (0..6)
                .map(|_| core::array::from_fn(|_| rng.next(amplitude)))
                .collect();
            let mut out = vec![[0; 16]; 6];
            dsp.h264_inverse_4x4(&coeffs, &mut out);
            for (c, o) in coeffs.iter().zip(&out) {
                assert_eq!(*o, h264_inverse_4x4_ref(c), "inverse");
                assert_eq!(dsp.h264_hadamard_4x4(c), hadamard_ref(c), "hadamard");
            }
        }
    }
}

#[test]
fn h264_quantisers_match_the_backend() {
    let mut rng = Rng(0xfeed_f00d_0000_0001);
    for dsp in dsps() {
        for qp in [0u8, 5, 12, 26, 33, 40, 51] {
            for intra in [false, true] {
                let q = H264Quantizer::new(qp, intra);
                let blocks: Vec<[i32; 16]> = (0..3)
                    .map(|_| core::array::from_fn(|_| rng.next(9000)))
                    .collect();
                let mut got = blocks.clone();
                let nonzero = dsp.h264_quantize_4x4(&mut got, &q);
                let mut count = 0;
                for (g, b) in got.iter().zip(&blocks) {
                    let want: [i32; 16] =
                        core::array::from_fn(|p| h264_quantize_ref(b[p], p, qp, intra, false));
                    count += want.iter().filter(|&&l| l != 0).count();
                    assert_eq!(*g, want, "qp {qp}");
                }
                assert_eq!(nonzero as usize, count);
                for len in [4, 16] {
                    let dc: Vec<i32> = (0..len).map(|_| rng.next(32000)).collect();
                    let mut got = dc.clone();
                    dsp.h264_quantize_dc(&mut got, &q);
                    let want: Vec<i32> = dc
                        .iter()
                        .map(|&c| h264_quantize_ref(c, 0, qp, intra, true))
                        .collect();
                    assert_eq!(got, want, "dc qp {qp} len {len}");
                }
            }
            let q = H264Quantizer::new(qp, true);
            let levels: Vec<[i32; 16]> = (0..3)
                .map(|_| core::array::from_fn(|_| rng.next(200)))
                .collect();
            let mut out = vec![[0; 16]; 3];
            dsp.h264_dequantize_4x4(&levels, &q, &mut out);
            for (l, o) in levels.iter().zip(&out) {
                assert_eq!(*o, h264_dequantize_ref(l, qp), "dequantise qp {qp}");
            }
            let dc: [i32; 16] = core::array::from_fn(|_| rng.next(200));
            assert_eq!(
                dsp.h264_dequantize_luma_dc(&dc, &q),
                h264_dequantize_luma_dc_ref(&dc, qp),
                "luma dc qp {qp}"
            );
        }
    }
}

/// `levels` with each block made, at random, empty, DC-only or left full:
/// the shortcuts of the fused reconstruction.
fn sparse_blocks<const N: usize>(rng: &mut Rng, count: usize, amplitude: i32) -> Vec<[i32; N]> {
    (0..count)
        .map(|_| {
            let mut block: [i32; N] = core::array::from_fn(|_| rng.next(amplitude));
            match rng.next(1) {
                -1 => block = [0; N],
                0 => block[1..].fill(0),
                _ => {}
            }
            block
        })
        .collect()
}

/// A 16x16 prediction of random samples, edge values included.
fn prediction(rng: &mut Rng) -> Vec<u8> {
    (0..16 * 16)
        .map(|_| match rng.next(3) {
            -3 => 0,
            3 => 255,
            _ => (rng.next(127) + 128) as u8,
        })
        .collect()
}

#[test]
fn h264_add_inverse_matches_dequantise_inverse_and_add() {
    use nc_common::PlaneViewMut;
    let mut rng = Rng(0x5eed_0000_0000_0264);
    for dsp in dsps() {
        for qp in [0u8, 7, 20, 26, 36, 45, 51] {
            let q = H264Quantizer::new(qp, false);
            for with_dc in [false, true] {
                let levels = sparse_blocks::<16>(&mut rng, 16, 60);
                let dc: Vec<i32> = (0..16).map(|_| rng.next(1) * rng.next(3000)).collect();
                let pred = prediction(&mut rng);
                let mut fused = pred.clone();
                let mut view = PlaneViewMut::new(&mut fused, 16, 16, 16).unwrap();
                dsp.h264_add_inverse_4x4(&levels, with_dc.then_some(&dc[..]), &q, 4, &mut view);
                let mut scaled = vec![[0; 16]; 16];
                dsp.h264_dequantize_4x4(&levels, &q, &mut scaled);
                if with_dc {
                    for (block, &d) in scaled.iter_mut().zip(&dc) {
                        block[0] = d;
                    }
                }
                let mut residual = vec![[0; 16]; 16];
                dsp.h264_inverse_4x4(&scaled, &mut residual);
                for (b, block) in residual.iter().enumerate() {
                    for (i, &r) in block.iter().enumerate() {
                        let at = (b / 4 * 4 + i / 4) * 16 + b % 4 * 4 + i % 4;
                        let want = (i32::from(pred[at]) + r).clamp(0, 255);
                        assert_eq!(
                            i32::from(fused[at]),
                            want,
                            "4x4 qp {qp} dc {with_dc} block {b}"
                        );
                    }
                }
            }
            for levels in sparse_blocks::<64>(&mut rng, 6, 60) {
                let pred = prediction(&mut rng);
                let mut fused = pred.clone();
                let mut view = PlaneViewMut::new(&mut fused, 16, 8, 8).unwrap();
                dsp.h264_add_inverse_8x8(&levels, &q, &mut view);
                let (mut scaled, mut residual) = ([0; 64], [0; 64]);
                dsp.h264_dequantize_8x8(&levels, &q, &mut scaled);
                dsp.h264_inverse_8x8(&scaled, &mut residual);
                for (i, &r) in residual.iter().enumerate() {
                    let at = i / 8 * 16 + i % 8;
                    let want = (i32::from(pred[at]) + r).clamp(0, 255);
                    assert_eq!(i32::from(fused[at]), want, "8x8 qp {qp}");
                }
                assert_eq!(fused[8..16], pred[8..16], "8x8 writes only its block");
            }
        }
    }
}

#[test]
fn h264_add_scaled_matches_inverse_and_add() {
    use nc_common::PlaneViewMut;
    let mut rng = Rng(0x5eed_0000_0000_5ca1);
    for dsp in dsps() {
        let coeffs = sparse_blocks::<16>(&mut rng, 16, 3000);
        let pred = prediction(&mut rng);
        let mut fused = pred.clone();
        let mut view = PlaneViewMut::new(&mut fused, 16, 16, 16).unwrap();
        dsp.h264_add_scaled_4x4(&coeffs, 4, &mut view);
        let mut residual = vec![[0; 16]; 16];
        dsp.h264_inverse_4x4(&coeffs, &mut residual);
        for (b, block) in residual.iter().enumerate() {
            for (i, &r) in block.iter().enumerate() {
                let at = (b / 4 * 4 + i / 4) * 16 + b % 4 * 4 + i % 4;
                let want = (i32::from(pred[at]) + r).clamp(0, 255);
                assert_eq!(i32::from(fused[at]), want, "4x4 block {b}");
            }
        }
        for coeffs in sparse_blocks::<64>(&mut rng, 6, 3000) {
            let pred = prediction(&mut rng);
            let mut fused = pred.clone();
            let mut view = PlaneViewMut::new(&mut fused, 16, 8, 8).unwrap();
            dsp.h264_add_scaled_8x8(&coeffs, &mut view);
            let mut residual = [0; 64];
            dsp.h264_inverse_8x8(&coeffs, &mut residual);
            for (i, &r) in residual.iter().enumerate() {
                let at = i / 8 * 16 + i % 8;
                let want = (i32::from(pred[at]) + r).clamp(0, 255);
                assert_eq!(i32::from(fused[at]), want, "8x8 sample {i}");
            }
        }
    }
}

#[test]
fn h264_add_scaled_mb_matches_the_block_form() {
    use nc_common::PlaneViewMut;
    let mut rng = Rng(0x5eed_0000_0000_0a6b);
    for dsp in dsps() {
        let blocks = sparse_blocks::<16>(&mut rng, 16, 3000);
        let coeffs: [[i32; 16]; 16] = core::array::from_fn(|b| blocks[b]);
        let coded = blocks.iter().enumerate().fold(0u16, |m, (b, block)| {
            m | (u16::from(block.iter().any(|&c| c != 0)) << b)
        });
        let ac = blocks.iter().enumerate().fold(0u16, |m, (b, block)| {
            m | (u16::from(block[1..].iter().any(|&c| c != 0)) << b)
        });
        let pred = prediction(&mut rng);
        let (mut by_block, mut by_mask) = (pred.clone(), pred.clone());
        let mut view = PlaneViewMut::new(&mut by_block, 16, 16, 16).unwrap();
        dsp.h264_add_scaled_4x4(&blocks, 4, &mut view);
        let mut view = PlaneViewMut::new(&mut by_mask, 16, 16, 16).unwrap();
        dsp.h264_add_scaled_mb(&coeffs, coded, ac, &mut view);
        assert_eq!(by_mask, by_block);
    }
}

#[test]
fn h264_forward_quantize_chroma_matches_the_separate_steps() {
    let mut rng = Rng(0x5eed_0000_0000_c420);
    for dsp in dsps() {
        for qp in [0u8, 12, 26, 39, 51] {
            for intra in [false, true] {
                let q = H264Quantizer::new(qp, intra);
                let residual: Vec<i16> = (0..8 * 11).map(|_| rng.next(255) as i16).collect();
                let mut ac = [[0; 16]; 4];
                let (dc, count) = dsp.h264_forward_quantize_chroma(&residual, 11, &q, &mut ac);
                let mut want = [[0; 16]; 4];
                dsp.h264_forward_4x4(&residual, 11, 2, &mut want);
                let c: [i32; 4] = core::array::from_fn(|b| want[b][0]);
                let mut want_dc = [
                    c[0] + c[1] + c[2] + c[3],
                    c[0] - c[1] + c[2] - c[3],
                    c[0] + c[1] - c[2] - c[3],
                    c[0] - c[1] - c[2] + c[3],
                ];
                dsp.h264_quantize_dc(&mut want_dc, &q);
                dsp.h264_quantize_4x4(&mut want, &q);
                for block in &mut want {
                    block[0] = 0;
                }
                assert_eq!(ac, want, "ac qp {qp}");
                assert_eq!(dc, want_dc, "dc qp {qp}");
                let nonzero = want.as_flattened().iter().filter(|&&l| l != 0).count();
                assert_eq!(count as usize, nonzero, "count qp {qp}");
            }
        }
    }
}

// ------------------------------------------------------------ H.264 8x8

/// Rows of `F`: 8 times the basis of the inverse 8x8 transform.
const F8: [[i64; 8]; 8] = [
    [8, 8, 8, 8, 8, 8, 8, 8],
    [12, 10, 6, 3, -3, -6, -10, -12],
    [8, 4, -4, -8, -8, -4, 4, 8],
    [10, -3, -12, -6, 6, 12, 3, -10],
    [8, -8, -8, 8, 8, -8, -8, 8],
    [6, -12, 3, 10, -10, -3, 12, -6],
    [4, -8, 8, -4, -4, 8, -8, 4],
    [3, -6, 10, -12, 12, -10, 6, -3],
];

#[test]
fn h264_8x8_inverse_has_the_basis_of_the_forward_matrix() {
    // An impulse of 512 at (k, 0) reconstructs to row k of F down every
    // column: the row pass spreads it flat, the column pass gives 64 F[k][y]
    // exactly (512 is divisible by the transform's shifts), then >> 6.
    let dsp = Dsp::new();
    for k in 0..8 {
        let mut c = [0; 64];
        c[k * 8] = 512;
        let mut out = [0; 64];
        dsp.h264_inverse_8x8(&c, &mut out);
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(i64::from(out[y * 8 + x]), F8[k][y], "k {k} ({y}, {x})");
            }
        }
    }
}

#[test]
fn h264_8x8_forward_is_the_matrix_product_and_round_trips() {
    let mut rng = Rng(0x8888_1234_5678_9999);
    for dsp in dsps() {
        for amplitude in [1, 30, 255] {
            let residual: Vec<i16> = (0..64).map(|_| rng.next(amplitude) as i16).collect();
            let mut c = [0; 64];
            dsp.h264_forward_8x8(&residual, 8, &mut c);
            for v in 0..8 {
                for u in 0..8 {
                    let sum: i64 = (0..8)
                        .flat_map(|y| (0..8).map(move |x| (y, x)))
                        .map(|(y, x)| F8[v][y] * F8[u][x] * i64::from(residual[y * 8 + x]))
                        .sum();
                    assert_eq!(i64::from(c[v * 8 + u]), (sum + 32) >> 6, "({v}, {u})");
                }
            }
            // Quantise at QP 0, scale, invert: close to the residual.
            let q = H264Quantizer::new(0, true);
            let mut levels = c;
            dsp.h264_quantize_8x8(&mut levels, &q);
            let mut d = [0; 64];
            dsp.h264_dequantize_8x8(&levels, &q, &mut d);
            let mut back = [0; 64];
            dsp.h264_inverse_8x8(&d, &mut back);
            for (b, r) in back.iter().zip(&residual) {
                assert!((b - i32::from(*r)).abs() <= 2, "{b} vs {r}");
            }
        }
    }
}

#[test]
fn h264_8x8_multipliers_match_the_reference_encoder() {
    // JM `quant_coef8[0]` classes: (0,0), (1,1), (2,2), (0,1), (0,2), (1,2).
    let m = &QUANT_MF_8X8_POS[0];
    assert_eq!(
        [m[0], m[9], m[18], m[1], m[2], m[10]],
        [13107, 11428, 20972, 12222, 16777, 15481]
    );
}

/// Blocks at the zero-SAD thresholds always quantise to zero, including the
/// adversarial ones that concentrate the whole SAD where the transform
/// weights are largest; and the bounds are not vacuous.
#[test]
fn h264_zero_sad_thresholds_are_safe() {
    let mut rng = Rng(7);
    for divisor in [2, 3, 4, 6, 16] {
        for qp in 0..=51u8 {
            let q = H264Quantizer::with_rounding_divisor(qp, divisor);
            let limit = q.zero_sad_threshold_4x4();
            assert!(qp < 20 || limit > 0, "threshold vacuous at qp {qp}");
            for trial in 0..400 {
                // Spend exactly `limit` absolute residual over a few positions.
                let mut block = [0i32; 16];
                let mut left = limit as i32;
                while left > 0 {
                    let pos = (rng.next(1000).unsigned_abs() % 16) as usize;
                    let amount = (rng.next(255).abs() + 1).min(left);
                    let sign = if (trial + pos).is_multiple_of(2) {
                        1
                    } else {
                        -1
                    };
                    block[pos] = (block[pos] + sign * amount).clamp(-255, 255);
                    left -= amount;
                }
                let coeffs = [h264_forward_4x4_ref(&block)];
                for d in dsps() {
                    let mut c = coeffs;
                    d.h264_quantize_4x4(&mut c, &q);
                    assert!(c[0].iter().all(|&v| v == 0), "qp {qp} block {block:?}");
                }
            }
            // Chroma DC: the 2x2 Hadamard of four DCs summing to the bound.
            let dc_limit = q.zero_sad_threshold_chroma_dc() as i32;
            for d in dsps() {
                let mut dc = [dc_limit, 0, 0, 0];
                d.h264_quantize_dc(&mut dc, &q);
                assert_eq!(dc, [0; 4], "chroma dc qp {qp}");
            }
        }
    }
}

/// Fusion must preserve each level and count, including clearing reused
/// output for every combination of skipped quadrants.
#[test]
fn h264_macroblock_quantisation_matches_separate_blocks() {
    let reference = Dsp::with_isa(Isa::Scalar).unwrap();
    let mut rng = Rng(0xfade_1234_beef_5678);
    let stride = 23;
    for amplitude in [0, 1, 255] {
        let residual: Vec<i16> = (0..15 * stride + 16)
            .map(|_| rng.next(amplitude) as i16)
            .collect();
        for qp in [0, 26, 51] {
            for divisor in [3, 4, 6] {
                let q = H264Quantizer::with_rounding_divisor(qp, divisor);
                for zero_mask in 0..16 {
                    let mut want = [[0; 64]; 4];
                    let mut want_counts = [0; 4];
                    for (i, block) in want.iter_mut().enumerate() {
                        if zero_mask & (1 << i) == 0 {
                            let start = (i / 2) * 8 * stride + (i % 2) * 8;
                            reference.h264_forward_8x8(&residual[start..], stride, block);
                            want_counts[i] = reference.h264_quantize_8x8(block, &q);
                        }
                    }
                    for dsp in dsps() {
                        let mut got = [[12345; 64]; 4];
                        let counts = dsp
                            .h264_forward_quantize_mb(&residual, stride, zero_mask, &q, &mut got);
                        assert_eq!(got, want, "qp {qp} divisor {divisor} zero {zero_mask}");
                        assert_eq!(counts, want_counts);
                    }
                }
            }
        }
    }
}

#[test]
fn hevc_forward_column_bound_preserves_raw_fallback() {
    assert!(DCT32.iter().flatten().all(|&v| v.unsigned_abs() <= 90));
    assert!(DCT32[0].iter().all(|&v| v == 64));
    assert!(
        DCT32[1..]
            .iter()
            .all(|row| row.iter().map(|&v| i32::from(v)).sum::<i32>() == 0)
    );
    let [scalar, best] = dsps();
    let mut rng = Rng(0x1717_fefe_abab_5151);
    let mut hits = 0;
    for log2 in 2..=5 {
        let n = 1usize << log2;
        let t = HevcTransform::dct(log2).unwrap();
        for qp in 0..=51 {
            let q = HevcQuantizer::new(qp, log2, false);
            let limit = (((1u64 << q.qbits) - q.offset as u64 - 1) / q.scale as u64) as u32;
            for amplitude in [0, 1, 2, 4, 8, 32, 255] {
                let input: Vec<i16> = (0..n * n).map(|_| rng.next(amplitude) as i16).collect();
                let mut raw = vec![0; n * n];
                scalar.hevc_forward(t, &input, n, &mut raw);
                for dsp in [&scalar, &best] {
                    let mut got = vec![i32::MIN; n * n];
                    if dsp.hevc_forward_unless_zero(t, &input, n, limit, &mut got) {
                        assert_eq!(got, raw, "log2 {log2} qp {qp} amplitude {amplitude}");
                    } else {
                        hits += 1;
                        assert_eq!(got, vec![i32::MIN; n * n]);
                        let mut quantized = raw.clone();
                        assert_eq!(
                            dsp.hevc_quantize(&mut quantized, &q),
                            0,
                            "false zero log2 {log2} qp {qp} amplitude {amplitude}"
                        );
                    }
                }
            }
        }
    }
    assert!(hits > 100);
}

#[test]
fn hevc_separate_quantisation_matches_in_place_at_all_qps() {
    let mut rng = Rng(0x8b24_f671_01cc_3a90);
    for dsp in dsps() {
        for qp in 0..=51 {
            for log2 in 2..=5 {
                for rounding in [0, 85, 128, 171, 256] {
                    let q = HevcQuantizer::with_rounding(qp, log2, rounding);
                    let mut coeffs: Vec<i32> =
                        (0..1usize << (2 * log2)).map(|_| rng.next(32768)).collect();
                    coeffs[..5].copy_from_slice(&[-32768, 32767, 0, -1, 1]);
                    let mut expected = coeffs.clone();
                    let count = dsp.hevc_quantize(&mut expected, &q);
                    let mut actual = vec![i32::MIN; coeffs.len()];
                    assert_eq!(dsp.hevc_quantize_from(&coeffs, &q, &mut actual), count);
                    assert_eq!(actual, expected, "QP{qp} log2{log2} rounding{rounding}");
                }
            }
        }
    }
}

/// The encoder-side staged forwards under test.
const STAGINGS: [HevcForwardMode; 2] = [
    HevcForwardMode::staged(HevcForwardStaging::Rows),
    HevcForwardMode::staged(HevcForwardStaging::RowsAndColumns),
];

#[test]
fn hevc_staged_rows_bound_matches_full_approximation_and_isa() {
    let [scalar, best] = dsps();
    let mut rng = Rng(0x6b0f_c011_2026_55aa);
    for staging in STAGINGS {
        for log2 in [4, 5] {
            let n = 1usize << log2;
            let transform = HevcTransform::dct(log2).unwrap();
            for (bias, amplitude) in [
                (-255, 0),
                (-1, 0),
                (0, 0),
                (1, 0),
                (255, 0),
                (0, 1),
                (0, 16),
                (0, 255),
                (32, 2),
                (-32, 2),
            ] {
                let stride = n + 5;
                let mut input = vec![0i16; stride * n];
                for row in input.chunks_exact_mut(stride) {
                    for value in &mut row[..n] {
                        *value = (bias + rng.next(amplitude)).clamp(-255, 255) as i16;
                    }
                }
                let mut raw = vec![0; n * n];
                scalar.hevc_forward(transform, &input, stride, &mut raw);
                let mut expected = vec![0; n * n];
                scalar.hevc_forward_staged_unless_zero(
                    transform,
                    &input,
                    stride,
                    0,
                    staging,
                    &mut expected,
                );
                for limit in [0, 1, 7, 31, 127, 511, 4095] {
                    let mut reference = vec![i32::MIN; n * n];
                    let survives = scalar.hevc_forward_staged_unless_zero(
                        transform,
                        &input,
                        stride,
                        limit,
                        staging,
                        &mut reference,
                    );
                    let mut scalar_approx = None;
                    for dsp in [&scalar, &best] {
                        let mut got = vec![i32::MIN; n * n + 3];
                        assert_eq!(
                            dsp.hevc_forward_staged_unless_zero(
                                transform, &input, stride, limit, staging, &mut got
                            ),
                            survives
                        );
                        if survives {
                            // Staged rows keep the horizontal-DC column exact,
                            // staged columns the 2-D DC.
                            if staging.staging == HevcForwardStaging::Rows {
                                for row in 0..n {
                                    assert_eq!(
                                        got[row * n],
                                        raw[row * n],
                                        "horizontal DC column, n={n}"
                                    );
                                }
                            }
                            assert_eq!(got[0], raw[0], "DC, n={n}");
                            assert_eq!(&got[..n * n], &expected);
                            assert!(got[..n * n].iter().all(|&v| (-32768..=32767).contains(&v)));
                            if let Some(ref reference) = scalar_approx {
                                assert_eq!(&got[..n * n], reference);
                            } else {
                                scalar_approx = Some(got[..n * n].to_vec());
                            }
                        } else {
                            assert!(expected.iter().all(|&v| v.unsigned_abs() <= limit));
                            assert!(got[..n * n].iter().all(|&v| v == i32::MIN));
                        }
                        assert!(got[n * n..].iter().all(|&v| v == i32::MIN));
                    }
                }
            }
        }
    }
}

/// `zero_columns` changes only coefficients the bound proves quantise to
/// zero (to zero), in whole lane groups, identically on every ISA, and
/// never the proven-zero decision.
#[test]
fn hevc_zero_columns_only_zero_proven_coefficients() {
    let [scalar, best] = dsps();
    let mut rng = Rng(0x2c0f_5e11_2026_1109);
    for log2 in [4, 5] {
        let n = 1usize << log2;
        let transform = HevcTransform::dct(log2).unwrap();
        for amplitude in [2, 8, 40, 255] {
            let mut input = vec![0i16; n * n];
            for (r, row) in input.chunks_exact_mut(n).enumerate() {
                for (c, value) in row.iter_mut().enumerate() {
                    // Smooth horizontally on the right half, so high
                    // column groups are often proven zero.
                    let noise = if c < n / 2 {
                        rng.next(amplitude)
                    } else {
                        rng.next(2)
                    };
                    *value = (noise + (r as i32 % 3)).clamp(-255, 255) as i16;
                }
            }
            for limit in [0, 7, 31, 127, 511] {
                for staging in STAGINGS {
                    let skipping = HevcForwardMode {
                        zero_columns: true,
                        ..staging
                    };
                    let mut full = vec![0; n * n];
                    let survives = scalar.hevc_forward_staged_unless_zero(
                        transform, &input, n, limit, staging, &mut full,
                    );
                    let mut reference = None;
                    for dsp in [&scalar, &best] {
                        let mut got = vec![0; n * n];
                        assert_eq!(
                            dsp.hevc_forward_staged_unless_zero(
                                transform, &input, n, limit, skipping, &mut got
                            ),
                            survives
                        );
                        if !survives {
                            continue;
                        }
                        for (&g, &f) in got.iter().zip(&full) {
                            assert!(g == f || (g == 0 && f.unsigned_abs() <= limit));
                        }
                        match &reference {
                            Some(reference) => assert_eq!(&got, reference),
                            None => reference = Some(got),
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn hevc_staged_rows_report_small_coefficient_error() {
    let [scalar, best] = dsps();
    let mut rng = Rng(0x32c0_16c0_2026_9933);
    for staging in STAGINGS {
        for log2 in [4, 5] {
            let n = 1usize << log2;
            let transform = HevcTransform::dct(log2).unwrap();
            for amplitude in [1, 4, 16, 64, 255] {
                let (mut energy, mut error, mut max_error) = (0i64, 0i64, 0i32);
                for _ in 0..16 {
                    let input: Vec<i16> = (0..n * n).map(|_| rng.next(amplitude) as i16).collect();
                    let mut exact = vec![0; n * n];
                    let mut approximate = vec![0; n * n];
                    scalar.hevc_forward(transform, &input, n, &mut exact);
                    best.hevc_forward_staged_unless_zero(
                        transform,
                        &input,
                        n,
                        0,
                        staging,
                        &mut approximate,
                    );
                    for (&a, &b) in exact.iter().zip(&approximate) {
                        let delta = a - b;
                        energy += i64::from(a) * i64::from(a);
                        error += i64::from(delta) * i64::from(delta);
                        max_error = max_error.max(delta.abs());
                    }
                }
                eprintln!(
                    "HEVC_STAGED {staging:?} n={n} amplitude={amplitude} max_error={max_error} relative_l2={}",
                    (error as f64 / energy as f64).sqrt()
                );
                if amplitude >= 16 {
                    assert!(
                        error * 400 < energy,
                        "scale or axis mismatch, n={n}, amplitude={amplitude}"
                    );
                } else {
                    // Unit-sized rounding changes have unstable relative error
                    // close to zero; retain an absolute coefficient guard there.
                    assert!(max_error <= 2);
                }
            }
        }
    }
}

#[test]
fn hevc_staged_rows_handle_maximum_row_dc_and_high_contrast() {
    let [scalar, best] = dsps();
    let mut rng = Rng(0x1f2f_3f4f_5f6f_7f8f);
    for staging in STAGINGS {
        for log2 in [4, 5] {
            let n = 1usize << log2;
            let transform = HevcTransform::dct(log2).unwrap();
            // Every HEVC row basis has L1 norm at most64*N, hence first-pass
            // samples from8-bit residuals have magnitude at most32640.
            for row in 0..n {
                let norm: i32 = DCT32[row << (5 - log2)][..n]
                    .iter()
                    .map(|&v| i32::from(v).abs())
                    .sum();
                assert!(norm <= (64 * n) as i32);
            }
            for pattern in 0..6 {
                let mut input = vec![0i16; n * n];
                for (row, dst) in input.chunks_exact_mut(n).enumerate() {
                    let negative = match pattern {
                        0 => false,
                        1 => row % 2 != 0,
                        2 => row >= n / 2,
                        3 => row / 4 % 2 != 0,
                        _ => rng.next(1) < 0,
                    };
                    for (col, value) in dst.iter_mut().enumerate() {
                        *value = if negative ^ (pattern == 5 && col % 2 != 0) {
                            -255
                        } else {
                            255
                        };
                    }
                }
                let mut exact = vec![0; n * n];
                let mut a = vec![0; n * n];
                let mut b = vec![0; n * n];
                scalar.hevc_forward(transform, &input, n, &mut exact);
                scalar.hevc_forward_staged_unless_zero(transform, &input, n, 0, staging, &mut a);
                best.hevc_forward_staged_unless_zero(transform, &input, n, 0, staging, &mut b);
                assert_eq!(a, b);
                let (energy, error) =
                    exact
                        .iter()
                        .zip(&a)
                        .fold((0i64, 0i64), |(energy, error), (&x, &y)| {
                            (
                                energy + i64::from(x) * i64::from(x),
                                error + i64::from(x - y) * i64::from(x - y),
                            )
                        });
                assert!(
                    error * 400 < energy,
                    "extreme column scale/overflow n={n}, pattern={pattern}"
                );
            }
        }
    }
}

/// With exact passes, `zero_columns` gives the exact forward except zeros
/// for coefficients the bound proves quantise to zero, the same proven-zero
/// decision, and the same output on every ISA.
#[test]
fn hevc_exact_zero_columns_match_the_exact_forward() {
    let [scalar, best] = dsps();
    let mode = HevcForwardMode {
        zero_columns: true,
        ..HevcForwardMode::EXACT
    };
    let mut rng = Rng(0x0e8a_c7c0_2026_1109);
    for log2 in [4, 5] {
        let n = 1usize << log2;
        let transform = HevcTransform::dct(log2).unwrap();
        for amplitude in [2, 8, 40, 255] {
            // Noisy left half, nearly flat right half: high column groups
            // are often proven zero.
            let input: Vec<i16> = (0..n * n)
                .map(|i| rng.next(if i % n < n / 2 { amplitude } else { 2 }) as i16)
                .collect();
            for limit in [0, 7, 31, 127, 511] {
                let mut exact = vec![0; n * n];
                let survives =
                    scalar.hevc_forward_unless_zero(transform, &input, n, limit, &mut exact);
                let mut outputs = Vec::new();
                for dsp in [&scalar, &best] {
                    let mut got = vec![0; n * n];
                    assert_eq!(
                        dsp.hevc_forward_staged_unless_zero(
                            transform, &input, n, limit, mode, &mut got
                        ),
                        survives
                    );
                    if survives {
                        for (&g, &e) in got.iter().zip(&exact) {
                            assert!(g == e || (g == 0 && e.unsigned_abs() <= limit));
                        }
                    }
                    outputs.push(got);
                }
                assert_eq!(outputs[0], outputs[1]);
            }
        }
    }
}

/// `low_frequencies` gives the full forward's low `N/2 x N/2` quarter and
/// zeros elsewhere, unless the bound proves that quarter zero, and the same
/// output on every ISA, for exact and staged rows.
#[test]
fn hevc_low_frequencies_keep_the_low_quarter() {
    let [scalar, best] = dsps();
    let mut rng = Rng(0x10f9_ee4e_2026_1111);
    for log2 in [4, 5] {
        let n = 1usize << log2;
        let transform = HevcTransform::dct(log2).unwrap();
        for staging in [HevcForwardStaging::Exact, HevcForwardStaging::Rows] {
            let full_mode = HevcForwardMode {
                zero_columns: true,
                ..HevcForwardMode::staged(staging)
            };
            let low_mode = HevcForwardMode {
                low_frequencies: true,
                ..full_mode
            };
            for amplitude in [2, 8, 40, 255] {
                let input: Vec<i16> = (0..n * n).map(|_| rng.next(amplitude) as i16).collect();
                for limit in [0, 7, 31, 127, 511] {
                    let mut full = vec![0; n * n];
                    let full_survives = if staging == HevcForwardStaging::Exact {
                        scalar.hevc_forward_unless_zero(transform, &input, n, limit, &mut full)
                    } else {
                        scalar.hevc_forward_staged_unless_zero(
                            transform, &input, n, limit, full_mode, &mut full,
                        )
                    };
                    let mut outputs = Vec::new();
                    for dsp in [&scalar, &best] {
                        let mut got = vec![0; n * n];
                        let survives = dsp.hevc_forward_staged_unless_zero(
                            transform, &input, n, limit, low_mode, &mut got,
                        );
                        // Proving the low quarter zero is easier than
                        // proving the whole block zero.
                        assert!(!survives || full_survives);
                        if survives {
                            for (i, (&g, &f)) in got.iter().zip(&full).enumerate() {
                                let low = i / n < n / 2 && i % n < n / 2;
                                // Inside the quarter the full forward's
                                // zero-column skip may have zeroed a
                                // coefficient within the limit.
                                assert!(if low {
                                    g == f || (f == 0 && g.unsigned_abs() <= limit)
                                } else {
                                    g == 0
                                });
                            }
                        }
                        outputs.push((survives, got));
                    }
                    assert_eq!(outputs[0], outputs[1]);
                }
            }
        }
    }
}
