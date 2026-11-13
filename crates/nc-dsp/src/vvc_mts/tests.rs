//! The kernels against plain references: the normative scalar inverse
//! (`codec_txfm::vvc_primary`), the rectangular DCT-II forward
//! (`codec_txfm::vvc_forward`) and a 64-bit matrix product for the other
//! forward kinds.
use super::{basis, supported};
use crate::{Dsp, Isa, VvcTransformKind};
use VvcTransformKind::{Dct2, Dct8, Dst7};

const SIDES: [usize; 4] = [4, 8, 16, 32];
const KINDS: [VvcTransformKind; 3] = [Dct2, Dst7, Dct8];

/// Every supported (size, kinds) combination.
fn shapes() -> impl Iterator<Item = ((usize, usize), [VvcTransformKind; 2])> {
    SIDES.into_iter().flat_map(|w| {
        SIDES.into_iter().flat_map(move |h| {
            KINDS
                .into_iter()
                .flat_map(move |a| KINDS.into_iter().map(move |b| ((w, h), [a, b])))
        })
    })
}

/// A deterministic value stream (xorshift).
fn values(seed: u64, count: usize, range: i32) -> Vec<i32> {
    let mut state = seed | 1;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % (2 * range as u64 + 1)) as i32 - range
        })
        .collect()
}

fn dsps() -> Vec<Dsp> {
    vec![Dsp::with_isa(Isa::Scalar).unwrap(), Dsp::new()]
}

#[test]
fn inverse_matches_the_normative_reference() {
    for dsp in dsps() {
        for (size, kinds) in shapes() {
            let area = size.0 * size.1;
            for (seed, range, sparse) in [(1, 32767, false), (2, 300, true), (3, 32768, false)] {
                let mut coefficients = values(seed, area, range);
                if sparse {
                    for (i, c) in coefficients.iter_mut().enumerate() {
                        if i % size.0 > 5 || i / size.0 > 3 {
                            *c = 0;
                        }
                    }
                }
                let coefficients: Vec<i32> = coefficients
                    .into_iter()
                    .map(|c| c.clamp(-32768, 32767))
                    .collect();
                let mut expected = vec![0; area];
                crate::codec_txfm::vvc_primary::inverse(size, kinds, &coefficients, &mut expected);
                let mut actual = vec![7; area];
                dsp.vvc_inverse_kinds(size, kinds, &coefficients, &mut actual);
                assert_eq!(actual, expected, "{size:?} {kinds:?} {:?}", dsp.isa());
            }
        }
    }
}

#[test]
fn zero_coefficients_give_a_zero_residual() {
    let dsp = Dsp::new();
    let mut out = vec![1; 32 * 32];
    dsp.vvc_inverse_kinds((32, 32), [Dst7, Dct8], &[0; 32 * 32], &mut out);
    assert!(out.iter().all(|&v| v == 0));
}

#[test]
fn dct2_forward_matches_the_rectangular_forward() {
    for dsp in dsps() {
        for w in SIDES {
            for h in SIDES {
                let residual: Vec<i16> = values(5, w * h, 255).iter().map(|&v| v as i16).collect();
                let mut expected = vec![0; w * h];
                crate::codec_txfm::vvc_forward::forward(w, h, &residual, &mut expected);
                let mut actual = vec![7; w * h];
                dsp.vvc_forward_primary((w, h), [Dct2; 2], &residual, &mut actual);
                assert_eq!(actual, expected, "{w}x{h} {:?}", dsp.isa());
            }
        }
    }
}

/// The forward as a 64-bit product, rows first, with the kernels' shifts.
fn forward_reference(
    (width, height): (usize, usize),
    [horizontal, vertical]: [VvcTransformKind; 2],
    residual: &[i16],
) -> Vec<i32> {
    let round = |v: i64, shift: u32| (v + (1 << (shift - 1))) >> shift;
    let (bh, bv) = (basis(horizontal, width), basis(vertical, height));
    let (fh, fv) = (horizontal.frequencies(width), vertical.frequencies(height));
    let mut rows = vec![0i64; width * height];
    for y in 0..height {
        for k in 0..fh {
            let sum: i64 = (0..width)
                .map(|x| i64::from(bh[k][x]) * i64::from(residual[y * width + x]))
                .sum();
            rows[y * width + k] = round(sum, width.ilog2() - 1);
        }
    }
    let mut out = vec![0; width * height];
    for k in 0..fv {
        for x in 0..fh {
            let sum: i64 = (0..height)
                .map(|y| i64::from(bv[k][y]) * rows[y * width + x])
                .sum();
            out[k * width + x] = round(sum, height.ilog2() + 6) as i32;
        }
    }
    out
}

#[test]
fn forward_matches_the_matrix_product() {
    for dsp in dsps() {
        for (size, kinds) in shapes() {
            let residual: Vec<i16> = values(9, size.0 * size.1, 255)
                .iter()
                .map(|&v| v as i16)
                .collect();
            let mut actual = vec![7; size.0 * size.1];
            dsp.vvc_forward_primary(size, kinds, &residual, &mut actual);
            assert_eq!(
                actual,
                forward_reference(size, kinds, &residual),
                "{size:?} {kinds:?} {:?}",
                dsp.isa()
            );
        }
    }
}

#[test]
fn bases_are_the_normative_ones() {
    for (size, kinds) in shapes() {
        for (kind, n) in kinds.into_iter().zip([size.0, size.1]) {
            let table = basis(kind, n);
            for (k, row) in table.iter().enumerate() {
                for (s, &value) in row.iter().enumerate() {
                    let expected = if k < n && s < n {
                        crate::codec_txfm::vvc_primary::basis_entry(kind, n, k, s)
                    } else {
                        0
                    };
                    assert_eq!(i32::from(value), expected, "{kind:?} {n} {k} {s}");
                }
            }
        }
    }
    assert!(!supported((2, 4), [Dct2; 2]) && !supported((64, 4), [Dct2; 2]));
}

#[test]
fn forward_then_inverse_returns_the_residual() {
    // At QP 4 (unit step) every kind pair round-trips within the rounding a
    // DCT-II block of the same span shows; the 32-point DST-VII / DCT-VIII
    // keep 16 frequencies, so only smooth residuals survive them whole.
    let dsp = Dsp::new();
    for (size, kinds) in shapes() {
        let (w, h) = size;
        let residual: Vec<i16> = (0..w * h)
            .map(|i| ((i % w) as i16 * 7 - (i / w) as i16 * 5) % 60)
            .collect();
        let smooth = kinds.iter().zip([w, h]).any(|(k, n)| k.frequencies(n) < n);
        let residual: Vec<i16> = if smooth {
            (0..w * h).map(|i| ((i % w + i / w) as i16) - 20).collect()
        } else {
            residual
        };
        let mut coefficients = vec![0; w * h];
        dsp.vvc_forward_primary(size, kinds, &residual, &mut coefficients);
        let quantizer = crate::VvcQuantizer::rect(4, w.ilog2() as u8, h.ilog2() as u8);
        let mut levels = vec![0; w * h];
        dsp.vvc_quantize_from(&coefficients, &quantizer, &mut levels);
        let mut back = vec![0; w * h];
        dsp.vvc_inverse_primary(size, kinds, &levels, 4, false, &mut back);
        let worst = residual
            .iter()
            .zip(&back)
            .map(|(&a, &b)| (i32::from(a) - b).abs())
            .max()
            .unwrap();
        assert!(
            worst <= if smooth { 16 } else { 3 },
            "{size:?} {kinds:?}: {worst}"
        );
    }
}

#[test]
fn the_forward_row_pass_fits_16_bits() {
    for (size, kinds) in shapes() {
        for (kind, n) in kinds.into_iter().zip([size.0, size.1]) {
            let table = basis(kind, n);
            let largest: i32 = (0..kind.frequencies(n))
                .map(|k| table[k][..n].iter().map(|&b| i32::from(b).abs()).sum())
                .max()
                .unwrap();
            let shift = n.ilog2() - 1;
            assert!((255 * largest + (1 << (shift - 1))) >> shift <= i32::from(i16::MAX));
        }
    }
}
