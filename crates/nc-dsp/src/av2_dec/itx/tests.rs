//! Tests of [`super`]: fixed vectors from dav2d's C (commit 237229e, 8 bpc),
//! the type numbering, the coefficient clearing, the DC shortcut, and a
//! cross-check of the square types against [`crate::av2_txfm`]. The full
//! differential test against dav2d (every size, type, DPCM mode and end of
//! block; random extremes) was run outside the repository when porting.

use super::*;
use crate::av2_txfm::Av2TxSize;
use crate::av2_txfm::typed::{Av2Kernel, Av2TxType};
use crate::{Dsp, Isa};
use nc_common::{BitDepth, fnv1a};

const EIGHT: BitDepth = BitDepth::EIGHT;

/// The coefficient pattern of the C vector program: mixed signs, every
/// third one large.
pub(crate) fn pattern(k: usize) -> i16 {
    let base = ((k * 37 + 11) % 201) as i16 - 100;
    if k.is_multiple_of(3) { base * 40 } else { base }
}

fn pixel(x: usize, y: usize) -> u8 {
    ((x * 7 + y * 13) % 256) as u8
}

/// [`inverse_transform_add`] on the C program's inputs, then the hash of
/// the pixels.
fn run(size: RectTxSize, tx_type: TxType, eob: u16) -> u64 {
    let (w, h) = (size.width(), size.height());
    let mut coeffs: Vec<i16> = (0..size.coded_width() * size.coded_height())
        .map(pattern)
        .collect();
    let mut dst: Vec<u8> = (0..w * h).map(|i| pixel(i % w, i / w)).collect();
    inverse_transform_add(&mut dst, (0, w), &mut coeffs, (size, tx_type), eob, EIGHT);
    fnv1a(dst)
}

#[test]
fn matches_dav2d_vectors() {
    use RectTxSize::*;
    let cases = [
        (Tx4x4, TxType::DCT_DCT, 0, 0x8820_1fb9_60ff_6465),
        (Tx4x4, TxType::WHT_WHT, 5, 0x2281_587a_d672_db6d),
        (
            Tx4x4,
            TxType::IDTX.with_dpcm(Dpcm::Horizontal),
            15,
            0x0ed6_2c7b_93e6_c967,
        ),
        (Tx8x8, TxType::ADST_FLIPADST, 20, 0x826b_871a_a709_daf6),
        (Tx16x16, TxType::DDT_FLIPDDT, 100, 0xbcde_d86f_84b5_728f),
        (Tx8x4, TxType::FLIPADST_DDT, 15, 0x7a0a_7ae3_ab9e_0c66),
        (Tx16x64, TxType::DCT_DCT, 300, 0x651a_24ba_7e75_02f7),
        (Tx64x64, TxType::DCT_DCT, 600, 0x9ef5_91e8_755d_fcb2),
        (Tx32x64, TxType::DCT_DCT, 0, 0xab8f_75a7_a4a6_77a0),
        (Tx32x8, TxType::H_DCT, 20, 0x5850_db9d_d0c3_94b2),
        (Tx4x16, TxType::V_FLIPADST, 40, 0xad2d_ce66_9785_f38a),
        (
            Tx16x16,
            TxType::IDTX_INV.with_dpcm(Dpcm::Vertical),
            255,
            0xe6a3_22d7_ff06_0cb7,
        ),
        (Tx32x32, TxType::IDTX, 1000, 0xf676_1bc4_67bc_3d47),
    ];
    for (size, tx_type, eob, hash) in cases {
        assert_eq!(
            run(size, tx_type, eob),
            hash,
            "{size:?} {tx_type:?} eob {eob}"
        );
    }
}

#[test]
fn large_dct_ignores_rows_past_eob_without_truncating_live_columns() {
    for isa in [Isa::Avx2, Isa::Avx512] {
        let Some(dsp) = Dsp::with_isa(isa) else {
            continue;
        };
        for size in [RectTxSize::Tx32x32, RectTxSize::Tx64x64] {
            let stride = size.width() + 11;
            let initial: Vec<_> = (0..stride * (size.height() + 2))
                .map(|i| pixel(i % stride, i / stride))
                .collect();
            for eob in [1, 10, 36, 78, 136, 406, 1023] {
                let rows = first_pass_rows(size, TxClass::TwoD, eob);
                for whole_width in [false, true] {
                    let mut actual_coeffs = [0_i16; 1025];
                    for (column, coeffs) in actual_coeffs[..1024]
                        .as_chunks_mut::<32>()
                        .0
                        .iter_mut()
                        .enumerate()
                    {
                        for (row, coeff) in coeffs.iter_mut().enumerate() {
                            if row >= rows || whole_width || column < 8 {
                                *coeff = pattern(column * 32 + row);
                            }
                        }
                    }
                    actual_coeffs[1024] = 17;
                    let mut expected_coeffs = actual_coeffs;
                    let (mut actual, mut expected) = (initial.clone(), initial.clone());
                    inverse_transform_add(
                        &mut expected,
                        (3, stride),
                        &mut expected_coeffs,
                        (size, TxType::DCT_DCT),
                        eob,
                        EIGHT,
                    );
                    inverse_transform_add_with_dsp(
                        &dsp,
                        &mut actual,
                        (3, stride),
                        &mut actual_coeffs,
                        (size, TxType::DCT_DCT),
                        eob,
                        EIGHT,
                    );
                    assert_eq!(actual, expected, "{isa:?} {size:?} eob {eob}");
                    assert_eq!(actual_coeffs, expected_coeffs);
                }
            }
        }
    }
}

#[test]
fn cross_chroma_matches_dav2d_vector() {
    let mut u: Vec<i16> = (0..64).map(pattern).collect();
    let mut v: Vec<i16> = (500..564).map(pattern).collect();
    cross_chroma(&mut u, &mut v, &CCTX_ANGLES[2], EIGHT);
    let bytes = |c: &[i16]| fnv1a(c.iter().flat_map(|c| c.to_le_bytes()));
    assert_eq!(bytes(&u), 0xf252_6498_8d10_0865);
    assert_eq!(bytes(&v), 0xdd2e_6005_7298_a8df);
}

#[test]
fn numbering_matches_dav2d() {
    // dav2d `enum TxfmType` in declaration order (printed by its levels.h).
    let dav2d = [
        0, 64, 2, 66, 96, 3, 99, 67, 98, 33, 41, 25, 48, 89, 50, 121, 51, 198, 128, 160, 129, 161,
        132, 164, 4, 36, 133, 165, 5, 37, 130, 162, 131, 163, 68, 69, 100, 101,
    ];
    let ours = [
        TxType::DCT_DCT,
        TxType::ADST_DCT,
        TxType::DCT_ADST,
        TxType::ADST_ADST,
        TxType::FLIPADST_DCT,
        TxType::DCT_FLIPADST,
        TxType::FLIPADST_FLIPADST,
        TxType::ADST_FLIPADST,
        TxType::FLIPADST_ADST,
        TxType::IDTX,
        TxType::IDTX_INV,
        TxType::V_DCT,
        TxType::H_DCT,
        TxType::V_ADST,
        TxType::H_ADST,
        TxType::V_FLIPADST,
        TxType::H_FLIPADST,
        TxType::WHT_WHT,
        TxType::DDT_DCT,
        TxType::FLIPDDT_DCT,
        TxType::DDT_IDENTITY,
        TxType::FLIPDDT_IDENTITY,
        TxType::DDT_DDT,
        TxType::FLIPDDT_DDT,
        TxType::DCT_DDT,
        TxType::IDENTITY_DDT,
        TxType::DDT_FLIPDDT,
        TxType::FLIPDDT_FLIPDDT,
        TxType::DCT_FLIPDDT,
        TxType::IDENTITY_FLIPDDT,
        TxType::DDT_ADST,
        TxType::FLIPDDT_ADST,
        TxType::DDT_FLIPADST,
        TxType::FLIPDDT_FLIPADST,
        TxType::ADST_DDT,
        TxType::ADST_FLIPDDT,
        TxType::FLIPADST_DDT,
        TxType::FLIPADST_FLIPDDT,
    ];
    assert_eq!(ours.map(|t| t.0), dav2d);
    for (raw, size) in RectTxSize::ALL.iter().enumerate() {
        assert_eq!(RectTxSize::from_raw(raw as u8), Some(*size));
    }
    assert_eq!(RectTxSize::from_raw(RECT_TX_SIZES as u8), None);
    assert_eq!(
        TxType::ADST_FLIPADST.with_inter_ddt(RectTxSize::Tx8x8),
        TxType::DDT_FLIPDDT
    );
    assert_eq!(
        TxType::ADST_ADST.with_inter_ddt(RectTxSize::Tx4x8),
        TxType::DDT_ADST
    );
    assert_eq!(
        TxType::ADST_ADST.with_inter_ddt(RectTxSize::Tx32x32),
        TxType::ADST_ADST
    );
}

#[test]
fn clears_the_coefficients_it_reads() {
    for size in RectTxSize::ALL {
        let area = size.coded_width() * size.coded_height();
        let mut coeffs: Vec<i16> = (0..=area).map(pattern).collect();
        let mut dst = vec![128_u8; size.width() * size.height()];
        let last = coeffs[area];
        inverse_transform_add(
            &mut dst,
            (0, size.width()),
            &mut coeffs,
            (size, TxType::DCT_DCT),
            9,
            EIGHT,
        );
        assert!(coeffs[..area].iter().all(|&c| c == 0), "{size:?}");
        assert_eq!(coeffs[area], last, "{size:?} wrote past its coefficients");
    }
}

/// dav2d's folded DC-only rounding gives the two-pass result.
#[test]
fn dc_shortcut_matches_the_full_transform() {
    for size in RectTxSize::ALL {
        let area = size.coded_width() * size.coded_height();
        for dc in (-4000..=4000).step_by(37) {
            let [shortcut, general] = [0, 1].map(|eob| {
                let mut coeffs = vec![0; area];
                coeffs[0] = dc;
                let mut dst = vec![128_u8; size.width() * size.height()];
                inverse_transform_add(
                    &mut dst,
                    (0, size.width()),
                    &mut coeffs,
                    (size, TxType::DCT_DCT),
                    eob,
                    EIGHT,
                );
                dst
            });
            assert_eq!(shortcut, general, "{size:?} dc {dc}");
        }
    }
}

/// The square types of [`crate::av2_txfm`] (AVM's inverse, row-major
/// coefficients, a residual clamped to 9 bits) give the same pixels: the
/// clamp cannot change a sum clipped to `0..=255` from 128.
#[test]
fn squares_agree_with_av2_txfm() {
    let dsp = Dsp::with_isa(Isa::Scalar).expect("scalar DSP");
    let kernels = [
        (Tx1d::Dct, Av2Kernel::Dct),
        (Tx1d::Adst, Av2Kernel::Adst),
        (Tx1d::FlipAdst, Av2Kernel::FlipAdst),
        (Tx1d::Identity, Av2Kernel::Identity),
    ];
    let squares = [
        (RectTxSize::Tx4x4, Av2TxSize::Tx4),
        (RectTxSize::Tx8x8, Av2TxSize::Tx8),
        (RectTxSize::Tx16x16, Av2TxSize::Tx16),
        (RectTxSize::Tx32x32, Av2TxSize::Tx32),
        (RectTxSize::Tx64x64, Av2TxSize::Tx64),
    ];
    for (size, av2_size) in squares {
        let (n, coded) = (size.width(), size.coded_width());
        for (ver, av2_ver) in kernels {
            for (hor, av2_hor) in kernels {
                let tx_type = two_d(hor, ver);
                let av2_type = Av2TxType {
                    vertical: av2_ver,
                    horizontal: av2_hor,
                };
                if !av2_type.supports(av2_size) {
                    continue;
                }
                let coeffs: Vec<i16> = (0..coded * coded).map(|k| pattern(k) / 4).collect();
                // dav2d is column-major, av2_txfm row-major.
                let row_major: Vec<i16> = (0..coded * coded)
                    .map(|i| coeffs[(i % coded) * coded + i / coded])
                    .collect();
                let mut residual = vec![0; n * n];
                dsp.av2_inverse_typed(av2_size, av2_type, &row_major, &mut residual);
                let expected: Vec<u8> = residual
                    .iter()
                    .map(|&r| (128 + i32::from(r)).clamp(0, 255) as u8)
                    .collect();

                let mut dst = vec![128_u8; n * n];
                let mut ours = coeffs.clone();
                let eob = (coded * coded - 1) as u16;
                inverse_transform_add(&mut dst, (0, n), &mut ours, (size, tx_type), eob, EIGHT);
                assert_eq!(dst, expected, "{size:?} {tx_type:?}");
            }
        }
    }
}

#[test]
fn rectangular_dct_dispatch_matches_scalar() {
    let dsp = Dsp::new();
    for size in RectTxSize::ALL {
        let (width, height) = (size.width(), size.height());
        if width == height {
            continue;
        }
        let area = size.coded_width() * size.coded_height();
        let stride = width + 11;
        let input: Vec<i16> = (0..=area).map(pattern).collect();
        let prediction: Vec<u8> = (0..stride * (height + 2))
            .map(|i| pixel(i % stride, i / stride))
            .collect();
        let (mut scalar, mut simd) = (prediction.clone(), prediction);
        let (mut reference, mut coefficients) = (input.clone(), input);
        inverse_transform_add(
            &mut scalar,
            (3, stride),
            &mut reference,
            (size, TxType::DCT_DCT),
            9,
            EIGHT,
        );
        inverse_transform_add_with_dsp(
            &dsp,
            &mut simd,
            (3, stride),
            &mut coefficients,
            (size, TxType::DCT_DCT),
            9,
            EIGHT,
        );
        assert_eq!(simd, scalar, "{size:?}");
        assert_eq!(coefficients, reference, "{size:?}");
    }
}

/// The `u8` instance with `i16` coefficients is the 8-bit path, so the
/// `u16` instance at 8 bits (`i32` coefficients) must give the same pixels
/// and clear the same coefficients, for every size and type class.
#[test]
fn wide_instance_at_eight_bits_matches_u8() {
    let types = [
        TxType::DCT_DCT,
        TxType::ADST_FLIPADST,
        TxType::IDTX,
        TxType::H_DCT,
        TxType::V_DCT,
        TxType::IDTX_INV.with_dpcm(Dpcm::Vertical),
    ];
    for size in RectTxSize::ALL {
        let (w, h) = (size.width(), size.height());
        let area = size.coded_width() * size.coded_height();
        for tx_type in types {
            let supported = |kind, log2| itx_1d::transform(kind, log2).is_some();
            if !supported(tx_type.horizontal(), size.log2_width4())
                || !supported(tx_type.vertical(), size.log2_height4())
            {
                continue;
            }
            for eob in [0, 9, (area - 1) as u16] {
                let narrow_coeffs: Vec<i16> = (0..area).map(pattern).collect();
                let mut wide_coeffs: Vec<i32> = narrow_coeffs.iter().map(|&c| c.into()).collect();
                let mut narrow_coeffs = narrow_coeffs;
                let mut narrow: Vec<u8> = (0..w * h).map(|i| pixel(i % w, i / w)).collect();
                let mut wide: Vec<u16> = narrow.iter().map(|&p| p.into()).collect();
                let transform = (size, tx_type);
                inverse_transform_add(
                    &mut narrow,
                    (0, w),
                    &mut narrow_coeffs,
                    transform,
                    eob,
                    EIGHT,
                );
                inverse_transform_add(&mut wide, (0, w), &mut wide_coeffs, transform, eob, EIGHT);
                let widened: Vec<u16> = narrow.iter().map(|&p| p.into()).collect();
                assert_eq!(wide, widened, "{size:?} {tx_type:?} eob {eob}");
                let narrow_coeffs: Vec<i32> = narrow_coeffs.iter().map(|&c| c.into()).collect();
                assert_eq!(wide_coeffs, narrow_coeffs, "{size:?} {tx_type:?} eob {eob}");
            }
        }
    }
}

/// At 10 bits an all-zero-but-DC `DCT_DCT` block gives the DC shortcut's
/// result (for DCs whose row output stays inside the clamp, as dav2d's
/// shortcut assumes), every output stays in `0..=1023`, and coefficients
/// beyond the 8-bit `i16` range are kept.
#[test]
fn ten_bit_dc_and_range() {
    let ten = BitDepth::TEN;
    for size in RectTxSize::ALL {
        let (w, h) = (size.width(), size.height());
        let area = size.coded_width() * size.coded_height();
        for dc in [-65_000, -40_000, -4000, 0, 513, 40_000, 65_000] {
            let [shortcut, general] = [0, 1].map(|eob| {
                let mut coeffs = vec![0_i32; area];
                coeffs[0] = dc;
                let mut dst = vec![512_u16; w * h];
                let transform = (size, TxType::DCT_DCT);
                inverse_transform_add(&mut dst, (0, w), &mut coeffs, transform, eob, ten);
                dst
            });
            assert_eq!(shortcut, general, "{size:?} dc {dc}");
            assert!(general.iter().all(|&p| p <= 1023), "{size:?} dc {dc}");
        }
        let mut coeffs: Vec<i32> = (0..area).map(|k| i32::from(pattern(k)) * 16).collect();
        let mut dst: Vec<u16> = (0..w * h).map(|i| (i * 37 % 1024) as u16).collect();
        let eob = (area - 1) as u16;
        let transform = (size, TxType::ADST_ADST.with_inter_ddt(size));
        if itx_1d::transform(transform.1.horizontal(), size.log2_width4()).is_none()
            || itx_1d::transform(transform.1.vertical(), size.log2_height4()).is_none()
        {
            continue;
        }
        inverse_transform_add(&mut dst, (0, w), &mut coeffs, transform, eob, ten);
        assert!(dst.iter().all(|&p| p <= 1023), "{size:?}");
    }
}

/// The row pass clamps to `bd + 8` bits as dav2d (`row_clip_min =
/// ~bitdepth_max << 7`): `±2^17` at 10 bits, the `i16` range at 8.
#[test]
fn ten_bit_row_clamp_is_dav2ds() {
    let ten = BitDepth::TEN;
    let (low, high) = intermediate_bounds(ten);
    assert_eq!((low, high), (-(1 << 17), (1 << 17) - 1));
    assert_eq!(
        intermediate_bounds(EIGHT),
        (i16::MIN.into(), i16::MAX.into())
    );
    // `V_DCT` 4x4: the identity row pass keeps each coefficient (gain 128,
    // shift 7), then the clamp; the column DCT's first output is
    // `64 * (row 0 + row 2)`. Row 0 is `2^20`, clamped to `high`; row 2 is
    // `-(high - 100)`, inside the 10-bit clamp but not the 8-bit one. The
    // sum `6400`, after the 10-bit column shift, adds 6 to mid-grey; no
    // clamp would saturate, the 8-bit clamp would add nothing.
    let mut coeffs = vec![0_i32; 16];
    coeffs[0] = 1 << 20;
    coeffs[2] = -(high - 100);
    let mut dst = vec![512_u16; 16];
    inverse_transform_add(
        &mut dst,
        (0, 4),
        &mut coeffs,
        (RectTxSize::Tx4x4, TxType::V_DCT),
        15,
        ten,
    );
    assert_eq!(dst[0], 518);
}

/// The cross-chroma rotation clamps to `bd + 8` bits of the depth, and
/// agrees with the `i16` instance where nothing clamps.
#[test]
fn cross_chroma_clamps_to_the_depth() {
    let mut u = vec![1_i32 << 18; 16];
    let mut v = vec![-(1_i32 << 18); 16];
    // 45 degrees: u' = (u - v) / sqrt(2) beyond 2^17, v' = (u + v) / sqrt(2) = 0.
    cross_chroma(&mut u, &mut v, &CCTX_ANGLES[0], BitDepth::TEN);
    assert!(u.iter().all(|&c| c == (1 << 17) - 1));
    assert!(v.iter().all(|&c| c == 0));
    let mut u = vec![3000_i32; 16];
    let mut v = vec![-3000_i32; 16];
    let (mut narrow_u, mut narrow_v) = (vec![3000_i16; 16], vec![-3000_i16; 16]);
    cross_chroma(&mut u, &mut v, &CCTX_ANGLES[1], BitDepth::TEN);
    cross_chroma(&mut narrow_u, &mut narrow_v, &CCTX_ANGLES[1], EIGHT);
    assert_eq!(
        u,
        narrow_u.iter().map(|&c| i32::from(c)).collect::<Vec<_>>()
    );
    assert_eq!(
        v,
        narrow_v.iter().map(|&c| i32::from(c)).collect::<Vec<_>>()
    );
}
