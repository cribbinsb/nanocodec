//! Tests of [`super`]: fixed vectors from dav2d's C (commit 237229e, 8 bpc),
//! the type numbering, the coefficient clearing, the DC shortcut, and a
//! cross-check of the square types against [`crate::av2_txfm`]. The full
//! differential test against dav2d (every size, type, DPCM mode and end of
//! block; random extremes) was run outside the repository when porting.

use super::*;
use crate::av2_txfm::Av2TxSize;
use crate::av2_txfm::typed::{Av2Kernel, Av2TxType};
use crate::{Dsp, Isa};
use nc_common::fnv1a;

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
    inverse_transform_add(&mut dst, 0, w, &mut coeffs, size, tx_type, eob);
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
fn cross_chroma_matches_dav2d_vector() {
    let mut u: Vec<i16> = (0..64).map(pattern).collect();
    let mut v: Vec<i16> = (500..564).map(pattern).collect();
    cross_chroma(&mut u, &mut v, &CCTX_ANGLES[2]);
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
        let mut dst = vec![128; size.width() * size.height()];
        let last = coeffs[area];
        inverse_transform_add(
            &mut dst,
            0,
            size.width(),
            &mut coeffs,
            size,
            TxType::DCT_DCT,
            9,
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
                let mut dst = vec![128; size.width() * size.height()];
                inverse_transform_add(
                    &mut dst,
                    0,
                    size.width(),
                    &mut coeffs,
                    size,
                    TxType::DCT_DCT,
                    eob,
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

                let mut dst = vec![128; n * n];
                let mut ours = coeffs.clone();
                let eob = (coded * coded - 1) as u16;
                inverse_transform_add(&mut dst, 0, n, &mut ours, size, tx_type, eob);
                assert_eq!(dst, expected, "{size:?} {tx_type:?}");
            }
        }
    }
}
