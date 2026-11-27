//! Fixed-vector tests of the intra predictors: inputs from a small LCG,
//! outputs checked against FNV-1a hashes (and the MHCCP fit's values)
//! captured from dav2d 237229e's C. The full differential test against dav2d
//! (millions of random cases) is run outside the repository.

#![allow(
    clippy::decimal_bitwise_operands,
    reason = "angles are degrees OR'ed with the ANGLE_* flags, as dav2d passes them"
)]
#![allow(clippy::type_complexity, reason = "one fixture row per case")]

use super::*;
use nc_common::{BitDepth, Sample};

/// Deterministic sample content.
fn pattern(seed: u32, len: usize) -> Vec<u8> {
    let mut s = seed;
    (0..len)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (s >> 24) as u8
        })
        .collect()
}

/// FNV-1a 64.
fn fnv(data: &[u8]) -> u64 {
    nc_common::fnv1a(data.iter().copied())
}

const L: i32 = ANGLE_IS_LUMA;
const T: i32 = ANGLE_HAS_TOP_FLAG;
const LF: i32 = ANGLE_HAS_LEFT_FLAG;
const F: i32 = ANGLE_USE_EDGE_FILTER_FLAG;
const MRL1: i32 = 1 << ANGLE_MRL_IDX_SHIFT;
const MRL2: i32 = 2 << ANGLE_MRL_IDX_SHIFT;
const MRL3: i32 = 3 << ANGLE_MRL_IDX_SHIFT;
/// DIP's transpose bit in its mode argument.
const DIP_TRANSPOSE: i32 = 16;

/// (predictor, width, height, angle with flags, max width, max height, hash
/// of the 64x64 destination).
#[rustfmt::skip]
const PREDICT: [(ImplMode, usize, usize, i32, i32, i32, u64); 20] = [
    (ImplMode::Dc, 16, 8, ANGLE_IBP_FLAG, 64, 64, 0x5a4f_eadf_e5af_3de9),
    (ImplMode::TopDc, 8, 16, ANGLE_IBP_FLAG, 64, 64, 0xfda7_f13c_852e_9ab3),
    (ImplMode::LeftDc, 32, 8, 0, 64, 64, 0xc7ea_9de7_3e4f_a325),
    (ImplMode::Dc128, 4, 4, 0, 64, 64, 0xeb42_7ae0_bcfa_2f25),
    (ImplMode::Vert, 8, 8, 90 | ANGLE_MULTI_MRL_FLAG | MRL2 | L, 64, 64, 0xff55_5d40_5ec1_4325),
    (ImplMode::Hor, 16, 4, 180, 64, 64, 0x0032_dd53_7726_3be5),
    (ImplMode::Paeth, 8, 32, 0, 64, 64, 0xa75d_1741_edf1_bfb4),
    (ImplMode::Smooth, 16, 16, 0, 64, 64, 0xb4f2_4e9c_aee4_c74c),
    (ImplMode::SmoothV, 4, 16, 0, 64, 64, 0x5685_4c64_9173_500b),
    (ImplMode::SmoothH, 64, 16, 0, 64, 64, 0xd2e8_09e9_7d64_d20b),
    (ImplMode::Z1, 16, 16, 42 | L | T | LF | F | ANGLE_IBP_FLAG, 12, 64, 0xc65b_1c09_e8d5_59e8),
    (ImplMode::Z1, 8, 8, 67 | L | ANGLE_MULTI_MRL_FLAG | MRL2, 64, 64, 0x0857_e88b_32a6_1625),
    (ImplMode::Z2, 32, 16, 113 | L | T | LF | F | ANGLE_SMOOTH_TOP_EDGE_FLAG, 32, 8, 0x6355_c833_1e84_5878),
    (ImplMode::Z2, 8, 4, 157 | MRL1, 64, 64, 0x2144_86b1_b7ec_abbc),
    (ImplMode::Z3, 8, 32, 203 | L | LF | F | ANGLE_IBP_FLAG, 64, 20, 0x4d2a_a966_e6b8_ce8e),
    (ImplMode::Z3, 4, 16, 248, 64, 64, 0x1414_7268_bcd6_26e2),
    (ImplMode::Z3, 16, 64, 225 | L | LF | F | ANGLE_SMOOTH_LEFT_EDGE_FLAG | MRL3 | ANGLE_MULTI_MRL_FLAG, 64, 64, 0xa091_a3e5_36d1_cc1b),
    (ImplMode::Dip, 16, 8, 3 | DIP_TRANSPOSE, 64, 64, 0x3c59_289e_87d7_8423),
    (ImplMode::Dip, 64, 64, 5, 64, 64, 0x0a8b_a108_b078_bcb1),
    (ImplMode::Dip, 4, 32, 0, 64, 64, 0x908d_0e62_f326_0664),
];

#[test]
fn predictors_match_dav2d() {
    let edge = pattern(1, EDGE_LEN);
    for (i, &(mode, w, h, angle, max_w, max_h, expected)) in PREDICT.iter().enumerate() {
        let mut dst = vec![0; 64 * 64];
        predict(
            mode,
            &mut dst,
            0,
            64,
            &edge,
            EDGE_TOPLEFT,
            w,
            h,
            angle,
            max_w,
            max_h,
            BitDepth::EIGHT,
        );
        assert_eq!(fnv(&dst), expected, "case {i}: {mode:?} {w}x{h}");
    }
}

#[test]
fn dc_without_edges_is_mid_grey() {
    let edge = pattern(1, EDGE_LEN);
    let mut dst = vec![0; 16];
    predict(
        ImplMode::Dc128,
        &mut dst,
        0,
        4,
        &edge,
        EDGE_TOPLEFT,
        4,
        4,
        0,
        4,
        4,
        BitDepth::EIGHT,
    );
    assert!(dst.iter().all(|&p| p == 128));
}

/// (coded mode, x, y, top-right units, bottom-left units, tw4, th4, flags,
/// from the pre-filter row, returned predictor, hash of the edge buffer).
#[rustfmt::skip]
const PREPARE: [(IntraPredMode, i32, i32, i32, i32, i32, i32, i32, bool, ImplMode, u64); 6] = [
    (IntraPredMode::VertLeft, 4, 3, 2, 0, 2, 4, 64 | T | LF | F | ANGLE_SEQHDR_IBP_FLAG | L, false, ImplMode::Z1, 0x8af8_f998_1b9f_50fc),
    (IntraPredMode::Dc, 4, 3, 1, 1, 4, 2, 2 | ANGLE_DIP_FLAG | T | LF, false, ImplMode::Dip, 0x22b8_ca52_daee_210a),
    (IntraPredMode::DiagDownRight, 5, 6, 0, 3, 2, 2, 136 | MRL2 | ANGLE_MULTI_MRL_FLAG | L | T | LF, true, ImplMode::Z2, 0x1a4b_3615_cdd4_4f6f),
    (IntraPredMode::SmoothV, 0, 2, 1, 0, 4, 4, T, false, ImplMode::SmoothV, 0xe1cf_fe79_eb34_5047),
    (IntraPredMode::Paeth, 3, 0, 0, 1, 1, 4, LF, false, ImplMode::Hor, 0xd2c8_c4fa_032c_2b4c),
    (IntraPredMode::HorUp, 6, 8, 0, 2, 8, 2, 206 | T | LF | F | ANGLE_IBP_FLAG | L, true, ImplMode::Z3, 0x5017_1423_7754_4490),
];

#[test]
fn edge_preparation_matches_dav2d() {
    let picture = pattern(2, 128 * 128);
    let sb_row = pattern(3, 128);
    for (i, case) in PREPARE.iter().enumerate() {
        let &(mode, x, y, n_tr, n_bl, tw4, th4, intra_flags, use_sb, expected_mode, expected) =
            case;
        let req = EdgeRequest {
            x,
            y,
            end_x: 32,
            end_y: 32,
            n_tr,
            n_bl,
            mode,
            tw4,
            th4,
            intra_flags,
        };
        let offset = (y as usize * 4) * 128 + x as usize * 4;
        let sb = use_sb.then(|| SampleRef::new(&sb_row, 0));
        let mut edge = vec![0; EDGE_LEN];
        let got = prepare_intra_edges(
            &req,
            &picture,
            offset,
            128,
            sb,
            &mut edge,
            EDGE_TOPLEFT,
            BitDepth::EIGHT,
        );
        assert_eq!((got, fnv(&edge)), (expected_mode, expected), "case {i}");
    }
}

/// A 16x8 chroma block at (8, 8) of 64-wide chroma planes, its luma at
/// (16, 16) of a 128-wide plane, one 4-sample unit past the right edge.
fn run_cfl(implicit: bool, flags: u32) -> (u64, u64) {
    let luma = pattern(4, 128 * 64);
    let mut u = pattern(5, 64 * 32);
    let mut v = pattern(6, 64 * 32);
    let (luma_at, chroma_at) = (16 * 128 + 16, 8 * 64 + 8);
    let luma_top = SampleRef::new(&luma, luma_at - 2 * 128);
    let u_block = CflChroma {
        buf: &mut u,
        offset: chroma_at,
        top_sb_edge: None,
    };
    let v_block = CflChroma {
        buf: &mut v,
        offset: chroma_at,
        top_sb_edge: None,
    };
    let luma_block = SampleRef::new(&luma, luma_at);
    cfl_pred_420(
        implicit,
        luma_top,
        luma_block,
        128,
        u_block,
        v_block,
        64,
        1,
        0,
        16,
        8,
        flags,
        BitDepth::EIGHT,
    );
    (fnv(&u), fnv(&v))
}

#[test]
fn explicit_cfl_matches_dav2d() {
    let flags = CFL_FLT_TYPE_VSTRIP
        | CFL_HAS_TOP
        | CFL_HAS_LEFT
        | (5 << CFL_ALPHA_U_SHIFT)
        | (27 << CFL_ALPHA_V_SHIFT);
    assert_eq!(
        run_cfl(false, flags),
        (0xe674_0f54_59a9_8423, 0xce00_75da_ae9a_a37f)
    );
}

#[test]
fn implicit_cfl_matches_dav2d() {
    let flags = CFL_FLT_TYPE_GAUSS | CFL_HAS_TOP | CFL_HAS_LEFT;
    assert_eq!(
        run_cfl(true, flags),
        (0xbc98_872c_19b0_c86f, 0x0b03_9c78_6dd0_35ca)
    );
}

/// An 8x8 chroma block with both edges, one unit of top-right and of
/// bottom-left, the top direction and the GAUSS filter, through all four
/// MHCCP steps with the decoder's reference counts.
#[test]
fn mhccp_matches_dav2d() {
    let flags = CFL_HAS_TOP | CFL_HAS_LEFT | CflMhDir::Top as u32;
    let (refw, refh, needed_top, needed_left_rows) = (12, 11, 12, 11);
    let luma = pattern(4, 128 * 64);
    let mut buf = vec![0; CFL_MH_MAX_LUMA_SIZE];
    let src = SampleRef::new(&luma, 16 * 128 + 16);
    cfl_mh_gen_y_420(
        &mut buf,
        src,
        None,
        128,
        needed_top,
        needed_left_rows,
        8,
        8,
        flags,
        CFL_FLT_TYPE_GAUSS,
    );
    assert_eq!(fnv(&buf), 0x5e9d_df6f_794a_4b45);

    let mut mat = [[0; 4]; 3];
    let mut imat = [[0; CFL_MH_MAX_EDGE_SAMPLES]; 2];
    cfl_mh_gen_mat(
        &mut mat,
        &mut imat,
        &buf,
        refw,
        refh,
        flags,
        BitDepth::EIGHT,
    );
    let expected_mat = [
        [1_026_304, 574_316, 853_248, 0],
        [574_316, 412_330, 504_832, 0],
        [853_248, 504_832, 753_666, 0],
    ];
    assert_eq!(mat, expected_mat);

    let chroma = pattern(7, 64 * 32);
    let block = 8 * 64 + 8;
    let mut alpha = [0; 4];
    let chroma_ref = SampleRef::new(&chroma, block);
    cfl_mh_calc_alphas(
        &mut alpha,
        chroma_ref,
        None,
        64,
        refw,
        refh,
        &mat,
        &imat,
        flags,
        BitDepth::EIGHT,
    );
    assert_eq!(alpha, [2890, -11576, 76638, 0]);

    let mut dst = chroma.clone();
    let luma_block = SampleRef::new(&buf, 2 * CFL_MH_TOP_STRIDE);
    cfl_mh_pred(
        CflMhDir::Top,
        &mut dst,
        block,
        64,
        luma_block,
        8,
        8,
        &alpha,
        flags,
        BitDepth::EIGHT,
    );
    assert_eq!(fnv(&dst), 0x0cb9_cb4e_006a_62a1);
}

#[test]
fn palette_matches_dav2d() {
    let pal = [3, 40, 77, 120, 160, 200, 230, 255];
    let idx: Vec<u8> = pattern(8, 16 * 8 / 2).iter().map(|b| b & 0x77).collect();
    let mut dst = vec![0; 16 * 8];
    pal_pred(&mut dst, 0, 16, &pal, &idx, 16, 8);
    assert_eq!(fnv(&dst), 0x1e23_318b_3b36_d724);
}

#[test]
fn mode_numbering_is_dav2ds() {
    assert_eq!(IntraPredMode::from_index(12), Some(IntraPredMode::Paeth));
    assert_eq!(IntraPredMode::Cfl as usize, N_INTRA_PRED_MODES);
    assert_eq!(ImplMode::from_index(3), Some(ImplMode::LeftDc));
    assert_eq!(ImplMode::Dip as usize, N_INTRA_PRED_MODES);
    assert_eq!(ImplMode::from_index(N_IMPL_INTRA_PRED_MODES as u8), None);
}

/// `samples` widened to `u16`, shifted left by `shift`.
fn widen(samples: &[u8], shift: u32) -> Vec<u16> {
    samples.iter().map(|&s| u16::from(s) << shift).collect()
}

/// One fixture prediction at samples `T` and `depth`.
fn predict_case<T: Sample>(
    edge: &[T],
    &(mode, w, h, angle, max_w, max_h, _): &(ImplMode, usize, usize, i32, i32, i32, u64),
    depth: BitDepth,
) -> Vec<T> {
    let mut dst = vec![T::default(); 64 * 64];
    predict(
        mode,
        &mut dst,
        0,
        64,
        edge,
        EDGE_TOPLEFT,
        w,
        h,
        angle,
        max_w,
        max_h,
        depth,
    );
    dst
}

/// The `u16` instance at 8 bits does the `u8` instance's arithmetic.
#[test]
fn u16_at_eight_bits_matches_u8() {
    let edge = pattern(1, EDGE_LEN);
    let wide = widen(&edge, 0);
    for (i, case) in PREDICT.iter().enumerate() {
        let narrow = predict_case(&edge, case, BitDepth::EIGHT);
        let deep = predict_case(&wide, case, BitDepth::EIGHT);
        assert_eq!(widen(&narrow, 0), deep, "case {i}: {:?}", case.0);
    }
}

/// Flat 10-bit edges predict that flat value, past the 8-bit range. Not
/// for DIP, whose trained weights need not preserve it, nor for DC over a
/// non-power-of-two count, whose reciprocal division is approximate.
#[test]
fn flat_ten_bit_edges_predict_flat() {
    const VALUE: u16 = 1000;
    let edge = vec![VALUE; EDGE_LEN];
    let mut wrong = Vec::new();
    for (i, case) in PREDICT.iter().enumerate() {
        let (mode, w, h) = (case.0, case.1, case.2);
        let approximate_dc = mode == ImplMode::Dc && !(w + h).is_power_of_two();
        if matches!(mode, ImplMode::Dip | ImplMode::Dc128) || approximate_dc {
            continue;
        }
        let dst = predict_case(&edge, case, BitDepth::TEN);
        if !(0..h).all(|y| dst[y * 64..][..w].iter().all(|&p| p == VALUE)) {
            wrong.push((i, mode));
        }
    }
    assert!(wrong.is_empty(), "{wrong:?}");
}

/// Every 10-bit prediction stays within 10 bits, and DC without edges is
/// the 10-bit mid-grey.
#[test]
fn ten_bit_predictions_stay_in_range() {
    let max = BitDepth::TEN.max() as u16;
    let edge: Vec<u16> = widen(&pattern(1, EDGE_LEN), 2)
        .iter()
        .map(|&p| p | 3)
        .collect();
    for (i, case) in PREDICT.iter().enumerate() {
        let dst = predict_case(&edge, case, BitDepth::TEN);
        assert!(dst.iter().all(|&p| p <= max), "case {i}: {:?}", case.0);
    }
    let dc = (ImplMode::Dc128, 4, 4, 0, 4, 4, 0);
    let dst = predict_case(&edge, &dc, BitDepth::TEN);
    assert_eq!(dst[..4], [512; 4]);
}

/// Copying predictors commute with the scale from 8 to 10 bits: 8-bit
/// edges `<< 2` predict the 8-bit prediction `<< 2`.
#[test]
fn copying_predictors_scale_with_depth() {
    let edge = pattern(1, EDGE_LEN);
    let wide = widen(&edge, 2);
    let cases = [
        (ImplMode::Vert, 8, 8, 90, 64, 64, 0),
        (ImplMode::Hor, 16, 4, 180, 64, 64, 0),
        (ImplMode::Paeth, 8, 32, 0, 64, 64, 0),
    ];
    for case in &cases {
        let narrow = predict_case(&edge, case, BitDepth::EIGHT);
        let deep = predict_case(&wide, case, BitDepth::TEN);
        assert_eq!(widen(&narrow, 2), deep, "{:?}", case.0);
    }

    let pal = [3, 40, 77, 120, 160, 200, 230, 255];
    let idx: Vec<u8> = pattern(8, 16 * 8 / 2).iter().map(|b| b & 0x77).collect();
    let mut narrow = vec![0; 16 * 8];
    pal_pred(&mut narrow, 0, 16, &pal, &idx, 16, 8);
    let mut deep = vec![0; 16 * 8];
    pal_pred(
        &mut deep,
        0,
        16,
        &pal.map(|c| u16::from(c) << 2),
        &idx,
        16,
        8,
    );
    assert_eq!(widen(&narrow, 2), deep);
}

/// Edge preparation at `u16` and 8 bits matches `u8` on every fixture;
/// without either edge at 10 bits it fills around the 10-bit mid-grey.
#[test]
fn edge_preparation_by_depth() {
    let picture = pattern(2, 128 * 128);
    let sb_row = pattern(3, 128);
    let (wide_picture, wide_sb) = (widen(&picture, 0), widen(&sb_row, 0));
    for (i, case) in PREPARE.iter().enumerate() {
        let &(mode, x, y, n_tr, n_bl, tw4, th4, intra_flags, use_sb, _, _) = case;
        let req = EdgeRequest {
            x,
            y,
            end_x: 32,
            end_y: 32,
            n_tr,
            n_bl,
            mode,
            tw4,
            th4,
            intra_flags,
        };
        let offset = (y as usize * 4) * 128 + x as usize * 4;
        let mut edge = vec![0; EDGE_LEN];
        let sb = use_sb.then(|| SampleRef::new(&sb_row[..], 0));
        let narrow = prepare_intra_edges(
            &req,
            &picture,
            offset,
            128,
            sb,
            &mut edge,
            EDGE_TOPLEFT,
            BitDepth::EIGHT,
        );
        let mut wide_edge = vec![0; EDGE_LEN];
        let sb = use_sb.then(|| SampleRef::new(&wide_sb[..], 0));
        let deep = prepare_intra_edges(
            &req,
            &wide_picture,
            offset,
            128,
            sb,
            &mut wide_edge,
            EDGE_TOPLEFT,
            BitDepth::EIGHT,
        );
        assert_eq!((narrow, widen(&edge, 0)), (deep, wide_edge), "case {i}");
    }

    let req = EdgeRequest {
        x: 4,
        y: 4,
        end_x: 32,
        end_y: 32,
        n_tr: 0,
        n_bl: 0,
        mode: IntraPredMode::Smooth,
        tw4: 2,
        th4: 2,
        intra_flags: 0,
    };
    let mut edge = vec![0; EDGE_LEN];
    let mode = prepare_intra_edges(
        &req,
        &wide_picture,
        4 * 4 * 128 + 4 * 4,
        128,
        None,
        &mut edge,
        EDGE_TOPLEFT,
        BitDepth::TEN,
    );
    assert_eq!(mode, ImplMode::Smooth);
    let top = &edge[EDGE_TOPLEFT + 1..][..9];
    let left = &edge[EDGE_TOPLEFT - 9..EDGE_TOPLEFT];
    assert!(top.iter().all(|&p| p == 511), "{top:?}");
    assert!(left.iter().all(|&p| p == 513), "{left:?}");
}

/// `CfL` at `u16` and 8 bits matches `u8`; at 10 bits it stays in range,
/// and without edges or alphas it is the 10-bit mid-grey.
#[test]
fn cfl_by_depth() {
    fn run<T: Sample>(luma: &[T], u: &mut [T], v: &mut [T], flags: u32, depth: BitDepth) {
        let (luma_at, chroma_at) = (16 * 128 + 16, 8 * 64 + 8);
        cfl_pred_420(
            true,
            SampleRef::new(luma, luma_at - 2 * 128),
            SampleRef::new(luma, luma_at),
            128,
            CflChroma {
                buf: u,
                offset: chroma_at,
                top_sb_edge: None,
            },
            CflChroma {
                buf: v,
                offset: chroma_at,
                top_sb_edge: None,
            },
            64,
            1,
            0,
            16,
            8,
            flags,
            depth,
        );
    }
    let flags = CFL_FLT_TYPE_GAUSS | CFL_HAS_TOP | CFL_HAS_LEFT;
    let luma = pattern(4, 128 * 64);
    let (mut u, mut v) = (pattern(5, 64 * 32), pattern(6, 64 * 32));
    let (mut wu, mut wv) = (widen(&u, 0), widen(&v, 0));
    run(&luma, &mut u, &mut v, flags, BitDepth::EIGHT);
    run(&widen(&luma, 0), &mut wu, &mut wv, flags, BitDepth::EIGHT);
    assert_eq!((widen(&u, 0), widen(&v, 0)), (wu, wv));

    let max = BitDepth::TEN.max() as u16;
    let deep_luma = widen(&luma, 2);
    let (mut wu, mut wv) = (
        widen(&pattern(5, 64 * 32), 2),
        widen(&pattern(6, 64 * 32), 2),
    );
    run(&deep_luma, &mut wu, &mut wv, flags, BitDepth::TEN);
    assert!(wu.iter().chain(&wv).all(|&p| p <= max));

    // Explicit with zero alphas and no edges: flat mid-grey.
    run(&deep_luma, &mut wu, &mut wv, 0, BitDepth::TEN);
    assert!(wu[8 * 64 + 8..][..16].iter().all(|&p| p == 512));
}

/// MHCCP at `u16` and 8 bits matches `u8` through all four steps; at 10
/// bits its prediction stays in range.
#[test]
fn mhccp_by_depth() {
    fn run<T: Sample>(luma: &[T], chroma: &[T], depth: BitDepth) -> (Vec<T>, [i32; 4]) {
        let flags = CFL_HAS_TOP | CFL_HAS_LEFT | CflMhDir::Top as u32;
        let (refw, refh) = (12, 11);
        let mut buf = vec![T::default(); CFL_MH_MAX_LUMA_SIZE];
        let src = SampleRef::new(luma, 16 * 128 + 16);
        cfl_mh_gen_y_420(
            &mut buf,
            src,
            None,
            128,
            12,
            11,
            8,
            8,
            flags,
            CFL_FLT_TYPE_GAUSS,
        );
        let mut mat = [[0; 4]; 3];
        let mut imat = [[T::default(); CFL_MH_MAX_EDGE_SAMPLES]; 2];
        cfl_mh_gen_mat(&mut mat, &mut imat, &buf, refw, refh, flags, depth);
        let block = 8 * 64 + 8;
        let mut alpha = [0; 4];
        cfl_mh_calc_alphas(
            &mut alpha,
            SampleRef::new(chroma, block),
            None,
            64,
            refw,
            refh,
            &mat,
            &imat,
            flags,
            depth,
        );
        let mut dst = chroma.to_vec();
        cfl_mh_pred(
            CflMhDir::Top,
            &mut dst,
            block,
            64,
            SampleRef::new(&buf, 2 * CFL_MH_TOP_STRIDE),
            8,
            8,
            &alpha,
            flags,
            depth,
        );
        (dst, alpha)
    }
    let luma = pattern(4, 128 * 64);
    let chroma = pattern(7, 64 * 32);
    let (narrow, narrow_alpha) = run(&luma, &chroma, BitDepth::EIGHT);
    let (deep, deep_alpha) = run(&widen(&luma, 0), &widen(&chroma, 0), BitDepth::EIGHT);
    assert_eq!((widen(&narrow, 0), narrow_alpha), (deep, deep_alpha));

    let max = BitDepth::TEN.max() as u16;
    let (deep, _) = run(&widen(&luma, 2), &widen(&chroma, 2), BitDepth::TEN);
    assert!(deep.iter().all(|&p| p <= max));
}
