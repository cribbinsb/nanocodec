//! Bit-exactness of the decoder's intra prediction of every block shape
//! (`nc_dsp::intra_prediction::av1_block`) against libaom v3.15.3
//! (`build_non_directional_intra_predictors` and
//! `build_directional_and_filter_intra_predictors` of `reconintra.c`,
//! BSD-2-Clause, see `THIRD_PARTY.md`).
//!
//! For each of the 19 transform sizes, [`CASES`] seeded cases draw a random
//! picture, a picture edge that may cut the block and its extensions, the
//! four availability flags, a mode with its angle delta, and whether the
//! edge filter is on and which filter type applies. [`GOLDEN`] holds the
//! FNV-1a 64 hash of libaom's predictions over every case of a size,
//! produced once by a throwaway C harness linked against libaom that runs
//! this same generator (not kept in the repository).
//!
//! Filter intra of every shape up to 32x32 (`Dsp::av1_intra_predict_filter_block`
//! on every ISA) is checked against the same builder with a filter mode.
//! Chroma from luma (4:2:0) is checked the same way against libaom's
//! `cfl.c` subsampling, padding, averaging and prediction, with luma
//! extents that may stop short of the chroma block.

use nc_checkasm::rng::Rng;
use nc_common::{Plane, PlaneView, PlaneViewMut};
use nc_dsp::intra_prediction::av1_block::cfl::CflAc;
use nc_dsp::intra_prediction::av1_block::{BlockEdges, EdgeAvail, EdgeFilterCtx, IntraKind};
use nc_dsp::intra_prediction::filter::FilterIntraMode;
use nc_dsp::{Dsp, Isa};

/// Cases per size.
const CASES: u64 = 400;
/// Seed of (size index `s`, case `k`): `SEED_BASE + 4096 s + k`.
const SEED_BASE: u64 = 0x1D7A_0000;
/// Side of the random picture buffer; the block sits at `(ORIGIN, ORIGIN)`.
const BUFFER: usize = 256;
const ORIGIN: usize = 96;
/// Sizes in `TX_SIZES_ALL` order.
const SIZES: [(usize, usize); 19] = [
    (4, 4),
    (8, 8),
    (16, 16),
    (32, 32),
    (64, 64),
    (4, 8),
    (8, 4),
    (8, 16),
    (16, 8),
    (16, 32),
    (32, 16),
    (32, 64),
    (64, 32),
    (4, 16),
    (16, 4),
    (8, 32),
    (32, 8),
    (16, 64),
    (64, 16),
];
/// libaom `PREDICTION_MODE` values 0..13 and their nominal angles (0 for
/// the non-directional modes).
const MODES: u64 = 13;
const NOMINAL_ANGLE: [u16; 13] = [0, 90, 180, 45, 135, 113, 157, 203, 67, 0, 0, 0, 0];
const ANGLE_STEP: i32 = 3;
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn draw(rng: &mut Rng, bound: usize) -> usize {
    (rng.next_u64() % bound as u64) as usize
}

/// The prediction of one seeded case, row-major.
fn case(w: usize, h: usize, rng: &mut Rng) -> Vec<u8> {
    let frame: Vec<u8> = (0..BUFFER * BUFFER).map(|_| rng.next_u64() as u8).collect();
    let picture_w = ORIGIN + 4 + 4 * draw(rng, 2 * w / 4 + 1);
    let picture_h = ORIGIN + 4 + 4 * draw(rng, 2 * h / 4 + 1);
    let have_above = draw(rng, 4) != 0;
    let have_left = draw(rng, 4) != 0;
    let have_above_right = have_above && ORIGIN + w < picture_w && draw(rng, 2) == 1;
    let have_below_left = have_left && ORIGIN + h < picture_h && draw(rng, 2) == 1;
    let mode = draw(rng, MODES as usize);
    let delta = if (1..=8).contains(&mode) {
        draw(rng, 7) as i32 - 3
    } else {
        0
    };
    let edge_filter_on = draw(rng, 2) == 1;
    let smooth_type = draw(rng, 2) == 1;

    let mut plane = Plane::new(picture_w, picture_h, 0).expect("plane");
    {
        let mut view = plane.view_mut();
        for r in 0..picture_h {
            view.row_mut(r)[..picture_w]
                .copy_from_slice(&frame[r * BUFFER..r * BUFFER + picture_w]);
        }
    }
    let avail = EdgeAvail {
        have_above,
        have_left,
        have_above_right,
        have_below_left,
    };
    let kind = match mode {
        0 => IntraKind::Dc,
        9 => IntraKind::Smooth,
        10 => IntraKind::SmoothVertical,
        11 => IntraKind::SmoothHorizontal,
        12 => IntraKind::Paeth,
        _ => IntraKind::Directional {
            angle: (i32::from(NOMINAL_ANGLE[mode]) + ANGLE_STEP * delta) as u16,
            edge_filter: edge_filter_on.then_some(EdgeFilterCtx {
                smooth_above: smooth_type,
                smooth_left: false,
            }),
        },
    };
    let edges = BlockEdges::read(&plane, ORIGIN, ORIGIN, w, h, avail);
    let mut prediction = vec![0u8; w * h];
    let mut dst = PlaneViewMut::new(&mut prediction, w, w, h).expect("block");
    edges.predict(kind, &mut dst);
    prediction
}

#[test]
fn every_shape_mode_and_edge_setting_matches_libaom() {
    for (index, (&(w, h), expected)) in SIZES.iter().zip(GOLDEN).enumerate() {
        let mut hash = FNV_OFFSET;
        for k in 0..CASES {
            let mut rng = Rng::new(SEED_BASE + 4096 * index as u64 + k);
            for p in case(w, h, &mut rng) {
                hash = (hash ^ u64::from(p)).wrapping_mul(FNV_PRIME);
            }
        }
        assert_eq!(hash, expected, "{w}x{h}");
    }
}

/// Per size, in [`SIZES`] order: libaom's hash.
const GOLDEN: [u64; 19] = [
    0x19b5_7df3_9999_57ac,
    0x3aa2_4fce_0e9a_4915,
    0xe259_cc1f_421d_df6d,
    0x426d_3fe7_48b1_27f7,
    0xec71_3373_a060_c19d,
    0x72b4_aa4c_c998_ad9e,
    0x2a1b_7f1e_bc6a_c3ec,
    0x8240_46d8_6cc8_4f1c,
    0x6548_71bb_5c8d_2b04,
    0x3df1_5867_5b66_cabb,
    0xf1a4_664f_1a10_0787,
    0x17eb_b807_b617_3a55,
    0x61b9_baa1_5cf1_921d,
    0x17ce_c64c_9236_37b3,
    0x3f34_ee03_f7c5_cb49,
    0x40ec_a5a8_abc9_9bd1,
    0x0e90_6ac4_b783_ff48,
    0x1faf_4170_22ea_53c9,
    0x14c9_4004_048f_8473,
];

/// Chroma sizes chroma-from-luma predicts.
const CFL_SIZES: [(usize, usize); 14] = [
    (4, 4),
    (8, 8),
    (16, 16),
    (32, 32),
    (4, 8),
    (8, 4),
    (8, 16),
    (16, 8),
    (16, 32),
    (32, 16),
    (4, 16),
    (16, 4),
    (8, 32),
    (32, 8),
];
const CFL_CASES: u64 = 200;
const CFL_SEED_BASE: u64 = 0xCF1_0000;
/// Side of the random luma buffer.
const LUMA_BUFFER: usize = 64;
/// `alpha_q3` is drawn from `-CFL_ALPHA_MAX..=CFL_ALPHA_MAX`.
const CFL_ALPHA_MAX: i32 = 16;

#[test]
fn chroma_from_luma_matches_libaom() {
    for (index, (&(w, h), expected)) in CFL_SIZES.iter().zip(CFL_GOLDEN).enumerate() {
        let mut hash = FNV_OFFSET;
        for k in 0..CFL_CASES {
            let mut rng = Rng::new(CFL_SEED_BASE + 4096 * index as u64 + k);
            let luma: Vec<u8> = (0..LUMA_BUFFER * LUMA_BUFFER)
                .map(|_| rng.next_u64() as u8)
                .collect();
            let stored_w = 2 * (1 + draw(&mut rng, w / 2));
            let stored_h = 2 * (1 + draw(&mut rng, h / 2));
            let alpha = draw(&mut rng, 2 * CFL_ALPHA_MAX as usize + 1) as i32 - CFL_ALPHA_MAX;
            let mut prediction: Vec<u8> = (0..w * h).map(|_| rng.next_u64() as u8).collect();
            let view = PlaneView::new(&luma, LUMA_BUFFER, LUMA_BUFFER, LUMA_BUFFER).expect("luma");
            let ac = CflAc::from_luma(&view, 2 * stored_w, 2 * stored_h, w, h);
            let mut dst = PlaneViewMut::new(&mut prediction, w, w, h).expect("block");
            ac.predict(alpha, &mut dst);
            for p in prediction {
                hash = (hash ^ u64::from(p)).wrapping_mul(FNV_PRIME);
            }
        }
        assert_eq!(hash, expected, "{w}x{h}");
    }
}

/// Per size, in [`CFL_SIZES`] order: libaom's hash.
const CFL_GOLDEN: [u64; 14] = [
    0x9093_9b22_66e2_5076,
    0xe058_5e28_457e_6cf0,
    0xb57c_321f_c535_92eb,
    0xc3e7_faca_1cdf_0029,
    0x4b01_4fe4_f376_d573,
    0xc924_7dca_7e78_4d07,
    0xc701_d8f2_bd0f_dcd0,
    0xd7b1_5e32_7807_0685,
    0x25b6_a083_d567_dceb,
    0xdd2c_79cd_bb66_6c6a,
    0x6896_c19d_15e3_5825,
    0x4110_37d8_6034_547e,
    0x5031_532e_5c94_f1c2,
    0xac3c_0e51_435f_7879,
];

/// The top-left `width`x`height` of the random `frame` as a picture plane.
fn picture(frame: &[u8], width: usize, height: usize) -> Plane {
    let mut plane = Plane::new(width, height, 0).expect("plane");
    let mut view = plane.view_mut();
    for r in 0..height {
        view.row_mut(r)[..width].copy_from_slice(&frame[r * BUFFER..r * BUFFER + width]);
    }
    plane
}

/// Shapes filter intra allows, in the order of [`FILTER_GOLDEN`].
const FILTER_SIZES: [(usize, usize); 14] = [
    (4, 4),
    (8, 8),
    (16, 16),
    (32, 32),
    (4, 8),
    (8, 4),
    (8, 16),
    (16, 8),
    (16, 32),
    (32, 16),
    (4, 16),
    (16, 4),
    (8, 32),
    (32, 8),
];
const FILTER_CASES: u64 = 200;
const FILTER_SEED_BASE: u64 = 0xF17A_0000;

#[test]
fn filter_intra_of_every_shape_matches_libaom() {
    for isa in Isa::ALL {
        let Some(dsp) = Dsp::with_isa(isa) else {
            continue;
        };
        for (index, (&(w, h), expected)) in FILTER_SIZES.iter().zip(FILTER_GOLDEN).enumerate() {
            let mut hash = FNV_OFFSET;
            for k in 0..FILTER_CASES {
                let mut rng = Rng::new(FILTER_SEED_BASE + 4096 * index as u64 + k);
                let frame: Vec<u8> = (0..BUFFER * BUFFER).map(|_| rng.next_u64() as u8).collect();
                let picture_w = ORIGIN + 4 + 4 * draw(&mut rng, 2 * w / 4 + 1);
                let picture_h = ORIGIN + 4 + 4 * draw(&mut rng, 2 * h / 4 + 1);
                let avail = EdgeAvail {
                    have_above: draw(&mut rng, 4) != 0,
                    have_left: draw(&mut rng, 4) != 0,
                    ..EdgeAvail::default()
                };
                let mode = FilterIntraMode::ALL[draw(&mut rng, FilterIntraMode::ALL.len())];
                let plane = picture(&frame, picture_w, picture_h);
                let edges = BlockEdges::read(&plane, ORIGIN, ORIGIN, w, h, avail);
                let mut prediction = vec![0u8; w * h];
                let mut dst = PlaneViewMut::new(&mut prediction, w, w, h).expect("block");
                dsp.av1_intra_predict_filter_block(&edges, mode, &mut dst);
                for p in prediction {
                    hash = (hash ^ u64::from(p)).wrapping_mul(FNV_PRIME);
                }
            }
            assert_eq!(hash, expected, "{isa:?} {w}x{h}");
        }
    }
}

/// Per shape, in [`FILTER_SIZES`] order: libaom's hash.
const FILTER_GOLDEN: [u64; 14] = [
    0x8aca_39b9_1c34_25ce,
    0xc8c6_9aa0_ff68_17d3,
    0xdcc7_5cba_6709_1e87,
    0xbb82_1adb_4396_b507,
    0xd487_a4c6_a346_ac62,
    0x66db_3c2a_a134_4053,
    0x3654_dd0b_92e2_96ad,
    0xf8ee_36ca_83ec_9b64,
    0x353b_58f2_f18c_8f47,
    0xc597_ae56_00dc_de89,
    0x46e3_29d3_3ef1_ecf6,
    0xc211_78df_4055_e485,
    0xa1a3_eb9e_f69e_dc2d,
    0xdbfe_6b7e_47f5_deba,
];
