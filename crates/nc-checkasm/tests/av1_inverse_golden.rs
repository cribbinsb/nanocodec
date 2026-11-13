//! Bit-exactness of `Dsp::av1_inverse_add` against libaom v3.15.3's
//! `av1_inv_txfm2d_add_{W}x{H}_c` (BSD-2-Clause, see `THIRD_PARTY.md`), the
//! reference behind the AV1 conformance MD5s, for every size and every
//! transform type that fits it, on every ISA this machine supports; and of
//! the lossless WHT against `av1_highbd_iwht4x4_16_add_c`.
//!
//! The inputs come from a seeded generator (below); [`GOLDEN`] holds, per
//! (size, type), the FNV-1a 64 hash of libaom's reconstructed pixels over
//! all [`TRIALS`]. The hashes were produced once by a throwaway C harness
//! linked against libaom that runs this same generator (not kept in the
//! repository); a mismatch names the (size, type) and ISA to debug.
//!
//! Each trial's mode stresses one regime: full-range 16-bit coefficients
//! (every stage clamp), small dense coefficients, sparse coefficients,
//! DC only (`eob` 1), a low-frequency triangle with `eob` 36 (the
//! EOB-sparse DCT kernels), and alternating ±32768 extremes.

use nc_checkasm::rng::Rng;
use nc_common::PlaneViewMut;
use nc_dsp::{Av1TxSize, Av1TxType, Dsp, Isa};

/// Trials per (size, type); the mode is `trial % MODES`.
const TRIALS: u64 = 12;
const MODES: u64 = 6;
/// Seed of (size index `s`, type `t`, trial `k`): `SEED_BASE + 256 s + 16 t + k`.
const SEED_BASE: u64 = 0xA71_0000;
/// The low-frequency triangle `r + c < LOW_TRIANGLE` fills exactly the first
/// 36 positions of the default zig-zag scan.
const LOW_TRIANGLE: usize = 8;
const LOW_TRIANGLE_EOB: usize = 36;
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// One trial's coefficients (column-major coded area) and end-of-block.
fn coefficients(size: Av1TxSize, kind: Av1TxType, trial: u64, rng: &mut Rng) -> (Vec<i32>, usize) {
    let coded_h = size.coded_height();
    let area = size.coded_area();
    let mut coeffs = vec![0; area];
    let mode = trial % MODES;
    for (i, coeff) in coeffs.iter_mut().enumerate() {
        let (c, r) = (i / coded_h, i % coded_h);
        *coeff = match mode {
            0 => i32::from(rng.next_u64() as u16 as i16),
            1 => (rng.next_u64() % 2048) as i32 - 1024,
            2 => {
                let v = rng.next_u64();
                if v.is_multiple_of(8) {
                    ((v >> 8) % 8192) as i32 - 4096
                } else {
                    0
                }
            }
            3 if i == 0 => i32::from(rng.next_u64() as u16 as i16),
            4 if r + c < LOW_TRIANGLE => (rng.next_u64() % 4096) as i32 - 2048,
            3 | 4 => 0,
            _ => {
                if (i as u64 + trial).is_multiple_of(2) {
                    32767
                } else {
                    -32768
                }
            }
        };
    }
    let eob = match mode {
        3 => 1,
        4 if size.coded_width() >= LOW_TRIANGLE && coded_h >= LOW_TRIANGLE => {
            low_triangle_eob(size, kind)
        }
        _ => area,
    };
    (coeffs, eob)
}

/// The end of block of the low triangle `r + c < LOW_TRIANGLE` in `kind`'s
/// scan (AV1 §5.11.41): the first 36 positions of the 2-D zig-zag; for the
/// 1-D classes one past its last position in their raster scans, row 0 of
/// column 7 (`H_*`, column by column) or column 0 of row 7 (`V_*`, row by
/// row). libaom ignores the end of block, so the hashes do not depend on it;
/// our inverse bounds its row pass by it, so it must match the scan.
fn low_triangle_eob(size: Av1TxSize, kind: Av1TxType) -> usize {
    use Av1TxType::{HAdst, HDct, HFlipadst, VAdst, VDct, VFlipadst};
    match kind {
        HDct | HAdst | HFlipadst => (LOW_TRIANGLE - 1) * size.coded_height() + 1,
        VDct | VAdst | VFlipadst => (LOW_TRIANGLE - 1) * size.coded_width() + 1,
        _ => LOW_TRIANGLE_EOB,
    }
}

/// The hash of `dsp`'s reconstructions over every trial of (size, type).
fn case_hash(dsp: &Dsp, size_index: usize, kind: Av1TxType) -> u64 {
    let size = Av1TxSize::ALL[size_index];
    let (w, h) = (size.width(), size.height());
    let mut hash = FNV_OFFSET;
    for trial in 0..TRIALS {
        let mut rng = Rng::new(SEED_BASE + 256 * size_index as u64 + 16 * kind as u64 + trial);
        let (coeffs, eob) = coefficients(size, kind, trial, &mut rng);
        let mut pixels: Vec<u8> = (0..w * h).map(|_| rng.next_u64() as u8).collect();
        let mut dst = PlaneViewMut::new(&mut pixels, w, w, h).expect("block plane");
        dsp.av1_inverse_add(size, kind, &coeffs, eob, &mut dst);
        for &p in &pixels {
            hash = (hash ^ u64::from(p)).wrapping_mul(FNV_PRIME);
        }
    }
    hash
}

#[test]
fn every_size_and_type_matches_libaom() {
    let fitting = Av1TxSize::ALL
        .iter()
        .map(|&size| Av1TxType::ALL.iter().filter(|kind| kind.fits(size)).count())
        .sum::<usize>();
    assert_eq!(
        GOLDEN.len(),
        fitting,
        "one golden hash per fitting (size, type)"
    );
    for isa in Isa::ALL {
        let Some(dsp) = Dsp::with_isa(isa) else {
            continue;
        };
        for &(size_index, kind, expected) in GOLDEN {
            let kind = Av1TxType::from_value(kind).expect("type value");
            assert!(kind.fits(Av1TxSize::ALL[size_index]));
            assert_eq!(
                case_hash(&dsp, size_index, kind),
                expected,
                "{isa:?} {:?} {kind:?}",
                Av1TxSize::ALL[size_index]
            );
        }
    }
}

/// libaom `av1_highbd_iwht4x4_16_add_c` over [`WHT_TRIALS`] seeded trials
/// (full-range, small and DC-only coefficients, by `trial % 3`).
const WHT_GOLDEN: u64 = 0x61e9_8a2f_e6e8_2fc4;
const WHT_TRIALS: u64 = 64;
const WHT_SEED_BASE: u64 = 0x3A_7000;

#[test]
fn lossless_wht_matches_libaom() {
    let dsp = Dsp::with_isa(Isa::Scalar).expect("scalar");
    let mut hash = FNV_OFFSET;
    for trial in 0..WHT_TRIALS {
        let mut rng = Rng::new(WHT_SEED_BASE + trial);
        let coeffs: [i32; 16] = core::array::from_fn(|i| match trial % 3 {
            0 => i32::from(rng.next_u64() as u16 as i16),
            1 => (rng.next_u64() % 1024) as i32 - 512,
            _ if i == 0 => i32::from(rng.next_u64() as u16 as i16),
            _ => 0,
        });
        let mut pixels: Vec<u8> = (0..16).map(|_| rng.next_u64() as u8).collect();
        let mut dst = PlaneViewMut::new(&mut pixels, 4, 4, 4).expect("block plane");
        dsp.av1_inverse_wht_add(&coeffs, &mut dst);
        for &p in &pixels {
            hash = (hash ^ u64::from(p)).wrapping_mul(FNV_PRIME);
        }
    }
    assert_eq!(hash, WHT_GOLDEN);
}

/// (index into `Av1TxSize::ALL`, `TxType` value, libaom hash).
#[rustfmt::skip]
const GOLDEN: &[(usize, u8, u64)] = &[
    (0, 0, 0x9e4d_2021_895c_a09e),
    (0, 1, 0x1f3d_17fc_2fea_c02b),
    (0, 2, 0x0835_a3c6_eb76_daaf),
    (0, 3, 0x6ede_48b0_3ba7_a89c),
    (0, 4, 0x9958_5c65_4cd2_38aa),
    (0, 5, 0x2b1a_f2cb_cf58_ff17),
    (0, 6, 0xbaed_0126_8aec_8a1b),
    (0, 7, 0xff38_9fd7_2a75_af54),
    (0, 8, 0x31fe_68e4_88ba_dab3),
    (0, 9, 0xe1f7_8b25_f739_cf9f),
    (0, 10, 0x4ab7_43b8_9bb0_322a),
    (0, 11, 0x6b2e_0ae4_ff33_5cb7),
    (0, 12, 0x14da_b8b1_a6bf_c94d),
    (0, 13, 0x6113_d778_24d9_ba41),
    (0, 14, 0xe2fc_7914_d801_a1bc),
    (0, 15, 0x0752_b768_53a3_e4f5),
    (1, 0, 0x99c8_4212_4908_b8e6),
    (1, 1, 0x92a5_3248_3963_c1b2),
    (1, 2, 0xe48c_d17f_1d92_c6d8),
    (1, 3, 0xd28a_c754_8d40_b00c),
    (1, 4, 0xb553_7eb3_47fe_cfa1),
    (1, 5, 0x3d20_8603_351b_a8d1),
    (1, 6, 0xbdaf_e0fa_c856_074b),
    (1, 7, 0x5e6f_ee2f_32f2_4d9c),
    (1, 8, 0x6716_7b42_6a43_6885),
    (1, 9, 0xb453_964c_f108_6de6),
    (1, 10, 0x3c5f_47a0_88f7_3de5),
    (1, 11, 0xaf8b_d064_a485_3830),
    (1, 12, 0x5ddd_66eb_b5d4_5285),
    (1, 13, 0xc335_fd8c_6c38_d63c),
    (1, 14, 0xe751_b1e7_d27c_96c1),
    (1, 15, 0x5d47_66dd_82ef_f8f0),
    (2, 0, 0xf229_b86b_4f79_1f0e),
    (2, 1, 0x4b50_d13b_99f3_2ac4),
    (2, 2, 0xc23f_848c_448b_6a43),
    (2, 3, 0x3b2b_8cef_8d32_aae5),
    (2, 4, 0xbaf5_f794_058f_0a45),
    (2, 5, 0x0dd1_0bf0_2d35_0d65),
    (2, 6, 0xb55e_8e80_f7cb_7d2e),
    (2, 7, 0x4630_7bb9_69bb_3a52),
    (2, 8, 0x5fc4_b120_b82a_ddad),
    (2, 9, 0x6b15_0e9d_6ea8_ac7a),
    (2, 10, 0x91d7_ce75_9120_f7a5),
    (2, 11, 0x1f9b_ba48_6bf4_f80e),
    (2, 12, 0x75b4_c80d_d63f_f912),
    (2, 13, 0x5642_1cdd_5302_238c),
    (2, 14, 0x6e85_2935_87af_63df),
    (2, 15, 0xe47b_e0ef_9334_5b2b),
    (3, 0, 0x910c_5859_1f17_1003),
    (3, 9, 0x000c_04ea_2c6c_c9fa),
    (3, 10, 0x322f_a917_15d7_0ac1),
    (3, 11, 0x59b1_e5ee_ec1c_d1ba),
    (4, 0, 0x6b4f_7fe5_dbb3_bbd1),
    (5, 0, 0x248b_2964_6c40_f546),
    (5, 1, 0x2d87_8cb8_357d_c863),
    (5, 2, 0x663c_4f58_1915_7d64),
    (5, 3, 0x6b9f_138c_456e_3a82),
    (5, 4, 0x1174_8a90_7b87_f74a),
    (5, 5, 0x40e4_4526_6f11_230a),
    (5, 6, 0xac7f_dd87_828b_2b9a),
    (5, 7, 0x04b5_1a90_25cc_b0b3),
    (5, 8, 0x63f1_ff9e_34be_d4b2),
    (5, 9, 0x4d52_0787_9b91_847c),
    (5, 10, 0xf98f_60dc_38ae_1a85),
    (5, 11, 0x9be3_3d14_5eaa_5dd8),
    (5, 12, 0x37c7_3301_6211_c125),
    (5, 13, 0xb9c0_3c1b_522c_01b8),
    (5, 14, 0x4881_8e0c_b73a_b6e8),
    (5, 15, 0x227c_c9df_edbb_0bee),
    (6, 0, 0x0ad6_074b_e5a1_c699),
    (6, 1, 0xe50a_fe81_969e_6039),
    (6, 2, 0x69c4_8d72_ab6b_74a1),
    (6, 3, 0xd62e_f477_0277_98b7),
    (6, 4, 0x7453_1198_1755_ab5f),
    (6, 5, 0x4be4_0718_b844_8cee),
    (6, 6, 0x21f4_578b_bd6c_8932),
    (6, 7, 0x9ac6_5bd9_492b_df6a),
    (6, 8, 0x8d31_0256_3a04_c6de),
    (6, 9, 0xe63d_a602_0145_9e3b),
    (6, 10, 0x09df_55e1_788e_8df5),
    (6, 11, 0x7c73_25c8_746f_d9b7),
    (6, 12, 0xf2d7_73af_71fe_f064),
    (6, 13, 0x45d3_ad18_f5d2_9d82),
    (6, 14, 0x135e_6d60_623e_0263),
    (6, 15, 0x9432_1af5_b25c_464d),
    (7, 0, 0x7381_1f36_c32e_69f4),
    (7, 1, 0x7a0f_0298_8a9a_9fbf),
    (7, 2, 0x6982_1c45_7c17_5f5e),
    (7, 3, 0x815b_3060_2219_7282),
    (7, 4, 0xba20_9a33_16dc_19aa),
    (7, 5, 0xb106_2362_2d31_bf41),
    (7, 6, 0xb09d_009c_cfc5_5ac1),
    (7, 7, 0x7ccf_1d9e_d7d9_d01e),
    (7, 8, 0x48cd_4fdb_8a90_b80d),
    (7, 9, 0xe158_8f4c_3d3b_5f65),
    (7, 10, 0x823a_0351_3a85_7f78),
    (7, 11, 0x9ce8_3ffb_3d0e_fb6b),
    (7, 12, 0x8939_55e7_9b1f_6be8),
    (7, 13, 0x359e_e8f3_c443_c506),
    (7, 14, 0x989c_554c_16a2_fa59),
    (7, 15, 0xfa98_f0c9_cb64_28f6),
    (8, 0, 0xb6b8_053f_5e34_e80c),
    (8, 1, 0x9de1_f760_4fc3_5c96),
    (8, 2, 0xa836_4cf1_2dc4_2de7),
    (8, 3, 0x2c8a_f2cc_1515_229c),
    (8, 4, 0x4789_5c7c_ce29_aed4),
    (8, 5, 0x2dce_4259_b84e_f2af),
    (8, 6, 0xf54b_d578_7fda_3a72),
    (8, 7, 0x8b5b_41b4_fb1e_2e81),
    (8, 8, 0x04da_cdb0_7925_d0dc),
    (8, 9, 0x2990_3693_1c18_7478),
    (8, 10, 0x2b3a_d1d1_f5f1_8498),
    (8, 11, 0x3a32_291a_5a36_fe85),
    (8, 12, 0x7f04_a4b1_7399_5e49),
    (8, 13, 0xa442_847e_13c9_cfc8),
    (8, 14, 0xfc31_d40d_b10c_2924),
    (8, 15, 0x195f_21de_d66b_aa50),
    (9, 0, 0x79c0_97dd_fc0b_24fa),
    (9, 2, 0x82f7_6a7b_3c45_5b70),
    (9, 5, 0xc0ec_1b95_d2e6_1b62),
    (9, 9, 0x9a73_25f4_d4ba_6dc0),
    (9, 10, 0x2fc6_8b33_9d6f_fbf1),
    (9, 11, 0xc96b_6c97_6e55_a0f9),
    (9, 13, 0x32f3_4235_3a19_7ff7),
    (9, 15, 0xe214_59a4_b916_0abb),
    (10, 0, 0x2455_0469_e827_a66d),
    (10, 1, 0xd8ee_f85d_2579_0eec),
    (10, 4, 0x892e_73e4_40f2_3b1f),
    (10, 9, 0xbbe4_de8a_4f91_313d),
    (10, 10, 0x8e1e_9b68_31f0_3043),
    (10, 11, 0x7b84_fc01_9f1b_cfc1),
    (10, 12, 0x1fe5_c15e_a736_4535),
    (10, 14, 0x4769_1a11_0648_e7e6),
    (11, 0, 0x2e20_28b8_2f0c_fa90),
    (11, 10, 0x41c2_e7bd_c8d6_caba),
    (12, 0, 0x67da_6253_cdf4_8efa),
    (12, 11, 0x0da1_43ce_6fdb_f89b),
    (13, 0, 0x3c2c_e7f5_46cc_0276),
    (13, 1, 0x6345_0de5_af85_f9aa),
    (13, 2, 0xd1f5_5a03_4d84_4fb9),
    (13, 3, 0x5d6b_96c6_3f87_4622),
    (13, 4, 0x52e9_7c08_a058_7226),
    (13, 5, 0xa24e_da35_8684_f2bd),
    (13, 6, 0x6d5e_83eb_3f3a_8661),
    (13, 7, 0x5a6a_f672_ddbd_d9bf),
    (13, 8, 0xfc6d_fbf5_daf5_f8fd),
    (13, 9, 0x2ebb_bc5c_edd4_7088),
    (13, 10, 0xc43a_63e5_2a30_44e4),
    (13, 11, 0xa989_3729_b8e7_83e0),
    (13, 12, 0x942e_3676_0847_c66e),
    (13, 13, 0xe89c_858c_a1a3_c203),
    (13, 14, 0x5cdf_d45b_1fa2_fc82),
    (13, 15, 0xd5c0_7c93_805e_f90c),
    (14, 0, 0xeb92_eb28_7482_f934),
    (14, 1, 0xee36_5f78_3989_5800),
    (14, 2, 0xf542_f4bc_d6e2_f1e8),
    (14, 3, 0x9d2d_0304_9f97_7b37),
    (14, 4, 0x9ee5_0fba_6d53_7f6b),
    (14, 5, 0xf815_f5a7_2c27_42ea),
    (14, 6, 0x4170_48d4_beef_4a5b),
    (14, 7, 0x8f14_f485_6309_7628),
    (14, 8, 0x5a0f_fa1a_be4f_df0e),
    (14, 9, 0x1625_cc5e_02d6_5e5b),
    (14, 10, 0xaa9f_ba98_ebb5_7cb4),
    (14, 11, 0xbc0c_09ce_1bd8_f174),
    (14, 12, 0xbfca_0e80_9867_f72c),
    (14, 13, 0xfef5_6a8c_6247_21f5),
    (14, 14, 0xadc1_6698_0f11_41e7),
    (14, 15, 0x453a_c216_c77d_bc00),
    (15, 0, 0x4b84_9805_67d3_ada3),
    (15, 2, 0x5f43_b67d_e3a2_ad6f),
    (15, 5, 0x35f2_23ac_70a2_6461),
    (15, 9, 0xc53a_89a1_a9e9_11e5),
    (15, 10, 0x81bd_a99b_96c4_a5b6),
    (15, 11, 0x9506_94b2_b526_c138),
    (15, 13, 0x7d6d_b1fc_9986_4c43),
    (15, 15, 0x4ab2_dc18_6f2a_384a),
    (16, 0, 0xbc25_e2d1_86dc_4dd7),
    (16, 1, 0xdb20_5c02_dce7_3f30),
    (16, 4, 0xc861_0661_d676_f0fd),
    (16, 9, 0x14ee_8a3c_f6f3_d032),
    (16, 10, 0xed6c_e866_b43c_4f86),
    (16, 11, 0x15cd_ffa8_597e_2ffd),
    (16, 12, 0x0550_6af6_c293_d882),
    (16, 14, 0xcf6c_d624_f315_52cb),
    (17, 0, 0xdb10_e3e6_0863_b758),
    (17, 2, 0xfdc2_f22f_020a_ae4e),
    (17, 5, 0x2e88_3515_b87b_02a0),
    (17, 10, 0xf70a_1b55_e5f5_a3b7),
    (18, 0, 0x018f_a3a6_0d7d_4129),
    (18, 1, 0x45aa_4956_49ad_1433),
    (18, 4, 0x4b3f_4361_e1bc_fd12),
    (18, 11, 0x3596_5db2_ff72_adeb),
];
