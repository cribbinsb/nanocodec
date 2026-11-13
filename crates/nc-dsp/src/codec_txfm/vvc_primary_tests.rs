//! Primary-transform fixtures from pinned `VVdeC` bases and independent i64 scaling.
use super::*;
use VvcTransformKind::{Dct2, Dct8, Dst7};
const CASES: &[((usize, usize), [VvcTransformKind; 2], u64)] = &[
    ((1, 16), [Dct2, Dct2], 0x944d_d73a_38ad_a117),
    ((1, 16), [Dct2, Dst7], 0x92e1_d4f2_6bef_2943),
    ((1, 16), [Dct2, Dct8], 0x91db_022b_75af_8a8b),
    ((16, 1), [Dct2, Dct2], 0x944d_d73a_38ad_a117),
    ((16, 1), [Dst7, Dct2], 0x92e1_d4f2_6bef_2943),
    ((16, 1), [Dct8, Dct2], 0x91db_022b_75af_8a8b),
    ((2, 8), [Dct2, Dct2], 0x8b23_cd3e_f0c9_362e),
    ((2, 8), [Dct2, Dst7], 0x98d0_4a67_0553_b7f1),
    ((2, 8), [Dct2, Dct8], 0x8b29_04b3_5ee6_4dbc),
    ((4, 4), [Dct2, Dct2], 0x2107_61c7_eb01_ee22),
    ((4, 4), [Dct2, Dst7], 0xeea2_8896_7006_c4a7),
    ((4, 4), [Dct2, Dct8], 0x4205_f9ab_efd2_84d9),
    ((4, 4), [Dst7, Dct2], 0x1a2d_0429_5139_b3f2),
    ((4, 4), [Dst7, Dst7], 0x61bf_b0ca_236f_844f),
    ((4, 4), [Dst7, Dct8], 0x1626_0b78_9bc3_6047),
    ((4, 4), [Dct8, Dct2], 0xb2fe_09e9_f416_74bd),
    ((4, 4), [Dct8, Dst7], 0xc3a0_c03b_f4d2_ca7e),
    ((4, 4), [Dct8, Dct8], 0x8846_d67c_ba48_392d),
    ((4, 16), [Dct2, Dct2], 0xc1e8_4f47_f6a1_3f31),
    ((4, 16), [Dct2, Dst7], 0x5d05_a028_a8d4_3380),
    ((4, 16), [Dct2, Dct8], 0x46bf_c005_9995_5f79),
    ((4, 16), [Dst7, Dct2], 0xc1bd_9473_55ea_70d6),
    ((4, 16), [Dst7, Dst7], 0x17ba_5770_517a_c53f),
    ((4, 16), [Dst7, Dct8], 0xc4df_fe60_df1a_c416),
    ((4, 16), [Dct8, Dct2], 0x42e2_c8ff_518a_8f90),
    ((4, 16), [Dct8, Dst7], 0x44c1_d0a9_7c2a_e788),
    ((4, 16), [Dct8, Dct8], 0x32b6_c93b_456e_9662),
    ((8, 32), [Dct2, Dct2], 0x7167_0b32_0d33_fb02),
    ((8, 32), [Dct2, Dst7], 0x4ff5_2e01_9a31_48ef),
    ((8, 32), [Dct2, Dct8], 0x5272_2052_f653_70bd),
    ((8, 32), [Dst7, Dct2], 0x473b_4e2b_c098_db74),
    ((8, 32), [Dst7, Dst7], 0x73d3_73f7_3e4e_4e74),
    ((8, 32), [Dst7, Dct8], 0x8112_d7b6_5c3d_f4d4),
    ((8, 32), [Dct8, Dct2], 0x3caa_2bb2_8bc1_b6a4),
    ((8, 32), [Dct8, Dst7], 0x4850_042e_25b9_f14e),
    ((8, 32), [Dct8, Dct8], 0x9d3c_90c4_10f0_3a3f),
    ((16, 4), [Dct2, Dct2], 0xf378_a926_d1d1_2301),
    ((16, 4), [Dct2, Dst7], 0x32a7_049e_67d0_993d),
    ((16, 4), [Dct2, Dct8], 0x5438_6774_ff11_c0ab),
    ((16, 4), [Dst7, Dct2], 0x5e57_a4b8_2201_0ae9),
    ((16, 4), [Dst7, Dst7], 0x0ecb_d1de_583e_48ec),
    ((16, 4), [Dst7, Dct8], 0xb54b_218f_c06c_b527),
    ((16, 4), [Dct8, Dct2], 0xbd62_afcd_52e8_2b32),
    ((16, 4), [Dct8, Dst7], 0x343b_0c85_f6bd_431c),
    ((16, 4), [Dct8, Dct8], 0x2c8e_cf5d_62ff_7a17),
    ((16, 16), [Dct2, Dct2], 0x6dca_5c21_8106_edc0),
    ((16, 16), [Dct2, Dst7], 0xc22b_daee_4c9d_a197),
    ((16, 16), [Dct2, Dct8], 0x05ab_2ed6_f9a0_54c6),
    ((16, 16), [Dst7, Dct2], 0x0d21_b761_13c3_ebbf),
    ((16, 16), [Dst7, Dst7], 0xd56d_43d1_e8f8_e4b8),
    ((16, 16), [Dst7, Dct8], 0x0ef8_a18e_3308_a90c),
    ((16, 16), [Dct8, Dct2], 0x7bf5_1f8a_ce22_5816),
    ((16, 16), [Dct8, Dst7], 0x6694_d5d9_9123_1a12),
    ((16, 16), [Dct8, Dct8], 0x984d_c222_5089_d9cf),
    ((32, 8), [Dct2, Dct2], 0x24a9_fccc_5689_2faf),
    ((32, 8), [Dct2, Dst7], 0xdf01_8b5c_07ca_904d),
    ((32, 8), [Dct2, Dct8], 0x8701_2518_e111_afb3),
    ((32, 8), [Dst7, Dct2], 0x19ab_6ce7_7f37_8dec),
    ((32, 8), [Dst7, Dst7], 0x74f3_db82_cdf7_9508),
    ((32, 8), [Dst7, Dct8], 0x6329_42d5_f31f_141e),
    ((32, 8), [Dct8, Dct2], 0x2783_a139_5681_5aea),
    ((32, 8), [Dct8, Dst7], 0x21b0_027c_28d4_3c5e),
    ((32, 8), [Dct8, Dct8], 0x34dc_559c_b8b9_9484),
    ((32, 32), [Dct2, Dct2], 0x65a3_53bb_fc87_8037),
    ((32, 32), [Dct2, Dst7], 0xbaa9_bc05_a478_3883),
    ((32, 32), [Dct2, Dct8], 0x1ac6_5ffc_5574_9192),
    ((32, 32), [Dst7, Dct2], 0x842f_3922_89fa_7a05),
    ((32, 32), [Dst7, Dst7], 0xec79_7765_e377_a1ac),
    ((32, 32), [Dst7, Dct8], 0xfebc_d8cc_8d3e_b0f5),
    ((32, 32), [Dct8, Dct2], 0x55a4_e4f2_8613_7c67),
    ((32, 32), [Dct8, Dst7], 0x48da_673e_a237_270f),
    ((32, 32), [Dct8, Dct8], 0x0d92_c39c_a783_3d12),
    ((64, 4), [Dct2, Dct2], 0xcd1c_db47_bc4c_a4f6),
    ((64, 4), [Dct2, Dst7], 0x9efb_900d_3ef9_a5c7),
    ((64, 4), [Dct2, Dct8], 0xc9ab_f496_9adf_8ca6),
    ((4, 64), [Dct2, Dct2], 0x0c58_9dca_1a20_709e),
    ((4, 64), [Dst7, Dct2], 0x2fb6_a339_4ef3_477d),
    ((4, 64), [Dct8, Dct2], 0xf961_a6ab_b8fa_2e06),
    ((64, 64), [Dct2, Dct2], 0xfa97_298a_aed6_ecee),
];
#[test]
fn primary_matches_reference_bases_scaling_clipping_and_zero_out() {
    for dsp in [Dsp::new(), Dsp::with_isa(crate::Isa::Scalar).unwrap()] {
        for &(size, kinds, expected) in CASES {
            let length = size.0 * size.1;
            let mut digest = 0xcbf2_9ce4_8422_2325_u64;
            for qp in [0, 22, 63] {
                for dependent in [false, true] {
                    for pattern in 0..2 {
                        let levels: Vec<_> = (0..length)
                            .map(|i| {
                                if pattern == 0 {
                                    (i as i32 * 37 % 29) - 14
                                } else if i % 2 == 0 {
                                    -32768
                                } else {
                                    32767
                                }
                            })
                            .collect();
                        let mut actual = vec![0; length];
                        dsp.vvc_inverse_primary(size, kinds, &levels, qp, dependent, &mut actual);
                        for value in actual {
                            for byte in value.to_le_bytes() {
                                digest = (digest ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3);
                            }
                        }
                    }
                }
            }
            assert_eq!(digest, expected, "{size:?} {kinds:?} {:?}", dsp.isa());
        }
    }
}
