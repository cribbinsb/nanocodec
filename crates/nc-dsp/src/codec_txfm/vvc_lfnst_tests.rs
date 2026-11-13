//! LFNST impulse fixtures from pinned `VVdeC` tables and spectral rearrangement.
use super::*;
#[test]
fn lfnst_reference_impulses_cover_wide_angles_transpose_and_zero_out() {
    for (size, expected) in [
        ((4, 4), 0x5504_a004_98cd_8379),
        ((4, 16), 0xa2cf_7522_a844_1da8),
        ((8, 8), 0x4aa8_59c2_b8dc_4642),
        ((8, 16), 0xf586_b8f7_b956_3ac8),
        ((16, 4), 0x430f_5ebf_ab17_ebf0),
        ((16, 8), 0xac89_31e7_2a1a_f090),
        ((32, 32), 0xed89_4852_3c8e_4a20),
        ((64, 64), 0x3c30_0665_387f_8420),
    ] {
        let (width, height) = size;
        let mut digest = 0xcbf2_9ce4_8422_2325_u64;
        for direction in [0, 2, 18, 34, 50, 66] {
            for index in [1, 2] {
                let active = if width == height && [4, 8].contains(&width) {
                    8
                } else {
                    16
                };
                for impulse in 0..active {
                    let mut coefficients = vec![0; width * height];
                    let scan: Vec<_> = (0usize..7)
                        .flat_map(|d| {
                            (d.saturating_sub(3)..=d.min(3)).map(move |x| (d - x) * width + x)
                        })
                        .collect();
                    coefficients[scan[impulse]] = if impulse % 2 == 0 { -32768 } else { 32767 };
                    inverse(
                        size,
                        VvcLfnst {
                            index,
                            direction,
                            prediction_size: size,
                        },
                        &mut coefficients,
                    );
                    for value in coefficients {
                        for byte in value.to_le_bytes() {
                            digest = (digest ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3);
                        }
                    }
                }
            }
        }
        assert_eq!(digest, expected, "{size:?}");
    }
}
