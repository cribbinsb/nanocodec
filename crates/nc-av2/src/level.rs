//! The `seq_level_idx` an AV2 stream signals: the lowest Annex A level
//! whose picture size, dimensions, display, decode and header rates and
//! Main-tier bit rate the stream fits, or 31 (no constraints) where a
//! frame could miss the compression-ratio limit.
//!
//! AV2's table is its own (AVM v1.0.0 `av2/common/level.c`
//! `av2_level_defs`, BSD-3-Clause-Clear, `docs/licenses/avm`): its
//! dimension limits are square, unlike AV1's, so the AV1 picker
//! (`nc_av1` `obu.rs`) does not serve; the selection rule and its 31
//! fallbacks follow it.

use nc_encoder::FrameRate;

/// `seq_level_idx` 31: no level constraints.
pub const UNCONSTRAINED: u8 = 31;
/// Smallest picture area signalling a level rather than 31 (tiny test
/// pictures can miss the compression ratio on intra headers; as AV1).
const MIN_CONSTRAINED_AREA: usize = 128 * 128;
/// Lowest fixed `base_q_idx` signalling a level rather than 31 (as AV1).
const MIN_CONSTRAINED_QINDEX: u32 = 32;

/// One Annex A level's limits.
struct Limits {
    idx: u8,
    max_picture_size: u64,
    max_dimension: u64,
    max_display_rate: u64,
    max_decode_rate: u64,
    max_header_rate: u64,
    /// Main-tier `MaxBitrate`, bits per second.
    max_bitrate: u64,
}

const fn level(
    idx: u8,
    (max_picture_size, max_dimension): (u64, u64),
    (max_display_rate, max_decode_rate, max_header_rate): (u64, u64, u64),
    max_bitrate: u64,
) -> Limits {
    Limits {
        idx,
        max_picture_size,
        max_dimension,
        max_display_rate,
        max_decode_rate,
        max_header_rate,
        max_bitrate,
    }
}

/// AVM `av2_level_defs`, lowest level first.
#[rustfmt::skip]
const LEVELS: [Limits; 22] = [
    level(0, (147_456, 640), (4_423_680, 5_529_600, 150), 1_500_000),
    level(1, (278_784, 880), (8_363_520, 10_454_400, 150), 3_000_000),
    level(2, (665_856, 1360), (19_975_680, 24_969_600, 150), 6_000_000),
    level(3, (1_065_024, 1720), (31_950_720, 39_938_400, 150), 10_000_000),
    level(4, (2_359_296, 2560), (70_778_880, 77_856_768, 300), 12_000_000),
    level(5, (2_359_296, 2560), (141_557_760, 155_713_536, 300), 20_000_000),
    level(6, (8_912_896, 4975), (267_386_880, 273_715_200, 300), 30_000_000),
    level(7, (8_912_896, 4975), (534_773_760, 547_430_400, 300), 40_000_000),
    level(8, (8_912_896, 4975), (1_069_547_520, 1_094_860_800, 300), 60_000_000),
    level(9, (8_912_896, 4975), (1_069_547_520, 1_176_502_272, 300), 60_000_000),
    level(10, (35_651_584, 9951), (1_069_547_520, 1_176_502_272, 300), 60_000_000),
    level(11, (35_651_584, 9951), (2_139_095_040, 2_189_721_600, 300), 100_000_000),
    level(12, (35_651_584, 9951), (4_278_190_080, 4_379_443_200, 300), 160_000_000),
    level(13, (35_651_584, 9951), (4_278_190_080, 4_706_009_088, 300), 160_000_000),
    level(14, (142_606_336, 19902), (4_278_190_080, 4_706_009_088, 960), 160_000_000),
    level(15, (142_606_336, 19902), (8_556_380_160, 8_758_886_400, 960), 200_000_000),
    level(16, (142_606_336, 19902), (17_112_760_320, 17_517_772_800, 960), 320_000_000),
    level(17, (142_606_336, 19902), (17_112_760_320, 18_824_036_352, 960), 320_000_000),
    level(18, (530_841_600, 38400), (17_112_760_320, 18_824_036_352, 960), 320_000_000),
    level(19, (530_841_600, 38400), (34_225_520_640, 34_910_031_052, 960), 400_000_000),
    level(20, (530_841_600, 38400), (68_451_041_280, 69_820_062_105, 960), 640_000_000),
    level(21, (530_841_600, 38400), (68_451_041_280, 75_296_145_408, 960), 640_000_000),
];

/// What the level must cover.
#[derive(Debug, Clone, Copy)]
pub struct LevelBasis {
    /// Coded picture size.
    pub size: (usize, usize),
    /// Pictures per second.
    pub frame_rate: FrameRate,
    /// The fixed `base_q_idx`, `None` under rate control.
    pub fixed_qindex: Option<u32>,
    /// The peak rate under rate control.
    pub max_bitrate_bps: Option<u64>,
    /// Random-decision test streams signal no level.
    pub random_decisions: bool,
}

/// The `seq_level_idx` of a stream on `basis`.
#[must_use]
pub fn seq_level_idx(basis: LevelBasis) -> u8 {
    let (width, height) = basis.size;
    if basis.random_decisions
        || width * height < MIN_CONSTRAINED_AREA
        || basis
            .fixed_qindex
            .is_some_and(|q| q < MIN_CONSTRAINED_QINDEX)
    {
        return UNCONSTRAINED;
    }
    let picture = (width * height) as u64;
    let num = u64::from(basis.frame_rate.num);
    let den = u64::from(basis.frame_rate.den.max(1));
    let sample_rate = (picture * num).div_ceil(den);
    let headers = num.div_ceil(den);
    LEVELS
        .iter()
        .find(|l| {
            picture <= l.max_picture_size
                && width.max(height) as u64 <= l.max_dimension
                && sample_rate <= l.max_display_rate
                && sample_rate <= l.max_decode_rate
                && headers <= l.max_header_rate
                && basis.max_bitrate_bps.is_none_or(|b| b <= l.max_bitrate)
        })
        .map_or(UNCONSTRAINED, |l| l.idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(size: (usize, usize), max_bitrate_bps: Option<u64>) -> u8 {
        seq_level_idx(LevelBasis {
            size,
            frame_rate: FrameRate { num: 30, den: 1 },
            fixed_qindex: None,
            max_bitrate_bps,
            random_decisions: false,
        })
    }

    #[test]
    fn levels_of_common_sizes_and_rates() {
        assert_eq!(at((320, 184), None), 0); // 2.0
        assert_eq!(at((1280, 720), None), 3); // 3.1
        assert_eq!(at((1920, 1080), None), 4); // 4.0
        assert_eq!(at((320, 184), Some(2_000_000)), 1); // 2.1 for the rate
        assert_eq!(at((64, 64), None), UNCONSTRAINED);
    }
}
