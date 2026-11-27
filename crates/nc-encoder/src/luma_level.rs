//! The level a stream signals, for the codecs whose levels share one table
//! of picture-size and luma-sample-rate limits: HEVC (Table A.8) and VVC
//! (Tables A.1 and A.2; its levels 1 to 6.2 carry HEVC's limits). Each
//! codec writes the level in its own `general_level_idc` form.
//!
//! The level follows the coded picture size and the frame rate, as x265 and
//! `VVenC` choose theirs; a fixed high level makes hardware and browser
//! decoders reserve (or refuse) far more than a stream needs.

use crate::config::FrameRate;

/// A level number, `major.minor` (level 4.1 is `{ major: 4, minor: 1 }`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LumaLevel {
    /// The level's major number.
    pub major: u32,
    /// The level's minor number.
    pub minor: u32,
}

impl LumaLevel {
    /// HEVC `general_level_idc`: 30 times the level number (§A.4.1).
    #[must_use]
    pub const fn hevc_idc(self) -> u32 {
        30 * self.major + 3 * self.minor
    }

    /// VVC `general_level_idc`: 16 times the major number plus 3 times the
    /// minor (§A.4.1).
    #[must_use]
    pub const fn vvc_idc(self) -> u32 {
        16 * self.major + 3 * self.minor
    }
}

/// One level's limits: `MaxLumaPs` (samples per picture) and `MaxLumaSr`
/// (luma samples per second).
struct Limits {
    level: LumaLevel,
    max_luma_ps: u64,
    max_luma_sr: u64,
}

const fn limits(major: u32, minor: u32, max_luma_ps: u64, max_luma_sr: u64) -> Limits {
    Limits {
        level: LumaLevel { major, minor },
        max_luma_ps,
        max_luma_sr,
    }
}

/// HEVC Table A.8 / VVC Tables A.1-A.2, lowest level first.
const LEVELS: [Limits; 13] = [
    limits(1, 0, 36_864, 552_960),
    limits(2, 0, 122_880, 3_686_400),
    limits(2, 1, 245_760, 7_372_800),
    limits(3, 0, 552_960, 16_588_800),
    limits(3, 1, 983_040, 33_177_600),
    limits(4, 0, 2_228_224, 66_846_720),
    limits(4, 1, 2_228_224, 133_693_440),
    limits(5, 0, 8_912_896, 267_386_880),
    limits(5, 1, 8_912_896, 534_773_760),
    limits(5, 2, 8_912_896, 1_069_547_520),
    limits(6, 0, 35_651_584, 1_069_547_520),
    limits(6, 1, 35_651_584, 2_139_095_040),
    limits(6, 2, 35_651_584, 4_278_190_080),
];

/// The lowest level whose picture-size and sample-rate limits hold for
/// `width` x `height` coded luma samples at `frame_rate`, with each
/// dimension at most `sqrt(8 * MaxLumaPs)` (HEVC §A.4.1, VVC §A.4.1); the
/// highest level if none does.
#[must_use]
pub fn luma_level(width: usize, height: usize, frame_rate: FrameRate) -> LumaLevel {
    let picture = (width * height) as u64;
    let sample_rate =
        (picture * u64::from(frame_rate.num)).div_ceil(u64::from(frame_rate.den.max(1)));
    let longest = width.max(height) as u64;
    LEVELS
        .iter()
        .find(|l| {
            picture <= l.max_luma_ps
                && sample_rate <= l.max_luma_sr
                && longest * longest <= 8 * l.max_luma_ps
        })
        .map_or(LEVELS[LEVELS.len() - 1].level, |l| l.level)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(width: usize, height: usize, num: u32, den: u32) -> (u32, u32) {
        let level = luma_level(width, height, FrameRate { num, den });
        (level.hevc_idc(), level.vvc_idc())
    }

    #[test]
    fn levels_of_common_sizes() {
        assert_eq!(at(320, 184, 30, 1), (60, 32)); // 2
        assert_eq!(at(640, 360, 30, 1), (63, 35)); // 2.1
        assert_eq!(at(1280, 720, 30, 1), (93, 51)); // 3.1
        assert_eq!(at(1920, 1080, 30, 1), (120, 64)); // 4
        assert_eq!(at(1920, 1080, 60, 1), (123, 67)); // 4.1
        assert_eq!(at(1920, 1080, 30_000, 1001), (120, 64)); // 4
        assert_eq!(at(3840, 2160, 30, 1), (150, 80)); // 5
    }

    #[test]
    fn a_long_thin_picture_needs_the_dimension_limit() {
        // 2048x64 fits level 2's picture size but not its sqrt(8 * MaxLumaPs) side.
        assert_eq!(at(2048, 64, 30, 1), (90, 48)); // 3
    }
}
