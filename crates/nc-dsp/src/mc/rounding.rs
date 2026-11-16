//! Rounding schemes of the shared prediction passes (`av1_avx2`,
//! `av1_avx512`, `av1_simd128`): AV1's halved taps and roundings, and
//! HEVC/VVC's full taps (`docs/analysis/encode-kernel-sharing.md`
//! section 5). NEON's kernels do not use them yet.

/// Rounding multipliers of the shared passes ([`Rounding`]): `(x + 2) >>
/// 2`, `(x + 8) >> 4`, `(x + 32) >> 6`.
pub(crate) const ROUND_2: i16 = 1 << 13;
pub(crate) const ROUND_4: i16 = 1 << 11;
pub(crate) const ROUND_6: i16 = 1 << 9;
/// The vertical pass of a 2-D AV1 filter at compound precision: `(acc +
/// 32) >> 6` (`InterRound1` 7, less the halving).
pub(crate) const PREP_LAST_ROUND: i32 = 1 << 5;
pub(crate) const PREP_LAST_SHIFT: i32 = 6;

/// A codec's taps and roundings on the shared passes (section 5 of the
/// kernel-sharing record): the passes are the same multiply-adds; what
/// differs is whether the taps are halved and how each pass rounds.
/// Constants, so every ISA's passes round alike. A multiplier
/// `m = 2^(15 - k)` is one rounding multiply-high (`vpmulhrsw`,
/// `q15mulr`), which is `(x + 2^(k - 1)) >> k`.
pub(crate) trait Rounding {
    /// Taps are halved before use (AV1: every tap is even).
    const HALVED: bool;
    /// `put`'s vertical pass of a 2-D filter: `(acc + LAST_ROUND) >>
    /// LAST_SHIFT`.
    const LAST_ROUND: i32;
    const LAST_SHIFT: i32;
    /// The `vpmulhrsw` multiplier rounding the horizontal sums of a 2-D
    /// filter (and of `prep`'s horizontal-only filter); `None`: they pass
    /// unrounded.
    const FIRST: Option<i16>;
    /// The further `vpmulhrsw` multiplier of `put`'s horizontal-only
    /// filter, after `FIRST`.
    const HORIZONTAL_PUT: i16;
}

/// AV1 (module docs): halved taps; `(sum + 2) >> 2` first, `(acc + 512) >>
/// 10` last, `((sum + 2) >> 2 + 8) >> 4` horizontally.
pub(crate) struct Av1Rounding;

impl Rounding for Av1Rounding {
    const HALVED: bool = true;
    const LAST_ROUND: i32 = 1 << 9;
    const LAST_SHIFT: i32 = 10;
    const FIRST: Option<i16> = Some(ROUND_2);
    const HORIZONTAL_PUT: i16 = ROUND_4;
}

/// HEVC and VVC at 8 bits (H.265 §8.5.3.3.3, H.266 §8.5.6.3.2: `shift1`
/// 0, `shift2` 6, `shift3` 6): full taps (every HEVC/VVC tap fits a signed
/// byte; `separable_sums_fit_i16` bounds the sums); the horizontal sums
/// pass unrounded, `(acc + 2048) >> 12` last (`((acc >> 6) + 32) >> 6` by
/// nested floors), `(sum + 32) >> 6` in one direction. Only `put`.
pub(crate) struct SeparableRounding;

impl Rounding for SeparableRounding {
    const HALVED: bool = false;
    const LAST_ROUND: i32 = 1 << 11;
    const LAST_SHIFT: i32 = 12;
    const FIRST: Option<i16> = None;
    const HORIZONTAL_PUT: i16 = ROUND_6;
}
