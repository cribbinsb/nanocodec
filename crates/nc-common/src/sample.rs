//! Sample types: 8-bit samples in `u8`, deeper ones (9 to 16 bits) in `u16`.
//!
//! Every buffer and kernel that touches samples is generic over [`Sample`]
//! and monomorphised twice, the way dav1d builds its `bitdepth` templates
//! from one source: the `u8` instance is the 8-bit code path (encoders and
//! 8-bit streams), the `u16` instance decodes streams of more than 8 bits
//! (owner, 2026-11-15: decoders take high bit depth for conformance; the
//! encoders stay 8-bit). Code chooses the type once per sequence, never
//! per sample.
//!
//! The type only bounds the storage. The stream's actual depth, which
//! clips and shifts depend on, is a runtime [`BitDepth`]: a `u16` plane may
//! hold 9-, 10- or 12-bit samples, and luma and chroma may differ
//! (HEVC `bit_depth_luma_minus8` / `bit_depth_chroma_minus8`). In the `u8`
//! instance it is always [`BitDepth::EIGHT`] and folds away.

use std::fmt::Debug;

/// A stored sample: `u8` (8-bit) or `u16` (up to 16 bits). Sealed: kernels
/// rely on there being exactly these two instances.
pub trait Sample:
    Copy + Default + Debug + Eq + Ord + Send + Sync + Into<i32> + Into<u32> + sealed::Sealed + 'static
{
    /// The deepest samples the type stores.
    const MAX_BIT_DEPTH: u32;
    /// Bytes per sample, in memory and in raw (little-endian) YUV files.
    const BYTES: usize;

    /// `value` stored as a sample. The caller guarantees `0 <= value <=
    /// (1 << MAX_BIT_DEPTH) - 1` (a clipped value); the conversion
    /// truncates otherwise.
    fn from_i32(value: i32) -> Self;

    /// The sample as `i32`.
    #[inline]
    fn to_i32(self) -> i32 {
        self.into()
    }

    /// `samples` as raw little-endian bytes (`yuv420p` for `u8`,
    /// `yuv420p10le` and friends for `u16`): the samples themselves for
    /// `u8` (no copy), else converted into `scratch`.
    fn le_bytes<'a>(samples: &'a [Self], scratch: &'a mut Vec<u8>) -> &'a [u8];

    /// Borrow little-endian bytes when the sample representation needs no
    /// conversion. Wider samples use [`Self::le_bytes`] instead.
    fn borrowed_le_bytes(_samples: &[Self]) -> Option<&[u8]> {
        None
    }
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for u8 {}
    impl Sealed for u16 {}
}

impl Sample for u8 {
    const MAX_BIT_DEPTH: u32 = 8;
    const BYTES: usize = 1;

    #[inline(always)]
    #[allow(
        clippy::inline_always,
        reason = "per-sample conversion inside kernels; must fold into the caller"
    )]
    fn from_i32(value: i32) -> Self {
        value as u8
    }

    fn le_bytes<'a>(samples: &'a [Self], _scratch: &'a mut Vec<u8>) -> &'a [u8] {
        samples
    }

    fn borrowed_le_bytes(samples: &[Self]) -> Option<&[u8]> {
        Some(samples)
    }
}

impl Sample for u16 {
    const MAX_BIT_DEPTH: u32 = 16;
    const BYTES: usize = 2;

    #[inline(always)]
    #[allow(
        clippy::inline_always,
        reason = "per-sample conversion inside kernels; must fold into the caller"
    )]
    fn from_i32(value: i32) -> Self {
        value as u16
    }

    fn le_bytes<'a>(samples: &'a [Self], scratch: &'a mut Vec<u8>) -> &'a [u8] {
        scratch.clear();
        scratch.extend(samples.iter().flat_map(|s| s.to_le_bytes()));
        scratch
    }
}

/// The bit depth of one colour component of a stream: 8 to 16.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BitDepth(u8);

/// Shallowest supported depth.
const MIN_BITS: u32 = 8;
/// Deepest supported depth (the widest `u16` sample).
const MAX_BITS: u32 = 16;

impl BitDepth {
    /// This depth as code instantiated for samples `T` sees it: the
    /// constant [`Self::EIGHT`] for `u8` (so that depth arithmetic folds
    /// away in the 8-bit instance), `self` for `u16`.
    #[must_use]
    #[inline(always)]
    #[allow(
        clippy::inline_always,
        reason = "must fold to a constant in the 8-bit instance"
    )]
    pub fn of<T: Sample>(self) -> Self {
        if T::MAX_BIT_DEPTH == u8::MAX_BIT_DEPTH {
            Self::EIGHT
        } else {
            self
        }
    }

    /// 8 bits: the only depth of the `u8` instance.
    pub const EIGHT: Self = Self(8);
    /// 10 bits (HEVC Main 10, VVC Main 10, AV1 Main high bit depth).
    pub const TEN: Self = Self(10);

    /// `bits` as a depth, or `None` outside 8..=16.
    #[must_use]
    pub const fn new(bits: u32) -> Option<Self> {
        if bits >= MIN_BITS && bits <= MAX_BITS {
            Some(Self(bits as u8))
        } else {
            None
        }
    }

    /// Bits per sample.
    #[must_use]
    #[inline]
    pub const fn bits(self) -> u32 {
        self.0 as u32
    }

    /// The largest sample value, `(1 << bits) - 1` (`bitdepth_max` in
    /// dav1d, the upper bound of `Clip1` in H.264/H.265/H.266).
    #[must_use]
    #[inline]
    pub const fn max(self) -> i32 {
        (1 << self.0) - 1
    }

    /// Bits above 8: the shift that scales 8-bit thresholds and offsets to
    /// this depth (`BitDepth − 8` in the HEVC deblocking `β`/`tC` and
    /// weighted-prediction offsets).
    #[must_use]
    #[inline]
    pub const fn above_eight(self) -> u32 {
        self.0 as u32 - MIN_BITS
    }

    /// The mid-grey value `1 << (bits − 1)` (the intra reference when no
    /// neighbour is available).
    #[must_use]
    #[inline]
    pub const fn mid(self) -> i32 {
        1 << (self.0 - 1)
    }

    /// `Clip1`: `value` clamped to `0..=max`.
    #[must_use]
    #[inline(always)]
    #[allow(
        clippy::inline_always,
        reason = "per-sample clip inside kernels; must fold into the caller"
    )]
    pub fn clip(self, value: i32) -> i32 {
        // `max(0)` then `min(max)`, not `clamp`: the same value (`max` is
        // positive), but vectorised loops compile it to saturating packs at
        // 8 bits (two instructions for eight samples on x86), while `clamp`
        // stays a compare-and-select sequence even at a constant depth
        // (rustc 1.99; -14.5M instructions in the AV1 warp store alone on a
        // libaom 720p stream, 2026-11-19).
        value.max(0).min(self.max())
    }

    /// `Clip1(value)` stored as a sample of type `T`.
    ///
    /// # Panics
    /// In debug builds, if this depth is deeper than `T` stores.
    #[must_use]
    #[inline(always)]
    #[allow(
        clippy::inline_always,
        reason = "per-sample clip inside kernels; must fold into the caller"
    )]
    pub fn clip_to<T: Sample>(self, value: i32) -> T {
        debug_assert!(
            self.bits() <= T::MAX_BIT_DEPTH,
            "{self:?} in {}-bit samples",
            T::MAX_BIT_DEPTH
        );
        T::from_i32(self.clip(value))
    }

    /// Whether samples of this depth fit `T`.
    #[must_use]
    pub const fn fits<T: Sample>(self) -> bool {
        self.bits() <= T::MAX_BIT_DEPTH
    }
}

/// The bit depths of a picture's luma and chroma (equal except in streams
/// like HEVC's `TSUNEQBD`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BitDepths {
    /// Luma (`BitDepthY`).
    pub luma: BitDepth,
    /// Both chroma planes (`BitDepthC`).
    pub chroma: BitDepth,
}

impl BitDepths {
    /// 8-bit luma and chroma.
    pub const EIGHT: Self = Self {
        luma: BitDepth::EIGHT,
        chroma: BitDepth::EIGHT,
    };

    /// The depth of plane `index` (0 luma, 1 and 2 chroma).
    #[must_use]
    #[inline]
    pub fn plane(self, index: usize) -> BitDepth {
        if index == 0 { self.luma } else { self.chroma }
    }

    /// Whether every plane is 8-bit, so the `u8` instance applies.
    #[must_use]
    pub fn is_eight(self) -> bool {
        self == Self::EIGHT
    }

    /// These depths as code instantiated for samples `T` sees them: the
    /// constant [`Self::EIGHT`] for `u8` (so that depth arithmetic folds
    /// away in the 8-bit instance), `self` for `u16`.
    #[must_use]
    #[inline(always)]
    #[allow(
        clippy::inline_always,
        reason = "must fold to a constant in the 8-bit instance"
    )]
    pub fn of<T: Sample>(self) -> Self {
        if T::MAX_BIT_DEPTH == u8::MAX_BIT_DEPTH {
            Self::EIGHT
        } else {
            self
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_limits_and_clip() {
        assert!(BitDepth::new(7).is_none() && BitDepth::new(17).is_none());
        let ten = BitDepth::new(10).unwrap();
        assert_eq!((ten.max(), ten.mid(), ten.above_eight()), (1023, 512, 2));
        assert_eq!(ten.clip(-5), 0);
        assert_eq!(ten.clip(4000), 1023);
        assert_eq!(ten.clip_to::<u16>(700), 700u16);
        assert_eq!(BitDepth::EIGHT.clip_to::<u8>(300), 255u8);
        assert!(ten.fits::<u16>() && !ten.fits::<u8>());
    }

    #[test]
    fn little_endian_rows() {
        let mut scratch = Vec::new();
        assert_eq!(
            u16::le_bytes(&[0x0102, 0x03ff], &mut scratch),
            [0x02, 0x01, 0xff, 0x03]
        );
        assert_eq!(u8::le_bytes(&[1, 2], &mut scratch), [1, 2]);
    }
}
