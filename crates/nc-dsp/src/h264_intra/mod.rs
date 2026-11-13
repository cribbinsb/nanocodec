//! H.264 `Intra_8x8` luma prediction (H.264 §8.3.2.2): the reference sample
//! filter and the nine `Intra8x8PredMode`s.
//!
//! The codec gathers the unfiltered neighbours and their availability into
//! [`H264Intra8x8Edges`]; [`H264Intra8x8Edges::filtered`] applies the
//! filter of §8.3.2.2.1 (scalar, once per block) and lays the filtered
//! samples `p′` out as one border, which every mode then reads:
//!
//! ```text
//! index  0    1 ..= 8         9        10 ..= 25          26 ..= 31
//!        l7   l7, l6 .. l0    corner   t0, t1 .. t15      t15 (repeated)
//! ```
//!
//! where `t[i] = p′[i, −1]` and `l[j] = p′[−1, j]`. [`reference`] is the
//! scalar oracle, a transcription of the per-sample formulas of
//! §8.3.2.2.2–10. [`rows`] computes the same predictions as the diagonal
//! modes really are: 2- and 3-tap averages along that border, of which
//! every output row is a contiguous slice at a fixed offset; its loops are
//! over byte arrays, so each ISA's instantiation vectorises them.

mod reference;
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
mod rows;

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod avx2;
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod neon;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(unsafe_code)]
mod simd128;

use crate::Isa;

/// Block side.
pub const H264_INTRA_8X8_SIDE: usize = 8;

/// `Intra8x8PredMode` values (H.264 Table 8-3).
pub mod h264_intra_8x8_mode {
    /// The row above, repeated.
    pub const VERTICAL: u8 = 0;
    /// The left column, repeated.
    pub const HORIZONTAL: u8 = 1;
    /// The mean of the available neighbours.
    pub const DC: u8 = 2;
    /// 45° down-left from the row above.
    pub const DIAGONAL_DOWN_LEFT: u8 = 3;
    /// 45° down-right from the corner.
    pub const DIAGONAL_DOWN_RIGHT: u8 = 4;
    /// Steep down-right.
    pub const VERTICAL_RIGHT: u8 = 5;
    /// Shallow down-right.
    pub const HORIZONTAL_DOWN: u8 = 6;
    /// Steep down-left.
    pub const VERTICAL_LEFT: u8 = 7;
    /// Up-right from the left column.
    pub const HORIZONTAL_UP: u8 = 8;
    /// Number of modes.
    pub const COUNT: u8 = 9;
}
use h264_intra_8x8_mode as mode;

/// Border length (module docs).
pub(crate) const BORDER: usize = 32;
/// Border index of `l[0]`, the corner and `t[0]`.
pub(crate) const LEFT0: usize = 8;
pub(crate) const CORNER: usize = 9;
pub(crate) const TOP0: usize = 10;
/// Number of top samples (the row above and the row above-right).
const TOP_LEN: usize = 2 * H264_INTRA_8X8_SIDE;
/// Prediction value without neighbours (`1 << (BitDepth − 1)`).
const NO_NEIGHBOUR: u8 = 128;

/// The unfiltered neighbours of an 8x8 block, `p[x, y]` with the block's
/// top-left sample at (0, 0), and which of them exist (decoded already and
/// inside the picture, §6.4.11.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "one availability per neighbouring sample group"
)]
pub struct H264Intra8x8Edges {
    /// `p[x, −1]` for `x` in 0..16; `8..16` only read if `has_top_right`.
    pub top: [u8; TOP_LEN],
    /// `p[−1, y]` for `y` in 0..8.
    pub left: [u8; H264_INTRA_8X8_SIDE],
    /// `p[−1, −1]`.
    pub corner: u8,
    /// The row above exists.
    pub has_top: bool,
    /// The row above-right (`p[8..16, −1]`) exists.
    pub has_top_right: bool,
    /// The left column exists.
    pub has_left: bool,
    /// The above-left sample exists.
    pub has_corner: bool,
}

/// The filtered reference samples `p′` of an 8x8 block in border order
/// (module docs), with the availability the modes depend on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "one availability per neighbouring sample group"
)]
pub struct H264Intra8x8Refs {
    pub(crate) border: [u8; BORDER],
    pub(crate) has_top: bool,
    pub(crate) has_left: bool,
    pub(crate) has_corner: bool,
}

/// The rounded 2-tap average.
#[inline]
pub(crate) fn avg2(a: u8, b: u8) -> u8 {
    ((u16::from(a) + u16::from(b) + 1) >> 1) as u8
}

/// The rounded [1, 2, 1] average.
#[inline]
pub(crate) fn avg3(a: u8, b: u8, c: u8) -> u8 {
    ((u16::from(a) + 2 * u16::from(b) + u16::from(c) + 2) >> 2) as u8
}

impl H264Intra8x8Edges {
    /// The reference sample filter (§8.3.2.2.1). A missing top-right half
    /// repeats `p[7, −1]` first (§8.3.2.2); missing groups stay zero and
    /// only modes that do not read them may be predicted.
    #[must_use]
    pub fn filtered(&self) -> H264Intra8x8Refs {
        let mut border = [0u8; BORDER];
        let n = H264_INTRA_8X8_SIDE;
        if self.has_top {
            let mut top = self.top;
            if !self.has_top_right {
                let last = top[n - 1];
                top[n..].fill(last);
            }
            let first = if self.has_corner { self.corner } else { top[0] };
            let t = &mut border[TOP0..TOP0 + TOP_LEN];
            t[0] = avg3(first, top[0], top[1]);
            for i in 1..TOP_LEN - 1 {
                t[i] = avg3(top[i - 1], top[i], top[i + 1]);
            }
            t[TOP_LEN - 1] = avg3(top[TOP_LEN - 2], top[TOP_LEN - 1], top[TOP_LEN - 1]);
            let last = border[TOP0 + TOP_LEN - 1];
            border[TOP0 + TOP_LEN..].fill(last);
        }
        if self.has_corner {
            border[CORNER] = match (self.has_top, self.has_left) {
                (true, true) => avg3(self.top[0], self.corner, self.left[0]),
                (true, false) => avg3(self.corner, self.corner, self.top[0]),
                (false, true) => avg3(self.corner, self.corner, self.left[0]),
                (false, false) => self.corner,
            };
        }
        if self.has_left {
            let l = &self.left;
            let first = if self.has_corner { self.corner } else { l[0] };
            let mut filtered = [0u8; H264_INTRA_8X8_SIDE];
            filtered[0] = avg3(first, l[0], l[1]);
            for j in 1..n - 1 {
                filtered[j] = avg3(l[j - 1], l[j], l[j + 1]);
            }
            filtered[n - 1] = avg3(l[n - 2], l[n - 1], l[n - 1]);
            for (j, &value) in filtered.iter().enumerate() {
                border[LEFT0 - j] = value;
            }
            border[0] = filtered[n - 1];
        }
        H264Intra8x8Refs {
            border,
            has_top: self.has_top,
            has_left: self.has_left,
            has_corner: self.has_corner,
        }
    }
}

impl H264Intra8x8Refs {
    /// `p′[i, −1]` for `i` in 0..16.
    pub(crate) fn t(&self, i: usize) -> u8 {
        self.border[TOP0 + i]
    }

    /// `p′[−1, j]` for `j` in 0..8.
    pub(crate) fn l(&self, j: usize) -> u8 {
        self.border[LEFT0 - j]
    }

    /// `p′[−1, −1]`.
    pub(crate) fn corner(&self) -> u8 {
        self.border[CORNER]
    }

    /// Whether `mode` has the samples it reads (§8.3.2.2.2–10; DC works
    /// with none).
    #[must_use]
    pub fn allows(&self, m: u8) -> bool {
        match m {
            mode::VERTICAL | mode::DIAGONAL_DOWN_LEFT | mode::VERTICAL_LEFT => self.has_top,
            mode::HORIZONTAL | mode::HORIZONTAL_UP => self.has_left,
            mode::DC => true,
            mode::DIAGONAL_DOWN_RIGHT | mode::VERTICAL_RIGHT | mode::HORIZONTAL_DOWN => {
                self.has_top && self.has_left && self.has_corner
            }
            _ => false,
        }
    }

    /// The DC prediction (§8.3.2.2.4).
    pub(crate) fn dc(&self) -> u8 {
        let n = H264_INTRA_8X8_SIDE;
        let sum = |samples: &[u8]| samples.iter().map(|&s| u32::from(s)).sum::<u32>();
        let top = sum(&self.border[TOP0..TOP0 + n]);
        let left = sum(&self.border[1..=LEFT0]);
        match (self.has_top, self.has_left) {
            (true, true) => ((top + left + 8) >> 4) as u8,
            (true, false) => ((top + 4) >> 3) as u8,
            (false, true) => ((left + 4) >> 3) as u8,
            (false, false) => NO_NEIGHBOUR,
        }
    }
}

/// Predict an 8x8 block in a mode the references allow into `out`
/// (raster, stride 8).
pub(crate) type PredictFn = unsafe fn(&H264Intra8x8Refs, u8, &mut [u8; 64]);

/// The kernel of `isa`: the scalar reference, or the border-and-rows form
/// compiled for the ISA.
pub(crate) fn kernel(isa: Isa) -> PredictFn {
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => avx2::predict,
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => neon::predict,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Isa::Simd128 => simd128::predict,
        _ => reference::predict,
    }
}
