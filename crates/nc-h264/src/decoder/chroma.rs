//! The chroma format of the macroblock layer: `ChromaArrayType` (H.264
//! §7.4.2.1.1, Table 6-1), with what each format's chroma syntax and
//! sample arrays look like, and the 4:2:2 chroma DC transform (§8.5.11).
//!
//! - 4:0:0, and 4:4:4 with `separate_colour_plane_flag` (each colour plane
//!   decoded as a monochrome picture, §7.4.2.1.1): no chroma syntax.
//! - 4:2:0 and 4:2:2: chroma DC and AC blocks of their own (§7.3.5.3), the
//!   chroma intra modes (§8.3.4), the bilinear chroma interpolation and
//!   chroma deblocking.
//! - 4:4:4: Cb and Cr coded, predicted and filtered like luma
//!   (`residual_luma()`, §7.3.5.3; §8.3.4.5; §8.4.2.2; §8.7).

use nc_common::ChromaFormat;

/// `ChromaArrayType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ChromaArray {
    /// 0: monochrome, or one colour plane of separately coded 4:4:4.
    Mono,
    /// 1: 4:2:0.
    Yuv420,
    /// 2: 4:2:2.
    Yuv422,
    /// 3: 4:4:4 with Cb and Cr coded like luma.
    Yuv444,
}

/// Coefficients of a 4:2:0 and of a 4:2:2 chroma DC block (`4 * NumC8x8`).
pub(super) const CHROMA_DC_420: usize = 4;
pub(super) const CHROMA_DC_422: usize = 8;
/// Most 4x4 chroma blocks of one component (4:2:2: two 8x8 blocks).
pub(super) const MAX_CHROMA_BLOCKS: usize = 8;
/// Side of a 4x4 block.
const SIDE_4: usize = 4;
/// Macroblock side in luma samples.
const MB_SIDE: usize = 16;

/// The raster index (row * 2 + column) in the 4x2 matrix `c` of each 4:2:2
/// chroma DC level in decoding order (eq. 8-330).
const CHROMA_DC_422_RASTER: [usize; CHROMA_DC_422] = [0, 2, 1, 4, 6, 3, 5, 7];

impl ChromaArray {
    /// Whether there is no chroma syntax (4:0:0, separate colour planes).
    pub fn is_mono(self) -> bool {
        self == Self::Mono
    }

    /// Whether the macroblock layer codes chroma DC and AC blocks and a
    /// chroma intra mode (4:2:0 and 4:2:2).
    pub fn has_chroma_blocks(self) -> bool {
        matches!(self, Self::Yuv420 | Self::Yuv422)
    }

    /// Whether Cb and Cr are coded like luma (4:4:4).
    pub fn is_444(self) -> bool {
        self == Self::Yuv444
    }

    /// `(log2 SubWidthC, log2 SubHeightC)` of the chroma planes.
    pub fn subsampling(self) -> (usize, usize) {
        match self {
            Self::Mono | Self::Yuv420 => (1, 1),
            Self::Yuv422 => (1, 0),
            Self::Yuv444 => (0, 0),
        }
    }

    /// `MbWidthC` and `MbHeightC`.
    pub fn mb_size(self) -> (usize, usize) {
        let (sx, sy) = self.subsampling();
        (MB_SIDE >> sx, MB_SIDE >> sy)
    }

    /// The 4x4 chroma blocks of one component (`4 * NumC8x8`; also the
    /// chroma DC coefficients).
    pub fn chroma_blocks(self) -> usize {
        if self == Self::Yuv422 {
            CHROMA_DC_422
        } else {
            CHROMA_DC_420
        }
    }

    /// `I_PCM` chroma samples of one component (`MbWidthC * MbHeightC`).
    pub fn pcm_chroma_samples(self) -> usize {
        if self.is_mono() {
            return 0;
        }
        let (w, h) = self.mb_size();
        w * h
    }
}

/// The picture format of a sequence: its chroma planes (4:0:0 is output
/// as 4:2:0 with mid-grey chroma, as JM writes it; separately coded
/// planes as 4:4:4).
pub(super) fn picture_format(chroma_format_idc: u32) -> ChromaFormat {
    match chroma_format_idc {
        2 => ChromaFormat::Yuv422,
        3 => ChromaFormat::Yuv444,
        _ => ChromaFormat::Yuv420,
    }
}

/// The position (in chroma samples) of 4x4 chroma block `blk`
/// (`chroma4x4BlkIdx`, §6.4.7: two blocks per row).
pub(super) fn chroma_block_position(blk: usize) -> (usize, usize) {
    ((blk % 2) * SIDE_4, (blk / 2) * SIDE_4)
}

/// The 4:2:2 chroma DC levels (decoding order) as the raster 4x2 matrix
/// `c` (eq. 8-330), then `f = A · c · B` with the 4x4 Hadamard rows `A`
/// and the 2x2 `B` (eq. 8-329): one value per 4x4 block, in
/// `chroma4x4BlkIdx` order.
pub(super) fn chroma_dc_422(levels: impl Iterator<Item = (usize, i32)>) -> [i32; CHROMA_DC_422] {
    let mut c = [0i32; CHROMA_DC_422];
    for (scan, level) in levels {
        c[CHROMA_DC_422_RASTER[scan]] = level;
    }
    // Columns: the 4-point transform `A` down each of the two columns.
    let mut columns = [0i32; CHROMA_DC_422];
    for col in 0..2 {
        let v = [c[col], c[2 + col], c[4 + col], c[6 + col]];
        let (s0, s1, d0, d1) = (v[0] + v[1], v[2] + v[3], v[0] - v[1], v[2] - v[3]);
        let out = [s0 + s1, s0 - s1, d0 - d1, d0 + d1];
        for (row, value) in out.into_iter().enumerate() {
            columns[row * 2 + col] = value;
        }
    }
    // Rows: `B` across each row's two values.
    let mut f = [0i32; CHROMA_DC_422];
    for row in 0..4 {
        let (a, b) = (columns[row * 2], columns[row * 2 + 1]);
        f[row * 2] = a + b;
        f[row * 2 + 1] = a - b;
    }
    f
}

/// The 4:2:2 chroma DC levels as `c` itself, for transform bypass
/// (§8.5.11.1: `dcC = c`), in `chroma4x4BlkIdx` order.
pub(super) fn chroma_dc_422_bypass(
    levels: impl Iterator<Item = (usize, i32)>,
) -> [i32; CHROMA_DC_422] {
    let mut c = [0i32; CHROMA_DC_422];
    for (scan, level) in levels {
        c[CHROMA_DC_422_RASTER[scan]] = level;
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `A · c · B` written out as matrix products.
    #[test]
    fn chroma_dc_422_matches_the_matrix_product() {
        let a = [[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, -1, 1], [1, -1, 1, -1]];
        let b = [[1, 1], [1, -1]];
        let levels: [i32; 8] = [3, -1, 4, 1, -5, 9, 2, -6];
        let mut c = [[0i32; 2]; 4];
        for (scan, &level) in levels.iter().enumerate() {
            let raster = CHROMA_DC_422_RASTER[scan];
            c[raster / 2][raster % 2] = level;
        }
        let mut want = [0i32; 8];
        for i in 0..4 {
            for j in 0..2 {
                for k in 0..4 {
                    for l in 0..2 {
                        want[i * 2 + j] += a[i][k] * c[k][l] * b[l][j];
                    }
                }
            }
        }
        let got = chroma_dc_422(levels.iter().copied().enumerate());
        assert_eq!(got, want);
    }

    #[test]
    fn block_geometry() {
        assert_eq!(ChromaArray::Yuv422.mb_size(), (8, 16));
        assert_eq!(ChromaArray::Yuv422.chroma_blocks(), 8);
        assert_eq!(chroma_block_position(7), (4, 12));
        assert_eq!(ChromaArray::Yuv420.pcm_chroma_samples(), 64);
        assert_eq!(ChromaArray::Yuv444.pcm_chroma_samples(), 256);
    }
}
