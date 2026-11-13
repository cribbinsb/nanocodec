//! Scaling (dequantisation) of transform coefficient levels with the
//! active scaling matrices (§8.5.6..§8.5.13), and the inverse transforms
//! added to the prediction (nc-dsp kernels).
//!
//! Levels are dequantised as they are parsed, straight into raster-order
//! coefficient blocks of a per-picture [`Residual`] scratch. Only the
//! blocks that received coefficients are transformed, and they are zeroed
//! again afterwards, so no per-macroblock clearing is needed.

use nc_common::PlaneViewMut;
use nc_dsp::Dsp;

use super::scaling::LevelScale;
use super::syntax::Coeffs;
use crate::tables::{ZIGZAG_4X4, ZIGZAG_8X8, chroma_qp};

/// `QP % 6` period of the scaling factors.
const QP_PERIOD: i32 = 6;
/// 4x4 list indices (Table 7-2): Intra Y, Inter Y; Cb, Cr follow each.
const LIST_INTRA_Y: usize = 0;
const LIST_INTER_Y: usize = 3;
/// 8x8 list indices: Intra Y, Inter Y.
const LIST_8X8_INTRA: usize = 0;
const LIST_8X8_INTER: usize = 1;
/// Dequantised coefficients of a conforming 8-bit stream lie in
/// `-2^15..2^15` (§8.5.12.1); hostile input is clamped there so the
/// transforms cannot overflow.
const MIN_COEFF: i64 = -(1 << 15);
const MAX_COEFF: i64 = (1 << 15) - 1;

/// The scaling factors and chroma QP offsets of the active PPS.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Dequantiser {
    scale: LevelScale,
    chroma_qp_offset: [i8; 2],
}

/// Dequantised coefficients of one macroblock and which blocks have any.
#[derive(Debug, Clone)]
pub(super) struct Residual {
    /// Luma 4x4 blocks by raster index (`y4 * 4 + x4`).
    luma: [[i32; 16]; 16],
    luma_coded: u16,
    /// Luma 4x4 blocks with a non-zero coefficient other than the DC.
    luma_ac: u16,
    /// Luma 8x8 blocks.
    luma_8x8: [[i32; 64]; 4],
    luma_8x8_coded: u8,
    /// Chroma 4x4 blocks of Cb and Cr.
    chroma: [[[i32; 16]; 4]; 2],
    chroma_coded: [u8; 2],
}

/// `(level * scale) << (qp / 6)`, shifted down by `shift` with rounding
/// (eqs. 8-336, 8-337 for 4x4 with shift 4; 8-338, 8-339 for 8x8 with
/// shift 6; 8-326 for Intra16x16 DC with shift 6).
fn scale_level(level: i32, scale: i32, qp: i32, shift: i32) -> i32 {
    Scaler::new(qp, shift).apply(level, scale)
}

/// [`scale_level`] for one block: the shift direction and rounding depend
/// only on `QP / 6`, so they are chosen once and each level is one
/// multiply, shift pair and clamp.
#[derive(Clone, Copy)]
struct Scaler {
    left: u32,
    right: u32,
    round: i64,
}

impl Scaler {
    fn new(qp: i32, shift: i32) -> Self {
        let up = qp / QP_PERIOD;
        if up >= shift {
            Self {
                left: (up - shift) as u32,
                right: 0,
                round: 0,
            }
        } else {
            let right = (shift - up) as u32;
            Self {
                left: 0,
                right,
                round: 1 << (right - 1),
            }
        }
    }

    #[inline]
    fn apply(self, level: i32, scale: i32) -> i32 {
        let product = i64::from(level) * i64::from(scale);
        (((product << self.left) + self.round) >> self.right).clamp(MIN_COEFF, MAX_COEFF) as i32
    }
}

impl Dequantiser {
    pub fn new(scale: LevelScale, chroma_qp_offset: [i8; 2]) -> Self {
        Self {
            scale,
            chroma_qp_offset,
        }
    }

    /// `QP'c` of component `comp` (§8.5.8, Table 8-15).
    fn chroma_qp(&self, qp: i32, comp: usize) -> i32 {
        i32::from(chroma_qp(qp as u8, self.chroma_qp_offset[comp]))
    }

    /// `Intra16x16DCLevel`: inverse Hadamard (eq. 8-320) and DC scaling
    /// (eq. 8-326) into the DC of every luma 4x4 block.
    pub fn luma_dc(&self, levels: &Coeffs, qp: i32, out: &mut Residual) {
        if levels.count == 0 {
            return;
        }
        let mut c = [0i32; 16];
        for (scan, level) in levels.iter() {
            c[ZIGZAG_4X4[scan]] = level;
        }
        let f = hadamard_4x4(&c);
        let scale = self.scale.scale_4x4[LIST_INTRA_Y][(qp % QP_PERIOD) as usize][0];
        for (pos, &value) in f.iter().enumerate() {
            // dcY_ij belongs to the 4x4 block in row i, column j.
            let dc = scale_level(value, scale, qp, 6);
            out.luma[pos][0] = dc;
            if dc != 0 {
                out.luma_coded |= 1 << pos;
            }
        }
    }

    /// A luma 4x4 block (`ac`: Intra16x16 AC, scan from position 1) at
    /// raster index `block`.
    pub fn luma_4x4(
        &self,
        levels: &Coeffs,
        ac: bool,
        intra: bool,
        qp: i32,
        block: usize,
        out: &mut Residual,
    ) {
        if levels.count == 0 {
            return;
        }
        let list = if intra { LIST_INTRA_Y } else { LIST_INTER_Y };
        let scale = &self.scale.scale_4x4[list][(qp % QP_PERIOD) as usize];
        let first = usize::from(ac);
        let coeffs = &mut out.luma[block];
        let scaler = Scaler::new(qp, 4);
        for (scan, level) in levels.iter() {
            let pos = ZIGZAG_4X4[scan + first];
            coeffs[pos] = scaler.apply(level, scale[pos]);
        }
        out.luma_coded |= 1 << block;
        // Only a lone level at scan position 0 leaves the AC zero.
        let dc_only = levels.count == 1 && levels.scan[0] as usize + first == 0;
        out.luma_ac |= u16::from(!dc_only) << block;
    }

    /// Luma 8x8 block `b8`.
    pub fn luma_8x8(&self, levels: &Coeffs, intra: bool, qp: i32, b8: usize, out: &mut Residual) {
        if levels.count == 0 {
            return;
        }
        let list = if intra {
            LIST_8X8_INTRA
        } else {
            LIST_8X8_INTER
        };
        let scale = &self.scale.scale_8x8[list][(qp % QP_PERIOD) as usize];
        let coeffs = &mut out.luma_8x8[b8];
        let scaler = Scaler::new(qp, 6);
        for (scan, level) in levels.iter() {
            let pos = ZIGZAG_8X8[scan];
            coeffs[pos] = scaler.apply(level, scale[pos]);
        }
        out.luma_8x8_coded |= 1 << b8;
    }

    /// 4:2:0 `ChromaDCLevel` of `comp`: 2x2 inverse transform (eq. 8-328)
    /// and scaling (eq. 8-330) into the DC of its four 4x4 blocks.
    pub fn chroma_dc(
        &self,
        levels: &Coeffs,
        intra: bool,
        qp: i32,
        comp: usize,
        out: &mut Residual,
    ) {
        if levels.count == 0 {
            return;
        }
        let mut c = [0i32; 4];
        for (scan, level) in levels.iter() {
            c[scan] = level;
        }
        let f = [
            c[0] + c[1] + c[2] + c[3],
            c[0] - c[1] + c[2] - c[3],
            c[0] + c[1] - c[2] - c[3],
            c[0] - c[1] - c[2] + c[3],
        ];
        let qpc = self.chroma_qp(qp, comp);
        let list = if intra { LIST_INTRA_Y } else { LIST_INTER_Y } + 1 + comp;
        let scale = i64::from(self.scale.scale_4x4[list][(qpc % QP_PERIOD) as usize][0]);
        for (b, &value) in f.iter().enumerate() {
            let dc = ((i64::from(value) * scale) << (qpc / QP_PERIOD)) >> 5;
            let dc = dc.clamp(MIN_COEFF, MAX_COEFF) as i32;
            out.chroma[comp][b][0] = dc;
            if dc != 0 {
                out.chroma_coded[comp] |= 1 << b;
            }
        }
    }

    /// `ChromaACLevel` of block `b` of `comp`.
    pub fn chroma_ac(
        &self,
        levels: &Coeffs,
        intra: bool,
        qp: i32,
        comp: usize,
        b: usize,
        out: &mut Residual,
    ) {
        if levels.count == 0 {
            return;
        }
        let qpc = self.chroma_qp(qp, comp);
        let list = if intra { LIST_INTRA_Y } else { LIST_INTER_Y } + 1 + comp;
        let scale = &self.scale.scale_4x4[list][(qpc % QP_PERIOD) as usize];
        let coeffs = &mut out.chroma[comp][b];
        let scaler = Scaler::new(qpc, 4);
        for (scan, level) in levels.iter() {
            let pos = ZIGZAG_4X4[scan + 1];
            coeffs[pos] = scaler.apply(level, scale[pos]);
        }
        out.chroma_coded[comp] |= 1 << b;
    }
}

/// The 4x4 Hadamard transform of eq. 8-320, `H · c · H` with rows of `H`
/// (1, 1, 1, 1), (1, 1, -1, -1), (1, -1, -1, 1), (1, -1, 1, -1).
fn hadamard_4x4(c: &[i32; 16]) -> [i32; 16] {
    let butterfly = |v: [i32; 4]| {
        let (s0, s1, d0, d1) = (v[0] + v[1], v[2] + v[3], v[0] - v[1], v[2] - v[3]);
        [s0 + s1, s0 - s1, d0 - d1, d0 + d1]
    };
    let mut rows = [0i32; 16];
    for i in 0..4 {
        let row = butterfly([c[i * 4], c[i * 4 + 1], c[i * 4 + 2], c[i * 4 + 3]]);
        rows[i * 4..i * 4 + 4].copy_from_slice(&row);
    }
    let mut out = [0i32; 16];
    for j in 0..4 {
        let col = butterfly([rows[j], rows[4 + j], rows[8 + j], rows[12 + j]]);
        for (i, value) in col.into_iter().enumerate() {
            out[i * 4 + j] = value;
        }
    }
    out
}

impl Residual {
    pub fn new() -> Self {
        Self {
            luma: [[0; 16]; 16],
            luma_coded: 0,
            luma_ac: 0,
            luma_8x8: [[0; 64]; 4],
            luma_8x8_coded: 0,
            chroma: [[[0; 16]; 4]; 2],
            chroma_coded: [0; 2],
        }
    }

    /// Luma 4x4 blocks with coefficients, for the deblocking filter (bit
    /// `y4 * 4 + x4`; every block of a coded 8x8 transform block).
    pub fn luma_nonzero(&self) -> u16 {
        let mut mask = self.luma_coded;
        for b8 in 0..4 {
            if (self.luma_8x8_coded >> b8) & 1 != 0 {
                let (x, y) = ((b8 % 2) * 2, (b8 / 2) * 2);
                mask |= 0b11_0011 << (y * 4 + x);
            }
        }
        mask
    }

    /// Whether the luma 4x4 block at raster index `block` has coefficients.
    pub fn has_luma_4x4(&self, block: usize) -> bool {
        (self.luma_coded >> block) & 1 != 0
    }

    pub fn has_luma_8x8(&self, b8: usize) -> bool {
        (self.luma_8x8_coded >> b8) & 1 != 0
    }

    /// Add luma 4x4 block `block` to `dst` (its 4x4 samples) and clear it.
    pub fn add_luma_4x4(&mut self, dsp: &Dsp, block: usize, dst: &mut PlaneViewMut<'_>) {
        if !self.has_luma_4x4(block) {
            return;
        }
        add_4x4(dsp, &mut self.luma[block], dst);
        self.luma_coded &= !(1 << block);
        self.luma_ac &= !(1 << block);
    }

    /// Add luma 8x8 block `b8` to `dst` (its 8x8 samples) and clear it.
    pub fn add_luma_8x8(&mut self, dsp: &Dsp, b8: usize, dst: &mut PlaneViewMut<'_>) {
        if !self.has_luma_8x8(b8) {
            return;
        }
        dsp.h264_add_scaled_8x8(&self.luma_8x8[b8], dst);
        self.luma_8x8[b8] = [0; 64];
        self.luma_8x8_coded &= !(1 << b8);
    }

    /// Add every coded luma 4x4 block to `dst` (the macroblock's 16x16
    /// samples) in one call, and clear them.
    pub fn add_luma_4x4_all(&mut self, dsp: &Dsp, dst: &mut PlaneViewMut<'_>) {
        if self.luma_coded == 0 {
            return;
        }
        dsp.h264_add_scaled_mb(&self.luma, self.luma_coded, self.luma_ac, dst);
        for block in BitIter(self.luma_coded) {
            self.luma[block] = [0; 16];
        }
        self.luma_coded = 0;
        self.luma_ac = 0;
    }

    /// Whether any luma block has coefficients.
    pub fn has_luma(&self) -> bool {
        self.luma_coded != 0 || self.luma_8x8_coded != 0
    }

    /// Whether any chroma block of either component has coefficients.
    pub fn has_chroma_any(&self) -> bool {
        self.chroma_coded != [0; 2]
    }

    /// Whether chroma block `b` of `comp` has coefficients.
    pub fn has_chroma(&self, comp: usize, b: usize) -> bool {
        (self.chroma_coded[comp] >> b) & 1 != 0
    }

    /// Add chroma block `b` of `comp` to `dst` and clear it.
    pub fn add_chroma_4x4(&mut self, dsp: &Dsp, comp: usize, b: usize, dst: &mut PlaneViewMut<'_>) {
        if (self.chroma_coded[comp] >> b) & 1 == 0 {
            return;
        }
        add_4x4(dsp, &mut self.chroma[comp][b], dst);
        self.chroma_coded[comp] &= !(1 << b);
    }
}

/// Inverse transform `coeffs` (then zeroed) and add it to `dst`.
fn add_4x4(dsp: &Dsp, coeffs: &mut [i32; 16], dst: &mut PlaneViewMut<'_>) {
    dsp.h264_add_scaled_4x4(core::slice::from_ref(coeffs), 1, dst);
    *coeffs = [0; 16];
}

/// The indices of the set bits of a block mask, lowest first.
struct BitIter(u16);

impl Iterator for BitIter {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        if self.0 == 0 {
            return None;
        }
        let index = self.0.trailing_zeros() as usize;
        self.0 &= self.0 - 1;
        Some(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::scaling::ScalingMatrix;

    #[test]
    fn flat_scaling_matches_the_classic_formula() {
        // LevelScale(0, 0, 0) = 16 * 10: level 3 at QP 28 -> (3*160) << 0.
        assert_eq!(scale_level(3, 160, 28, 4), 480);
        // QP 10: (3*160 + 2) >> 3 = 60.
        assert_eq!(scale_level(3, 160, 10, 4), 60);
    }

    #[test]
    fn hadamard_matches_the_matrix_product() {
        let c: [i32; 16] = core::array::from_fn(|i| (i as i32 * 7) % 5 - 2);
        let h = [[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, -1, 1], [1, -1, 1, -1]];
        let mut want = [0i32; 16];
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    for l in 0..4 {
                        want[i * 4 + j] += h[i][k] * c[k * 4 + l] * h[l][j];
                    }
                }
            }
        }
        assert_eq!(hadamard_4x4(&c), want);
    }

    #[test]
    fn hadamard_of_a_dc_spreads_evenly() {
        let mut c = [0; 16];
        c[0] = 2;
        assert_eq!(hadamard_4x4(&c), [2; 16]);
        let dq = Dequantiser::new(LevelScale::new(&ScalingMatrix::FLAT), [0, 0]);
        let mut levels = Coeffs::new();
        levels.push(0, 2);
        let mut residual = Residual::new();
        dq.luma_dc(&levels, 28, &mut residual);
        assert_eq!(residual.luma_coded, u16::MAX);
    }
}
