//! Scaling (dequantisation) of transform coefficient levels with the
//! active scaling matrices (§8.5.6..§8.5.13), and the inverse transforms
//! added to the prediction (nc-dsp kernels).
//!
//! Levels are dequantised as they are parsed, straight into raster-order
//! coefficient blocks of a per-picture [`Residual`] scratch. Only the
//! blocks that received coefficients are transformed, and they are zeroed
//! again afterwards, so no per-macroblock clearing is needed.

use nc_common::{BitDepth, PlaneViewMut};
use nc_dsp::Dsp;

use super::chroma::{self, CHROMA_DC_420, ChromaArray, MAX_CHROMA_BLOCKS};
use super::sample::{Coefficient, H264Sample};
use super::scaling::LevelScale;
use super::syntax::Coeffs;
use crate::tables::{FIELD_SCAN_4X4, FIELD_SCAN_8X8, ZIGZAG_4X4, ZIGZAG_8X8, chroma_qp};

/// The inverse scans of a frame (zig-zag) or field macroblock (§8.5.6,
/// §8.5.7): scan position to raster index.
#[derive(Debug)]
pub(super) struct Scan {
    four: [usize; 16],
    eight: [usize; 64],
}

impl Scan {
    const FRAME: Self = Self {
        four: ZIGZAG_4X4,
        eight: ZIGZAG_8X8,
    };
    const FIELD: Self = Self {
        four: FIELD_SCAN_4X4,
        eight: FIELD_SCAN_8X8,
    };

    /// The scans with the scans `scan`.
    pub fn of(field: bool) -> &'static Self {
        if field { &Self::FIELD } else { &Self::FRAME }
    }

    /// The raster position of each 4x4 scan index.
    pub fn four(&self) -> &[usize; 16] {
        &self.four
    }
}

/// `QP % 6` period of the scaling factors.
const QP_PERIOD: i32 = 6;
/// 4x4 list indices (Table 7-2): Intra Y, Cb, Cr, then Inter Y, Cb, Cr.
const LIST_INTRA_Y: usize = 0;
const LIST_INTER_Y: usize = 3;
/// 8x8 lists per component (Intra, Inter; Y, Cb, Cr).
const LISTS_8X8_PER_PLANE: usize = 2;
/// `QP′c,DC` of 4:2:2 chroma DC is `QP′c + 3` (eq. 8-330), and scales up
/// from `QP′c,DC / 6` 6 (eqs. 8-331, 8-332).
const CHROMA_DC_422_QP_ADD: i32 = 3;
const CHROMA_DC_422_SHIFT: i32 = 6;
/// The scaling factors, chroma QP offsets and `QpBdOffset`s of the active
/// parameter sets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Dequantiser {
    scale: LevelScale,
    chroma_qp_offset: [i8; 2],
    /// `QpBdOffsetY` and `QpBdOffsetC` (eqs. 7-4, 7-6): `QP′ = QP +
    /// QpBdOffset` indexes the scaling.
    qp_bd_offset: [i32; 2],
    /// The chroma format: the shape of the chroma DC blocks.
    chroma: ChromaArray,
    /// `colour_plane_id` of separately coded planes, else 0: a plane coded
    /// as monochrome scales with its own component's lists and the luma
    /// QP (`iYCbCr`, §8.5.9).
    colour_plane: usize,
}

/// The 4x4 and 8x8 luma-like blocks of one component: luma, or Cb or Cr
/// where they are coded like luma (4:4:4).
#[derive(Debug, Clone)]
pub(super) struct LumaBlocks<C> {
    /// 4x4 blocks by raster index (`y4 * 4 + x4`).
    blocks: [[C; 16]; 16],
    coded: u16,
    /// 4x4 blocks with a non-zero coefficient other than the DC.
    ac: u16,
    /// 8x8 blocks.
    blocks_8x8: [[C; 64]; 4],
    coded_8x8: u8,
}

/// Dequantised coefficients of one macroblock and which blocks have any,
/// in `C`: 16 bits for 8-bit streams (a conforming one stays within
/// `-2^15..2^15`, §8.5.12.1; hostile input is clamped there), 32 at higher
/// depths. Under transform bypass they are the levels themselves.
#[derive(Debug, Clone)]
pub(super) struct Residual<C> {
    /// Luma, then Cb and Cr where they are coded like luma (4:4:4).
    planes: [LumaBlocks<C>; 3],
    /// Chroma 4x4 blocks of Cb and Cr (4:2:0, 4:2:2), by `chroma4x4BlkIdx`.
    chroma: [[[C; 16]; MAX_CHROMA_BLOCKS]; 2],
    chroma_coded: [u8; 2],
    /// The macroblock is coded with transform bypass (`TransformBypassModeFlag`,
    /// lossless; §8.5.15).
    bypass: bool,
}

/// `(level * scale) << (qp / 6)`, shifted down by `shift` with rounding
/// (eqs. 8-336, 8-337 for 4x4 with shift 4; 8-338, 8-339 for 8x8 with
/// shift 6; 8-326 for Intra16x16 DC with shift 6).
#[cfg(test)]
fn scale_level(level: i32, scale: i32, qp: i32, shift: i32) -> i16 {
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
    fn apply<C: Coefficient>(self, level: i32, scale: i32) -> C {
        let product = i64::from(level) * i64::from(scale);
        C::clamped(((product << self.left) + self.round) >> self.right)
    }
}

/// `QPc` of a macroblock with `QPY` `qp` (§8.5.8, Table 8-15): `qPI` clipped
/// to `−QpBdOffsetC..=51` (`bd_offset_c`), mapped by the table from 30 on.
pub(super) fn chroma_qp_at(qp: i32, offset: i8, bd_offset_c: i32) -> i32 {
    let qpi = (qp + i32::from(offset)).clamp(-bd_offset_c, MAX_QP);
    if qpi < 0 {
        qpi
    } else {
        i32::from(chroma_qp(qpi as u8, 0))
    }
}

/// Largest `QPY` and `qPI`.
const MAX_QP: i32 = 51;

impl Dequantiser {
    /// The dequantiser of `scale` at `depths` (luma, chroma), for chroma
    /// format `chroma` and colour plane `colour_plane` (0 unless the planes
    /// are coded separately).
    pub fn new(
        scale: LevelScale,
        chroma_qp_offset: [i8; 2],
        depths: [BitDepth; 2],
        (chroma, colour_plane): (ChromaArray, usize),
    ) -> Self {
        Self {
            scale,
            chroma_qp_offset,
            qp_bd_offset: depths.map(|d| 6 * d.above_eight() as i32),
            chroma,
            colour_plane,
        }
    }

    /// `QP′` of component `plane` of a macroblock with `QPY` `qp`: `QP′Y`
    /// for luma, `QP′c` for Cb and Cr coded like luma (4:4:4).
    fn plane_qp(&self, qp: i32, plane: usize) -> i32 {
        if plane == 0 {
            qp + self.qp_bd_offset[0]
        } else {
            self.chroma_qp(qp, plane - 1)
        }
    }

    /// `QP′c` of component `comp` (§8.5.8, Table 8-15).
    fn chroma_qp(&self, qp: i32, comp: usize) -> i32 {
        let bd_offset = self.qp_bd_offset[1];
        chroma_qp_at(qp, self.chroma_qp_offset[comp], bd_offset) + bd_offset
    }

    /// The 4x4 list of component `plane` (`iYCbCr`: the colour plane of a
    /// separately coded one).
    fn list_4x4(&self, plane: usize, intra: bool) -> usize {
        let first = if intra { LIST_INTRA_Y } else { LIST_INTER_Y };
        first + self.colour_plane + plane
    }

    /// The 8x8 list of component `plane`.
    fn list_8x8(&self, plane: usize, intra: bool) -> usize {
        LISTS_8X8_PER_PLANE * (self.colour_plane + plane) + usize::from(!intra)
    }

    /// `Intra16x16DCLevel` of component `PLANE`: inverse Hadamard (eq.
    /// 8-320) and DC scaling (eq. 8-326) into the DC of every 4x4 block,
    /// or the levels themselves under transform bypass (`dcY = c`).
    pub fn luma_dc<C: Coefficient, const PLANE: usize>(
        &self,
        levels: &Coeffs,
        (qp, scan): (i32, &Scan),
        out: &mut Residual<C>,
    ) {
        if levels.count == 0 {
            return;
        }
        let mut c = [0i32; 16];
        for (position, level) in levels.iter() {
            c[scan.four[position]] = level;
        }
        let blocks = &mut out.planes[PLANE];
        let mut place = |pos: usize, dc: C| {
            // dcY_ij belongs to the 4x4 block in row i, column j.
            blocks.blocks[pos][0] = dc;
            if dc != C::ZERO {
                blocks.coded |= 1 << pos;
            }
        };
        if out.bypass {
            for (pos, &value) in c.iter().enumerate() {
                place(pos, C::clamped(i64::from(value)));
            }
            return;
        }
        let qp = self.plane_qp(qp, PLANE);
        let f = hadamard_4x4(&c);
        let scale = self.scale.scale_4x4[self.list_4x4(PLANE, true)][(qp % QP_PERIOD) as usize][0];
        let scaler = Scaler::new(qp, 6);
        for (pos, &value) in f.iter().enumerate() {
            place(pos, scaler.apply(value, scale));
        }
    }

    /// A 4x4 block of component `PLANE` (`ac`: Intra16x16 AC, scan from
    /// position 1) at raster index `block` with the scans `scan`.
    pub fn luma_4x4<C: Coefficient, const PLANE: usize>(
        &self,
        levels: &Coeffs,
        (ac, intra): (bool, bool),
        (qp, scan): (i32, &Scan),
        block: usize,
        out: &mut Residual<C>,
    ) {
        if levels.count == 0 {
            return;
        }
        let first = usize::from(ac);
        let order = &scan.four;
        let bypass = out.bypass;
        let blocks = &mut out.planes[PLANE];
        let coeffs = &mut blocks.blocks[block];
        if bypass {
            for (position, level) in levels.iter() {
                coeffs[order[position + first]] = C::clamped(i64::from(level));
            }
        } else {
            let qp = self.plane_qp(qp, PLANE);
            let scale =
                &self.scale.scale_4x4[self.list_4x4(PLANE, intra)][(qp % QP_PERIOD) as usize];
            let scaler = Scaler::new(qp, 4);
            for (position, level) in levels.iter() {
                let pos = order[position + first];
                coeffs[pos] = scaler.apply(level, scale[pos]);
            }
        }
        blocks.coded |= 1 << block;
        // Only a lone level at scan position 0 leaves the AC zero.
        let dc_only = levels.count == 1 && levels.scan[0] as usize + first == 0;
        blocks.ac |= u16::from(!dc_only) << block;
    }

    /// 8x8 block `b8` of component `PLANE` with the scans `scan`.
    pub fn luma_8x8<C: Coefficient, const PLANE: usize>(
        &self,
        levels: &Coeffs,
        intra: bool,
        (qp, scan): (i32, &Scan),
        b8: usize,
        out: &mut Residual<C>,
    ) {
        if levels.count == 0 {
            return;
        }
        let order = &scan.eight;
        let bypass = out.bypass;
        let blocks = &mut out.planes[PLANE];
        let coeffs = &mut blocks.blocks_8x8[b8];
        if bypass {
            for (position, level) in levels.iter() {
                coeffs[order[position]] = C::clamped(i64::from(level));
            }
        } else {
            let qp = self.plane_qp(qp, PLANE);
            let scale =
                &self.scale.scale_8x8[self.list_8x8(PLANE, intra)][(qp % QP_PERIOD) as usize];
            let scaler = Scaler::new(qp, 6);
            for (position, level) in levels.iter() {
                let pos = order[position];
                coeffs[pos] = scaler.apply(level, scale[pos]);
            }
        }
        blocks.coded_8x8 |= 1 << b8;
    }

    /// `ChromaDCLevel` of `comp` into the DC of its 4x4 blocks: 4:2:0's 2x2
    /// inverse transform (eq. 8-328) and scaling (eq. 8-330), 4:2:2's 2x4
    /// one (eq. 8-329) and scaling at `QP′c,DC` (eqs. 8-331, 8-332), or the
    /// levels themselves under transform bypass (`dcC = c`).
    pub fn chroma_dc<C: Coefficient>(
        &self,
        levels: &Coeffs,
        intra: bool,
        qp: i32,
        comp: usize,
        out: &mut Residual<C>,
    ) {
        if levels.count == 0 {
            return;
        }
        let qpc = self.chroma_qp(qp, comp);
        let list = self.list_4x4(1 + comp, intra);
        let mut dc = [0i64; MAX_CHROMA_BLOCKS];
        let blocks = self.chroma.chroma_blocks();
        if blocks == CHROMA_DC_420 {
            let mut c = [0i32; CHROMA_DC_420];
            for (scan, level) in levels.iter() {
                c[scan] = level;
            }
            if out.bypass {
                for (out, value) in dc.iter_mut().zip(c) {
                    *out = i64::from(value);
                }
            } else {
                let f = [
                    c[0] + c[1] + c[2] + c[3],
                    c[0] - c[1] + c[2] - c[3],
                    c[0] + c[1] - c[2] - c[3],
                    c[0] - c[1] - c[2] + c[3],
                ];
                let scale = i64::from(self.scale.scale_4x4[list][(qpc % QP_PERIOD) as usize][0]);
                for (out, value) in dc.iter_mut().zip(f) {
                    *out = ((i64::from(value) * scale) << (qpc / QP_PERIOD)) >> 5;
                }
            }
        } else if out.bypass {
            let c = chroma::chroma_dc_422_bypass(levels.iter());
            for (out, value) in dc.iter_mut().zip(c) {
                *out = i64::from(value);
            }
        } else {
            let f = chroma::chroma_dc_422(levels.iter());
            let qp_dc = qpc + CHROMA_DC_422_QP_ADD;
            let scale = i64::from(self.scale.scale_4x4[list][(qp_dc % QP_PERIOD) as usize][0]);
            let per = qp_dc / QP_PERIOD;
            for (out, value) in dc.iter_mut().zip(f) {
                let product = i64::from(value) * scale;
                *out = if per >= CHROMA_DC_422_SHIFT {
                    product << (per - CHROMA_DC_422_SHIFT)
                } else {
                    let shift = CHROMA_DC_422_SHIFT - per;
                    (product + (1 << (shift - 1))) >> shift
                };
            }
        }
        for (b, &value) in dc.iter().enumerate().take(blocks) {
            let value = C::clamped(value);
            out.chroma[comp][b][0] = value;
            if value != C::ZERO {
                out.chroma_coded[comp] |= 1 << b;
            }
        }
    }

    /// `ChromaACLevel` of block `b` of `comp` with the scans `scan`.
    pub fn chroma_ac<C: Coefficient>(
        &self,
        levels: &Coeffs,
        intra: bool,
        (qp, scan): (i32, &Scan),
        (comp, b): (usize, usize),
        out: &mut Residual<C>,
    ) {
        if levels.count == 0 {
            return;
        }
        let order = &scan.four;
        let coeffs = &mut out.chroma[comp][b];
        if out.bypass {
            for (position, level) in levels.iter() {
                coeffs[order[position + 1]] = C::clamped(i64::from(level));
            }
        } else {
            let qpc = self.chroma_qp(qp, comp);
            let scale =
                &self.scale.scale_4x4[self.list_4x4(1 + comp, intra)][(qpc % QP_PERIOD) as usize];
            let scaler = Scaler::new(qpc, 4);
            for (position, level) in levels.iter() {
                let pos = order[position + 1];
                coeffs[pos] = scaler.apply(level, scale[pos]);
            }
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

impl<C: Coefficient> LumaBlocks<C> {
    const EMPTY: Self = Self {
        blocks: [[C::ZERO; 16]; 16],
        coded: 0,
        ac: 0,
        blocks_8x8: [[C::ZERO; 64]; 4],
        coded_8x8: 0,
    };

    /// Whether 4x4 block `block` (raster index) has coefficients.
    pub fn has_4x4(&self, block: usize) -> bool {
        (self.coded >> block) & 1 != 0
    }

    pub fn has_8x8(&self, b8: usize) -> bool {
        (self.coded_8x8 >> b8) & 1 != 0
    }

    /// Whether any 4x4 or 8x8 block has coefficients.
    pub fn has_any(&self) -> bool {
        self.coded != 0 || self.coded_8x8 != 0
    }

    /// Add 4x4 block `block` to `dst` (its 4x4 samples) and clear it.
    pub fn add_4x4<T: H264Sample<Coeff = C>>(
        &mut self,
        (dsp, depth): (&Dsp, BitDepth),
        block: usize,
        dst: &mut PlaneViewMut<'_, T>,
    ) {
        if !self.has_4x4(block) {
            return;
        }
        add_4x4(dsp, depth, &mut self.blocks[block], dst);
        self.coded &= !(1 << block);
        self.ac &= !(1 << block);
    }

    /// Add 8x8 block `b8` to `dst` (its 8x8 samples) and clear it.
    pub fn add_8x8<T: H264Sample<Coeff = C>>(
        &mut self,
        (dsp, depth): (&Dsp, BitDepth),
        b8: usize,
        dst: &mut PlaneViewMut<'_, T>,
    ) {
        if !self.has_8x8(b8) {
            return;
        }
        T::add_8x8(dsp, depth, &self.blocks_8x8[b8], dst);
        self.blocks_8x8[b8] = [C::ZERO; 64];
        self.coded_8x8 &= !(1 << b8);
    }

    /// Add every coded 4x4 block to `dst` (the macroblock's 16x16 samples)
    /// in one call, and clear them.
    pub fn add_4x4_all<T: H264Sample<Coeff = C>>(
        &mut self,
        (dsp, depth): (&Dsp, BitDepth),
        dst: &mut PlaneViewMut<'_, T>,
    ) {
        if self.coded == 0 {
            return;
        }
        T::add_4x4_mb(dsp, depth, &self.blocks, (self.coded, self.ac), dst);
        for block in BitIter(self.coded) {
            self.blocks[block] = [C::ZERO; 16];
        }
        self.coded = 0;
        self.ac = 0;
    }

    /// The levels of 4x4 block `block` (transform bypass), cleared; zero
    /// when the block has none.
    pub fn take_4x4(&mut self, block: usize) -> [C; 16] {
        self.coded &= !(1 << block);
        self.ac &= !(1 << block);
        core::mem::replace(&mut self.blocks[block], [C::ZERO; 16])
    }

    /// The levels of 8x8 block `b8` (transform bypass), cleared.
    pub fn take_8x8(&mut self, b8: usize) -> [C; 64] {
        self.coded_8x8 &= !(1 << b8);
        core::mem::replace(&mut self.blocks_8x8[b8], [C::ZERO; 64])
    }
}

impl<C: Coefficient> Residual<C> {
    pub fn new() -> Self {
        Self {
            planes: [LumaBlocks::EMPTY, LumaBlocks::EMPTY, LumaBlocks::EMPTY],
            chroma: [[[C::ZERO; 16]; MAX_CHROMA_BLOCKS]; 2],
            chroma_coded: [0; 2],
            bypass: false,
        }
    }

    /// Start a macroblock coded with (`bypass`, lossless) or without
    /// transform bypass.
    pub fn set_bypass(&mut self, bypass: bool) {
        self.bypass = bypass;
    }

    /// Whether the macroblock's blocks hold levels to add untransformed.
    pub fn bypass(&self) -> bool {
        self.bypass
    }

    /// The luma-like blocks of component `plane`.
    pub fn luma(&mut self, plane: usize) -> &mut LumaBlocks<C> {
        &mut self.planes[plane]
    }

    /// Luma 4x4 blocks with coefficients, for the deblocking filter (bit
    /// `y4 * 4 + x4`; every block of a coded 8x8 transform block). Only
    /// luma counts, in 4:4:4 as well (JM's `s_cbp[0]`).
    pub fn luma_nonzero(&self) -> u16 {
        let luma = &self.planes[0];
        let mut mask = luma.coded;
        for b8 in 0..4 {
            if (luma.coded_8x8 >> b8) & 1 != 0 {
                let (x, y) = ((b8 % 2) * 2, (b8 / 2) * 2);
                mask |= 0b11_0011 << (y * 4 + x);
            }
        }
        mask
    }

    /// Whether any chroma block of either component has coefficients.
    pub fn has_chroma_any(&self) -> bool {
        self.chroma_coded != [0; 2]
    }

    /// Whether chroma block `b` of `comp` has coefficients.
    pub fn has_chroma(&self, comp: usize, b: usize) -> bool {
        (self.chroma_coded[comp] >> b) & 1 != 0
    }

    /// The levels of chroma block `b` of `comp` (transform bypass), cleared.
    pub fn take_chroma_4x4(&mut self, comp: usize, b: usize) -> [C; 16] {
        self.chroma_coded[comp] &= !(1 << b);
        core::mem::replace(&mut self.chroma[comp][b], [C::ZERO; 16])
    }

    /// Add chroma block `b` of `comp` to `dst` and clear it.
    pub fn add_chroma_4x4<T: H264Sample<Coeff = C>>(
        &mut self,
        dsp: &Dsp,
        depth: BitDepth,
        (comp, b): (usize, usize),
        dst: &mut PlaneViewMut<'_, T>,
    ) {
        if (self.chroma_coded[comp] >> b) & 1 == 0 {
            return;
        }
        add_4x4(dsp, depth, &mut self.chroma[comp][b], dst);
        self.chroma_coded[comp] &= !(1 << b);
    }
}

/// Inverse transform `coeffs` (then zeroed) and add it to `dst`.
fn add_4x4<T: H264Sample>(
    dsp: &Dsp,
    depth: BitDepth,
    coeffs: &mut [T::Coeff; 16],
    dst: &mut PlaneViewMut<'_, T>,
) {
    T::add_4x4(dsp, depth, coeffs, dst);
    *coeffs = [T::Coeff::ZERO; 16];
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
        let dq = Dequantiser::new(
            LevelScale::new(&ScalingMatrix::FLAT, 2),
            [0, 0],
            [BitDepth::EIGHT; 2],
            (ChromaArray::Yuv420, 0),
        );
        let mut levels = Coeffs::new();
        levels.push(0, 2);
        let mut residual = Residual::<i16>::new();
        dq.luma_dc::<i16, 0>(&levels, (28, Scan::of(false)), &mut residual);
        assert_eq!(residual.planes[0].coded, u16::MAX);
    }

    /// A 4:2:2 chroma DC block at `QP′c,DC` 31 (`QP′c` 28, eq. 8-331's
    /// rounded right shift by `6 − 31 / 6 = 1`): the flat scale 16 · 16
    /// (`LevelScale4x4(1, 0, 0)` is 16 · 11) on the transformed value.
    #[test]
    fn chroma_dc_422_scaling() {
        let dq = Dequantiser::new(
            LevelScale::new(&ScalingMatrix::FLAT, 2),
            [0, 0],
            [BitDepth::EIGHT; 2],
            (ChromaArray::Yuv422, 0),
        );
        let mut levels = Coeffs::new();
        levels.push(0, 1);
        let mut residual = Residual::<i16>::new();
        // QPY 28 maps to QPc 28 (Table 8-15 is the identity below 30).
        dq.chroma_dc(&levels, true, 28, 0, &mut residual);
        // f is 1 everywhere; (1 * 176 + 1) >> 1 = 88.
        assert_eq!(residual.chroma_coded[0], 0xff);
        assert!(residual.chroma[0].iter().all(|block| block[0] == 88));
    }
}
