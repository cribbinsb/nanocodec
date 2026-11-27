//! SP and SI slices (Extended profile, Annex A.2.3): their slice header
//! fields (§7.3.3, §7.4.3) and the reconstruction of the macroblocks
//! decoded by §8.6, the P macroblocks of SP slices and the SI macroblocks
//! of SI slices.
//!
//! Such a macroblock's prediction is not added to a residual: it is
//! transformed (eq. 8-415), combined with the coefficient levels and
//! quantised with `QS` (§8.6.1 for SP slices, §8.6.2 for switching SP and
//! for SI slices), and the result is scaled with `QS` and inverse
//! transformed like any residual (§8.5.12), replacing the prediction. So
//! the levels are kept unscaled ([`SwitchingLevels`]) until the prediction
//! exists. The forward and inverse 4x4 transforms are nc-dsp's shared H.264
//! kernels; the decoder's sample type is 8-bit here (the Extended profile
//! has no deeper samples).

use nc_common::{BitDepth, PlaneViewMut, Sample};
use nc_dsp::Dsp;

use super::pps::Pps;
use super::reader::Reader;
use super::scaling::norm_adjust_4x4;
use super::sps::Sps;
use super::syntax::Coeffs;
use super::transform::{Scan, chroma_qp_at};
use super::{DecodeError, Result};

/// `QSY` range (§7.4.3, `slice_qs_delta`).
const MAX_QS: i32 = 51;
/// `QP % 6` period of the scaling factors.
const QP_PERIOD: i32 = 6;
/// `LevelScale2(m, i, j)` (eqs. 8-418, 8-419): `w` per `QS % 6`, columns
/// for positions with both coordinates even, both odd, and mixed.
const LEVEL_SCALE2: [[i32; 3]; QP_PERIOD as usize] = [
    [13107, 5243, 8066],
    [11916, 4660, 7490],
    [10082, 4194, 6554],
    [9362, 3647, 5825],
    [8192, 3355, 5243],
    [7282, 2893, 4559],
];
/// `Aij` (eq. 8-417), by the same three position classes.
const A: [i32; 3] = [16, 25, 20];
/// Samples of a luma 4x4 block, and of a 4:2:0 chroma component of a
/// macroblock (four 4x4 blocks).
const BLOCK: usize = 4;
const CHROMA_SIDE: usize = 8;
const CHROMA_BLOCKS: usize = 4;
/// The raster index of the transposed position in a 2x2 block.
const TRANSPOSED_2X2: [usize; 4] = [0, 2, 1, 3];

/// Which requantisation a slice's §8.6 macroblocks use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Requantise {
    /// SP slices with `sp_for_switch_flag` 0 (§8.6.1): the levels are
    /// scaled with `QPY` and added to the prediction's coefficients.
    NonSwitching,
    /// SP slices with `sp_for_switch_flag` 1 and SI slices (§8.6.2): the
    /// prediction is quantised with `QSY` and the levels added to it.
    Switching,
}

/// The QS fields of an SP or SI slice header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Switching {
    /// `QSY = 26 + pic_init_qs_minus26 + slice_qs_delta` (eq. 7-31).
    pub qs: i32,
    pub requantise: Requantise,
    /// `chroma_qp_index_offset` for `QPC` and `QSC` (§8.5.8).
    chroma_qp_offset: i8,
}

/// Parse `sp_for_switch_flag` (SP slices, `sp`) and `slice_qs_delta`,
/// after `slice_qp_delta`. SP and SI slices exist only in the Extended
/// profile: CAVLC, 8-bit samples and no 8x8 transform; with any of those
/// the stream belongs to no profile.
pub(super) fn parse(r: &mut Reader<'_>, sps: &Sps, pps: &Pps, sp: bool) -> Result<Switching> {
    if pps.cabac || pps.transform_8x8_mode || !sps.depths.is_eight() {
        return Err(DecodeError::Unsupported(
            "H264 SP/SI slices outside the Extended profile",
        ));
    }
    let for_switch = sp && r.flag()?;
    let qs = pps.pic_init_qs + r.se()?;
    if !(0..=MAX_QS).contains(&qs) {
        return Err(DecodeError::Malformed("slice QS"));
    }
    Ok(Switching {
        qs,
        requantise: if sp && !for_switch {
            Requantise::NonSwitching
        } else {
            Requantise::Switching
        },
        chroma_qp_offset: pps.chroma_qp_offset[0],
    })
}

/// The position class of raster position `pos` of a 4x4 block: both
/// coordinates even, both odd, or mixed (eqs. 8-417, 8-418).
fn class(pos: usize) -> usize {
    match (pos % 4 % 2, pos / 4 % 2) {
        (0, 0) => 0,
        (1, 1) => 1,
        _ => 2,
    }
}

/// `Sign(x) * ((Abs(x) * LevelScale2(qs % 6, pos) + (1 << (shift - 1))) >>
/// shift)`: the quantisation of eqs. 8-420, 8-432 (`shift` 15 + `qs` / 6)
/// and, for chroma DC, 8-429, 8-439 (16 + `qs` / 6).
fn quantise(x: i64, qs: i32, class: usize, shift: i32) -> i64 {
    let scale = i64::from(LEVEL_SCALE2[(qs % QP_PERIOD) as usize][class]);
    let magnitude = (x.abs() * scale + (1 << (shift - 1))) >> shift;
    x.signum() * magnitude
}

/// A requantised level in the range of eq. 8-335's input (`±2^15` at 8
/// bits, §8.5.12.1), which a conforming stream never leaves; hostile levels
/// are clamped so the scaling cannot overflow.
fn clamp_coefficient(c: i64) -> i32 {
    c.clamp(i64::from(i16::MIN), i64::from(i16::MAX)) as i32
}

/// The coefficient levels of a §8.6 macroblock, unscaled until its
/// prediction exists: luma 4x4 blocks and chroma AC blocks in raster order
/// (16-bit: levels lie in `−2^15..2^15`, §8.5.12.1), chroma DC by
/// `chroma4x4BlkIdx`. Cleared once used, so a `P_Skip` macroblock sees
/// zero levels (§8.6).
#[derive(Debug, Clone)]
pub(super) struct SwitchingLevels {
    luma: [[i16; 16]; 16],
    chroma_dc: [[i16; 4]; 2],
    chroma_ac: [[[i16; 16]; 4]; 2],
    /// Luma 4x4 blocks with a non-zero level (bit `y4 * 4 + x4`).
    luma_coded: u16,
}

impl SwitchingLevels {
    pub const fn new() -> Self {
        Self {
            luma: [[0; 16]; 16],
            chroma_dc: [[0; 4]; 2],
            chroma_ac: [[[0; 16]; 4]; 2],
            luma_coded: 0,
        }
    }

    /// `LumaLevel4x4` of the block at raster index `block`, in the
    /// picture's coefficient scan.
    pub fn luma_4x4(&mut self, levels: &Coeffs, scan: &Scan, block: usize) {
        for (index, level) in levels.iter() {
            self.luma[block][scan.four()[index]] = clamp_level(level);
        }
        self.luma_coded |= u16::from(levels.count > 0) << block;
    }

    /// `ChromaDCLevel` of component `comp`.
    pub fn chroma_dc(&mut self, levels: &Coeffs, comp: usize) {
        for (scan, level) in levels.iter() {
            self.chroma_dc[comp][scan] = clamp_level(level);
        }
    }

    /// `ChromaACLevel` of block `b` of component `comp`.
    pub fn chroma_ac(&mut self, levels: &Coeffs, scan: &Scan, comp: usize, b: usize) {
        for (index, level) in levels.iter() {
            self.chroma_ac[comp][b][scan.four()[index + 1]] = clamp_level(level);
        }
    }

    /// Luma 4x4 blocks with levels.
    pub fn luma_coded(&self) -> u16 {
        self.luma_coded
    }
}

/// A parsed level in the 16 bits a conforming one fits (§8.5.12.1).
fn clamp_level(level: i32) -> i16 {
    level.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
}

/// The predicted samples of a `side` x `side` block as 16-bit values for
/// the forward transform.
fn prediction<T: Sample, const N: usize>(block: &PlaneViewMut<'_, T>, side: usize) -> [i16; N] {
    let mut out = [0i16; N];
    for (y, row) in out.chunks_exact_mut(side).enumerate() {
        let samples = &block.as_view().row(y)[..side];
        for (value, &sample) in row.iter_mut().zip(samples) {
            *value = sample.to_i32() as i16;
        }
    }
    out
}

/// Write the inverse-transformed residual `r` of the 4x4 block at
/// (`x`, `y`) of `block`, clipped (eqs. 8-421, 8-426, 8-434, 8-438).
fn write_4x4<T: Sample>(
    block: &mut PlaneViewMut<'_, T>,
    (x, y): (usize, usize),
    r: &[i32; 16],
    depth: BitDepth,
) {
    for (row, values) in r.as_chunks::<BLOCK>().0.iter().enumerate() {
        let samples = &mut block.row_mut(y + row)[x..x + BLOCK];
        for (sample, &value) in samples.iter_mut().zip(values) {
            *sample = T::from_i32(value.clamp(0, depth.max()));
        }
    }
}

impl Switching {
    /// Replace the predicted luma 4x4 block `block` (its samples) by its
    /// §8.6 reconstruction from the levels of raster block `index` of a
    /// macroblock with `QPY` `qp` (§8.6.1.1, §8.6.2.1); the levels are
    /// cleared.
    pub fn luma_4x4<T: Sample>(
        self,
        (dsp, depth): (&Dsp, BitDepth),
        levels: &mut SwitchingLevels,
        (index, qp): (usize, i32),
        block: &mut PlaneViewMut<'_, T>,
    ) {
        let mut predicted = [[0i16; 16]];
        dsp.h264_forward_4x4(&prediction::<T, 16>(block, BLOCK), BLOCK, 1, &mut predicted);
        let mut c = [0i32; 16];
        for (pos, out) in c.iter_mut().enumerate() {
            let level = i32::from(levels.luma[index][pos]);
            *out = self.combine(i32::from(predicted[0][pos]), level, qp, class(pos));
        }
        levels.luma[index] = [0; 16];
        levels.luma_coded &= !(1 << index);
        let mut r = [[0i32; 16]];
        dsp.h264_inverse_4x4(&[scale(c, self.qs, false)], &mut r);
        write_4x4(block, (0, 0), &r[0], depth);
    }

    /// Replace the predicted 8x8 samples `plane` of chroma component `comp`
    /// of a macroblock with `QPY` `qp` by their §8.6 reconstruction
    /// (§8.6.1.2, §8.6.2.2); the component's levels are cleared.
    pub fn chroma<T: Sample>(
        self,
        (dsp, depth): (&Dsp, BitDepth),
        levels: &mut SwitchingLevels,
        (comp, qp): (usize, i32),
        plane: &mut PlaneViewMut<'_, T>,
    ) {
        let qpc = chroma_qp_at(qp, self.chroma_qp_offset, 0);
        let qsc = chroma_qp_at(self.qs, self.chroma_qp_offset, 0);
        let mut predicted = [[0i16; 16]; CHROMA_BLOCKS];
        let samples = prediction::<T, 64>(plane, CHROMA_SIDE);
        dsp.h264_forward_4x4(&samples, CHROMA_SIDE, 2, &mut predicted);
        let dc = self.chroma_dc(&predicted, levels.chroma_dc[comp], (qpc, qsc));
        let mut blocks = [[0i32; 16]; CHROMA_BLOCKS];
        for (b, block) in blocks.iter_mut().enumerate() {
            let ac = &levels.chroma_ac[comp][b];
            for pos in 1..16 {
                let (p, level) = (i32::from(predicted[b][pos]), i32::from(ac[pos]));
                block[pos] = self.combine(p, level, qpc, class(pos));
            }
            *block = scale(*block, qsc, true);
            // eq. 8-431 (or 8-441 scaled likewise): the DC is scaled already.
            block[0] = dc[b];
        }
        levels.chroma_dc[comp] = [0; 4];
        levels.chroma_ac[comp] = [[0; 16]; 4];
        let mut r = [[0i32; 16]; CHROMA_BLOCKS];
        dsp.h264_inverse_4x4(&blocks, &mut r);
        for (b, r) in r.iter().enumerate() {
            write_4x4(plane, ((b % 2) * BLOCK, (b / 2) * BLOCK), r, depth);
        }
    }

    /// The level of one coefficient that the inverse scaling with `QS`
    /// receives: prediction coefficient `p` and parsed level `level` at a
    /// position of class `class`, in a macroblock with `QPY` (or `QPC`)
    /// `qp`.
    fn combine(self, p: i32, level: i32, qp: i32, class: usize) -> i32 {
        let shift = 15 + self.qs / QP_PERIOD;
        let (p, level) = (i64::from(p), i64::from(level));
        clamp_coefficient(match self.requantise {
            // eqs. 8-416, 8-420 (8-424, 8-425): LevelScale4x4 is 16 · v
            // with flat weights, so `>> 10` is `>> 6` of v.
            Requantise::NonSwitching => {
                let v = i64::from(norm_adjust_4x4((qp % QP_PERIOD) as usize, class));
                let sum = p + (((level * v * i64::from(A[class])) << (qp / QP_PERIOD)) >> 6);
                quantise(sum, self.qs, class, shift)
            }
            // eqs. 8-432, 8-433 (8-435, 8-437).
            Requantise::Switching => quantise(p, self.qs, class, shift) + level,
        })
    }

    /// The chroma DC of the four blocks of a component, scaled (eqs.
    /// 8-427..8-431, or 8-439..8-441), from the prediction's coefficients
    /// and the DC levels.
    fn chroma_dc(
        self,
        predicted: &[[i16; 16]; CHROMA_BLOCKS],
        levels: [i16; 4],
        (qpc, qsc): (i32, i32),
    ) -> [i32; 4] {
        let p: [i32; 4] = core::array::from_fn(|b| i32::from(predicted[b][0]));
        // eq. 8-427: dcp = H · [p0 p1; p2 p3] · H, H = [1 1; 1 −1].
        let dcp = hadamard_2x2(p);
        let shift = 16 + qsc / QP_PERIOD;
        let quantised: [i32; 4] = core::array::from_fn(|k| {
            // eq. 8-428: dc_ij takes ChromaDCLevel[j * 2 + i], the level at
            // the transposed position of raster index `k`.
            let level = i64::from(levels[TRANSPOSED_2X2[k]]);
            let p = i64::from(dcp[k]);
            clamp_coefficient(match self.requantise {
                Requantise::NonSwitching => {
                    let v = i64::from(norm_adjust_4x4((qpc % QP_PERIOD) as usize, 0));
                    // eq. 8-428: LevelScale4x4 · A00 `>> 9` is v · A00 `>> 5`.
                    let sum = p + (((level * v * i64::from(A[0])) << (qpc / QP_PERIOD)) >> 5);
                    quantise(sum, qsc, 0, shift)
                }
                Requantise::Switching => quantise(p, qsc, 0, shift) + level,
            })
        });
        // eqs. 8-430, 8-431: f = H · dcr · H, scaled with QSC, f_ij to the
        // block of Figure 8-7 (raster), as the reference decoder (JM) and
        // so the suite's SP streams place it.
        let v = norm_adjust_4x4((qsc % QP_PERIOD) as usize, 0);
        hadamard_2x2(quantised).map(|f| ((f * v) << (qsc / QP_PERIOD)) >> 1)
    }
}

/// Scale levels `c` with `qs` (§8.5.12.1 with flat weights: `c · v <<
/// (qs / 6)`, exact at every `qs`), the DC left alone for `chroma`.
fn scale(mut c: [i32; 16], qs: i32, chroma: bool) -> [i32; 16] {
    let m = (qs % QP_PERIOD) as usize;
    for (pos, value) in c.iter_mut().enumerate().skip(usize::from(chroma)) {
        *value = (*value * norm_adjust_4x4(m, class(pos))) << (qs / QP_PERIOD);
    }
    c
}

/// The 2x2 transform `[1 1; 1 −1] · c · [1 1; 1 −1]` of a 2x2 block in
/// raster order (eqs. 8-427, 8-430).
fn hadamard_2x2(c: [i32; 4]) -> [i32; 4] {
    [
        c[0] + c[1] + c[2] + c[3],
        c[0] - c[1] + c[2] - c[3],
        c[0] + c[1] - c[2] - c[3],
        c[0] - c[1] - c[2] + c[3],
    ]
}
