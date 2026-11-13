//! CAVLC slice data (H.264 `entropy_coding_mode_flag` 0): `mb_skip_run`
//! (§7.3.4), the macroblock layer with exp-Golomb elements (§7.3.5, §9.1)
//! and `residual_block_cavlc()` (§7.3.5.3.2, §9.2), with `nC` from the
//! neighbouring blocks' `TotalCoeff` (§9.2.1). The macroblock types are
//! those of the CABAC writer (`syntax.rs`); an 8x8 transform block is coded
//! as four interleaved 4x4 blocks (§7.3.5.3.2).
//!
//! CAVLC codes are table lookups where CABAC renormalises per bin: about 6%
//! less Jetson time for h264-fastest, at +10% bytes at QP 32 growing to
//! +19% at QP 44, so no preset uses it (`docs/handoff/h264-fastest.md`).
//! Mode decision prices with the CABAC model either way.

use nc_common::BitWriter;
use nc_encoder::Mv;

use crate::cabac::SliceKind;
use crate::intra4x4::block_position;
use crate::intra8x8::modes_per_4x4;
use crate::macroblock::{CBP_CHROMA_AC, CBP_CHROMA_NONE, Residual, ResidualKind};
use crate::mbinfo::{MbInfo, MbKind, MbPartition, Neighbours};
use crate::mbqp::QpTracker;
use crate::mvpred::predict_mv;
use crate::residual::{SCAN_4X4, SCAN_4X4_AC, SCAN_8X8, SCAN_CHROMA_DC, Scan};
use crate::syntax::{CodedMb, carries_qp_delta, new_info, skip_info};
use crate::tables::cavlc::{
    CODED_BLOCK_PATTERN_420, COEFF_TOKEN, RUN_BEFORE, TOTAL_ZEROS, TOTAL_ZEROS_CHROMA_DC,
};

/// `mb_type` of the first intra type in P slices (Table 7-13: intra types
/// follow the five P types, numbered as in I slices, Table 7-11).
const P_SLICE_INTRA_MB_TYPE: u32 = 5;
/// `mb_type` of `I_16x16_0_0_0` in I slices; the prediction mode, chroma
/// pattern (x4) and luma pattern (x12) are added (Table 7-11).
const I_16X16_MB_TYPE: u32 = 1;
const I_16X16_CHROMA_STEP: u32 = 4;
const I_16X16_LUMA_STEP: u32 = 12;
/// `rem_intra4x4_pred_mode` / `rem_intra8x8_pred_mode` width.
const REM_INTRA_PRED_MODE_BITS: u32 = 3;
/// `nC` from which `coeff_token` is a 6-bit fixed-length code (Table 9-5),
/// and that code's value for `TotalCoeff` 0.
const FLC_NC: i32 = 8;
const FLC_BITS: u32 = 6;
const FLC_NO_COEFFS: u32 = 3;
/// `coeff_token` table class of 4:2:0 chroma DC (`nC` == -1).
const CHROMA_DC_CLASS: usize = 3;
const CHROMA_DC_NC: i32 = -1;
/// Most trailing ones coded as signs alone (§9.2.1).
const MAX_TRAILING_ONES: usize = 3;
/// `level_prefix` limits (§9.2.2.1): 14 with `suffixLength` 0 takes a 4-bit
/// suffix; 15 and above escape to a `level_prefix - 3` bit suffix.
const LEVEL_PREFIX_LONG_SUFFIX: u32 = 14;
const LEVEL_PREFIX_LONG_SUFFIX_BITS: u32 = 4;
const LEVEL_PREFIX_ESCAPE: u32 = 15;
/// `suffixLength` stops growing here (§9.2.2.1).
const MAX_SUFFIX_LENGTH: u32 = 6;
/// Number of `run_before` tables (`zerosLeft` 1..=6, and more than 6).
const RUN_BEFORE_TABLES: usize = 7;
/// Coefficients per 4x4 block, and the parts of an interleaved 8x8 block.
const BLOCK_COEFFS: usize = 16;
const PARTS_8X8: usize = 4;

/// `codeNum` of each `coded_block_pattern` value (`cbp_luma | cbp_chroma << 4`),
/// intra (`Intra_4x4`/`Intra_8x8`) and inter columns: the inverse of Table
/// 9-4 for 4:2:0.
const CBP_CODE_NUM: [[u8; 48]; 2] = [
    invert_cbp_table(&CODED_BLOCK_PATTERN_420[0]),
    invert_cbp_table(&CODED_BLOCK_PATTERN_420[1]),
];

const fn invert_cbp_table(table: &[u8; 48]) -> [u8; 48] {
    let mut inverse = [0; 48];
    let mut code = 0;
    while code < table.len() {
        inverse[table[code] as usize] = code as u8;
        code += 1;
    }
    inverse
}

/// `TotalCoeff` of every 4x4 block of a coded macroblock, the `nC` context
/// of later blocks (zero for skipped macroblocks and uncoded blocks).
#[derive(Debug, Clone, Copy, Default)]
struct TotalCoeffs {
    /// Luma blocks, raster (`by * 4 + bx`); the AC block for `I_16x16`.
    luma: [u8; 16],
    /// Chroma AC blocks per component, raster (`by * 2 + bx`).
    chroma: [[u8; 4]; 2],
}

/// The CAVLC writer of one slice: header, then slice data.
#[derive(Debug)]
pub struct CavlcEncoder {
    out: BitWriter,
    slice: SliceKind,
    /// `num_ref_idx_l0_active_minus1` + 1 (`ref_idx_l0` is `te(v)`).
    active_references: u8,
    /// Skipped macroblocks not yet counted by an `mb_skip_run`.
    skip_run: u32,
    /// Per macroblock, raster; reset per slice.
    counts: Vec<TotalCoeffs>,
    mb_cols: usize,
}

impl CavlcEncoder {
    /// Start a slice of `kind` after `header` (not aligned: CAVLC slice data
    /// follows the header directly) in a picture of `mb_count` macroblocks,
    /// `mb_cols` wide.
    pub fn new(
        header: BitWriter,
        kind: SliceKind,
        active_references: u8,
        (mb_count, mb_cols): (usize, usize),
    ) -> Self {
        Self {
            out: header,
            slice: kind,
            active_references,
            skip_run: 0,
            counts: vec![TotalCoeffs::default(); mb_count],
            mb_cols,
        }
    }

    /// Write macroblock `index` (raster) coded as `mb`, its residual
    /// quantized at `mb_qp`, and return its [`MbInfo`] (as
    /// [`crate::syntax::write_macroblock`]).
    pub fn write(
        &mut self,
        mb: &CodedMb<'_>,
        nb: &Neighbours<'_>,
        (qp, mb_qp): (&mut QpTracker, u8),
        index: usize,
    ) -> MbInfo {
        self.counts[index] = TotalCoeffs::default();
        if let CodedMb::Skip(mv) = *mb {
            self.skip_run += 1;
            let mut info = skip_info(mv);
            info.qp = qp.uncoded();
            return info;
        }
        if self.slice == SliceKind::P {
            self.out.put_ue(self.skip_run);
            self.skip_run = 0;
        }
        let mut info = self.write_layer(mb, nb, (qp, mb_qp), index);
        if !carries_qp_delta(&info) {
            info.qp = qp.uncoded();
        }
        info
    }

    /// The slice data's end (a final `mb_skip_run`, then
    /// `rbsp_slice_trailing_bits`) and the slice RBSP.
    pub fn finish(mut self) -> Vec<u8> {
        if self.skip_run > 0 {
            self.out.put_ue(self.skip_run);
        }
        self.out.trailing_bits();
        self.out.into_bytes()
    }

    fn intra_mb_type_offset(&self) -> u32 {
        match self.slice {
            SliceKind::I => 0,
            SliceKind::P => P_SLICE_INTRA_MB_TYPE,
        }
    }

    /// `macroblock_layer()` of a coded (not skipped) macroblock.
    fn write_layer(
        &mut self,
        mb: &CodedMb<'_>,
        nb: &Neighbours<'_>,
        (qp, mb_qp): (&mut QpTracker, u8),
        index: usize,
    ) -> MbInfo {
        let (mut info, residual) = match *mb {
            CodedMb::Skip(_) => unreachable!("skipped macroblocks only extend the skip run"),
            CodedMb::Intra16x16 { modes, residual } => {
                let mb_type = I_16X16_MB_TYPE
                    + modes.luma as u32
                    + I_16X16_CHROMA_STEP * u32::from(residual.cbp_chroma)
                    + I_16X16_LUMA_STEP * u32::from(residual.cbp_luma != 0);
                self.out.put_ue(self.intra_mb_type_offset() + mb_type);
                self.out.put_ue(modes.chroma as u32); // intra_chroma_pred_mode
                let mut info = new_info(MbKind::Intra16x16, [Mv::ZERO; 4], [[0; 2]; 4], residual);
                info.chroma_pred_mode = modes.chroma;
                (info, residual)
            }
            CodedMb::Intra4x4 {
                luma_modes,
                chroma,
                residual,
            } => {
                self.write_intra_nxn_header(false);
                write_intra_4x4_modes(&mut self.out, &luma_modes, nb);
                self.out.put_ue(chroma as u32);
                let mut info = new_info(MbKind::Intra4x4, [Mv::ZERO; 4], [[0; 2]; 4], residual);
                info.chroma_pred_mode = chroma;
                info.intra_modes = luma_modes;
                (info, residual)
            }
            CodedMb::Intra8x8 {
                luma_modes,
                chroma,
                residual,
            } => {
                self.write_intra_nxn_header(true);
                let intra_modes = write_intra_8x8_modes(&mut self.out, luma_modes, nb);
                self.out.put_ue(chroma as u32);
                let mut info = new_info(MbKind::Intra8x8, [Mv::ZERO; 4], [[0; 2]; 4], residual);
                info.chroma_pred_mode = chroma;
                info.intra_modes = intra_modes;
                info.transform_8x8 = true;
                (info, residual)
            }
            CodedMb::Inter {
                partition,
                mvs,
                ref_idx,
                mvp,
                residual,
            } => {
                self.write_inter_header(partition, &mvs, (ref_idx, mvp), nb);
                let mut info = new_info(MbKind::Inter, mvs, [[0; 2]; 4], residual);
                info.ref_idx = ref_idx.unwrap_or(0);
                (info, residual)
            }
        };
        if info.kind != MbKind::Intra16x16 {
            let column = usize::from(info.kind == MbKind::Inter);
            let cbp = residual.cbp_luma | (residual.cbp_chroma << 4);
            self.out
                .put_ue(u32::from(CBP_CODE_NUM[column][usize::from(cbp)]));
            if info.kind == MbKind::Inter && residual.cbp_luma != 0 {
                // transform_size_8x8_flag: every partition is at least 8x8.
                info.transform_8x8 = residual.kind == ResidualKind::Inter8x8;
                self.out.put_bit(info.transform_8x8);
            }
        }
        if carries_qp_delta(&info) {
            let delta = qp.delta_to(mb_qp);
            self.out.put_se(delta); // mb_qp_delta
            info.qp = qp.coded(delta);
            self.write_residual(residual, &mut info, index);
        }
        info
    }

    /// `mb_type` `I_NxN` and its `transform_size_8x8_flag` (the PPS enables
    /// the 8x8 transform).
    fn write_intra_nxn_header(&mut self, transform_8x8: bool) {
        self.out.put_ue(self.intra_mb_type_offset());
        self.out.put_bit(transform_8x8);
    }

    /// `mb_type`, `sub_mb_type`s, `ref_idx_l0`s and `mvd_l0`s of an inter
    /// macroblock whose partitions all use `ref_idx` (`None`: one active
    /// reference, not coded); `known_mvp` as in [`CodedMb::Inter`].
    fn write_inter_header(
        &mut self,
        partition: MbPartition,
        mvs: &[Mv; 4],
        (ref_idx, known_mvp): (Option<u8>, Option<Mv>),
        nb: &Neighbours<'_>,
    ) {
        let out = &mut self.out;
        // Table 7-13 order: P_L0_16x16, P_L0_L0_16x8, P_L0_L0_8x16, P_8x8.
        out.put_ue(match partition {
            MbPartition::P16x16 => 0,
            MbPartition::P16x8 => 1,
            MbPartition::P8x16 => 2,
            MbPartition::P8x8 => 3,
        });
        if partition == MbPartition::P8x8 {
            for _ in 0..4 {
                out.put_ue(0); // sub_mb_type P_L0_8x8
            }
        }
        if let Some(ref_idx) = ref_idx {
            for _ in 0..partition.count() {
                write_te(out, ref_idx, self.active_references - 1);
            }
        }
        let ref_idx = ref_idx.unwrap_or(0);
        let mut own = [None; 4];
        for part in 0..partition.count() {
            let (x8, y8, w8, h8) = partition.rect(part);
            let mv = mvs[y8 * 2 + x8];
            let mvp = match known_mvp {
                Some(mvp) if partition == MbPartition::P16x16 => mvp,
                _ => predict_mv(nb, &own, partition, part, ref_idx),
            };
            out.put_se(i32::from(mv.x) - i32::from(mvp.x));
            out.put_se(i32::from(mv.y) - i32::from(mvp.y));
            for qy in y8..y8 + h8 {
                for qx in x8..x8 + w8 {
                    own[qy * 2 + qx] = Some(mv);
                }
            }
        }
    }

    /// `residual()` (§7.3.5.3): luma, chroma DC, chroma AC; records each
    /// block's `TotalCoeff` and the deblocking filter's non-zero flags.
    fn write_residual(&mut self, res: &Residual, info: &mut MbInfo, index: usize) {
        let mut counts = TotalCoeffs::default();
        if res.kind == ResidualKind::Intra16x16 {
            let nc = self.luma_nc(&counts, index, 0, 0);
            let mut dc = [0; BLOCK_COEFFS];
            scan_levels(&res.luma_dc, &SCAN_4X4, &mut dc);
            write_block(&mut self.out, &dc, nc);
        }
        for b8 in 0..4 {
            if (res.cbp_luma >> b8) & 1 == 0 {
                continue;
            }
            let mut parts = [[0; BLOCK_COEFFS]; PARTS_8X8];
            let scan: &Scan = match res.kind {
                ResidualKind::Intra16x16 => &SCAN_4X4_AC,
                _ => &SCAN_4X4,
            };
            if res.kind.uses_8x8_transform() {
                interleave_8x8(&res.luma_8x8[b8], &mut parts);
            }
            for (part, levels) in parts.iter_mut().enumerate() {
                let blk = b8 * PARTS_8X8 + part;
                let (bx, by) = block_position(blk);
                let coeffs = if res.kind.uses_8x8_transform() {
                    &levels[..]
                } else {
                    let len = scan.len();
                    scan_levels(&res.luma[by * 4 + bx], scan, &mut levels[..len]);
                    &levels[..len]
                };
                let nc = self.luma_nc(&counts, index, bx, by);
                let total = write_block(&mut self.out, coeffs, nc);
                counts.luma[by * 4 + bx] = total;
                if total > 0 {
                    info.luma_coded |= 1 << (by * 4 + bx);
                }
            }
            if res.kind.uses_8x8_transform() {
                // The deblocking filter tests the 8x8 block as a whole
                // (§8.7.2.1), as for CABAC's 8x8 blocks.
                let quadrant = crate::macroblock::blocks_of_8x8(b8);
                if quadrant.iter().any(|&blk| info.luma_coded >> blk & 1 != 0) {
                    for blk in quadrant {
                        info.luma_coded |= 1 << blk;
                    }
                }
            }
        }
        if res.cbp_chroma != CBP_CHROMA_NONE {
            for dc in &res.chroma_dc {
                let mut levels = [0; 4];
                scan_levels(dc, &SCAN_CHROMA_DC, &mut levels);
                write_block(&mut self.out, &levels, CHROMA_DC_NC);
            }
        }
        if res.cbp_chroma == CBP_CHROMA_AC {
            for c in 0..2 {
                for blk in 0..4 {
                    let mut levels = [0; BLOCK_COEFFS - 1];
                    scan_levels(&res.chroma_ac[c][blk], &SCAN_4X4_AC, &mut levels);
                    let nc = self.chroma_nc(&counts, index, c, blk);
                    counts.chroma[c][blk] = write_block(&mut self.out, &levels, nc);
                }
            }
        }
        self.counts[index] = counts;
    }

    /// `nC` of luma block `(bx, by)` of macroblock `index`, whose blocks
    /// coded so far have `own` counts (§9.2.1: the left and above blocks'
    /// `TotalCoeff`; every earlier macroblock of the single slice is
    /// available).
    fn luma_nc(&self, own: &TotalCoeffs, index: usize, bx: usize, by: usize) -> i32 {
        let left = if bx > 0 {
            Some(own.luma[by * 4 + bx - 1])
        } else {
            self.left_of(index).map(|m| m.luma[by * 4 + 3])
        };
        let above = if by > 0 {
            Some(own.luma[(by - 1) * 4 + bx])
        } else {
            self.above_of(index).map(|m| m.luma[12 + bx])
        };
        predicted_nc(left, above)
    }

    /// `nC` of chroma AC block `blk` of component `c` (§9.2.1).
    fn chroma_nc(&self, own: &TotalCoeffs, index: usize, c: usize, blk: usize) -> i32 {
        let (bx, by) = (blk % 2, blk / 2);
        let left = if bx > 0 {
            Some(own.chroma[c][by * 2])
        } else {
            self.left_of(index).map(|m| m.chroma[c][by * 2 + 1])
        };
        let above = if by > 0 {
            Some(own.chroma[c][bx])
        } else {
            self.above_of(index).map(|m| m.chroma[c][2 + bx])
        };
        predicted_nc(left, above)
    }

    fn left_of(&self, index: usize) -> Option<&TotalCoeffs> {
        (!index.is_multiple_of(self.mb_cols)).then(|| &self.counts[index - 1])
    }

    fn above_of(&self, index: usize) -> Option<&TotalCoeffs> {
        index
            .checked_sub(self.mb_cols)
            .map(|above| &self.counts[above])
    }
}

/// `nC` from the counts of the available neighbouring blocks (§9.2.1).
fn predicted_nc(left: Option<u8>, above: Option<u8>) -> i32 {
    match (left, above) {
        (Some(a), Some(b)) => (i32::from(a) + i32::from(b) + 1) >> 1,
        (Some(n), None) | (None, Some(n)) => i32::from(n),
        (None, None) => 0,
    }
}

/// `te(v)` with range `max` (§9.1: one inverted bit for range 1).
fn write_te(out: &mut BitWriter, value: u8, max: u8) {
    if max == 1 {
        out.put_bit(value == 0);
    } else {
        out.put_ue(u32::from(value));
    }
}

/// The levels of a block (raster within the block) in `scan` order.
fn scan_levels(raster: &[i32], scan: &Scan, out: &mut [i32]) {
    for (i, level) in out.iter_mut().enumerate().take(scan.len()) {
        *level = raster[scan.position(i)];
    }
}

/// The four 4x4 parts of an 8x8 block (raster within the block) for CAVLC:
/// level `k` of part `i` is 8x8 scan position `4k + i` (§7.3.5.3.2).
fn interleave_8x8(raster: &[i32; 64], parts: &mut [[i32; BLOCK_COEFFS]; PARTS_8X8]) {
    for scan_index in 0..SCAN_8X8.len() {
        parts[scan_index % PARTS_8X8][scan_index / PARTS_8X8] =
            raster[SCAN_8X8.position(scan_index)];
    }
}

/// `residual_block_cavlc()` of `coeffs` (scan order, `maxNumCoeff` long)
/// with context `nc`; returns `TotalCoeff`.
fn write_block(out: &mut BitWriter, coeffs: &[i32], nc: i32) -> u8 {
    // Non-zero levels and the zeros below each, highest frequency first.
    let mut levels = [0i32; BLOCK_COEFFS];
    let mut runs = [0u8; BLOCK_COEFFS];
    let mut total = 0;
    let mut zeros = 0u8;
    let mut total_zeros = 0usize;
    for &level in coeffs.iter().rev() {
        if level != 0 {
            if total > 0 {
                runs[total - 1] = zeros;
                total_zeros += usize::from(zeros);
            }
            levels[total] = level;
            total += 1;
            zeros = 0;
        } else if total > 0 {
            zeros += 1;
        }
    }
    // The zeros below the lowest coefficient count too (its run is implied).
    total_zeros += usize::from(zeros);
    let trailing_ones = levels[..total]
        .iter()
        .take(MAX_TRAILING_ONES)
        .take_while(|level| level.abs() == 1)
        .count();
    write_coeff_token(out, nc, total, trailing_ones);
    if total == 0 {
        return 0;
    }
    for &level in &levels[..trailing_ones] {
        out.put_bit(level < 0); // trailing_ones_sign_flag
    }
    write_levels(out, &levels[trailing_ones..total], trailing_ones);
    if total < coeffs.len() {
        let (code, length) = if coeffs.len() == SCAN_CHROMA_DC.len() {
            TOTAL_ZEROS_CHROMA_DC[total - 1][total_zeros]
        } else {
            TOTAL_ZEROS[total - 1][total_zeros]
        };
        debug_assert!(length > 0, "no total_zeros code");
        out.put_bits(u32::from(code), u32::from(length));
    }
    let mut zeros_left = total_zeros;
    for &run in &runs[..total - 1] {
        if zeros_left == 0 {
            break;
        }
        let (code, length) = RUN_BEFORE[zeros_left.min(RUN_BEFORE_TABLES) - 1][usize::from(run)];
        debug_assert!(length > 0, "no run_before code");
        out.put_bits(u32::from(code), u32::from(length));
        zeros_left -= usize::from(run);
    }
    total as u8
}

/// `coeff_token` (Table 9-5) for `total` coefficients with `trailing_ones`.
fn write_coeff_token(out: &mut BitWriter, nc: i32, total: usize, trailing_ones: usize) {
    if nc >= FLC_NC {
        let code = if total == 0 {
            FLC_NO_COEFFS
        } else {
            (((total - 1) << 2) | trailing_ones) as u32
        };
        out.put_bits(code, FLC_BITS);
        return;
    }
    let class = match nc {
        CHROMA_DC_NC => CHROMA_DC_CLASS,
        0 | 1 => 0,
        2 | 3 => 1,
        _ => 2,
    };
    let (code, length) = COEFF_TOKEN[class][total][trailing_ones];
    debug_assert!(length > 0, "no coeff_token code");
    out.put_bits(u32::from(code), u32::from(length));
}

/// The levels after the trailing ones, highest frequency first
/// (`level_prefix` / `level_suffix`, §9.2.2): the inverse of §9.2.2.1.
fn write_levels(out: &mut BitWriter, levels: &[i32], trailing_ones: usize) {
    let total = levels.len() + trailing_ones;
    let mut suffix_length = u32::from(total > 10 && trailing_ones < MAX_TRAILING_ONES);
    for (i, &level) in levels.iter().enumerate() {
        let mut code = if level > 0 {
            2 * level.unsigned_abs() - 2
        } else {
            2 * level.unsigned_abs() - 1
        };
        // A level right after fewer than three trailing ones cannot be ±1,
        // so its code is shifted down by 2.
        if i == 0 && trailing_ones < MAX_TRAILING_ONES {
            code -= 2;
        }
        write_level_code(out, code, suffix_length);
        if suffix_length == 0 {
            suffix_length = 1;
        }
        if level.unsigned_abs() > (3 << (suffix_length - 1)) && suffix_length < MAX_SUFFIX_LENGTH {
            suffix_length += 1;
        }
    }
}

/// `levelCode` as `level_prefix` (that many zeros, then a one) and
/// `level_suffix` under `suffix_length`.
fn write_level_code(out: &mut BitWriter, code: u32, suffix_length: u32) {
    let (prefix, suffix, suffix_bits) = if suffix_length == 0 && code < LEVEL_PREFIX_LONG_SUFFIX {
        (code, 0, 0)
    } else if suffix_length == 0
        && code < LEVEL_PREFIX_LONG_SUFFIX + (1 << LEVEL_PREFIX_LONG_SUFFIX_BITS)
    {
        (
            LEVEL_PREFIX_LONG_SUFFIX,
            code - LEVEL_PREFIX_LONG_SUFFIX,
            LEVEL_PREFIX_LONG_SUFFIX_BITS,
        )
    } else if suffix_length > 0 && code < (LEVEL_PREFIX_ESCAPE << suffix_length) {
        (
            code >> suffix_length,
            code & ((1 << suffix_length) - 1),
            suffix_length,
        )
    } else {
        // Escape: prefix 15 holds the next 2^12 codes, each longer prefix
        // twice as many as the one before (the `(1 << (prefix - 3)) - 4096`
        // offset of §9.2.2.1).
        let mut rest = code - (LEVEL_PREFIX_ESCAPE << suffix_length);
        if suffix_length == 0 {
            rest -= LEVEL_PREFIX_ESCAPE;
        }
        let mut prefix = LEVEL_PREFIX_ESCAPE;
        while rest >= 1 << (prefix - 3) {
            rest -= 1 << (prefix - 3);
            prefix += 1;
        }
        (prefix, rest, prefix - 3)
    };
    out.put_bits(0, prefix);
    out.put_bit(true);
    if suffix_bits > 0 {
        out.put_bits(suffix, suffix_bits);
    }
}

/// `prev_intra4x4_pred_mode_flag` / `rem_intra4x4_pred_mode` (u(1), u(3))
/// of the 16 blocks in decoding order (§7.3.5.1).
fn write_intra_4x4_modes(out: &mut BitWriter, modes: &[u8; 16], nb: &Neighbours<'_>) {
    let mut decoded = [2u8; 16];
    for blk in 0..16 {
        let (bx, by) = block_position(blk);
        let mode = modes[by * 4 + bx];
        write_intra_pred_mode(out, mode, nb.predicted_intra_4x4_mode(bx, by, &decoded));
        decoded[by * 4 + bx] = mode;
    }
}

/// The four `Intra8x8PredMode`s (raster = decoding order) as for
/// `Intra_4x4`; returns the mode of every 4x4 block.
fn write_intra_8x8_modes(out: &mut BitWriter, modes: [u8; 4], nb: &Neighbours<'_>) -> [u8; 16] {
    let mut decoded = [2u8; 4];
    for (b8, &mode) in modes.iter().enumerate() {
        let own = modes_per_4x4(decoded);
        let predicted = nb.predicted_intra_4x4_mode(2 * (b8 % 2), 2 * (b8 / 2), &own);
        write_intra_pred_mode(out, mode, predicted);
        decoded[b8] = mode;
    }
    modes_per_4x4(modes)
}

fn write_intra_pred_mode(out: &mut BitWriter, mode: u8, predicted: u8) {
    out.put_bit(mode == predicted);
    if mode != predicted {
        let rem = if mode < predicted { mode } else { mode - 1 };
        out.put_bits(u32::from(rem), REM_INTRA_PRED_MODE_BITS);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reads back what [`write_block`] wrote, by the decoder's rules
    /// (§9.2), to check the level and run coding round-trips.
    fn round_trip(coeffs: &[i32], nc: i32) {
        let mut out = BitWriter::new();
        let total = write_block(&mut out, coeffs, nc);
        assert_eq!(
            usize::from(total),
            coeffs.iter().filter(|&&c| c != 0).count()
        );
        out.trailing_bits();
        let bytes = out.into_bytes();
        let decoded = crate::decoder::cavlc_block_for_tests(&bytes, nc, coeffs.len());
        assert_eq!(decoded, coeffs, "nC {nc}");
    }

    #[test]
    fn residual_blocks_round_trip() {
        let mut blocks: Vec<Vec<i32>> = vec![
            vec![0; 16],
            vec![1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 3, -1, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![12, -7, 5, 3, -2, 1, 1, -1, 0, 1, 0, 0, 0, 0, 0, 1],
            vec![200, -3000, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            vec![-1; 16],
            vec![5; 16],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -9],
        ];
        // Fifteen-coefficient (AC) and four-coefficient (chroma DC) blocks.
        blocks.push(vec![0, 2, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]);
        for block in &blocks {
            for nc in [0, 1, 2, 3, 4, 7, 8, 16] {
                round_trip(block, nc);
            }
        }
        for dc in [[0, 0, 0, 0], [1, 0, 0, -1], [-4, 2, 0, 0], [0, 0, 0, 30]] {
            round_trip(&dc, CHROMA_DC_NC);
        }
        // Every level magnitude across the escape boundaries, each suffix length.
        for level in [
            14, 15, 16, 22, 23, 30, 31, 45, 2063, 2064, 2065, 6000, 30000,
        ] {
            for lead in [1, 4, 10, 40] {
                round_trip(
                    &[lead, -level, level, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    0,
                );
            }
        }
    }

    #[test]
    fn cbp_code_numbers_invert_table_9_4() {
        for column in 0..2 {
            for (code, &cbp) in CODED_BLOCK_PATTERN_420[column].iter().enumerate() {
                assert_eq!(usize::from(CBP_CODE_NUM[column][usize::from(cbp)]), code);
            }
        }
    }
}
