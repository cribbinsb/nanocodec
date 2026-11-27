//! CABAC coding of one residual block: `residual_block_cabac()` (H.264
//! §7.3.5.3.3) with the context index derivations of §9.3.3.1.1.9 and
//! §9.3.3.1.3.

use nc_dsp::Dsp;

use crate::cabac::{BinRun, BinSink, encode_signed_ueg_suffix};
use crate::tables::{LAST_CTX_INC_8X8, SIGNIFICANT_CTX_INC_8X8, ZIGZAG_4X4, ZIGZAG_8X8};

/// A coefficient scan (H.264 §8.5.6, §8.5.7) prepared for the residual
/// writers: they find a block's coded levels from a bit mask of its non-zero
/// raster positions (a `Dsp::nonzero_mask` kernel, without the scan) and
/// visit only those, instead of reading every position in scan order
/// (finding the last level that way was a third of the residual writer:
/// Jetson instructions -1.9% on crowd).
#[derive(Debug)]
pub struct Scan {
    /// Raster position of each scan index.
    order: &'static [usize],
    /// Scan index of each raster position (only those in `covered`).
    index: [u8; MAX_COEFFS],
    /// Raster positions the scan visits, one bit each.
    covered: u64,
}

/// Coefficients of the largest block (8x8).
const MAX_COEFFS: usize = 64;

impl Scan {
    const fn new(order: &'static [usize]) -> Self {
        let mut index = [0; MAX_COEFFS];
        let mut covered = 0;
        let mut i = 0;
        while i < order.len() {
            index[order[i]] = i as u8;
            covered |= 1 << order[i];
            i += 1;
        }
        Self {
            order,
            index,
            covered,
        }
    }

    /// Number of coefficients the scan visits (`maxNumCoeff`).
    pub fn len(&self) -> usize {
        self.order.len()
    }

    /// The raster position of scan index `i`.
    pub fn position(&self, i: usize) -> usize {
        self.order[i]
    }

    /// The non-zero levels of a block as a mask over scan indices: bit `i`
    /// is set when the level at scan index `i` is non-zero, given `nonzero`,
    /// the block's non-zero raster positions ([`Dsp::nonzero_mask`]).
    fn significance(&self, nonzero: u64) -> u64 {
        if nonzero & self.covered == 0 {
            return 0;
        }
        match self.order.len() {
            // Chroma DC is already in scan order. The 4x4 AC scan drops
            // the DC bit after permuting the two raster bytes.
            4 => nonzero & self.covered,
            len @ (15 | 16) => {
                let low = ZIGZAG_MASKS_4X4[0][usize::from(nonzero as u8)];
                let high = ZIGZAG_MASKS_4X4[1][usize::from((nonzero >> 8) as u8)];
                u64::from((low | high) >> (16 - len))
            }
            _ => self.sparse_significance(nonzero),
        }
    }

    fn sparse_significance(&self, nonzero: u64) -> u64 {
        // Only the non-zero positions, few at our rates, are mapped.
        let mut significant = 0;
        let mut rest = nonzero & self.covered;
        while rest != 0 {
            let pos = rest.trailing_zeros() as usize % MAX_COEFFS;
            significant |= 1 << self.index[pos];
            rest &= rest - 1;
        }
        significant
    }
}

/// Each byte of a 4x4 raster significance mask, permuted into zig-zag
/// order. Two loads replace the per-nonzero bit walk during 4x4 RD pricing.
const ZIGZAG_MASKS_4X4: [[u16; 256]; 2] = {
    let mut table = [[0; 256]; 2];
    let mut index = 0;
    while index < ZIGZAG_4X4.len() {
        let pos = ZIGZAG_4X4[index];
        let (byte, bit) = (pos / 8, pos % 8);
        let mut mask = 0;
        while mask < 256 {
            if mask & (1 << bit) != 0 {
                table[byte][mask] |= 1 << index;
            }
            mask += 1;
        }
        index += 1;
    }
    table
};

/// The 4x4 frame zig-zag scan (all 16 coefficients).
pub static SCAN_4X4: Scan = Scan::new(&ZIGZAG_4X4);
/// The 4x4 zig-zag scan without the DC (`Intra16x16ACLevel`,
/// `ChromaACLevel`: scan positions 1..=15).
pub static SCAN_4X4_AC: Scan = Scan::new(ZIGZAG_4X4_AC);
/// The 2x2 chroma DC block of 4:2:0, in raster order (§8.5.11.1).
pub static SCAN_CHROMA_DC: Scan = Scan::new(&[0, 1, 2, 3]);
/// The 8x8 frame zig-zag scan.
pub static SCAN_8X8: Scan = Scan::new(&ZIGZAG_8X8);

const ZIGZAG_4X4_AC: &[usize] = ZIGZAG_4X4.split_at(1).1;

/// `ctxBlockCat` (H.264 Table 9-42) of the block kinds used so far.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockCat {
    /// `Intra16x16DCLevel`: 16 coefficients.
    LumaDc = 0,
    /// `Intra16x16ACLevel`: 15 coefficients (scan positions 1..=15).
    LumaAc = 1,
    /// `LumaLevel4x4` (inter 4x4 blocks): 16 coefficients.
    Luma4x4 = 2,
    /// `ChromaDCLevel` (4:2:0): 4 coefficients.
    ChromaDc = 3,
    /// `ChromaACLevel`: 15 coefficients.
    ChromaAc = 4,
}

impl BlockCat {
    /// `maxNumCoeff` of the block.
    pub const fn max_num_coeff(self) -> usize {
        match self {
            Self::LumaDc | Self::Luma4x4 => 16,
            Self::LumaAc | Self::ChromaAc => 15,
            Self::ChromaDc => 4,
        }
    }

    /// `ctxIdxBlockCatOffset` for `coded_block_flag` (Table 9-40).
    const fn cbf_offset(self) -> usize {
        4 * self as usize
    }

    /// `ctxIdxBlockCatOffset` for `significant/last_significant_coeff_flag`.
    const fn significance_offset(self) -> usize {
        match self {
            Self::LumaDc => 0,
            Self::LumaAc => 15,
            Self::Luma4x4 => 29,
            Self::ChromaDc => 44,
            Self::ChromaAc => 47,
        }
    }

    /// `ctxIdxBlockCatOffset` for `coeff_abs_level_minus1`.
    const fn level_offset(self) -> usize {
        match self {
            Self::LumaDc => 0,
            Self::LumaAc => 10,
            Self::Luma4x4 => 20,
            Self::ChromaDc => 30,
            Self::ChromaAc => 39,
        }
    }
}

/// `ctxIdxOffset` values for frame-coded macroblocks (H.264 Table 9-34).
const CODED_BLOCK_FLAG_OFFSET: usize = 85;
const SIGNIFICANT_COEFF_OFFSET: usize = 105;
const LAST_SIGNIFICANT_COEFF_OFFSET: usize = 166;
const COEFF_ABS_LEVEL_OFFSET: usize = 227;

/// `cMax` of the truncated-unary prefix of `coeff_abs_level_minus1` (UEG0).
const LEVEL_PREFIX_MAX: u32 = 14;

/// Write `coded_block_flag` and, if set, the significance map and levels of
/// one 4x4-transform block (or DC block). The block's levels in coding order
/// are `levels[scan.position(i)]`, with exactly `cat.max_num_coeff()` scan
/// entries (no scan arrays are built: the levels are read in place);
/// `cbf_ctx_inc` is the `coded_block_flag` context increment from the
/// neighbouring blocks (§9.3.3.1.1.9). Returns the `coded_block_flag` written.
pub fn write_residual_block(
    cabac: &mut impl BinSink,
    dsp: &Dsp,
    cat: BlockCat,
    levels: &[i16],
    scan: &Scan,
    cbf_ctx_inc: usize,
) -> bool {
    debug_assert_eq!(scan.len(), cat.max_num_coeff());
    let significant = scan.significance(dsp.nonzero_mask(levels));
    cabac.encode_decision(
        CODED_BLOCK_FLAG_OFFSET + cat.cbf_offset() + cbf_ctx_inc,
        significant != 0,
    );
    if significant == 0 {
        return false;
    }
    let contexts = LevelContexts {
        significant: SIGNIFICANT_COEFF_OFFSET + cat.significance_offset(),
        last: LAST_SIGNIFICANT_COEFF_OFFSET + cat.significance_offset(),
        level: COEFF_ABS_LEVEL_OFFSET + cat.level_offset(),
        gt1_cap: if cat == BlockCat::ChromaDc { 3 } else { 4 },
    };
    let inc = move |i: usize| significance_ctx_inc(cat, i);
    cabac.code_run(SignificanceAndLevels {
        levels,
        scan,
        significant,
        contexts,
        significant_inc: inc,
        last_inc: inc,
    });
    true
}

/// `ctxIdxOffset`s of a frame-coded 8x8 luma block (`ctxBlockCat` 5, Table
/// 9-34). For 4:2:0 its `coded_block_flag` is not coded: the block is coded
/// exactly when its `CodedBlockPatternLuma` bit is set (§7.3.5.3.3).
const SIGNIFICANT_COEFF_8X8_OFFSET: usize = 402;
const LAST_SIGNIFICANT_COEFF_8X8_OFFSET: usize = 417;
const COEFF_ABS_LEVEL_8X8_OFFSET: usize = 426;

/// Write the significance map and levels of one 8x8 luma block
/// (`ctxBlockCat` 5). `levels` are its 64 levels in raster order, at least
/// one of them non-zero; they are coded in zig-zag order.
pub fn write_luma_8x8_block(cabac: &mut impl BinSink, dsp: &Dsp, levels: &[i16; 64]) {
    let significant = SCAN_8X8.significance(dsp.nonzero_mask(levels));
    assert_ne!(significant, 0, "a coded 8x8 block has a non-zero level");
    let contexts = LevelContexts {
        significant: SIGNIFICANT_COEFF_8X8_OFFSET,
        last: LAST_SIGNIFICANT_COEFF_8X8_OFFSET,
        level: COEFF_ABS_LEVEL_8X8_OFFSET,
        gt1_cap: 4,
    };
    cabac.code_run(SignificanceAndLevels {
        levels,
        scan: &SCAN_8X8,
        significant,
        contexts,
        significant_inc: |i| usize::from(SIGNIFICANT_CTX_INC_8X8[i]),
        last_inc: |i| usize::from(LAST_CTX_INC_8X8[i]),
    });
}

/// Context bases of one block's significance map and levels.
struct LevelContexts {
    /// `significant_coeff_flag` ctxIdxOffset + ctxIdxBlockCatOffset.
    significant: usize,
    /// `last_significant_coeff_flag` ctxIdxOffset + ctxIdxBlockCatOffset.
    last: usize,
    /// `coeff_abs_level_minus1` ctxIdxOffset + ctxIdxBlockCatOffset.
    level: usize,
    /// Cap on `numDecodAbsLevelGt1` in the ctxIdxInc of the later bins.
    gt1_cap: usize,
}

/// The significance map of one block (scan order; bit `i` of `significant`
/// is set when scan index `i` holds a non-zero level, at least one is) and
/// then its levels in reverse scan order (§7.3.5.3.3, §9.3.3.1.3), with the
/// ctxIdxInc of the flags by scan index.
struct SignificanceAndLevels<'a, F, G> {
    levels: &'a [i16],
    scan: &'a Scan,
    significant: u64,
    contexts: LevelContexts,
    significant_inc: F,
    last_inc: G,
}

impl<F: Fn(usize) -> usize, G: Fn(usize) -> usize> BinRun for SignificanceAndLevels<'_, F, G> {
    type Output = ();

    fn code(self, cabac: &mut impl BinSink) {
        let Self {
            levels,
            scan,
            significant,
            contexts,
            significant_inc,
            last_inc,
        } = self;
        let last = (u64::BITS - 1 - significant.leading_zeros()) as usize;
        for i in 0..last {
            let coded = (significant >> i) & 1 != 0;
            cabac.encode_decision(contexts.significant + significant_inc(i), coded);
            if coded {
                cabac.encode_decision(contexts.last + last_inc(i), false);
            }
        }
        // The flags of the final scan position are never coded (it is
        // significant if the scan gets that far).
        if last + 1 < scan.len() {
            cabac.encode_decision(contexts.significant + significant_inc(last), true);
            cabac.encode_decision(contexts.last + last_inc(last), true);
        }

        // Track the contexts themselves. The first becomes zero after
        // the first level above one, and the suffix context saturates.
        let mut first_inc = 1usize;
        let mut rest_inc = 5usize;
        let rest_cap = 5 + contexts.gt1_cap;
        let mut rest = significant;
        while rest != 0 {
            let i = (u64::BITS - 1 - rest.leading_zeros()) as usize;
            rest ^= 1 << i;
            let c = levels[scan.position(i)];
            let abs_minus1 = u32::from(c.unsigned_abs()) - 1;
            let prefix = abs_minus1.min(LEVEL_PREFIX_MAX);
            cabac.encode_decision(contexts.level + first_inc, prefix > 0);
            if prefix > 0 {
                cabac.encode_repeated_decision(contexts.level + rest_inc, true, prefix - 1);
                if prefix < LEVEL_PREFIX_MAX {
                    cabac.encode_decision(contexts.level + rest_inc, false);
                }
            }
            if abs_minus1 >= LEVEL_PREFIX_MAX {
                encode_signed_ueg_suffix(cabac, (abs_minus1 - LEVEL_PREFIX_MAX) as u16, 0, c < 0);
            } else {
                cabac.encode_bypass(c < 0);
            }

            if prefix == 0 {
                first_inc = (first_inc + usize::from(first_inc != 0)).min(4);
            } else {
                first_inc = 0;
                rest_inc = (rest_inc + 1).min(rest_cap);
            }
        }
    }
}

/// ctxIdxInc of `significant/last_significant_coeff_flag` for scan position
/// `i` of a frame-coded block (§9.3.3.1.3).
fn significance_ctx_inc(cat: BlockCat, i: usize) -> usize {
    if cat == BlockCat::ChromaDc {
        // NumC8x8 = 1 for 4:2:0.
        i.min(2)
    } else {
        i
    }
}

#[path = "residual_trellis.rs"]
pub(crate) mod trellis;

#[cfg(test)]
mod mask_tests {
    use super::*;

    #[test]
    fn short_scan_masks_follow_scan_order() {
        for scan in [&SCAN_CHROMA_DC, &SCAN_4X4, &SCAN_4X4_AC] {
            for mask in 0..(1 << 16) {
                assert_eq!(scan.significance(mask), scan.sparse_significance(mask));
            }
        }
    }
}
