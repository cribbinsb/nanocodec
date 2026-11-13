//! Neighbour contexts for one transform block (AVM `get_txb_ctx`) and the
//! entropy-context byte a coded block leaves for its neighbours (AVM
//! `cul_level` with `set_dc_sign`, stored by `av2_set_entropy_contexts`).
//!
//! Each byte of the above/left arrays describes one 4×4 unit: the low three
//! bits are the block's level sum capped at seven, the next two its DC sign.
use super::{PLANE_U, PLANE_V, PLANE_Y, TxSize};
use nc_common::decode::{DecodeError, Result};

/// Bits holding the capped level sum (AVM `COEFF_CONTEXT_BITS`).
const LEVEL_BITS: u32 = 3;
/// Mask of the capped level sum (AVM `COEFF_CONTEXT_MASK`).
const LEVEL_MASK: u8 = (1 << LEVEL_BITS) - 1;
/// DC sign codes stored above the level sum.
const DC_NEGATIVE: u8 = 1;
const DC_POSITIVE: u8 = 2;
/// Luma skip contexts saturate each neighbour's level sum at four.
const LUMA_SKIP_LEVEL_CAP: u8 = 4;
/// U skip contexts follow the six luma ones in the shared Y/U models.
const U_SKIP_OFFSET: u8 = 6;
/// V offset when the plane block is larger than the transform
/// (AVM `V_TXB_SKIP_CONTEXT_OFFSET >> 1`).
const V_SKIP_SPLIT_OFFSET: u8 = 3;
/// V offset when the co-located U block had coefficients
/// (AVM `V_TXB_SKIP_CONTEXT_OFFSET`, added from `xd->eob_u_flag`).
const V_SKIP_U_CODED_OFFSET: u8 = 6;
/// Skip contexts of the shared Y/U models (AVM `TXB_SKIP_CONTEXTS`).
const SKIP_CONTEXTS: u8 = 10;
/// Skip contexts of the V model (AVM `V_TXB_SKIP_CONTEXTS`).
const V_SKIP_CONTEXTS: u8 = 12;
/// DC sign contexts: balanced, mostly negative, mostly positive neighbours.
const DC_SIGN_BALANCED: u8 = 0;
const DC_SIGN_NEGATIVE: u8 = 1;
const DC_SIGN_POSITIVE: u8 = 2;

/// Neighbour-derived contexts for one transform block.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Context {
    /// Inter prediction (selects skip/EOB models and the transform-type symbol).
    pub is_inter: bool,
    /// Plane index: zero for Y, one for U, two for V.
    pub plane: u8,
    /// All-zero flag context; for V it includes the U-coded offset of six.
    pub skip: u8,
    /// Luma DC sign context, in 0..3.
    pub dc_sign: u8,
}

impl Context {
    /// Derive the contexts of one transform block (AVM `get_txb_ctx`, plus the
    /// `eob_u_flag` offset `av2_read_sig_txtype` adds for V).
    ///
    /// `block_log2` is the square plane block side, `above` and `left` the
    /// entropy contexts starting at the transform (at least
    /// [`TxSize::units`] each; zero outside the picture), and `u_coded`
    /// whether the most recent U block had coefficients (used for V only).
    /// # Errors
    /// Rejects bad planes, transforms larger than the block or than chroma
    /// allows, short neighbour slices and corrupt sign fields.
    pub fn derive(
        plane: u8,
        block_log2: u32,
        tx: TxSize,
        is_inter: bool,
        above: &[u8],
        left: &[u8],
        u_coded: bool,
    ) -> Result<Self> {
        if plane > PLANE_V || block_log2 < tx.log2() {
            return Err(DecodeError::Malformed("AV2 transform context geometry"));
        }
        let units = tx.units();
        let (Some(above), Some(left)) = (above.get(..units), left.get(..units)) else {
            return Err(DecodeError::Malformed("AV2 transform context neighbours"));
        };
        let skip = match plane {
            PLANE_Y if block_log2 == tx.log2() => 0,
            PLANE_Y => luma_skip(above, left),
            _ => {
                let nonzero = u8::from(above.iter().any(|&a| a != 0))
                    + u8::from(left.iter().any(|&l| l != 0));
                nonzero
                    + if plane == PLANE_U {
                        U_SKIP_OFFSET
                    } else {
                        u8::from(block_log2 > tx.log2()) * V_SKIP_SPLIT_OFFSET
                            + u8::from(u_coded) * V_SKIP_U_CODED_OFFSET
                    }
            }
        };
        let mut balance = 0i32;
        for &value in above.iter().chain(left) {
            balance += match value >> LEVEL_BITS {
                0 => 0,
                DC_NEGATIVE => -1,
                DC_POSITIVE => 1,
                _ => return Err(DecodeError::Malformed("AV2 entropy context DC sign")),
            };
        }
        let dc_sign = match balance.cmp(&0) {
            std::cmp::Ordering::Equal => DC_SIGN_BALANCED,
            std::cmp::Ordering::Less => DC_SIGN_NEGATIVE,
            std::cmp::Ordering::Greater => DC_SIGN_POSITIVE,
        };
        Ok(Self {
            is_inter,
            plane,
            skip,
            dc_sign,
        })
    }

    pub(super) fn validate(self) -> Result<()> {
        let skip_contexts = if self.plane == PLANE_V {
            V_SKIP_CONTEXTS
        } else {
            SKIP_CONTEXTS
        };
        if self.plane > PLANE_V || self.skip >= skip_contexts || self.dc_sign > DC_SIGN_POSITIVE {
            return Err(DecodeError::Malformed(
                "AV2 coefficient context outside range",
            ));
        }
        Ok(())
    }
}

/// Luma skip context when the block holds several transforms: AVM's
/// `skip_contexts[top][left]` table, which equals `(top + left + 3) / 2`.
fn luma_skip(above: &[u8], left: &[u8]) -> u8 {
    let level = |units: &[u8]| {
        (units.iter().fold(0, |acc, &u| acc | u) & LEVEL_MASK).min(LUMA_SKIP_LEVEL_CAP)
    };
    (level(above) + level(left) + 3) >> 1
}

/// The entropy-context byte of a coded block (also of an all-zero one):
/// its level sum capped at seven, plus the sign of its DC coefficient.
#[must_use]
pub fn entropy_context(coefficients: &[i32]) -> u8 {
    // Clipping each magnitude first gives the same capped sum (any term at
    // the cap already caps it) without saturating adds, so the loop
    // vectorizes and cannot overflow.
    let sum: u32 = coefficients.iter().map(|c| capped(*c)).sum();
    context_byte(sum, coefficients.first().copied().unwrap_or(0))
}

/// [`entropy_context`] built from a block's nonzero levels as a coder
/// visits them (in any order), without another pass over the block.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct EntropyContext {
    sum: u32,
    dc: i32,
}

impl EntropyContext {
    /// Count the level `value` at raster position `pos`.
    pub(crate) fn add(&mut self, pos: usize, value: i32) {
        self.sum += capped(value);
        if pos == 0 {
            self.dc = value;
        }
    }

    /// The context byte of the levels added so far.
    pub(crate) fn value(self) -> u8 {
        context_byte(self.sum, self.dc)
    }
}

fn capped(level: i32) -> u32 {
    level.unsigned_abs().min(u32::from(LEVEL_MASK))
}

fn context_byte(sum: u32, dc: i32) -> u8 {
    let level = sum.min(u32::from(LEVEL_MASK)) as u8;
    let sign = match dc.signum() {
        -1 => DC_NEGATIVE,
        1 => DC_POSITIVE,
        _ => 0,
    };
    level | sign << LEVEL_BITS
}
