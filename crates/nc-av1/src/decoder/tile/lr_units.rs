//! Loop restoration unit syntax (AV1 §5.11.57 `read_lr`, §5.11.58
//! `read_lr_unit`): each superblock, before its partition, codes the
//! filter of every restoration unit whose top-left corner it holds. The
//! filters go into [`RestorationUnits`], which `decoder::restoration`
//! applies after CDEF.
use super::super::cdf::Cdfs;
use super::super::frame_header::{FrameSize, LrType, Restoration, SUPERRES_NUM};
use super::super::restoration::RestorationUnits;
use super::MI_SIZE;
use crate::decoder_range::SymbolReader;
use nc_common::PlaneIndex;
use nc_common::decode::DecodeError;
use nc_dsp::av1_restoration::{Av1LrFilter, Av1SgrProj, Av1Wiener};

/// Wiener taps coded per pass (the outer three; §5.11.58).
const WIENER_COEFFS: usize = 3;
/// `Wiener_Taps_Min`, `Wiener_Taps_Max`, `Wiener_Taps_K` and
/// `Wiener_Taps_Mid` (§3): range, subexponential parameter and initial
/// reference of each coded tap.
const WIENER_TAPS_MIN: [i32; WIENER_COEFFS] = [-5, -23, -17];
const WIENER_TAPS_MAX: [i32; WIENER_COEFFS] = [10, 8, 46];
const WIENER_TAPS_K: [u32; WIENER_COEFFS] = [1, 2, 3];
const WIENER_TAPS_MID: [i32; WIENER_COEFFS] = [3, -7, 15];
/// `Sgrproj_Xqd_Min`, `Sgrproj_Xqd_Max` and `Sgrproj_Xqd_Mid` (§3).
const SGRPROJ_XQD_MIN: [i32; 2] = [-96, -32];
const SGRPROJ_XQD_MAX: [i32; 2] = [31, 95];
const SGRPROJ_XQD_MID: [i32; 2] = [-32, 31];
/// `SGRPROJ_PRJ_SUBEXP_K`, `SGRPROJ_PRJ_BITS`, `SGRPROJ_PARAMS_BITS`
/// (§3).
const SGRPROJ_PRJ_SUBEXP_K: u32 = 4;
const SGRPROJ_PRJ_BITS: u32 = 7;
const SGRPROJ_PARAMS_BITS: u32 = 4;
/// `restoration_type` symbols of switchable planes (§6.10.27): none,
/// Wiener, self-guided.
const RESTORE_SWITCHABLE_TYPES: usize = 3;
const RESTORE_WIENER: usize = 1;
const RESTORE_SGRPROJ: usize = 2;

/// The restoration units of a frame and the references their
/// coefficients are coded against (`RefLrWiener`, `RefSgrXqd`).
#[derive(Debug)]
pub(crate) struct LrUnitReader {
    lr: Restoration,
    size: FrameSize,
    units: RestorationUnits,
    wiener: [[[i32; WIENER_COEFFS]; 2]; 3],
    xqd: [[i32; 2]; 3],
}

impl LrUnitReader {
    /// The reader of a frame of `size` with `lr` (one with at least one
    /// restored plane), its references as a tile starts them.
    pub(crate) fn new(lr: Restoration, size: FrameSize) -> Self {
        Self {
            units: RestorationUnits::new(&lr, &size),
            lr,
            size,
            wiener: [[WIENER_TAPS_MID; 2]; 3],
            xqd: [SGRPROJ_XQD_MID; 3],
        }
    }

    /// Start a tile: the references return to their initial values
    /// (§5.11.2 `decode_tile`).
    pub(crate) fn start_tile(&mut self) {
        self.wiener = [[WIENER_TAPS_MID; 2]; 3];
        self.xqd = [SGRPROJ_XQD_MID; 3];
    }

    /// The filters read so far.
    pub(crate) fn units(&self) -> &RestorationUnits {
        &self.units
    }

    /// `read_lr()` of the superblock of `sb_mi` 4x4 units at (`mi_col`,
    /// `mi_row`): the units whose top-left corner falls inside it.
    pub(crate) fn read_superblock(
        &mut self,
        r: &mut SymbolReader<'_>,
        cdfs: &mut Cdfs,
        (mi_col, mi_row): (usize, usize),
        sb_mi: usize,
    ) -> Result<(), DecodeError> {
        for plane in PlaneIndex::ALL {
            let kind = self.lr.kind[plane as usize];
            if kind == LrType::None {
                continue;
            }
            let unit_size = self.lr.unit_size[plane as usize];
            let sample = MI_SIZE >> usize::from(plane != PlaneIndex::Y);
            // Columns count in upscaled samples: a coded 4x4 column spans
            // `SuperresDenom / SUPERRES_NUM` of them.
            let first_row = |mi: usize| (mi * sample).div_ceil(unit_size);
            let denom = self.size.superres_denom;
            let first_column = |mi: usize| (mi * sample * denom).div_ceil(unit_size * SUPERRES_NUM);
            let (columns, rows) = self.units.grid(plane);
            let row_range = first_row(mi_row)..first_row(mi_row + sb_mi).min(rows);
            let column_range = first_column(mi_col)..first_column(mi_col + sb_mi).min(columns);
            for row in row_range {
                for column in column_range.clone() {
                    let filter = self.read_unit(r, cdfs, plane, kind)?;
                    self.units.set(plane, (row, column), filter);
                }
            }
        }
        Ok(())
    }

    /// `read_lr_unit()` of one unit of `plane`, whose frame type is `kind`.
    fn read_unit(
        &mut self,
        r: &mut SymbolReader<'_>,
        cdfs: &mut Cdfs,
        plane: PlaneIndex,
        kind: LrType,
    ) -> Result<Av1LrFilter, DecodeError> {
        let restoration_type = match kind {
            LrType::Wiener => usize::from(r.boolean(&mut cdfs.use_wiener)) * RESTORE_WIENER,
            LrType::SgrProj => usize::from(r.boolean(&mut cdfs.use_sgrproj)) * RESTORE_SGRPROJ,
            LrType::Switchable => r.symbol(&mut cdfs.restoration_type, RESTORE_SWITCHABLE_TYPES),
            LrType::None => return Ok(Av1LrFilter::None),
        };
        Ok(match restoration_type {
            RESTORE_WIENER => Av1LrFilter::Wiener(self.read_wiener(r, plane)?),
            RESTORE_SGRPROJ => Av1LrFilter::SgrProj(self.read_sgrproj(r, plane)?),
            _ => Av1LrFilter::None,
        })
    }

    /// The Wiener taps of a unit: vertical (pass 0), then horizontal;
    /// chroma codes no outermost tap.
    fn read_wiener(
        &mut self,
        r: &mut SymbolReader<'_>,
        plane: PlaneIndex,
    ) -> Result<Av1Wiener, DecodeError> {
        let first = usize::from(plane != PlaneIndex::Y);
        let mut taps = [[0i8; WIENER_COEFFS]; 2];
        for (pass, coded) in taps.iter_mut().enumerate() {
            let reference = &mut self.wiener[plane as usize][pass];
            for j in first..WIENER_COEFFS {
                let v = signed_subexp_with_ref(
                    r,
                    (WIENER_TAPS_MIN[j], WIENER_TAPS_MAX[j] + 1),
                    WIENER_TAPS_K[j],
                    reference[j],
                )?;
                coded[j] = v as i8;
                reference[j] = v;
            }
        }
        let [vertical, horizontal] = taps;
        Ok(Av1Wiener {
            vertical,
            horizontal,
        })
    }

    /// The self-guided set and projection weights of a unit; a pass the
    /// set skips codes no weight (the second then follows from the first).
    fn read_sgrproj(
        &mut self,
        r: &mut SymbolReader<'_>,
        plane: PlaneIndex,
    ) -> Result<Av1SgrProj, DecodeError> {
        let set = r.literal(SGRPROJ_PARAMS_BITS) as u8;
        let reference = &mut self.xqd[plane as usize];
        let mut xqd = [0i8; 2];
        for (i, pass_runs) in Av1SgrProj::passes(set).into_iter().enumerate() {
            let (min, max) = (SGRPROJ_XQD_MIN[i], SGRPROJ_XQD_MAX[i]);
            let v = if pass_runs {
                signed_subexp_with_ref(r, (min, max + 1), SGRPROJ_PRJ_SUBEXP_K, reference[i])?
            } else if i == 1 {
                ((1 << SGRPROJ_PRJ_BITS) - reference[0]).clamp(min, max)
            } else {
                0
            };
            xqd[i] = v as i8;
            reference[i] = v;
        }
        Ok(Av1SgrProj { set, xqd })
    }
}

/// `decode_signed_subexp_with_ref_bool(low, high, k, r)` (§5.11.58).
fn signed_subexp_with_ref(
    r: &mut SymbolReader<'_>,
    (low, high): (i32, i32),
    k: u32,
    reference: i32,
) -> Result<i32, DecodeError> {
    let x = unsigned_subexp_with_ref(r, (high - low) as u32, k, (reference - low) as u32)?;
    Ok(x as i32 + low)
}

/// `decode_unsigned_subexp_with_ref_bool(mx, k, r)`.
fn unsigned_subexp_with_ref(
    r: &mut SymbolReader<'_>,
    mx: u32,
    k: u32,
    reference: u32,
) -> Result<u32, DecodeError> {
    let v = subexp(r, mx, k)?;
    Ok(if reference << 1 <= mx {
        inverse_recenter(reference, v)
    } else {
        mx - 1 - inverse_recenter(mx - 1 - reference, v)
    })
}

/// `decode_subexp_bool(numSyms, k)`: a subexponential code of a value
/// below `symbols`.
fn subexp(r: &mut SymbolReader<'_>, symbols: u32, k: u32) -> Result<u32, DecodeError> {
    let mut i = 0;
    let mut mk = 0;
    loop {
        let b2 = if i > 0 { k + i - 1 } else { k };
        let a = 1 << b2;
        if symbols <= mk + 3 * a {
            return Ok(non_symmetric(r, symbols - mk) + mk);
        }
        if r.literal(1) == 0 {
            return Ok(r.literal(b2) + mk);
        }
        i += 1;
        mk += a;
    }
}

/// `NS(n)` read with literal bits (§4.10.10): a value below `n`, the
/// first `2^w - n` values one bit shorter.
fn non_symmetric(reader: &mut SymbolReader<'_>, n: u32) -> u32 {
    let bits = n.ilog2() + 1;
    let short = (1 << bits) - n;
    let value = reader.literal(bits - 1);
    if value < short {
        return value;
    }
    (value << 1) - short + reader.literal(1)
}

/// `inverse_recenter(r, v)` (§5.9.28).
fn inverse_recenter(r: u32, v: u32) -> u32 {
    if v > 2 * r {
        v
    } else if v & 1 == 1 {
        r - v.div_ceil(2)
    } else {
        r + v / 2
    }
}
