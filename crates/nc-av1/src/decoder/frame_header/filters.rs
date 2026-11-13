//! Loop filter, CDEF and loop restoration syntax of the frame header (AV1
//! §5.9.11, §5.9.19, §5.9.20).
use super::TOTAL_REFS;
use crate::backend::decoder::sequence::Sequence;
use nc_common::decode::{BitReader, DecodeError};

/// Default `loop_filter_ref_deltas[]` (`setup_past_independence`, §7.20):
/// INTRA 1, LAST..LAST3 and BWDREF 0, GOLDEN, ALTREF2 and ALTREF -1.
const DEFAULT_REF_DELTAS: [i8; TOTAL_REFS] = [1, 0, 0, 0, -1, 0, -1, -1];
/// Bits of a coded loop filter delta (`su(1+6)`).
const LF_DELTA_BITS: u32 = 7;
/// Most CDEF strength sets (`cdef_bits` 3).
pub(crate) const MAX_CDEF_SETS: usize = 8;
/// `RESTORATION_TILESIZE_MAX`.
const RESTORATION_TILESIZE_MAX: usize = 256;

/// `loop_filter_ref_deltas[]` and `loop_filter_mode_deltas[]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LoopFilterDeltas {
    pub refs: [i8; TOTAL_REFS],
    pub modes: [i8; 2],
}
impl Default for LoopFilterDeltas {
    fn default() -> Self {
        Self {
            refs: DEFAULT_REF_DELTAS,
            modes: [0; 2],
        }
    }
}

/// `loop_filter_params()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct LoopFilter {
    /// `loop_filter_level[0..4]`: luma vertical, luma horizontal, U, V.
    pub level: [u8; 4],
    pub sharpness: u8,
    pub delta_enabled: bool,
    pub deltas: LoopFilterDeltas,
}

/// `cdef_params()`; secondary strengths are stored decoded (3 means 4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CdefParams {
    pub damping: u8,
    pub bits: u8,
    pub y_pri: [u8; MAX_CDEF_SETS],
    pub y_sec: [u8; MAX_CDEF_SETS],
    pub uv_pri: [u8; MAX_CDEF_SETS],
    pub uv_sec: [u8; MAX_CDEF_SETS],
}
impl Default for CdefParams {
    /// CDEF off: one all-zero set, damping 3.
    fn default() -> Self {
        Self {
            damping: 3,
            bits: 0,
            y_pri: [0; MAX_CDEF_SETS],
            y_sec: [0; MAX_CDEF_SETS],
            uv_pri: [0; MAX_CDEF_SETS],
            uv_sec: [0; MAX_CDEF_SETS],
        }
    }
}

/// `FrameRestorationType` (AV1 §6.10.15).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum LrType {
    #[default]
    None,
    Wiener,
    SgrProj,
    Switchable,
}

/// `lr_params()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct Restoration {
    pub kind: [LrType; 3],
    /// `LoopRestorationSize[]` per plane.
    pub unit_size: [usize; 3],
}
impl Restoration {
    pub(crate) fn any(&self) -> bool {
        self.kind.iter().any(|&k| k != LrType::None)
    }
}

/// `secondary` as coded: 3 means 4.
fn secondary(coded: u32) -> u8 {
    if coded == 3 { 4 } else { coded as u8 }
}

fn read_su(r: &mut BitReader<'_>, bits: u32) -> Result<i8, DecodeError> {
    let value = r.read_bits(bits)? as i32;
    let shift = 32 - bits;
    Ok(((value << shift) >> shift) as i8)
}

/// `previous` holds the primary reference's deltas (`load_previous`).
pub(crate) fn loop_filter(
    r: &mut BitReader<'_>,
    seq: &Sequence,
    off: bool,
    previous: Option<LoopFilterDeltas>,
) -> Result<LoopFilter, DecodeError> {
    let mut lf = LoopFilter {
        deltas: previous.unwrap_or_default(),
        ..LoopFilter::default()
    };
    if off {
        lf.deltas = LoopFilterDeltas::default();
        return Ok(lf);
    }
    lf.level[0] = r.read_bits(6)? as u8;
    lf.level[1] = r.read_bits(6)? as u8;
    if !seq.mono_chrome && (lf.level[0] != 0 || lf.level[1] != 0) {
        lf.level[2] = r.read_bits(6)? as u8;
        lf.level[3] = r.read_bits(6)? as u8;
    }
    lf.sharpness = r.read_bits(3)? as u8;
    lf.delta_enabled = r.read_bit()?;
    if lf.delta_enabled && r.read_bit()? {
        for delta in &mut lf.deltas.refs {
            if r.read_bit()? {
                *delta = read_su(r, LF_DELTA_BITS)?;
            }
        }
        for delta in &mut lf.deltas.modes {
            if r.read_bit()? {
                *delta = read_su(r, LF_DELTA_BITS)?;
            }
        }
    }
    Ok(lf)
}

pub(crate) fn cdef(
    r: &mut BitReader<'_>,
    seq: &Sequence,
    off: bool,
) -> Result<CdefParams, DecodeError> {
    let mut c = CdefParams::default();
    if off {
        return Ok(c);
    }
    c.damping = r.read_bits(2)? as u8 + 3;
    c.bits = r.read_bits(2)? as u8;
    for i in 0..1 << c.bits {
        c.y_pri[i] = r.read_bits(4)? as u8;
        c.y_sec[i] = secondary(r.read_bits(2)?);
        if !seq.mono_chrome {
            c.uv_pri[i] = r.read_bits(4)? as u8;
            c.uv_sec[i] = secondary(r.read_bits(2)?);
        }
    }
    Ok(c)
}

pub(crate) fn restoration(
    r: &mut BitReader<'_>,
    seq: &Sequence,
    off: bool,
) -> Result<Restoration, DecodeError> {
    let mut lr = Restoration::default();
    if off {
        return Ok(lr);
    }
    let planes = if seq.mono_chrome { 1 } else { 3 };
    for kind in &mut lr.kind[..planes] {
        // Remap_Lr_Type.
        *kind = match r.read_bits(2)? {
            0 => LrType::None,
            1 => LrType::Switchable,
            2 => LrType::Wiener,
            _ => LrType::SgrProj,
        };
    }
    if !lr.any() {
        return Ok(lr);
    }
    // lr_unit_shift, plus lr_unit_extra_shift with 64x64 superblocks.
    let first = r.read_bits(1)?;
    let shift = if seq.use_128x128_superblock {
        first + 1
    } else if first == 1 {
        first + r.read_bits(1)?
    } else {
        0
    };
    lr.unit_size[0] = RESTORATION_TILESIZE_MAX >> (2 - shift);
    let chroma_lr = lr.kind[1..].iter().any(|&k| k != LrType::None);
    let uv_shift = usize::from(chroma_lr && r.read_bit()?);
    lr.unit_size[1] = lr.unit_size[0] >> uv_shift;
    lr.unit_size[2] = lr.unit_size[1];
    Ok(lr)
}
