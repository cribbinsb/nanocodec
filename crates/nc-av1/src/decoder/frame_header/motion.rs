//! Reference-order syntax of the frame header: short reference signalling
//! (AV1 §7.8), skip mode frames (§5.9.22) and global motion parameters
//! (§5.9.24–25).
use super::{
    ALTREF_FRAME, FrameHeader, GOLDEN_FRAME, LAST_FRAME, NUM_REF_FRAMES, REFS_PER_FRAME, TOTAL_REFS,
};
use crate::backend::decoder::sequence::Sequence;
use nc_common::decode::{BitReader, DecodeError};

/// `WARPEDMODEL_PREC_BITS`.
pub(crate) const WARPEDMODEL_PREC_BITS: u32 = 16;
/// `GM_ABS_ALPHA_BITS`, `GM_ALPHA_PREC_BITS`, `GM_ABS_TRANS_ONLY_BITS`,
/// `GM_TRANS_ONLY_PREC_BITS`, `GM_ABS_TRANS_BITS`, `GM_TRANS_PREC_BITS`.
const GM_ABS_ALPHA_BITS: u32 = 12;
const GM_ALPHA_PREC_BITS: u32 = 15;
const GM_ABS_TRANS_ONLY_BITS: u32 = 9;
const GM_TRANS_ONLY_PREC_BITS: u32 = 3;
const GM_ABS_TRANS_BITS: u32 = 12;
const GM_TRANS_PREC_BITS: u32 = 6;
/// Reference names filled from forward frames, in order (`Ref_Frame_List`).
const REF_FRAME_LIST: [usize; REFS_PER_FRAME - 2] = [2, 3, 5, 6, 7];
const BWDREF_FRAME: usize = 5;
const ALTREF2_FRAME: usize = 6;

/// `GmType` (AV1 §6.8.17), ordered by model complexity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub(crate) enum WarpType {
    #[default]
    Identity,
    Translation,
    RotZoom,
    Affine,
}

/// One reference's global motion: type and `gm_params[]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GlobalMotion {
    pub kind: WarpType,
    pub params: [i32; 6],
}
impl GlobalMotion {
    pub(crate) const IDENTITY: Self = Self {
        kind: WarpType::Identity,
        params: [
            0,
            0,
            1 << WARPEDMODEL_PREC_BITS,
            0,
            0,
            1 << WARPEDMODEL_PREC_BITS,
        ],
    };
}

/// `get_relative_dist(a, b)` (§7.12.4... §5.9.2 order hints).
pub(crate) fn relative_distance(seq: &Sequence, a: u32, b: u32) -> i32 {
    if !seq.enable_order_hint {
        return 0;
    }
    let diff = a as i32 - b as i32;
    let m = 1 << (seq.order_hint_bits - 1);
    (diff & (m - 1)) - (diff & m)
}

/// `set_frame_refs()` (§7.8): `ref_frame_idx[]` from LAST and GOLDEN.
pub(crate) fn set_frame_refs(
    seq: &Sequence,
    order_hint: u32,
    slot_hints: &[u32; NUM_REF_FRAMES],
    last: u8,
    gold: u8,
) -> [u8; REFS_PER_FRAME] {
    let mut refs: [Option<u8>; REFS_PER_FRAME] = [None; REFS_PER_FRAME];
    refs[0] = Some(last);
    refs[GOLDEN_FRAME - LAST_FRAME] = Some(gold);
    let mut used = [false; NUM_REF_FRAMES];
    used[usize::from(last)] = true;
    used[usize::from(gold)] = true;
    let current = 1 << (seq.order_hint_bits - 1);
    let shifted: [i32; NUM_REF_FRAMES] =
        std::array::from_fn(|i| current + relative_distance(seq, slot_hints[i], order_hint));
    // The unused slot whose hint is best by `better`, among those `keep`s.
    let mut pick = |keep: &dyn Fn(i32) -> bool, better: &dyn Fn(i32, i32) -> bool| {
        let mut best: Option<usize> = None;
        for i in 0..NUM_REF_FRAMES {
            if !used[i] && keep(shifted[i]) && best.is_none_or(|b| better(shifted[i], shifted[b])) {
                best = Some(i);
            }
        }
        if let Some(i) = best {
            used[i] = true;
        }
        best.map(|i| i as u8)
    };
    let backward = |hint: i32| hint >= current;
    let forward = |hint: i32| hint < current;
    // ALTREF: the latest backward frame; BWDREF, ALTREF2: the earliest.
    refs[ALTREF_FRAME - LAST_FRAME] = pick(&backward, &|a, b| a >= b);
    refs[BWDREF_FRAME - LAST_FRAME] = pick(&backward, &|a, b| a < b);
    refs[ALTREF2_FRAME - LAST_FRAME] = pick(&backward, &|a, b| a < b);
    for name in REF_FRAME_LIST {
        if refs[name - LAST_FRAME].is_none() {
            refs[name - LAST_FRAME] = pick(&forward, &|a, b| a >= b);
        }
    }
    // The rest: the earliest frame of all, used or not.
    let mut earliest = 0;
    for i in 1..NUM_REF_FRAMES {
        if shifted[i] < shifted[earliest] {
            earliest = i;
        }
    }
    refs.map(|r| r.unwrap_or(earliest as u8))
}

/// `skip_mode_params()` (§5.9.22): the two `SkipModeFrame[]` if skip mode
/// is allowed (the caller reads `skip_mode_present`).
pub(crate) fn skip_mode_frames(
    seq: &Sequence,
    h: &FrameHeader,
    slot_hints: &[u32; NUM_REF_FRAMES],
) -> Option<[u8; 2]> {
    if h.frame_type.is_intra() || !h.reference_select || !seq.enable_order_hint {
        return None;
    }
    let dist = |a: u32, b: u32| relative_distance(seq, a, b);
    let hint = |i: usize| slot_hints[usize::from(h.ref_frame_idx[i])];
    let mut forward: Option<usize> = None;
    let mut backward: Option<usize> = None;
    for i in 0..REFS_PER_FRAME {
        let d = dist(hint(i), h.order_hint);
        if d < 0 {
            if forward.is_none_or(|f| dist(hint(i), hint(f)) > 0) {
                forward = Some(i);
            }
        } else if d > 0 && backward.is_none_or(|b| dist(hint(i), hint(b)) < 0) {
            backward = Some(i);
        }
    }
    let forward = forward?;
    let second = if let Some(b) = backward {
        b
    } else {
        let mut second: Option<usize> = None;
        for i in 0..REFS_PER_FRAME {
            if dist(hint(i), hint(forward)) < 0 && second.is_none_or(|s| dist(hint(i), hint(s)) > 0)
            {
                second = Some(i);
            }
        }
        second?
    };
    let (lo, hi) = (forward.min(second), forward.max(second));
    Some([(LAST_FRAME + lo) as u8, (LAST_FRAME + hi) as u8])
}

/// `global_motion_params()` (§5.9.24) of an inter frame; `previous` is
/// `PrevGmParams`.
pub(crate) fn global_motion(
    r: &mut BitReader<'_>,
    previous: &[GlobalMotion; TOTAL_REFS],
    allow_high_precision_mv: bool,
) -> Result<[GlobalMotion; TOTAL_REFS], DecodeError> {
    let mut out = [GlobalMotion::IDENTITY; TOTAL_REFS];
    for name in LAST_FRAME..=ALTREF_FRAME {
        let kind = warp_type(r)?;
        let gm = &mut out[name];
        gm.kind = kind;
        let prev = &previous[name].params;
        let mut read = |idx: usize| -> Result<i32, DecodeError> {
            read_global_param(r, kind, idx, prev[idx], allow_high_precision_mv)
        };
        if kind >= WarpType::RotZoom {
            gm.params[2] = read(2)?;
            gm.params[3] = read(3)?;
            if kind == WarpType::Affine {
                gm.params[4] = read(4)?;
                gm.params[5] = read(5)?;
            } else {
                gm.params[4] = -gm.params[3];
                gm.params[5] = gm.params[2];
            }
        }
        if kind >= WarpType::Translation {
            gm.params[0] = read(0)?;
            gm.params[1] = read(1)?;
        }
    }
    Ok(out)
}

/// `is_global`, `is_rot_zoom`, `is_translation` (§5.9.24).
fn warp_type(r: &mut BitReader<'_>) -> Result<WarpType, DecodeError> {
    if !r.read_bit()? {
        return Ok(WarpType::Identity);
    }
    if r.read_bit()? {
        return Ok(WarpType::RotZoom);
    }
    Ok(if r.read_bit()? {
        WarpType::Translation
    } else {
        WarpType::Affine
    })
}

/// `read_global_param()` (§5.9.25).
fn read_global_param(
    r: &mut BitReader<'_>,
    kind: WarpType,
    idx: usize,
    previous: i32,
    allow_high_precision_mv: bool,
) -> Result<i32, DecodeError> {
    let (abs_bits, prec_bits) = if idx >= 2 {
        (GM_ABS_ALPHA_BITS, GM_ALPHA_PREC_BITS)
    } else if kind == WarpType::Translation {
        let low = u32::from(!allow_high_precision_mv);
        (GM_ABS_TRANS_ONLY_BITS - low, GM_TRANS_ONLY_PREC_BITS - low)
    } else {
        (GM_ABS_TRANS_BITS, GM_TRANS_PREC_BITS)
    };
    let prec_diff = WARPEDMODEL_PREC_BITS - prec_bits;
    let diagonal = idx % 3 == 2;
    let round = if diagonal {
        1 << WARPEDMODEL_PREC_BITS
    } else {
        0
    };
    let sub = if diagonal { 1 << prec_bits } else { 0 };
    let mx = 1i32 << abs_bits;
    let reference = (previous >> prec_diff) - sub;
    let value = signed_subexp_with_ref(r, -mx, mx + 1, reference)?;
    Ok((value << prec_diff) + round)
}

/// `decode_signed_subexp_with_ref(low, high, r)`.
fn signed_subexp_with_ref(
    r: &mut BitReader<'_>,
    low: i32,
    high: i32,
    reference: i32,
) -> Result<i32, DecodeError> {
    let x = unsigned_subexp_with_ref(r, (high - low) as u32, (reference - low) as u32)?;
    Ok(x as i32 + low)
}

fn unsigned_subexp_with_ref(
    r: &mut BitReader<'_>,
    mx: u32,
    reference: u32,
) -> Result<u32, DecodeError> {
    let v = subexp(r, mx)?;
    Ok(if (reference << 1) <= mx {
        inverse_recenter(reference, v)
    } else {
        mx - 1 - inverse_recenter(mx - 1 - reference, v)
    })
}

/// `decode_subexp(numSyms)`.
fn subexp(r: &mut BitReader<'_>, num_syms: u32) -> Result<u32, DecodeError> {
    const K: u32 = 3;
    let mut i = 0;
    let mut mk = 0;
    loop {
        let b2 = if i > 0 { K + i - 1 } else { K };
        let a = 1 << b2;
        if num_syms <= mk + 3 * a {
            return Ok(super::tiles::read_ns(r, num_syms - mk)? + mk);
        }
        if !r.read_bit()? {
            return Ok(r.read_bits(b2)? + mk);
        }
        i += 1;
        mk += a;
    }
}

/// `inverse_recenter(r, v)`.
fn inverse_recenter(r: u32, v: u32) -> u32 {
    if v > 2 * r {
        v
    } else if v & 1 != 0 {
        r - ((v + 1) >> 1)
    } else {
        r + (v >> 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recentering_alternates_around_the_reference() {
        let values: Vec<u32> = (0..8).map(|v| inverse_recenter(3, v)).collect();
        assert_eq!(values, [3, 2, 4, 1, 5, 0, 6, 7]);
    }

    #[test]
    fn subexponential_codes_small_values_in_four_bits() {
        // more_bits 0, then subexp_bits = 5 in 3 bits.
        let mut r = BitReader::new(&[0b0101_0000]);
        assert_eq!(subexp(&mut r, 1 << 13).unwrap(), 5);
    }
}
