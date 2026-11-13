//! Motion vector prediction for macroblock partitions (H.264 §8.4.1.3) and
//! the `P_Skip` vector (§8.4.1.1). Every partition of a macroblock uses the
//! macroblock's one reference index (`MbInfo::ref_idx`; 0 unless the slice
//! has the engine's second reference), so a neighbour's index is its
//! macroblock's.
//!
//! Partitions are at least 8x8 (no sub-8x8 shapes), so neighbouring
//! partitions are located at 8x8 granularity: quadrant `(qx, qy)` of the
//! current macroblock, with `-1` / `2` reaching into the neighbouring
//! macroblocks (§6.4.11.7).

use nc_encoder::Mv;

use crate::mbinfo::{MbInfo, MbPartition, Neighbours};

/// A neighbouring partition as seen by MV prediction (§8.4.1.3.2), in the
/// spec's form: `mvL0N` and `refIdxL0N`, both zero / -1 for an intra or
/// absent (outside the picture, or not yet decoded) partition, plus whether
/// it is available, which only the rules of §6.4.11.7 and §8.4.1.3.1 read.
/// Plain fields rather than an enum: prediction runs for most macroblocks,
/// and comparing enum variants cost more than the median itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Neighbour {
    mv: Mv,
    /// `refIdxL0N`: -1 for intra and absent partitions.
    ref_idx: i16,
    available: bool,
}

/// `refIdxL0N` of an intra or absent partition.
const NO_REFERENCE: i16 = -1;

impl Neighbour {
    const UNAVAILABLE: Self = Self {
        mv: Mv::ZERO,
        ref_idx: NO_REFERENCE,
        available: false,
    };

    /// An inter partition of the current macroblock.
    fn inter(mv: Mv, ref_idx: u8) -> Self {
        Self {
            mv,
            ref_idx: i16::from(ref_idx),
            available: true,
        }
    }

    /// Quadrant `(qx, qy)` of a neighbouring macroblock.
    fn of(mb: Option<&MbInfo>, qx: usize, qy: usize) -> Self {
        match mb {
            None => Self::UNAVAILABLE,
            Some(mb) if mb.is_intra() => Self {
                available: true,
                ..Self::UNAVAILABLE
            },
            Some(mb) => Self::inter(mb.mvs[qy * 2 + qx], mb.ref_idx),
        }
    }

    /// Whether `refIdxL0N` equals `ref_idx`.
    fn has_ref(self, ref_idx: u8) -> bool {
        self.ref_idx == i16::from(ref_idx)
    }
}

/// The partition covering quadrant `(qx, qy)` relative to the current
/// macroblock, whose quadrants decoded so far have the vectors in `own`
/// and reference index `ref_idx`.
fn neighbour_at(
    nb: &Neighbours<'_>,
    own: &[Option<Mv>; 4],
    ref_idx: u8,
    qx: isize,
    qy: isize,
) -> Neighbour {
    match (qx, qy) {
        (0..=1, 0..=1) => own[(qy * 2 + qx) as usize]
            .map_or(Neighbour::UNAVAILABLE, |mv| Neighbour::inter(mv, ref_idx)),
        (-1, 0..=1) => Neighbour::of(nb.left, 1, qy as usize),
        (0..=1, -1) => Neighbour::of(nb.above, qx as usize, 1),
        (2, -1) => Neighbour::of(nb.above_right, 0, 1),
        (-1, -1) => Neighbour::of(nb.above_left, 1, 1),
        // To the right of the macroblock: not decoded yet.
        _ => Neighbour::UNAVAILABLE,
    }
}

fn median(a: i16, b: i16, c: i16) -> i16 {
    a.max(b).min(a.min(b).max(c))
}

/// The predicted vector `mvpL0` of partition `part` of a macroblock of
/// shape `shape` predicting from reference index `ref_idx` (§8.4.1.3),
/// given the vectors of the quadrants of this macroblock decoded so far
/// (`own`, `None` where not yet decoded).
pub fn predict_mv(
    nb: &Neighbours<'_>,
    own: &[Option<Mv>; 4],
    shape: MbPartition,
    part: usize,
    ref_idx: u8,
) -> Mv {
    let (x8, y8, w8, _) = shape.rect(part);
    let (x8, y8, w8) = (x8 as isize, y8 as isize, w8 as isize);
    let at = |qx, qy| neighbour_at(nb, own, ref_idx, qx, qy);
    let a = at(x8 - 1, y8);
    let b = at(x8, y8 - 1);
    // C is replaced by D when C is not available (§6.4.11.7).
    let c = match at(x8 + w8, y8 - 1) {
        n if !n.available => at(x8 - 1, y8 - 1),
        n => n,
    };
    // Directional prediction of 16x8 and 8x16 partitions (§8.4.1.3).
    let directional = match (shape, part) {
        (MbPartition::P16x8, 0) => Some(b),
        (MbPartition::P16x8, _) | (MbPartition::P8x16, 0) => Some(a),
        (MbPartition::P8x16, _) => Some(c),
        _ => None,
    };
    if let Some(n) = directional.filter(|n| n.has_ref(ref_idx)) {
        return n.mv;
    }
    median_prediction(a, b, c, ref_idx)
}

/// The predicted vector of a `P_L0_16x16` partition predicting from
/// `ref_idx`: [`predict_mv`] for that shape, with its neighbours A (left),
/// B (above) and C (above right, else D above left) read straight from the
/// neighbouring macroblocks. Mode decision asks for it for most
/// macroblocks; the general form spent most of its time locating the
/// neighbours.
pub fn predict_16x16(nb: &Neighbours<'_>, ref_idx: u8) -> Mv {
    let a = Neighbour::of(nb.left, 1, 0);
    let b = Neighbour::of(nb.above, 0, 1);
    let c = match nb.above_right {
        Some(_) => Neighbour::of(nb.above_right, 0, 1),
        None => Neighbour::of(nb.above_left, 1, 1),
    };
    median_prediction(a, b, c, ref_idx)
}

/// The prediction from neighbours A, B and C once no directional rule
/// applies (§8.4.1.3, §8.4.1.3.1): the one neighbour using `ref_idx` if
/// exactly one does, otherwise the component-wise median.
fn median_prediction(a: Neighbour, mut b: Neighbour, mut c: Neighbour, ref_idx: u8) -> Mv {
    // Only A available: B and C take A's values (§8.4.1.3.1).
    if !b.available && !c.available && a.available {
        b = a;
        c = a;
    }
    match [a, b, c].map(|n| n.has_ref(ref_idx)) {
        [true, false, false] => return a.mv,
        [false, true, false] => return b.mv,
        [false, false, true] => return c.mv,
        _ => {}
    }
    let (a, b, c) = (a.mv, b.mv, c.mv);
    Mv {
        x: median(a.x, b.x, c.x),
        y: median(a.y, b.y, c.y),
    }
}

/// The `P_Skip` vector (§8.4.1.1, reference index 0): zero at the
/// picture's top or left edge or next to a stationary neighbour predicting
/// from index 0 (A: left of the top-left block, B: above it), otherwise the
/// 16x16 predicted vector.
#[cfg(test)]
pub fn skip_mv(nb: &Neighbours<'_>) -> Mv {
    skip_mv_from(nb, || predict_16x16(nb, 0))
}

/// [`skip_mv`] with the 16x16 predicted vector supplied by `mvp16`, called
/// only when the skip vector is that prediction (not zero), so an encoder
/// can compute and keep it lazily.
pub fn skip_mv_from(nb: &Neighbours<'_>, mvp16: impl FnOnce() -> Mv) -> Mv {
    let a = Neighbour::of(nb.left, 1, 0);
    let b = Neighbour::of(nb.above, 0, 1);
    if !a.available || !b.available {
        return Mv::ZERO;
    }
    let still = Neighbour::inter(Mv::ZERO, 0);
    if a == still || b == still {
        return Mv::ZERO;
    }
    mvp16()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mbinfo::MbKind;

    fn inter(mv: Mv) -> MbInfo {
        MbInfo {
            kind: MbKind::Inter,
            cbp_luma: 0,
            cbp_chroma: 0,
            luma_dc_coded: false,
            luma_coded: 0,
            chroma_dc_coded: [false; 2],
            chroma_ac_coded: [0; 2],
            mvs: [mv; 4],
            abs_mvd: [[0; 2]; 4],
            transform_8x8: false,
            chroma_pred_mode: crate::predict::ChromaMode::Dc,
            intra_modes: [2; 16],
            qp: 0,
            ref_idx: 0,
        }
    }

    fn predict_16x16(nb: &Neighbours<'_>) -> Mv {
        let mv = super::predict_16x16(nb, 0);
        assert_eq!(mv, predict_mv(nb, &[None; 4], MbPartition::P16x16, 0, 0));
        mv
    }

    #[test]
    fn median_of_three_neighbours() {
        let (a, b, c) = (
            inter(Mv { x: 4, y: -8 }),
            inter(Mv { x: 12, y: 0 }),
            inter(Mv { x: 8, y: 4 }),
        );
        let nb = Neighbours {
            left: Some(&a),
            above: Some(&b),
            above_right: Some(&c),
            above_left: None,
        };
        assert_eq!(predict_16x16(&nb), Mv { x: 8, y: 0 });
    }

    #[test]
    fn top_row_uses_the_left_vector() {
        let a = inter(Mv { x: 5, y: 3 });
        let nb = Neighbours {
            left: Some(&a),
            ..Neighbours::default()
        };
        assert_eq!(predict_16x16(&nb), Mv { x: 5, y: 3 });
        assert_eq!(skip_mv(&nb), Mv::ZERO, "no B: skip vector is zero");
    }

    #[test]
    fn single_inter_neighbour_is_taken_directly() {
        let mut intra = inter(Mv::ZERO);
        intra.kind = MbKind::Intra16x16;
        let b = inter(Mv { x: -6, y: 2 });
        let nb = Neighbours {
            left: Some(&intra),
            above: Some(&b),
            above_right: Some(&intra),
            above_left: None,
        };
        assert_eq!(predict_16x16(&nb), Mv { x: -6, y: 2 });
    }

    #[test]
    fn partitions_predict_directionally() {
        let (a, b) = (inter(Mv { x: 4, y: 0 }), inter(Mv { x: 0, y: 8 }));
        let nb = Neighbours {
            left: Some(&a),
            above: Some(&b),
            ..Neighbours::default()
        };
        // 16x8: top from B, bottom from A.
        assert_eq!(
            predict_mv(&nb, &[None; 4], MbPartition::P16x8, 0, 0),
            Mv { x: 0, y: 8 }
        );
        let own = [Some(Mv { x: 0, y: 8 }), Some(Mv { x: 0, y: 8 }), None, None];
        assert_eq!(
            predict_mv(&nb, &own, MbPartition::P16x8, 1, 0),
            Mv { x: 4, y: 0 }
        );
        // 8x16: left from A; right from C, here unavailable, so D (above).
        assert_eq!(
            predict_mv(&nb, &[None; 4], MbPartition::P8x16, 0, 0),
            Mv { x: 4, y: 0 }
        );
    }

    #[test]
    fn neighbours_with_another_reference_index_are_not_taken_directly() {
        let a = inter(Mv { x: 4, y: 0 });
        let mut b = inter(Mv { x: 20, y: 8 });
        b.ref_idx = 1;
        let c = inter(Mv { x: 8, y: 4 });
        let nb = Neighbours {
            left: Some(&a),
            above: Some(&b),
            above_right: Some(&c),
            above_left: None,
        };
        // Index 1: only B uses it, so its vector is the prediction.
        assert_eq!(
            predict_mv(&nb, &[None; 4], MbPartition::P16x16, 0, 1),
            Mv { x: 20, y: 8 }
        );
        // Index 0: two neighbours use it, so the median of all three.
        assert_eq!(predict_16x16(&nb), Mv { x: 8, y: 4 });
    }

    #[test]
    fn the_16x16_form_matches_the_general_one() {
        // Every mix of absent, intra and inter neighbours (index 0 or 1)
        // with distinct vectors, for both reference indices.
        let kinds = |k: usize, mv: Mv| -> Option<MbInfo> {
            let mut m = inter(mv);
            match k {
                0 => return None,
                1 => m.kind = MbKind::Intra16x16,
                2 => {}
                _ => m.ref_idx = 1,
            }
            Some(m)
        };
        for code in 0..4usize.pow(4) {
            let digit = |n: usize| (code / 4usize.pow(n as u32)) % 4;
            let mbs: [Option<MbInfo>; 4] = core::array::from_fn(|n| {
                let v = n as i16;
                kinds(
                    digit(n),
                    Mv {
                        x: 4 * v - 6,
                        y: 7 - 3 * v,
                    },
                )
            });
            let nb = Neighbours {
                left: mbs[0].as_ref(),
                above: mbs[1].as_ref(),
                above_right: mbs[2].as_ref(),
                above_left: mbs[3].as_ref(),
            };
            for ref_idx in 0..2 {
                assert_eq!(
                    super::predict_16x16(&nb, ref_idx),
                    predict_mv(&nb, &[None; 4], MbPartition::P16x16, 0, ref_idx),
                    "neighbour kinds {code:o}, ref_idx {ref_idx}"
                );
            }
        }
    }
}
