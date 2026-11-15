//! H.266 temporal motion vector prediction (TMVP): the collocated
//! picture's motion field, kept at the 8x8 granularity the collocated
//! derivation reads (§8.5.2.11: the block covering `((x >> 3) << 3,
//! (y >> 3) << 3)`), and a CU's collocated candidate for the regular merge
//! list (§8.5.2.2) and AMVP (§8.5.2.8).
//!
//! Collocated vectors are read at reduced precision: each component keeps
//! a 6-bit mantissa and a 4-bit exponent (§8.5.2.12; `VVdeC`
//! `roundMvComp`), applied here when a picture's motion is stored.
//!
//! Temporal layers use the shared engine reference slots; candidates are scaled
//! by current and collocated reference distances (§8.5.2.12). A field keeps
//! its picture's POC distance to each of its active references; only a
//! picture with two (the second reference) also keeps each block's
//! reference index, so a single-reference picture stores and reads its
//! vectors alone.
//!
//! Derived from `VVdeC` `UnitTools::getColocatedMVP` and `DecCu`'s
//! collocated motion store (BSD-3-Clause-Clear; `THIRD_PARTY.md`).
use crate::motion::{ActiveReferences, MAX_ACTIVE_REFERENCES, Mv16};
use nc_common::decode::{DecodeError, Result};
use nc_encoder::Block;

/// log2 of the collocated motion granularity (§8.5.2.11).
pub(crate) const FIELD_LOG2: u32 = 3;
/// §8.5.2.2, §8.5.2.8 (`VVdeC`: width + height > 12): CUs of this many luma
/// samples or fewer (4x8, 8x4) have no collocated candidate.
const MAX_AREA_WITHOUT_TEMPORAL: usize = 32;
/// Bits of a stored vector component's signed mantissa (§8.5.2.12).
const MANTISSA_BITS: u32 = 6;

/// A collocated vector (at storage precision) and the POC distance from
/// its picture to the reference it pointed into (`colPocDiff`).
pub(crate) type Collocated = (Mv16, u32);

/// The motion of a coded picture as a later picture's collocated picture
/// reads it: one vector per 8x8 block, the vector of its top-left 4x4
/// sub-block (at storage precision); `None` where that is intra.
#[derive(Debug, Clone, Default)]
pub(crate) struct MotionField {
    /// The coded picture's luma size (`pps_pic_width/height_in_luma_samples`).
    width: usize,
    height: usize,
    /// 8x8 blocks per row.
    cols: usize,
    mvs: Vec<Option<Mv16>>,
    /// Each block's reference index (`refIdxCol`), kept only when the
    /// picture had more than one active reference: empty, every vector
    /// points into reference 0.
    ref_indices: Vec<u8>,
    /// The picture's POC distance to each of its active references.
    distances: [u32; MAX_ACTIVE_REFERENCES],
}

impl MotionField {
    /// An all-intra field for a `width` x `height` coded picture.
    pub(crate) fn new(width: usize, height: usize) -> Self {
        let cols = width.div_ceil(1 << FIELD_LOG2);
        Self {
            width,
            height,
            cols,
            mvs: vec![None; cols * height.div_ceil(1 << FIELD_LOG2)],
            ref_indices: Vec::new(),
            distances: ActiveReferences::SINGLE.distances,
        }
    }

    /// Start storing a newly coded picture's motion, the picture having the
    /// active `references`: the entries to set to each 8x8 block's vector
    /// in raster order, at storage precision ([`at_storage_precision`]). A
    /// picture of more than one reference then stores its blocks'
    /// reference indices ([`Self::store_ref_indices`]).
    pub(crate) fn vectors_to_store(
        &mut self,
        references: &ActiveReferences,
    ) -> &mut [Option<Mv16>] {
        self.ref_indices.clear();
        self.distances = references.distances;
        &mut self.mvs
    }

    /// Store a newly coded picture's motion: each 8x8 block's vector in
    /// raster order, as [`Self::vectors_to_store`].
    #[cfg(test)]
    fn store(
        &mut self,
        vectors: impl IntoIterator<Item = Option<Mv16>>,
        references: &ActiveReferences,
    ) {
        for (entry, mv) in self.vectors_to_store(references).iter_mut().zip(vectors) {
            *entry = mv.map(at_storage_precision);
        }
    }

    /// Each 8x8 block's reference index in raster order (0 for intra), for
    /// a picture of more than one active reference.
    pub(crate) fn store_ref_indices(&mut self, ref_indices: impl IntoIterator<Item = u8>) {
        self.ref_indices
            .extend(ref_indices.into_iter().take(self.mvs.len()));
    }

    /// The stored motion of the 8x8 block covering luma sample `(x, y)`.
    fn at(&self, x: usize, y: usize) -> Option<Collocated> {
        let index = (y >> FIELD_LOG2) * self.cols + (x >> FIELD_LOG2);
        let mv = self.mvs[index]?;
        let ref_idx = self.ref_indices.get(index).copied().unwrap_or(0);
        Some((mv, self.distances[usize::from(ref_idx)]))
    }

    /// §8.5.2.11: the collocated vector of CU `cu` in a picture of
    /// `1 << ctu_log2` CTUs: the [`Self::bottom_right`] one, else the block
    /// at the CU's centre, else none.
    #[inline]
    pub(crate) fn candidate(&self, cu: &Block, ctu_log2: u8) -> Option<Collocated> {
        if cu.area() <= MAX_AREA_WITHOUT_TEMPORAL {
            return None;
        }
        self.bottom_right(cu, ctu_log2)
            .or_else(|| self.at(cu.x + cu.width() / 2, cu.y + cu.height() / 2))
    }

    /// The vector of the block below and right of CU `cu`, if that lies in
    /// the same CTU row and inside the picture and is inter (§8.5.2.11; the
    /// affine bottom-right control point alone, §8.5.5.6).
    #[inline]
    pub(crate) fn bottom_right(&self, cu: &Block, ctu_log2: u8) -> Option<Collocated> {
        let (right, below) = (cu.x + cu.width(), cu.y + cu.height());
        (below >> ctu_log2 == cu.y >> ctu_log2 && right < self.width && below < self.height)
            .then(|| self.at(right, below))
            .flatten()
    }
}

/// Vector `v` at the collocated storage precision.
pub(crate) fn at_storage_precision(v: Mv16) -> Mv16 {
    Mv16 {
        x: storage_precision(v.x),
        y: storage_precision(v.y),
    }
}

/// Vector component `c` at the collocated storage precision: rounded
/// (half up) to `MANTISSA_BITS` significant bits; components within
/// ±2^`MANTISSA_BITS` are exact. `VVdeC`'s mantissa-exponent encoding and
/// decoding (`convertMvFixedToFloat`, `convertMvFloatToFixed`) round trip
/// to this.
fn storage_precision(c: i32) -> i32 {
    // |c| for c >= 0, |c| - 1 below: the magnitude the mantissa must hold.
    let magnitude = (c ^ (c >> 31)) as u32;
    let mantissa_max = (1 << (MANTISSA_BITS - 1)) - 1;
    let shift = ((magnitude | mantissa_max).ilog2() + 1).saturating_sub(MANTISSA_BITS);
    ((c + ((1 << shift) >> 1)) >> shift) << shift
}

/// Normative short-term temporal scaling, shared by encoder and decoder.
pub(crate) fn scale(vector: Mv16, target: i32, source: i32) -> Result<Mv16> {
    if target == source {
        return Ok(vector.clip());
    }
    let (target, source) = (target.clamp(-128, 127), source.clamp(-128, 127));
    if source == 0 {
        return Err(DecodeError::Malformed("VVC zero temporal distance"));
    }
    let reciprocal = (16384 + (source / 2).abs()) / source;
    let factor = ((target * reciprocal + 32) >> 6).clamp(-4096, 4095);
    let component = |v: i32| {
        let product = v * factor;
        ((product + 128 - i32::from(product >= 0)) >> 8).clamp(-131_072, 131_071)
    };
    Ok(Mv16 {
        x: component(vector.x),
        y: component(vector.y),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `VVdeC`'s mantissa-exponent round trip, step by step.
    fn reference_round_trip(val: i32) -> i32 {
        let sign = val >> 31;
        let scale = ((val ^ sign) | 31).ilog2() as i32 - 5;
        let (exponent, mantissa) = if scale >= 0 {
            let n = (val + ((1 << scale) >> 1)) >> scale;
            (scale + ((n ^ sign) >> 5), (n & 31) | (sign * 32))
        } else {
            (0, val)
        };
        let packed = exponent | (mantissa * 16);
        let (exponent, mantissa) = (packed & 15, packed >> 4);
        if exponent == 0 {
            mantissa
        } else {
            (mantissa ^ 32) * (1 << (exponent - 1))
        }
    }

    #[test]
    fn storage_precision_matches_the_mantissa_exponent_round_trip() {
        for c in -64..64 {
            assert_eq!(storage_precision(c), c);
        }
        assert_eq!(storage_precision(-1000), -992);
        for c in -(1 << 17)..(1 << 17) {
            assert_eq!(storage_precision(c), reference_round_trip(c), "{c}");
        }
    }

    #[test]
    fn bottom_right_then_centre() {
        let mut field = MotionField::new(64, 64);
        let single = ActiveReferences::SINGLE;
        let vector = |x| Some(Mv16 { x, y: 0 });
        let mv = |x| Some((Mv16 { x, y: 0 }, 1));
        let cols = 64 >> FIELD_LOG2;
        // The 8x8 block at (16, 16) and the centre block of the 16x16 CU at
        // (0, 0).
        let mut entries = vec![None; cols * cols];
        entries[2 * cols + 2] = vector(4);
        entries[cols + 1] = vector(8);
        // The bottom-right block of a CU on the last row of its CTU row is
        // never read: (16, 32) is in the next 32x32 CTU row.
        entries[4 * cols + 2] = vector(12);
        entries[3 * cols + 1] = vector(16);
        field.store(entries.iter().copied(), &single);
        let cu = Block::square(0, 0, 4);
        assert_eq!(field.candidate(&cu, 5), mv(4));
        assert_eq!(field.candidate(&Block::square(0, 16, 4), 5), mv(16));
        // An intra bottom-right block falls back to the centre.
        entries[2 * cols + 2] = None;
        field.store(entries, &single);
        assert_eq!(field.candidate(&cu, 5), mv(8));
        // Nor outside the picture.
        assert_eq!(field.candidate(&Block::square(48, 48, 4), 6), None);
    }

    #[test]
    fn only_a_two_reference_picture_keeps_reference_indices() {
        let mut field = MotionField::new(16, 16);
        let v = Mv16 { x: 4, y: 0 };
        let both = ActiveReferences {
            distances: [1, 3],
            count: 2,
        };
        field.store([Some(v); 4], &both);
        field.store_ref_indices([0, 1, 0, 0]);
        assert_eq!(field.at(8, 0), Some((v, 3)));
        assert_eq!(field.at(0, 8), Some((v, 1)));
        // The next single-reference picture's vectors all span its one
        // distance.
        field.store([Some(v); 4], &ActiveReferences::SINGLE);
        assert_eq!(field.at(8, 0), Some((v, 1)));
    }
}
