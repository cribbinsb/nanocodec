//! Motion-vector difference coding (AV1 §5.11.32 `read_mv`, §5.11.33
//! `read_mv_component`), and its bit cost for mode decision.
//!
//! Ported from libaom `av1_encode_mv` / `encode_mv_component`
//! (`av1/encoder/encodemv.c`, BSD-2-Clause, Alliance for Open Media). Frames
//! set `allow_high_precision_mv = 0` and `force_integer_mv = 0`, so the
//! fractional part (`mv_fr`) is coded and the high-precision bit is not
//! (a decoder infers it as 1).
//!
//! The default CDFs are libaom's `default_nmv_context`, written out here in
//! the inverse convention used by [`crate::range_coder`] (32768 − cdf). Both
//! components start from the same defaults and then adapt separately.

use crate::cdf::{Cdf, FromDefaults, ResetCounters};
use crate::modeinfo::Av1Mv;
use crate::range_coder::SymbolWriter;

/// `mv_joint` inverse CDF: which components are non-zero.
const JOINT_CDF: [u16; MV_JOINTS] = [28672, 21504, 13440, 0];
/// `mv_class` inverse CDF (11 classes).
const CLASS_CDF: [u16; MV_CLASSES] = [4096, 1792, 910, 448, 217, 112, 28, 11, 6, 1, 0];
/// `mv_class0_bit` inverse CDF.
const CLASS0_CDF: [u16; 2] = [5120, 0];
/// `mv_bit[i]` inverse CDFs (classes above 0).
const BITS_CDF: [[u16; 2]; 10] = [
    [15360, 0],
    [14848, 0],
    [13824, 0],
    [12288, 0],
    [10240, 0],
    [8192, 0],
    [4096, 0],
    [2816, 0],
    [2816, 0],
    [2048, 0],
];
/// `mv_class0_fr[class0_bit]` inverse CDFs.
const CLASS0_FR_CDF: [[u16; MV_FR_SYMBOLS]; 2] = [[16384, 8192, 6144, 0], [20480, 11520, 8640, 0]];
/// `mv_fr` inverse CDF (classes above 0).
const FR_CDF: [u16; MV_FR_SYMBOLS] = [24576, 15360, 11520, 0];
/// `mv_sign` inverse CDF.
const SIGN_CDF: [u16; 2] = [16384, 0];

/// Size of class 0 in integer samples (libaom `CLASS0_SIZE`).
const CLASS0_SIZE: i32 = 2;
/// Magnitudes from `CLASS0_SIZE << 12` (1/8 units) up are all class 10.
const MV_CLASS_10: u32 = 10;
/// Number of classes, and of `mv_class` symbols.
const MV_CLASSES: usize = 11;
/// Symbols of `mv_joint`, `mv_class0_fr` and `mv_fr`.
const MV_JOINTS: usize = 4;
const MV_FR_SYMBOLS: usize = 4;

/// The CDFs of one vector component (AV1 `MvClassCdf[MvCtx][comp]` etc.).
#[derive(Debug, Clone)]
struct ComponentCdfs {
    sign: Cdf<2>,
    class: Cdf<MV_CLASSES>,
    class0: Cdf<2>,
    bits: [Cdf<2>; 10],
    class0_fr: [Cdf<MV_FR_SYMBOLS>; 2],
    fr: Cdf<MV_FR_SYMBOLS>,
}

impl Default for ComponentCdfs {
    fn default() -> Self {
        Self {
            sign: SIGN_CDF.cdfs(),
            class: CLASS_CDF.cdfs(),
            class0: CLASS0_CDF.cdfs(),
            bits: BITS_CDF.cdfs(),
            class0_fr: CLASS0_FR_CDF.cdfs(),
            fr: FR_CDF.cdfs(),
        }
    }
}

impl ResetCounters for ComponentCdfs {
    fn reset_counters(&mut self) {
        self.sign.reset_counters();
        self.class.reset_counters();
        self.class0.reset_counters();
        self.bits.reset_counters();
        self.class0_fr.reset_counters();
        self.fr.reset_counters();
    }
}

/// Motion-vector difference CDFs for `MvCtx` 0: the joint, and each
/// component (row first, AV1 §5.11.32).
#[derive(Debug, Clone)]
pub struct MvCdfs {
    joint: Cdf<MV_JOINTS>,
    components: [ComponentCdfs; 2],
}

impl Default for MvCdfs {
    fn default() -> Self {
        Self {
            joint: JOINT_CDF.cdfs(),
            components: [ComponentCdfs::default(), ComponentCdfs::default()],
        }
    }
}

impl ResetCounters for MvCdfs {
    fn reset_counters(&mut self) {
        self.joint.reset_counters();
        self.components.reset_counters();
    }
}

/// Which components of a difference are non-zero (AV1 `MvJoint`).
fn joint([row, col]: [i32; 2]) -> usize {
    match (row != 0, col != 0) {
        (false, false) => 0, // MV_JOINT_ZERO
        (false, true) => 1,  // MV_JOINT_HNZVZ
        (true, false) => 2,  // MV_JOINT_HZVNZ
        (true, true) => 3,   // MV_JOINT_HNZVNZ
    }
}

/// The symbols that code one non-zero component (`comp` in 1/8 samples).
struct ComponentSymbols {
    sign: bool,
    class: u32,
    /// Integer offset within the class.
    int_part: u32,
    /// Fractional part, in quarter samples (0..4).
    fr: usize,
}

impl ComponentSymbols {
    fn of(comp: i32) -> Self {
        debug_assert_ne!(comp, 0);
        let z = comp.unsigned_abs() - 1;
        // libaom `av1_get_mv_class`: class = floor(log2(z >> 3)), class 0
        // for z < 16, capped at 10.
        let class = if z >= (CLASS0_SIZE as u32) << 12 {
            MV_CLASS_10
        } else {
            (z >> 3).max(1).ilog2()
        };
        let base = if class == 0 {
            0
        } else {
            (CLASS0_SIZE as u32) << (class + 2)
        };
        let offset = z - base;
        Self {
            sign: comp < 0,
            class,
            int_part: offset >> 3,
            fr: ((offset >> 1) & 3) as usize,
        }
    }
}

impl MvCdfs {
    /// Code `mv - pred` (AV1 `read_mv` with `MvCtx` 0).
    pub fn write(&mut self, w: &mut SymbolWriter, mv: Av1Mv, pred: Av1Mv) {
        let diff = difference(mv, pred);
        w.write_symbol(joint(diff), &mut self.joint, MV_JOINTS);
        for (cdfs, comp) in self.components.iter_mut().zip(diff) {
            if comp != 0 {
                cdfs.write(w, comp);
            }
        }
    }

    /// Cost in bits of coding `mv - pred` with the CDFs as they stand.
    #[must_use]
    #[inline]
    pub fn bits(&self, mv: Av1Mv, pred: Av1Mv) -> f32 {
        let diff = difference(mv, pred);
        let mut bits = self.joint.bits(joint(diff));
        for (cdfs, comp) in self.components.iter().zip(diff) {
            if comp != 0 {
                bits += cdfs.bits(comp);
            }
        }
        bits
    }
}

/// `mv - pred` per component (row, column). Widened: the difference of two
/// valid vectors can reach twice their range.
fn difference(mv: Av1Mv, pred: Av1Mv) -> [i32; 2] {
    [
        i32::from(mv.row) - i32::from(pred.row),
        i32::from(mv.col) - i32::from(pred.col),
    ]
}

impl ComponentCdfs {
    /// `read_mv_component` (AV1 §5.11.33) for a non-zero component.
    fn write(&mut self, w: &mut SymbolWriter, comp: i32) {
        let s = ComponentSymbols::of(comp);
        w.write_bool(s.sign, &mut self.sign);
        w.write_symbol(s.class as usize, &mut self.class, MV_CLASSES);
        if s.class == 0 {
            w.write_bool(s.int_part != 0, &mut self.class0);
            w.write_symbol(
                s.fr,
                &mut self.class0_fr[s.int_part as usize],
                MV_FR_SYMBOLS,
            );
        } else {
            for (i, cdf) in self.bits[..s.class as usize].iter_mut().enumerate() {
                w.write_bool((s.int_part >> i) & 1 != 0, cdf);
            }
            w.write_symbol(s.fr, &mut self.fr, MV_FR_SYMBOLS);
        }
    }

    fn bits(&self, comp: i32) -> f32 {
        let s = ComponentSymbols::of(comp);
        let mut bits = self.sign.bits(usize::from(s.sign)) + self.class.bits(s.class as usize);
        if s.class == 0 {
            bits += self.class0.bits(s.int_part as usize);
            bits += self.class0_fr[s.int_part as usize].bits(s.fr);
        } else {
            for (i, cdf) in self.bits[..s.class as usize].iter().enumerate() {
                bits += cdf.bits(((s.int_part >> i) & 1) as usize);
            }
            bits += self.fr.bits(s.fr);
        }
        bits
    }
}

#[path = "decoder_mv.rs"]
mod decoder;

pub(crate) use decoder::{DecoderMvCdfs, MvPrecision};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classes_match_libaom() {
        // One whole sample: z = 7 -> class 0, integer part 0, fr 3.
        let s = ComponentSymbols::of(8);
        assert_eq!((s.class, s.int_part, s.fr), (0, 0, 3));
        // Two samples: z = 15 -> class 0, integer part 1.
        let s = ComponentSymbols::of(-16);
        assert!(s.sign);
        assert_eq!((s.class, s.int_part, s.fr), (0, 1, 3));
        // Three samples: z = 23 -> class 1 (base 16), offset 7.
        let s = ComponentSymbols::of(24);
        assert_eq!((s.class, s.int_part, s.fr), (1, 0, 3));
        // 64 samples: z = 511 -> class 5 (base 256), offset 255.
        let s = ComponentSymbols::of(512);
        assert_eq!((s.class, s.int_part, s.fr), (5, 31, 3));
    }

    #[test]
    fn longer_differences_cost_more() {
        let cdfs = MvCdfs::default();
        let small = cdfs.bits(Av1Mv { row: 0, col: 8 }, Av1Mv::ZERO);
        let large = cdfs.bits(
            Av1Mv {
                row: 256,
                col: -512,
            },
            Av1Mv::ZERO,
        );
        assert!(small > 1.0 && large > small);
    }
}
