//! Independent AV1 signed motion-difference parsing (AV1 §5.11.32
//! `read_mv`, §5.11.33 `read_mv_component`) with the decoder's own MV CDFs
//! (the encoder's [`super::MvCdfs`] carry symbol-cost caches and no
//! high-precision bits).
use super::{
    Av1Mv, BITS_CDF, CLASS_CDF, CLASS0_CDF, CLASS0_FR_CDF, FR_CDF, JOINT_CDF, MV_CLASSES,
    MV_FR_SYMBOLS, MV_JOINTS, SIGN_CDF,
};
use crate::cdf::ResetCounters;
use crate::decoder_range::{Cdf, DecoderDefaults, SymbolReader};
use nc_common::decode::DecodeError;

/// `mv_class0_hp` and `mv_hp` inverse CDFs (libaom `default_nmv_context`;
/// our encoder never allows high precision, so only the decoder has them).
const CLASS0_HP_CDF: [u16; 2] = [12288, 0];
const HP_CDF: [u16; 2] = [16384, 0];
/// `mv_bit[i]`: one per integer bit of the largest class.
const MV_BITS: usize = 10;
/// `mv_fr` of a vector coded in whole samples (`force_integer_mv`).
const INTEGER_FR: i32 = 3;

/// The precision differences are coded at (§5.11.32 `MvCtx` aside).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MvPrecision {
    /// `force_integer_mv`: whole samples, no fraction coded.
    Integer,
    /// Quarter samples (`allow_high_precision_mv` 0).
    Quarter,
    /// Eighth samples.
    Eighth,
}

/// The decoder's CDFs of one vector component (`MvClassCdf[MvCtx][comp]`
/// etc.).
#[derive(Debug, Clone)]
struct ComponentCdfs {
    sign: Cdf<2>,
    class: Cdf<MV_CLASSES>,
    class0: Cdf<2>,
    bits: [Cdf<2>; MV_BITS],
    class0_fr: [Cdf<MV_FR_SYMBOLS>; 2],
    fr: Cdf<MV_FR_SYMBOLS>,
    class0_hp: Cdf<2>,
    hp: Cdf<2>,
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
            class0_hp: CLASS0_HP_CDF.cdfs(),
            hp: HP_CDF.cdfs(),
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
        self.class0_hp.reset_counters();
        self.hp.reset_counters();
    }
}

/// The decoder's motion-vector difference CDFs for `MvCtx` 0: the joint,
/// and each component (row first, §5.11.32).
#[derive(Debug, Clone)]
pub(crate) struct DecoderMvCdfs {
    joint: Cdf<MV_JOINTS>,
    components: [ComponentCdfs; 2],
}

impl Default for DecoderMvCdfs {
    fn default() -> Self {
        Self {
            joint: JOINT_CDF.cdfs(),
            components: [ComponentCdfs::default(), ComponentCdfs::default()],
        }
    }
}

impl ResetCounters for DecoderMvCdfs {
    fn reset_counters(&mut self) {
        self.joint.reset_counters();
        self.components.reset_counters();
    }
}

impl DecoderMvCdfs {
    /// `read_mv`: a difference at `precision` added to `pred`.
    pub(crate) fn read(
        &mut self,
        r: &mut SymbolReader<'_>,
        pred: Av1Mv,
        precision: MvPrecision,
    ) -> Result<Av1Mv, DecodeError> {
        let joint = r.symbol(&mut self.joint, 4);
        let row = if joint & 2 != 0 {
            self.components[0].read(r, precision)
        } else {
            0
        };
        let col = if joint & 1 != 0 {
            self.components[1].read(r, precision)
        } else {
            0
        };
        Av1Mv::checked(i32::from(pred.row) + row, i32::from(pred.col) + col)
            .ok_or(DecodeError::Malformed("AV1 motion vector range"))
    }
}

impl ComponentCdfs {
    fn read(&mut self, r: &mut SymbolReader<'_>, precision: MvPrecision) -> i32 {
        let negative = r.boolean(&mut self.sign);
        let class = r.symbol(&mut self.class, MV_CLASSES);
        let class0 = class == 0;
        let integer = if class0 {
            i32::from(r.boolean(&mut self.class0))
        } else {
            let mut integer = 0;
            for i in 0..class {
                integer |= i32::from(r.boolean(&mut self.bits[i])) << i;
            }
            integer
        };
        let (fraction, high) = if precision == MvPrecision::Integer {
            (INTEGER_FR, 1)
        } else {
            let fr = if class0 {
                &mut self.class0_fr[integer as usize]
            } else {
                &mut self.fr
            };
            let fraction = r.symbol(fr, MV_FR_SYMBOLS) as i32;
            // Without high precision the eighth bit is inferred as 1.
            let high = if precision == MvPrecision::Eighth {
                let hp = if class0 {
                    &mut self.class0_hp
                } else {
                    &mut self.hp
                };
                i32::from(r.boolean(hp))
            } else {
                1
            };
            (fraction, high)
        };
        let base = if class0 { 0 } else { 2 << (class + 2) };
        let magnitude = base + ((integer << 3) | (fraction << 1) | high) + 1;
        if negative { -magnitude } else { magnitude }
    }
}
