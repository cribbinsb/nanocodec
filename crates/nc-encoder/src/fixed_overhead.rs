//! The fixed per-frame overhead of inter frames (`rate.fixed_overhead`): the
//! bits a frame costs whatever its quantizer (frame and tile headers, OBU
//! framing, end-of-frame symbols), which the rate model would otherwise read
//! as content that scales with the step.
//!
//! Rate control models a frame's bits as `m · hint / step^e`. On a still
//! scene the smallest inter frames are 26-42 bytes, almost all overhead, so
//! a byte or three more per frame (an AV1 OBU extension when layers are
//! configured, `av1.frame_ids`) inflated the fitted `m`, the level drifted
//! coarser from frame 30-70, and six static clips lost 0.01-0.71 dB PSNR-Y
//! at equal bytes (unified-structure probes, 2026-11-15). With the overhead
//! `F` the model is `F + m · hint / step^e`: the content model learns from
//! `bits − F` and the level is solved for `budget − F`.
//!
//! `F` is learnt only from inter frames whose blocks are all but entirely
//! unchanged, where the content part is near zero: it follows their
//! smallest size down at once and rises slowly, so a refining still frame
//! does not lift it. Busy content never updates it and keeps `F = 0`.

/// Share of unchanged 8x8 blocks from which an inter frame's size is taken
/// as mostly overhead.
const STILL_SHARE: f64 = 0.98;
/// Fraction of the gap to a larger still frame the estimate rises by per
/// frame: a quiet still frame rarely sits above the overhead by much, a
/// refining one by a lot, so the estimate tracks the low end.
const RISE: f64 = 1. / 32.;
/// Largest overhead estimate, as a share of the uniform frame budget: what
/// is left must still plan the frame's content.
const MAX_BUDGET_SHARE: f64 = 0.5;

/// The learnt overhead of one stream (0 while off or not yet observed).
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct FixedOverhead {
    enabled: bool,
    bits: f64,
}

impl FixedOverhead {
    pub fn new(enabled: bool) -> Self {
        Self { enabled, bits: 0. }
    }

    /// The overhead to add to an inter frame's prediction, in bits.
    pub fn bits(&self) -> f64 {
        self.bits
    }

    /// An inter frame of `bits` whose share of unchanged blocks was
    /// `still_share`, in a stream whose uniform frame budget is `uniform`.
    pub fn observe(&mut self, bits: f64, still_share: f64, uniform: f64) {
        if !self.enabled || still_share < STILL_SHARE {
            return;
        }
        let bits = bits.min(MAX_BUDGET_SHARE * uniform);
        self.bits = if self.bits <= 0. || bits < self.bits {
            bits
        } else {
            self.bits + RISE * (bits - self.bits)
        };
    }

    /// Forget the estimate (the picture size changed, and so did its headers).
    pub fn reset(&mut self) {
        self.bits = 0.;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learns_the_low_end_of_still_frames_only() {
        let mut overhead = FixedOverhead::new(true);
        overhead.observe(5000., 0.5, 10_000.);
        assert_eq!(overhead.bits(), 0.);
        overhead.observe(300., 1., 10_000.);
        overhead.observe(260., 1., 10_000.);
        assert_eq!(overhead.bits(), 260.);
        // A refining still frame lifts it only slowly.
        overhead.observe(4260., 1., 10_000.);
        assert_eq!(overhead.bits(), 260. + RISE * 4000.);
        let mut off = FixedOverhead::new(false);
        off.observe(260., 1., 10_000.);
        assert_eq!(off.bits(), 0.);
    }
}
