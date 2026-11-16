//! H.266 §9.3 binary arithmetic encoder with dual probability estimates.
//!
//! The range coder is the engine shared with HEVC
//! ([`nc_common::cabac_encoder`], kvazaar, BSD-3-Clause).
//! VVC probability model follows `VVenC` (BSD-3-Clause-Clear); see `THIRD_PARTY.md`.
use crate::contexts::Context;
use nc_common::BitWriter;
use nc_common::cabac_encoder::CabacEncoder;

/// Encodes bins into one slice's `slice_data()`: the shared engine
/// ([`CabacEncoder`]) with H.266's two-rate model.
#[derive(Debug, Default)]
pub struct CabacWriter(CabacEncoder);

impl CabacWriter {
    /// A fresh encoder (H.266 §9.3.2.5, encoder side).
    #[must_use]
    pub fn new() -> Self {
        Self(CabacEncoder::new())
    }

    /// Encode one context-coded bin (`EncodeDecision`, H.266 §9.3.4.3.2).
    pub fn encode_bin(&mut self, ctx: &mut Context, bin: bool) {
        let lps = ctx.lps(self.0.range());
        let mps = bin == ctx.mps();
        ctx.update(bin);
        self.0.encode_decision(lps, mps);
    }

    /// Encode equiprobable bins (`EncodeBypass`, H.266 §9.3.4.3.4): the
    /// `count` low bits of `value`, most significant first.
    pub fn encode_bypass(&mut self, value: u32, count: u32) {
        self.0.encode_bypass(value, count);
    }

    /// Encode a terminating bin (`EncodeTerminate`, H.266 §9.3.4.3.5), e.g.
    /// `end_of_slice_flag`.
    pub fn encode_terminate(&mut self, bin: bool) {
        self.0.encode_terminate(bin);
    }

    /// Flush the encoder after the terminating bin of the slice
    /// (`EncodeFlush`, H.266 §9.3.4.3.5) and return the bits written. The
    /// caller appends `rbsp_slice_trailing_bits()`.
    #[must_use]
    pub fn finish(self) -> BitWriter {
        self.0.finish()
    }
}
