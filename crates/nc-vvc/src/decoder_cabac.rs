//! H.266 §9.3.4.3 arithmetic decoding on the shared CABAC engine
//! (`nc_common::cabac::CabacReader`: a 64-bit window, branch-free
//! decisions, bypass runs as one division). H.266's two-rate context model
//! stays here: each decision derives its LPS range from the context
//! (`Context::lps`) and the engine's range, and the context adapts on the
//! outcome. Bins never fail one by one: data read past the end decodes as
//! zeros, and [`Reader::terminate`] rejects a slice that ran over.
use crate::contexts::Context;
use nc_common::cabac::{CabacReader, MAX_BYPASS_BITS};
use nc_common::decode::DecodeError;

pub(super) struct Reader<'a> {
    engine: CabacReader<'a>,
}
#[allow(
    clippy::unnecessary_wraps,
    reason = "the syntax readers' `?` call form, shared with the header parsers; \
              infallible bins inline to no check, overrun is checked at `terminate`"
)]
impl<'a> Reader<'a> {
    pub fn new(bytes: &'a [u8]) -> Result<Self, DecodeError> {
        Ok(Self {
            engine: CabacReader::new(bytes, 0)?,
        })
    }
    #[inline]
    pub fn bin(&mut self, ctx: &mut Context) -> Result<bool, DecodeError> {
        let lps = ctx.lps(self.engine.range());
        let value = ctx.mps() != self.engine.decide_lps(lps);
        ctx.update(value);
        Ok(value)
    }
    #[inline]
    pub fn bypass(&mut self) -> Result<bool, DecodeError> {
        Ok(self.engine.bypass())
    }
    #[inline]
    pub fn bypass_bits(&mut self, count: u32) -> Result<u32, DecodeError> {
        if count >= MAX_BYPASS_BITS {
            return Err(DecodeError::Malformed("CABAC bypass width"));
        }
        Ok(self.engine.bypass_bits(count))
    }
    /// `end_of_slice_segment_flag` and the like; a slice whose bins ran
    /// past its data is rejected here.
    pub fn terminate(&mut self) -> Result<bool, DecodeError> {
        let end = self.engine.terminate();
        if self.engine.overrun() {
            return Err(DecodeError::Truncated);
        }
        Ok(end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cabac::CabacWriter;
    #[test]
    fn arithmetic_bins_round_trip() {
        let mut w = CabacWriter::new();
        let mut c = Context::new(45, 6, 28);
        let values: Vec<_> = (0..600).map(|i| (i * 19 + i / 7) % 11 < 4).collect();
        for (i, &v) in values.iter().enumerate() {
            w.encode_bin(&mut c, v);
            w.encode_bypass((i % 8) as u32, 3);
        }
        w.encode_terminate(true);
        let mut out = w.finish();
        out.trailing_bits();
        let bytes = out.into_bytes();
        let mut r = Reader::new(&bytes).unwrap();
        let mut c = Context::new(45, 6, 28);
        for (i, &v) in values.iter().enumerate() {
            assert_eq!(r.bin(&mut c).unwrap(), v, "{i}");
            assert_eq!(r.bypass_bits(3).unwrap(), (i % 8) as u32);
        }
        assert!(r.terminate().unwrap());
    }
}
